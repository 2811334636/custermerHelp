# 电商智能客服系统 · ch03 设计文档（知识库与向量检索）

- 日期：2026-10-19
- 状态：待用户评审
- 前置：ch01（纯对话）、ch02（Function Calling 工具链）已交付并合入 `main`

---

## 1. 目标

给客服系统建知识库，把 `query_faq` 的内部实现从**关键词查表**升级成 **Milvus dense 向量语义检索**，**工具的入参出参契约保持不变**。

交付五件事：

1. **文档处理**：知识 Markdown 按标题层级做结构感知切分，超长递归切，chunk 间加重叠且裁到最近句末标点，表格按行切且每块复述表头
2. **对话挖知识**：独立 CLI 定时任务，分批喂 LLM 抽取问答对，先进暂存表、再整体去重入库
3. **落库结构**：`knowledge_chunks`（MySQL 权威源）带 `category / questions / answer` 与四类只存不向量的元数据
4. **双写落库**：先写 MySQL 记【待向量化】→ 写 Milvus → 回填 `vector_id` → 【已向量化】；按主键幂等，挂了能重跑
5. **在线检索**：问题向量化 → Milvus Top-k → 替换 `query_faq` 的关键词查表

## 2. 非目标（明确不做）

- **关键词召回、混合检索、重排** —— 本章只跑 dense 向量单路
- 文档编辑后的**孤儿向量清理**（见 §14.1）
- 真实业务文档 —— 本章用样例 Markdown 跑通（用户确认）
- 多租户知识库隔离、权限控制

## 3. 前置条件与实测结论

### 3.1 Embedding 上游（本章新增）

- 端点：`https://api.siliconflow.cn/v1`（OpenAI 兼容）
- 模型：`BAAI/bge-m3`
- **实测：HTTP 200，约 0.84s，返回 1024 维向量**，响应形状为 OpenAI 标准的 `{object, data:[{embedding}], model, usage}`
- 现有 DeepSeek 上游 `/embeddings` 返回 **404**，不能用于向量化

### 3.2 语义检索可行性实测（本章设计的经验基础）

**这份数据决定了验收 1 能不能过，必须记录。**

#### 反例：用 ch02 的 12 条短问答当知识库 → 失败

```
查询「邮费是多少」
  0.5853  [咨询] 发什么快递        ← 错的排第一
  0.5651  [售后] 换货运费谁承担    ← 该召回的排第二
```

原因：`换货运费谁承担` 的答案里只有"运费"，从不出现"邮费"；而 `发什么快递` 与"邮费"共享"物流配送"语义。**短问答的措辞太贫瘠。**

#### 正例：用贴近真实手册的 chunk → 通过

用 `售后政策 > 退换货 > 运费\n运费说明\n因商品质量问题导致的退换货，往返运费由平台承担……`（正文**只出现"运费"、不出现"邮费"**）：

```
「邮费是多少」
   0.6376  运费说明      ← ✅ top1，领先第二名 0.069
   0.5679  发什么快递
```

同一批 chunk 的其他查询也都排对：`退货政策是什么` → 0.6989、`怎么开发票` → 0.7502、`什么时候发货` → 0.7540。

#### 查询形态对效果的影响

| 查询形态 | top1 分数 | 领先第二名 |
|---|---|---|
| 2 字「邮费」 | 0.6350 | +0.0504 |
| 4 字「退货运费」 | 0.7874 | **+0.1345** |
| 完整问句「邮费是多少」 | 0.6376 | +0.0691 |
| 口语换说法「退货要自己出邮费吗」 | 0.7015 | +0.0822 |

**结论**：四种都能排对，但 2 字核心词的领先幅度只有约一半。这是 §9.2 改 `query_faq` docstring 指引的直接依据。

#### 混合场景：短 FAQ 与手册 chunk 共存（最终确认）

用户决定把 ch02 的 12 条短 FAQ 也作为知识来源导入（§5.5）。但上一小节的反例表明这些短问答单独使用时会把"邮费"判错，所以必须验证**共存时**验收 1 是否仍成立。

实测（14 条：12 条短 FAQ + 2 条手册 chunk）：

```
「邮费是多少」  0.6377  运费说明[MD]      ← ✅ 仍排第一
                0.5856  发什么快递（短 FAQ）
                0.5681  快递说明[MD]
「运费怎么算」  0.6823  运费说明[MD]      ← ✅ 仍排第一
                0.6074  换货运费谁承担（短 FAQ）
```

**结论：共存不破坏验收 1。** 内容丰富的手册 chunk 稳定压过措辞贫瘠的短问答。§5.5 的"把 12 条 FAQ 作为来源之一导入"这个决策是安全的。

### 3.3 依赖版本（实测解析）

`pymilvus==3.0.2`、`langchain-milvus==0.4.0`（不使用）、`milvus-lite==3.2.2`（不使用）、`grpcio==1.84.0`、`protobuf==7.36.2`。

### 3.4 环境

- 宿主 **3306 被占用**（ch02 已确认），MySQL 容器映射到 **3307**
- Milvus 走 **Standalone（Docker）**，需要 milvus + etcd + minio 三个服务；Docker 可用内存 7.6 GB
- HuggingFace 可达但不使用本地模型

## 4. 架构与模块划分

### 4.1 目录结构

```
app/knowledge/                  ← 新增：知识库子系统
├── __init__.py
├── splitter.py                 ★ 自研 Markdown 结构感知切分器（纯函数）
├── chunk.py                    Chunk 数据类（切分器输出契约）
├── embedding.py                bge-m3 客户端（langchain_openai.OpenAIEmbeddings）
├── vector_store.py             ★ Milvus 封装：ensure / upsert / search / count
├── builder.py                  ★ 双写编排与状态机
├── retrieval.py                ★ 在线检索
└── sources/
    ├── __init__.py
    ├── markdown.py             目录扫描 → Chunk[]
    ├── faq_table.py            faq 表 → Chunk[]
    └── mined.py                挖掘暂存表 → Chunk[]

app/repositories/
└── knowledge_chunk.py          新增：knowledge_chunks / mined_qa_staging 读写

jobs/                           ← 新增：离线任务（独立 CLI，cron 调）
├── __init__.py
├── build_knowledge.py          uv run python -m jobs.build_knowledge
└── mine_knowledge.py           uv run python -m jobs.mine_knowledge

knowledge/                      ← 新增：样例 Markdown 文档
├── 售后政策.md
├── 商品FAQ.md
└── 平台规则.md

app/tools/knowledge.py          ← 改：改调 retrieval（签名不变，只改 docstring）
app/db/models.py                ← 改：加 KnowledgeChunk + MinedQaStaging
scripts/schema.sql              ← 改：加两张表
docker-compose.yml              ← 改：加 milvus + etcd + minio
```

### 4.2 分层边界

| 模块 | 知道什么 | 不知道什么 |
|---|---|---|
| `splitter.py` | Markdown 语法、标题栈、句子边界 | 数据库、Milvus、embedding |
| `sources/` | 从某种来源读出 `Chunk` | 向量化、Milvus |
| `builder.py` | 双写顺序、状态机、幂等 | Markdown 怎么切、向量怎么算 |
| `vector_store.py` | Milvus 的集合/索引/字段 | 业务语义 |
| `retrieval.py` | 一次检索的完整流程 | 工具层、HTTP |
| `app/tools/knowledge.py` | 只调 `retrieval.search(query)` | Milvus、embedding、切分 |

### 4.3 客户端选型

- **Embedding**：`langchain_openai.OpenAIEmbeddings`，`base_url` 指向 SiliconFlow。**Task 0 必须先实测它能正确打到该端点**（OpenAI 兼容不等价于一切字段都兼容）。
- **Milvus**：`pymilvus` **直连**，不用 `langchain-milvus`。

  **理由（实测依据）**：`langchain-milvus` 的官方文档示例是 `auto_id=True` + `add_documents`，**重跑会重复插入**，直接破坏验收 2；而 `pymilvus` 的 `client.upsert(collection, data)` 原生按主键幂等。本章的硬需求正好落在高层封装掉的那一层。

## 5. 切分规则

### 5.1 行扫描 → 块识别

| 块类型 | 识别 | 切分策略 |
|---|---|---|
| heading | `#`~`######` | 不入 chunk，只更新**标题栈** |
| paragraph | 连续非空行 | 超长时递归降级（§5.2） |
| list | 连续 `-` / `*` / `1.` | 同 paragraph |
| **table** | 连续以 `\|` 开头的行 | **按行切，每块复述表头** |
| code fence | ` ``` ` 包围 | 整体一块，**绝不切** |
| blockquote | 连续 `>` | 整体一块；`> [!IMPORTANT]` 同时置 `is_key_clause=true` |

### 5.2 超长递归切 —— 三条硬规则

参数：`CHUNK_MAX_CHARS=400`、`CHUNK_OVERLAP_CHARS=80`（中文按字计，可配）。

```
① 块内按段落边界（空行）切
② 单段仍超长 → 按句子边界（。！？；）切
③ 单句仍超长 → 才在字数上限处硬切
```

**硬规则：任何切点都不许落在句子中间。** 在 `MAX_CHARS` 处取候选窗口后，**从切点往回找最近的句末标点**；找不到才允许硬切。

**重叠**：从**前一块的某个句子起点**开始，不用"第 N 个字符"。重叠部分也必须是完整句子。

### 5.3 表格切分

```
原表：| 商品 | 尺码 | 库存 |
      |---|---|---|
      | 跑步鞋 | 42 | 12 |
      | 运动袜 | L  | 88 |

块1 = "| 商品 | 尺码 | 库存 |\n|---|---|---|\n| 跑步鞋 | 42 | 12 |"
块2 = "| 商品 | 尺码 | 库存 |\n|---|---|---|\n| 运动袜 | L  | 88 |"
```

每条数据行独立成块，**表头（含分隔行）必须复述**。

### 5.4 `chunk_id` 必须确定性

```
chunk_id = f"{source_key}:{ordinal}"
例： "售后政策.md:12"   "faq:97"   "mined:123"
```

同一份输入、同样的切分参数 → 同样的 chunk_id → `upsert` 命中同一主键 → 幂等。

**这是验收 2 成立的前提，是硬约束。** 随机 UUID 会让每次重跑都插入重复向量。

### 5.5 三源的字段填充

| 来源 | `questions` | `category` | `answer` |
|---|---|---|---|
| **policy**（Markdown） | 叶子标题，如「运费说明」 | 上级标题路径，如「售后政策 > 退换货」 | 该块正文（表格块含表头） |
| **faq**（存量表） | `faq.question` 原文 | `faq.category` | `faq.answer` |
| **mined**（挖掘） | LLM 抽出的 question | LLM 抽出的分类 | LLM 抽出的 answer |

`content_type` 与来源前缀一致，不单独推导。

### 5.6 四类元数据的来源

| 元数据 | 推导规则 | 例子 |
|---|---|---|
| `section_path` | 切分时维护的标题栈，` > ` 连接 | `售后政策 > 退换货 > 运费` |
| `content_type` | 来自知识来源 | `policy` / `faq` / `mined` |
| `is_key_clause` | 识别 `> [!IMPORTANT]` 或以 `**关键条款**` 开头的段落 | `true` / `false` |
| `prev_chunk_id` / `next_chunk_id` | 同一文档内的切分顺序；首块 prev=null，末块 next=null | 块 #3：prev=#2、next=#4 |

**这四者全部由文档结构推导，确定性、可单测、不花 token。**

## 6. 数据模型

### 6.1 MySQL（用户提供方向，本章落地）

```sql
CREATE TABLE IF NOT EXISTS knowledge_chunks (
  chunk_id      VARCHAR(64)  NOT NULL COMMENT '确定性主键：source:ordinal',
  category      VARCHAR(255) NOT NULL,
  questions     VARCHAR(512) NOT NULL,
  answer        TEXT         NOT NULL,
  section_path  VARCHAR(512) NOT NULL,
  content_type  ENUM('policy','faq','mined') NOT NULL,
  is_key_clause TINYINT(1)   NOT NULL DEFAULT 0,
  prev_chunk_id VARCHAR(64)  NULL,
  next_chunk_id VARCHAR(64)  NULL,
  status        ENUM('待向量化','已向量化') NOT NULL DEFAULT '待向量化',
  vector_id     VARCHAR(64)  NULL COMMENT '回填的 Milvus 主键',
  content_hash  CHAR(64)     NOT NULL COMMENT 'embedding 文本的 sha256',
  created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (chunk_id),
  KEY idx_status (status),
  KEY idx_content_type (content_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS mined_qa_staging (
  id              BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  question        VARCHAR(512) NOT NULL,
  answer          TEXT         NOT NULL,
  category        VARCHAR(64)  NOT NULL,
  conversation_id BIGINT UNSIGNED NULL,
  status          ENUM('待入库','已入库','已丢弃') NOT NULL DEFAULT '待入库',
  created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 6.2 Milvus 集合

```python
schema = client.create_schema(auto_id=False)
schema.add_field("id",     DataType.VARCHAR, is_primary=True, max_length=64)
schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)

index_params = IndexParams()
index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
```

**只存 `id` + `vector`。** 四类元数据留在 MySQL：它们只存不进向量，本章也不做标量过滤，放进 Milvus 没有消费方（YAGNI）。

### 6.3 向量化文本

```
f"{category}\n{questions}\n{answer}"
```

`content_hash = sha256(该文本)`。

> **注意**：§3.2 的实测表明，这个拼法在**内容足够丰富**时可用（`邮费是多少` → 运费说明 top1 0.6376）。但同一拼法用在 ch02 那 12 条短问答上会失败。**这是内容质量问题，不是拼法问题**——样例文档与真实文档都必须把政策写完整。

## 7. 双写状态机与断点续跑

```
① 三源切分 → Chunk[]
② UPSERT 进 MySQL：
     新块                      → status='待向量化'
     已存在且 content_hash 相同 → 保持原 status
     已存在但 content_hash 变了 → status='待向量化'
③ SELECT WHERE status='待向量化' → 待办集
④ 分批（每批 16 条）embedding
⑤ Milvus upsert(id=chunk_id, vector=...)
⑥ MySQL UPDATE status='已向量化', vector_id=chunk_id
```

**`content_hash` 是幂等的前提之一**：没有它，每次重建都会把已向量化的块重置为待向量化，等于每次全量重跑。

**中断重跑矩阵**：

| 中断点 | 重跑行为 |
|---|---|
| ② 后、⑤ 前 | 仍是【待向量化】→ 被 ③ 捞出补齐 |
| ⑤ 后、⑥ 前 | 仍是【待向量化】→ 重新向量化并 upsert；id 相同 ⇒ 覆盖而非重复 |
| ⑤ 一批中途 | 已写的留下，未写的下次补；已写的再 upsert 无害 |
| ② 中途 | 该批 MySQL 写入在一个事务内 ⇒ 全成或全回滚，下次幂等重写 |

## 8. 对话挖知识

```
jobs/mine_knowledge.py
  ① 选未处理会话（已有 staging 行的会话跳过 ⇒ 重跑不重复挖）
  ② 分批（默认 10 个会话/批）喂 LLM，抽 (question, answer, category)
  ③ 写 mined_qa_staging，status='待入库'
  ④ 整体去重入库：
       规范化文本（去空白/标点/大小写）→ 批内去重 + 与 knowledge_chunks 现有记录去重
       存活者 → knowledge_chunks（content_type='mined'，chunk_id='mined:{staging_id}'）
       staging 行标 '已入库' / '已丢弃'
  ⑤ 新 chunk 保持 status='待向量化'，由下一次 build_knowledge 捡起
```

**去重是确定的**（规范化后精确比较），可单测。语义近似去重不在本章范围。

**抽取的 Prompt 是纯 Prompt 任务**，按用户要求用标注样例验证，不写语义单测。

调度形态：**独立 CLI + cron**（用户确认）。应用进程不承载长时间任务。

## 9. 在线检索与 `query_faq`

### 9.1 检索流程

```python
async def search(query: str, top_k: int = 3) -> list[Chunk]:
    qv = await embed([query])
    hits = client.search("knowledge", data=[qv], limit=top_k, output_fields=["id"])
    ids = [h["id"] for h in hits[0]]
    rows = await repo.get_by_ids(ids)
    return rows_in_milvus_order(rows, ids)     # 必须保持 Milvus 的相关性顺序
```

### 9.2 `query_faq`（契约不变）

```python
@tool
async def query_faq(keyword: str) -> str:      # 签名一个字不改
    """按语义检索常见问题与政策手册。

    keyword 请传**用户的完整问题或其中的核心说法**——
    例如「邮费是多少」「退货运费谁承担」「怎么开发票」。
    """
```

**只改 docstring 的依据**：§3.2 实测——完整问句/口语换说法的领先幅度（+0.069/+0.082）大于 2 字核心词（+0.050），而旧 docstring 写着「不要传整句话」，在向量检索下是**反向指引**。

**返回格式不变**：`问：{questions}\n答：{answer}`，多条用空行连接。

### 9.3 Milvus 故障降级

Milvus 不可用时**不抛异常**，返回「知识库暂时不可用，请稍后再试或转人工」。

理由：与现有「查不到不是错误」的原则一致；抛异常会让 `ToolRegistry` 把一次基础设施故障当成工具故障重试 3 次，并让模型拿到 `upstream_unavailable` 式的错误码。

## 10. 配置

```ini
APP_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
APP_EMBEDDING_API_KEY=sk-...
APP_EMBEDDING_MODEL=BAAI/bge-m3
APP_EMBEDDING_DIM=1024
APP_MILVUS_URI=http://127.0.0.1:19530
APP_MILVUS_COLLECTION=knowledge
APP_RETRIEVAL_TOP_K=3
APP_CHUNK_MAX_CHARS=400
APP_CHUNK_OVERLAP_CHARS=80
APP_EMBED_BATCH_SIZE=16
```

`.env` 不进 git；`.env.example` 只放占位符。

## 11. 错误处理

| 故障 | 表现 |
|---|---|
| Milvus 连不上 | `query_faq` 返回降级文案（§9.3），不抛异常 |
| Embedding 端点失败（建库时） | 该批不写状态 → 仍是【待向量化】→ 下次重跑补齐 |
| Embedding 端点失败（在线检索） | 同上，返回降级文案 |
| MySQL 写入失败 | 该批事务回滚，下次幂等重写 |
| 切分器遇到非法 Markdown | 该文件记日志并跳过，**不中断整个建库** |
| 挖掘任务 LLM 返回非法 JSON | 该批标记失败并跳过，不写 staging |

## 12. 测试与评估策略

### 12.1 TDD（确定性、可单测）

| 模块 | 测什么 |
|---|---|
| `splitter.py` | 标题栈；超长切分**每个 chunk 以句末标点结尾**；表格每块**含表头**；重叠从句子起点开始；`> [!IMPORTANT]` → `is_key_clause=true`；代码块不被切开 |
| `chunk.py` | `chunk_id` 确定性：同输入两次相同；不同来源不冲突 |
| `sources/` | faq 表 → chunk 字段映射；Markdown 目录扫描 |
| `builder.py` | 新增→待向量化；hash 未变→不改状态；hash 变→重置；中断后只处理待办 |
| `retrieval.py` | 假 vector store：结果按分数排序；空结果；Milvus 故障降级 |
| `dedupe` | 规范化规则；批内与跨批去重 |

### 12.2 集成测试（真 MySQL + 真 Milvus + 真 embedding）

- 建库 → 断言 MySQL 与 Milvus 条数一致
- 中断重跑 → 验收 2

### 12.3 评估集（不可单测的部分）

`evals/kb_cases.yaml` —— **20 条标注查询**，每条标注期望命中的章节标题，断言 Top-3 内包含：

```yaml
- query: 邮费是多少
  expect_section: 运费说明        # ← 验收 1
- query: 退货运费谁承担
  expect_section: 运费说明
- query: 七天无理由怎么算
  expect_section: 退货政策
- query: 发票丢了能重开吗
  expect_section: 发票
```

**这是本章唯一验证检索质量的手段。**

### 12.4 挖知识抽取的验证

5 段带已知问答对的会话样本，跑抽取，比对抽出的 Q&A 是否命中（措辞可不同，事实要一致）。作为评估项，不写语义单测。

## 13. 验收标准对应

| 验收 | 复现方式 |
|---|---|
| 1. 「邮费是多少」能召回运费说明并答对 | `evals/kb_cases.yaml` 中该条 Top-3 含「运费说明」；再经 `/api/chat` 端到端确认回答正确 |
| 2. 中断建库再重跑，漏向量化的块能补齐 | 处理 N 条后主动中断 → 重跑 → 断言 `status='已向量化'` 数 == 总数，且 **Milvus 实体数 == MySQL 条数（无重复）** |

## 14. 已知限制

### 14.1 孤儿向量不清理

文档编辑后某节变短，后面块的 ordinal 顺移，旧 `chunk_id` 对应的向量会留在 Milvus 里继续被检索到。

本章不做删除：正确清理需要「文档 → 该文档所有 chunk_id」的反向索引 + 差集删除，属于知识库运维的独立课题。

**本章的幂等只承诺"同样输入重跑不重复"，不承诺"文档编辑后自动清理旧块"。**

### 14.2 其他

- 只用 dense 单路，不做关键词/混合/重排
- 短查询（2 字）也能用，但领先幅度约为完整问句的一半
- 知识库内容质量直接决定召回质量（§3.2 的实测对比）

## 15. 未决项与风险

| 项 | 性质 | 处置 |
|---|---|---|
| `OpenAIEmbeddings` 是否完全兼容 SiliconFlow | **阻塞性** | Task 0 实测；不通则退回裸 `openai` SDK 的 `client.embeddings.create` |
| Milvus Standalone 三容器的启动耗时与稳定性 | 中 | Task 0 验证起停；必要时改用 `milvusdb/milvus:v2.x` 的 standalone 镜像 |
| 样例文档能否复现 §3.2 的召回效果 | 中 | 评估集 20 条会暴露；若召不回，先修文档内容再调切分 |
| 挖掘任务抽取质量 | 中 | 用 5 段标注会话验证 |
| `pymilvus 3.0.2` 与 Milvus 服务端版本匹配 | 中 | Task 0 对齐版本；版本不匹配是常见故障 |
| 孤儿向量 | 已知 | §14.1，本章不处理 |
