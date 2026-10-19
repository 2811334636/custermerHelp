# 混合检索与重排系统设计

## 1. 项目背景与目标

### 1.1 当前状态

ch03 已实现基于 Milvus dense 向量的单路语义检索：
- 知识来源：Markdown 文档、FAQ 表、对话挖掘
- 检索方式：纯 dense 向量（bge-m3，1024维）
- Top-k：默认 3
- 已知限制：换说法召回能力受限、无关键词精确匹配、无重排

### 1.2 本章目标

将检索质量提升一个台阶：
1. **混合检索**：dense 向量 + BM25 全文检索，RRF 融合
2. **重排**：bge-reranker-v2-m3 精排 Top-10
3. **生成质量控制**：引用溯源、拒答机制、低置信度问题池
4. **评估体系**：标注评估集 + 四策略对比（纯向量/纯BM25/混合/混合+Rerank）
5. **前端配套**：引用编号可点、满意度反馈

## 2. 技术选型

### 2.1 混合检索

**Milvus 原生 BM25**（Milvus 2.5+ 支持）：
- `knowledge_chunks.content` 列挂 BM25 函数
- 中文分词：`chinese` analyzer（内置）
- Dense + BM25 各召回 Top-50
- `hybrid_search()` + RRF 融合

### 2.2 重排模型

**bge-reranker-v2-m3**：
- 输入：(query, passage) 对
- 输出：相关性分数
- 精排：Top-50 → Top-10

### 2.3 评估指标

| 阶段 | 指标 |
|---|---|
| 检索 | Recall@K、MRR（Mean Reciprocal Rank） |
| 生成 | Faithfulness（答案是否基于召回的知识） |

## 3. 功能需求详解

### 3.1 混合检索

**Milvus 版本确认**：3.0.0（已验证支持 BM25、hybrid_search、RRFRanker）

**Dense 召回**（现有）：
- embedding → Milvus COSINE 检索
- Top-50

**BM25 召回**（新增）：
- Milvus 3.0 原生 BM25 支持（通过 `sparse_vector` 字段存储 BM25 向量）
- 中文分词：使用 `"cn"` analyzer（Milvus 内置，别名指向 chinese analyzer）
- 在 collection schema 中需要：
  1. 添加 `sparse_vector` 字段（`SPARSE_FLOAT_VECTOR` 类型）
  2. 添加 `Function` 将 `content` 字段自动转换为 BM25 向量存入 `sparse_vector`
- Top-50

**RRF 融合**：
```python
from pymilvus import AnnSearchRequest, RRFRanker

# Dense 请求
dense_req = AnnSearchRequest(
    data=[query_embedding],
    anns_field="vector",
    param={"metric_type": "COSINE"},
    limit=50
)

# BM25 请求（直接传文本，Function 自动转换）
sparse_req = AnnSearchRequest(
    data=[query_text],
    anns_field="sparse_vector",
    param={
        "metric_type": "BM25",
        "analyzer_name": "cn"  # 中文分词
    },
    limit=50
)

# 混合检索
results = client.hybrid_search(
    collection_name="knowledge",
    reqs=[dense_req, sparse_req],
    ranker=RRFRanker(k=60),  # RRF 参数（可调，初始值 60）
    limit=50,
    output_fields=["chunk_id", "content", ...]
)
```

**实施要点**：
- `sparse_vector` 字段由 Milvus Function 自动维护，建库时无需手动计算 BM25 向量
- 检索时传入文本 query，Milvus 自动转换为 BM25 向量后检索
- 现有 `build_knowledge.py` 需要更新：建表时添加 Function 定义

### 3.2 Query 改写

**执行逻辑**：每次检索前先执行改写，改写成功则替换原 query 进行检索

**口语归一化**：
- "邮费咋算" → "邮费如何计算"
- "能退吗" → "退货政策"

**同义词扩展**（本章暂不实现，留作后续优化）：
- 检索侧：query 扩展为多个同义表达，分别检索后合并
- 不在入库侧拆成多份（避免存储膨胀）

**实现方式**：
- LLM 驱动的 query rewrite（单轮调用，prompt 要求归一化口语表达）
- 改写失败时回退到原 query（不阻塞检索）
- 同义词词典暂不实现

### 3.3 元数据过滤

支持按 `category` 等字段先过滤再检索：
```python
search(
    query="退货运费",
    filter_expr='category == "售后政策"'
)
```

**执行时机**：过滤在 Milvus 检索层应用（`filter=` 参数），不是检索后在 Python 侧过滤

**降级策略**：
- 如果过滤后结果为空，**不自动放宽过滤**（保持语义明确性）
- 返回空结果并触发拒答机制

### 3.4 重排

**输入**：混合检索的 Top-50
**模型**：bge-reranker-v2-m3
**输出**：Top-10

**Prompt 组装优化**：
- 最相关的放首尾（首因效应 + 近因效应）
- 不按分数顺序堆中间

排列策略（10 个块）：
```
[1st, 3rd, 5th, 7th, 9th, 10th, 8th, 6th, 4th, 2nd]
```
即：最相关在最前，次相关在最后，中间按交替降序排列。

### 3.5 生成质量控制

#### 3.5.1 引用编号

**生成侧**：
- System Prompt 要求模型在回答中标注引用编号 `[1]`、`[2]`
- 编号对应召回的 chunk 顺序（重排后）

**模型不标注或标注错误的处理**：
- 不做后处理修正（避免引入新错误）
- 信任模型输出（模型在 System Prompt 指导下标注准确率高）
- 如果回答中没有任何引用编号，`metadata.citations` 为空数组（不阻塞回答）

**存储**：
- 每条 message 存 `metadata` JSON：
  ```json
  {
    "citations": [
      {"index": 1, "chunk_id": "售后政策.md:1", "score": 0.85},
      {"index": 2, "chunk_id": "商品FAQ.md:3", "score": 0.78}
    ]
  }
  ```

**前端渲染**：
- 解析回答中的 `[数字]` 模式
- 与 `metadata.citations` 交叉验证：编号存在且在数组范围内才渲染为可点击
- 超出范围的编号渲染为普通文本（不可点击）

#### 3.5.2 拒答机制

**触发条件**：
1. 检索结果为空（reason=`no_results`）
2. 重排后最高分 < 阈值（如 0.5，reason=`low_score`）
3. 模型自评：召回的知识不足以回答（reason=`model_insufficient`）

**模型自评实现**：
- System Prompt 中要求模型在知识不足时输出特殊标记：`<INSUFFICIENT_KNOWLEDGE>`
- 后端检测到该标记后：
  1. 替换为标准拒答话术
  2. 记录到低置信度问题池（reason=`model_insufficient`）
- 不依赖自然语言解析（如"我不确定"），避免误判

**拒答话术**：
```
抱歉，我暂时没有找到相关信息。您可以：
1. 换个说法再问一次
2. 联系人工客服（输入"转人工"）
```

#### 3.5.3 低置信度问题池

**表结构**：`low_confidence_questions`
| 列 | 类型 | 说明 |
|---|---|---|
| id | BIGINT | 主键 |
| question | TEXT | 原话 |
| session_id | VARCHAR(255) | 来源会话 |
| reason | VARCHAR(50) | `no_results` / `low_score` / `model_insufficient` |
| created_at | DATETIME | 入池时间 |

**入池入口**：
- 检索为空：reason=`no_results`
- 重排最高分低：reason=`low_score`
- 模型自评不足：reason=`model_insufficient`

#### 3.5.4 负面知识列表

在 System Prompt 中列明禁止承诺的内容：
- 不承诺到账时间
- 不承诺具体库存
- 不承诺物流异常赔偿金额

### 3.6 评估体系

#### 3.6.1 评估集

**文件位置**：`evals/hybrid_cases.yaml`（需新建）

**结构**：
```yaml
- query: "邮费是多少"
  ground_truth_ids: ["售后政策.md:1"]
  difficulty: easy
  query_type: fact

- query: "我买了个手机壳但是颜色不喜欢能退吗"
  ground_truth_ids: ["售后政策.md:2", "售后政策.md:3"]
  difficulty: medium
  query_type: procedure
```

**ground_truth_ids 获取方式**：
- 从现有知识库 `knowledge_chunks` 表读取所有 `chunk_id`
- 人工标注：针对每个 query，标注哪些 chunk_id 是正确答案
- 可以通过 SQL 查询辅助标注：
  ```sql
  SELECT chunk_id, LEFT(content, 100) FROM knowledge_chunks WHERE content LIKE '%退货%';
  ```

**难度梯度**：
- easy：单跳、字面匹配（如"邮费是多少"能直接匹配"邮费"）
- medium：需要改写或同义（如"能退吗"需要理解为"退货政策"）
- hard：多跳、需要推理（如"买错了尺码怎么办"需要推理出"退货"+"换货"）

**Query 类型**：
- fact：事实查询（"多少"、"什么"）
- procedure：流程查询（"怎么办"）
- comparison：对比查询（"区别"）

#### 3.6.2 评估指标

**Recall@K**：
```python
def recall_at_k(retrieved_ids, ground_truth_ids, k):
    top_k = retrieved_ids[:k]
    hit = len(set(top_k) & set(ground_truth_ids))
    return hit / len(ground_truth_ids)
```

**MRR（Mean Reciprocal Rank）**：
```python
def mrr(retrieved_ids, ground_truth_ids):
    for i, id in enumerate(retrieved_ids, 1):
        if id in ground_truth_ids:
            return 1 / i
    return 0
```

**Faithfulness**（生成段）：
- LLM-as-Judge：模型回答是否基于召回的知识
- 人工标注（抽样）

#### 3.6.3 四策略对比

| 策略 | Dense | BM25 | RRF | Rerank |
|---|---|---|---|---|
| 纯向量 | ✓ | ✗ | ✗ | ✗ |
| 纯BM25 | ✗ | ✓ | ✗ | ✗ |
| 混合 | ✓ | ✓ | ✓ | ✗ |
| 混合+Rerank | ✓ | ✓ | ✓ | ✓ |

**对比报告**：
```
=== 策略对比报告 ===
按 query_type 分桶：

[fact]
策略          Recall@3  Recall@10  MRR
纯向量        0.65      0.82       0.71
纯BM25        0.58      0.75       0.63
混合          0.78      0.91       0.84
混合+Rerank   0.85      0.94       0.89

[procedure]
...
```

### 3.7 前端配套

#### 3.7.1 引用编号可点

**UI 行为**：
- 回答中的 `[1]`、`[2]` 渲染为可点击的徽章
- 点击后弹出 modal 或侧边栏，显示：
  - 来源 chunk 的原文
  - 章节路径（如 `售后政策.md > 运费说明`）
  - 相关性分数

**数据流**：
1. 后端在 message.metadata 存 citations
2. 前端解析 message.content 中的 `[数字]`
3. 点击时根据编号查 metadata 取 chunk_id
4. 调接口 `GET /api/chunks/{chunk_id}` 取原文

**新增接口**：`GET /api/chunks/{chunk_id}`（见 §5.1）

#### 3.7.2 满意度反馈

**UI 组件**：
- 每条助手回答左下角显示 👍 / 👎
- 点击后：
  - 图标点亮（填充色）
  - 显示"已反馈"文本
  - 锁定（不可再点击另一个）

**数据采集**：
- 纯前端状态（本章不落库）
- 为后续数据飞轮留接口：`POST /api/feedback`
- 接口实现：**本章暂不实现后端落库**，前端可以发请求但后端只返回 `{"status": "ok"}`（不存储）

## 4. 数据库变更

### 4.1 新增表

**low_confidence_questions**：
```sql
CREATE TABLE low_confidence_questions (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    reason VARCHAR(50) NOT NULL COMMENT 'no_results/low_score/model_insufficient',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session (session_id),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 4.2 现有表变更

**messages**：
- 已有 `metadata` JSON 列，无需变更
- 存储格式扩展：
  ```json
  {
    "citations": [...],
    "retrieval_strategy": "hybrid_rerank",
    "top_score": 0.85
  }
  ```

**knowledge_chunks**：
- 新增 BM25 函数（Milvus 侧，非 MySQL schema）

## 5. API 变更

### 5.1 新增接口

**GET /api/chunks/{chunk_id}**：
- 功能：根据 chunk_id 返回原文与元数据
- 返回：
  ```json
  {
    "chunk_id": "售后政策.md:1",
    "content": "...",
    "category": "运费说明",
    "source": "售后政策.md",
    "is_key_clause": true
  }
  ```
- 错误：chunk_id 不存在时返回 404

**POST /api/feedback**（预留接口）：
- 功能：接收满意度反馈
- 请求体：
  ```json
  {
    "message_id": 123,
    "feedback": "thumbs_up"  // 或 "thumbs_down"
  }
  ```
- 返回：`{"status": "ok"}`
- **本章实现**：接口存在但不落库，只返回成功（为前端开发提供端点）

**GET /api/low-confidence-questions**（可选，本章暂不实现）：
- 功能：查询低置信度问题池
- 用途：由管理员或数据分析人员使用
- 本章：只落库，由管理员用 SQL 查询（不提供 HTTP 接口）

### 5.2 现有接口行为变更

**POST /api/chat**：
- 生成的 message.metadata 增加 `citations` 字段
- SSE 事件中不变（citations 只在 done 后查库可见）

## 6. 配置项

新增 Settings 字段（通过环境变量 `APP_*` 配置，定义在 `app/config.py` 的 `Settings` 类）：

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `reranker_base_url` | str | `https://api.siliconflow.cn/v1` | 重排模型 API |
| `reranker_api_key` | str | `""` | API key |
| `reranker_model` | str | `bge-reranker-v2-m3` | 模型名 |
| `retrieval_dense_top_k` | int | `50` | Dense 召回数 |
| `retrieval_bm25_top_k` | int | `50` | BM25 召回数 |
| `retrieval_rerank_top_k` | int | `10` | 重排后保留数 |
| `retrieval_rerank_threshold` | float | `0.5` | 拒答阈值（重排最高分低于此值触发拒答） |
| `retrieval_strategy` | str | `hybrid_rerank` | 检索策略：`dense` / `bm25` / `hybrid` / `hybrid_rerank` |
| `rerank_arrangement_strategy` | str | `first_last` | 重排后排列策略：`score_desc`（按分数降序）/ `first_last`（首尾优先） |

**策略说明**：
- `dense`：纯向量检索（ch03 模式，兼容性回退）
- `bm25`：纯 BM25 全文检索
- `hybrid`：混合检索 + RRF 融合，不重排
- `hybrid_rerank`：混合检索 + 重排（默认，完整链路）

**排列策略说明**：
- `score_desc`：按重排分数降序排列（传统方式）
- `first_last`：最相关放首尾（首因+近因效应优化），排列为 `[1st, 3rd, 5th, 7th, 9th, 10th, 8th, 6th, 4th, 2nd]`

## 7. 实施计划概要

### 7.1 阶段划分

1. **基础设施**：Milvus BM25 函数 + 低置信度问题池表
2. **混合检索**：hybrid_search + RRF 融合
3. **重排**：bge-reranker-v2-m3 接入
4. **Query 改写**：LLM 驱动的口语归一化
5. **生成质量控制**：引用编号 + 拒答 + 低置信度池
6. **评估体系**：标注集 + 四策略对比脚本
7. **前端配套**：引用可点 + 满意度反馈

### 7.2 验收标准

1. ✅ 四策略对比报告能跑出数字（Recall@3/10、MRR）
2. ✅ 问"型号 ABC-123"，BM25 能命中包含该字符串的 chunk
3. ✅ 答案中的 `[1]` 可点击，能看到来源原文与章节路径
4. ✅ 问知识库没有的内容，得到明确拒答，且问题进了 `low_confidence_questions` 表

## 8. 非功能需求

### 8.1 兼容性

- 不破坏 ch03 的现有测试（257 passed）
- `retrieval_strategy=dense` 时回退到纯向量检索（与 ch03 等价）

### 8.2 性能

- 混合检索 + 重排的 P95 延迟 < 2s
- BM25 分词不阻塞写入（异步建索引）

### 8.3 可观测性

- 每次检索记 log：strategy / dense_count / bm25_count / rerank_top_score
- 低置信度问题池按天统计（便于发现知识盲区）

## 9. 本章不做

- **指代消解**："那个" → 上文提到的实体
- **多轮改写**："继续" → 结合上文重写 query
- **Faithfulness 自动评估**：LLM-as-Judge（评估集只做人工标注）
- **满意度反馈落库**：纯前端状态，不存后端
- **同义词词典**：先用 LLM 改写，词典是后续优化

## 10. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| Milvus BM25 中文分词效果差 | BM25 召回率低 | 实测验证，必要时换 jieba 外部分词 |
| 重排模型 API 不稳定 | 检索降级到混合不重排 | 降级策略 + 重试 |
| RRF 融合权重不合理 | 混合检索不如单路 | 评估集调参（k 值） |
| 引用编号与 chunk 对不上 | 用户点击看到错误内容 | 测试覆盖 citations 映射逻辑 |

## 11. 后续优化方向

- Query 理解：意图识别、槽位抽取
- 多路召回：加入关键词精确匹配（Elasticsearch）
- 个性化：根据用户历史调整召回权重
- A/B 测试：四策略在真实流量的效果对比

---

## 附录 A：Builder-Check 问题澄清

### A.1 Milvus BM25 具体实现

**版本**：Milvus 3.0.0（已确认支持 BM25、hybrid_search、RRFRanker）

**API 调用方式**（已在 §3.1 更新）：
1. Schema 中添加 `sparse_vector` 字段（`SPARSE_FLOAT_VECTOR` 类型）
2. 定义 Function 将 `content` 自动转换为 BM25 向量存入 `sparse_vector`
3. 检索时使用 `AnnSearchRequest` + `metric_type="BM25"` + `analyzer_name="cn"`

**Chinese analyzer**：`"cn"` 是 Milvus 内置别名，指向 chinese analyzer，无需额外配置（已通过 Context7 验证）

### A.2 Query 改写的执行逻辑

**每次查询都执行改写**（不依赖触发条件），改写成功则替换原 query 进行检索，改写失败回退到原 query。

**不并行检索**：只用改写后的 query 检索一次（降低延迟和成本）。

### A.3 引用编号的生成与验证

**信任模型输出**，不做后处理修正。

**模型不标注或标注错误的处理**：
- 没有引用编号：`metadata.citations` 为空数组，不阻塞回答
- 引用超出范围：前端渲染时校验，超出范围的编号渲染为普通文本（不可点击）

（已在 §3.5.1 更新）

### A.4 拒答机制的"模型自评不足"判断

**特殊标记方式**：System Prompt 要求模型在知识不足时输出 `<INSUFFICIENT_KNOWLEDGE>`。

**后端处理**：
1. 检测到该标记后替换为标准拒答话术
2. 记录到低置信度问题池（reason=`model_insufficient`）

**不依赖自然语言解析**（如"我不确定"），避免误判。

（已在 §3.5.2 更新）

### A.5 元数据过滤的执行时机

**检索前应用**：`filter_expr` 作为 Milvus 检索参数传入（不是检索后在 Python 侧过滤）。

**过滤后结果太少的处理**：不自动放宽过滤（保持语义明确性），返回空结果并触发拒答机制。

（已在 §3.3 更新）

### A.6 评估集的具体格式

**文件不存在**，需要新建 `evals/hybrid_cases.yaml`。

**ground_truth_ids 获取方式**：
1. 从 `knowledge_chunks` 表读取所有 `chunk_id`
2. 人工标注：针对每个 query 标注正确答案的 chunk_id
3. 可用 SQL 辅助标注（如 `SELECT chunk_id, LEFT(content, 100) FROM knowledge_chunks WHERE content LIKE '%退货%';`）

**格式确认**：`ground_truth_ids` 是 `chunk_id` 数组（如 `["售后政策.md:1"]`）

（已在 §3.6.1 更新）

### A.7 满意度反馈的接口实现

**需要实现 POST /api/feedback 接口**，但本章不落库，只返回 `{"status": "ok"}`（为前端开发提供端点）。

（已在 §5.1 更新）

### A.8 低置信度问题池的查询接口

**本章不提供 HTTP 接口**，只落库，由管理员用 SQL 查询（如 `SELECT * FROM low_confidence_questions ORDER BY created_at DESC LIMIT 100;`）。

（已在 §5.1 说明）

### A.9 当前系统的配置读取方式

**环境变量 + Settings 类**：
- 通过 `.env` 文件定义环境变量（如 `APP_RERANKER_BASE_URL`）
- 在 `app/config.py` 的 `Settings` 类中定义字段（`env_prefix="APP_"`）
- Pydantic Settings 自动读取并校验

（已在 §6 说明）

### A.10 重排后排列策略的控制

**通过配置项控制**：新增 `rerank_arrangement_strategy` 配置项（见 §6）。

**默认值**：`first_last`（首尾优先排列）。

**可选值**：
- `score_desc`：按分数降序（传统方式）
- `first_last`：首尾优先（首因+近因效应优化）

（已在 §6 更新）
