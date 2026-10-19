# 电商智能客服 · ch01 纯对话 → ch02 Function Calling → ch03 知识库与向量检索

## ch03 · 知识库与向量检索（当前章节）

`query_faq` 的内部实现从关键词查表升级为 Milvus dense 向量语义检索，工具的
名称、入参、出参契约保持不变。知识来自 Markdown 手册、存量 FAQ 表与从历史
对话挖掘的问答对。

### 启动

```bash
docker compose up -d                  # MySQL + Milvus Standalone
uv run python -m scripts.seed         # 灌 FAQ 种子 + 建测试库
uv run python -m jobs.build_knowledge # 切分 + 向量化
uv run python -m evals.run_kb_evals   # 检索质量评估
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**首次在干净机器上启动前先读「已知限制」里的 MinIO 条目**：官方镜像已下架，
`docker compose up -d` 依赖本机已有该 tag 的缓存，且**不要**执行
`docker compose pull`。

### 知识来源

| 来源 | 内容 | content_type |
|---|---|---|
| `knowledge/*.md` | 政策手册、商品 FAQ、平台规则 | `policy` |
| `faq` 表 | 存量 12 条 FAQ | `faq` |
| `mined_qa_staging` | 从历史对话挖掘的问答对 | `mined` |

### 定时任务

```bash
# 每天凌晨 3 点挖一次历史对话
0 3 * * * cd /home/zzx/custermerHelp && uv run python -m jobs.mine_knowledge && uv run python -m jobs.build_knowledge
```

### 已知限制

- 只用 dense 向量单路，无关键词/混合/重排
- 文档编辑后旧 chunk 的向量不自动清理（孤儿向量）
- 检索质量依赖知识内容本身是否写完整（见 spec §3.2 的实测对比）
- **MinIO 镜像只能从本机缓存获取**：`minio/minio:RELEASE.2024-05-28T17-19-04Z`
  已从公共仓库下架；干净机器需自备同版本镜像，且不要执行 `docker compose pull`。
- **挖掘任务的零产出对话会重复处理**：抽取结果为 0 条的对话不会留下暂存行，
  每次 cron 都会重新送 LLM。

## ch02 · Function Calling 工具链（上一章）

在 ch01 的对话链路上加一层工具链：模型自己决定调哪个工具，再按工具返回的结果作答。
工具调用在聊天页上可见（助手气泡上方的徽章）。

### 启动

前置（`.env` 与依赖）见下面 ch01 的「快速开始」。ch02 另需要 MySQL 与种子数据：

```bash
docker compose up -d          # 起 MySQL 8.4（宿主映射 3307，见 docker-compose.yml）
uv run python -m scripts.seed # 灌种子数据（faq 12 条）+ 建测试库
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

聊天页：<http://127.0.0.1:8000/chat>

服务必须**单 worker** 运行。历史现在落 MySQL（重启不丢），但本项目全程只验证过
单 worker，多 worker 未实测。

### 五个工具

| 工具 | 作用 | 数据来源 |
|---|---|---|
| `query_order` | 查订单 | 工具内确定性生成（不接真实系统） |
| `query_product` | 查商品 | 同上 |
| `query_logistics` | 查物流 | 同上 |
| `query_faq` | 查常见问题 | `faq` 表，SQL `LIKE`（**只搜 `question` 列**） |
| `create_ticket` | 建人工工单 | 写 `tickets` 表 + 会话转「已转人工」 |

三个「确定性生成」的工具由订单号/商品号派生种子（`sha256` → `random.Random`），
同一入参永远返回逐字节相同的内容——验收时才能拿模型回答与工具输出逐字对账。

### 已知限制

- **`query_faq` 只有关键词匹配、没有语义泛化。** 问「邮费政策是什么」会调工具、但查不到
  （`faq` 表里确实没有「邮费」条目）——这个漏召回是**预期结果**，留给下一步升级。
  措辞本身也影响工具选择：「邮费是多少」**连工具都不会触发**，模型会反问澄清。
- 刷新页面会丢当前对话显示（数据已落库，缺的是读接口）。
- 所有会话的 `user_id` 都是 `anonymous`（本章无鉴权）。
- 徽章的「⏳ 查询中…」中间态短到看不见：本地库里工具 0–2ms 就返回，
  用户实际等的是上游首个 token。

## ch01 · 纯对话跑通（上一章）

### 快速开始

```bash
cp .env.example .env      # 然后填入 APP_LLM_API_KEY
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

服务必须**单 worker** 运行（说明见 ch02 一节）。

## 演示

```bash
bash scripts/smoke.sh     # ch01 三条验收标准一键复现（需要服务已在 8000 跑着）
```

ch02 的验收在浏览器里做：打开聊天页，问「订单 1001 的物流到哪了」「退货政策是什么」
「邮费政策是什么」，看助手气泡上方是否出现工具徽章、正文是否与工具返回的数据对得上。

## 测试

**前置：MySQL 必须在跑**（`docker compose up -d`）。测试会话一开始就会连库清理
`customer_service_test`，MySQL 不在时会直接报错——包括 `pytest tests/test_config.py`
这种根本不碰数据库的选择。这是有意的：环境没备好就响亮地失败，不要静默跳过。

```bash
uv run pytest -q                       # 单元测试
uv run python -m evals.run_evals       # 评估集（Prompt 行为 + 抽取准确率）
uv run python -m evals.run_tool_evals  # ch02 工具选择 + 结果转述评估集
# ch03 检索质量评估（需先运行 uv run python -m jobs.build_knowledge 建库）
uv run python -m evals.run_kb_evals
```

**退出码会因模型抖动而偶发变红。** `run_evals` 的口径是「Prompt 15/15 且抽取字段级
≥90%」，判据本身是确定性的（禁用词正则 / 长度上限 / 语义下限 / 转述），但它**打的是
真实模型**：实测约每 7 次 run 里有 1 次掉到 14/15——是模型某一条的措辞或长度抖了
（行为用例里 150 字上限那条最常触发），**不是代码回归**。红了先重跑一次再下结论；
同理，一次绿也不代表稳定。`run_tool_evals` 走同一个模型，一样会抖。

三个评估集里，`run_evals` 与 `run_tool_evals` 连 `customer_service_test`
（先跑 `uv run python -m scripts.seed` 建库灌种子）；`run_kb_evals` 直接连生产库与
Milvus（先跑 `uv run python -m jobs.build_knowledge` 建库）。

测试库的留痕是**跨 run 有界**，不是「跑完不留痕」：`/api/chat` 路由会**真提交**本轮写入
（commit 不可撤销），每个驱动真路由的用例都会留下自己的会话行/消息行。所以每次 pytest
会话**开始前**清一次 `messages` / `tickets` / `conversations` 三张表（`faq` 一个字不动，
工具评估要读它），跑完会留十几行直到下次运行。放在「开始」而不是「结束」是为了自愈：
上一次跑崩（Ctrl-C / 被 kill）留下的垃圾，下一次照样从干净状态出发。

改造 Prompt 后要证明评估闸门还守得住，就跑一次负对照——把改动前的 Prompt 接进来，
它**必须**拿不到满分：

```bash
git show <改动前的 commit>:app/prompts.py > /tmp/old_prompts.py
EVAL_PROMPT_FILE=/tmp/old_prompts.py uv run python -m evals.run_evals   # 期望非 0 退出
```

脚本一律用 `python -m 包.模块` 调用，**不要**用 `python 路径/文件.py`。后者会把
`sys.path[0]` 设成脚本所在目录，导致 `import app` 报 `ModuleNotFoundError`。

## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/chat` | SSE 流式对话。事件序列 `meta? → (tool* → delta*)? → (done \| error)`；`tool` 帧成对出现（`phase:start` 声明调了谁，`phase:end` 声明结果如何） |
| POST | `/api/extract` | 售后描述 → 结构化工单 JSON |
| GET | `/healthz` | 健康检查 |

## 关键约束（改代码前必读）

- **必须关闭 thinking**：`APP_LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}`。
  开着 thinking 时上游会把 token 预算全烧在推理上，**正文一个字都不输出**。
- **结构化输出只能用 `function_calling`**：上游 `response_format: json_schema` 完全不可用。
- **换 provider** 只改 `.env` 的 `APP_LLM_BASE_URL` / `APP_LLM_API_KEY` / `APP_LLM_MODEL` /
  `APP_LLM_EXTRA_BODY` 四行，不要改代码。

设计文档：`docs/superpowers/specs/2026-09-18-ecom-customer-service-chat-design.md`
