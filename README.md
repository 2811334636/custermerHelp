# 电商智能客服 · ch01（纯对话跑通）

## 快速开始

```bash
cp .env.example .env      # 然后填入 APP_LLM_API_KEY
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

服务必须**单 worker** 运行——会话历史存在进程内存里。

## 演示

```bash
bash scripts/smoke.sh     # 三条验收标准一键复现
```

## 测试

```bash
uv run pytest -v                       # 单元测试
uv run python -m evals.run_evals       # 评估集（Prompt 行为 + 抽取准确率）
uv run python -m evals.run_tool_evals  # ch02 工具选择评估集
```

两个评估集都连 `customer_service_test`（先跑 `uv run python -m scripts.seed` 建库灌种子），
跑完回滚，库里不留痕。

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
| POST | `/api/chat` | SSE 流式对话，事件序列 `meta? → delta* → (done \| error)` |
| POST | `/api/extract` | 售后描述 → 结构化工单 JSON |
| GET | `/healthz` | 健康检查 |

## 关键约束（改代码前必读）

- **必须关闭 thinking**：`APP_LLM_EXTRA_BODY={"thinking":{"type":"disabled"}}`。
  开着 thinking 时上游会把 token 预算全烧在推理上，**正文一个字都不输出**。
- **结构化输出只能用 `function_calling`**：上游 `response_format: json_schema` 完全不可用。
- **换 provider** 只改 `.env` 的 `APP_LLM_BASE_URL` / `APP_LLM_API_KEY` / `APP_LLM_MODEL` /
  `APP_LLM_EXTRA_BODY` 四行，不要改代码。

设计文档：`docs/superpowers/specs/2026-09-18-ecom-customer-service-chat-design.md`
