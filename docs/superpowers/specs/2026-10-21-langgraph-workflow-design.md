# Workflow 确定性编排 + 主力 Agent 设计（ch05）

## 1. 背景与目标

ch01-04 已经把「一次请求 = 一次模型调用（带工具）」做出来了：`/api/chat` 先让模型决定是否调工具，有工具就执行并二次请求，最后落库。它的问题是**控制流在模型手里**：要不要检索、要不要查订单、什么时候收敛，全由一次 Function Calling 决定。

本章把控制流收回到代码里：

- 用 LangGraph 搭一张确定的图：指代消解 → 意图识别 → 分流 → 知识检索 → 置信度闸 → 主力 Agent → 日志。
- 分流规则硬编码：7 类意图分到 4 个出口（知识类 / 业务数据类 / 投诉 / 闲聊）。
- 主力 Agent 是图里的核心节点，内部跑 ReAct 循环；工具沿用第 2 章，不新写业务工具。
- 会话状态用 LangGraph State 贯穿，用官方 checkpointer 持久化。
- 先用裸循环祛魅，再用 LangGraph 重构。

### 1.1 当前状态

- `app/chat.py::stream_reply_with_tools` 是一次性的「模型→工具→模型」循环，单轮收敛。
- `app/tools/` 有 5 个工具：`query_order`、`query_logistics`、`query_product`、`query_faq`、`create_ticket`。
- ch04 提供 `retrieval.search_with_metadata(session, query, top_k, strategy=..., filter_expr=...) -> (chunks, meta)`，并写入低置信度池。
- `/api/chat` 以 SSE 输出 `meta` / `tool` / `delta` / `done` / `error` 事件；消息落 MySQL `messages` 表。
- 前端 `web/index.html` 是原生 HTML/CSS/JS，已有聊天流、工具徽章、引用徽章与反馈。

## 2. 非目标（本章不做）

- 指代消解、意图识别的正式版（本章分别做透传与单 prompt 分类）。
- 上下文管理策略升级、MCP 接入、数据飞轮入库。
- 真实人工客服系统对接（转人工只做前端模拟）。
- 修改第 2 章业务工具的签名或行为。
- 多轮子图恢复（LangGraph `interrupt`）——追问靠 State 标记，不用 interrupt。

## 3. 总体架构

```
                 ┌─────────────────────────────────────────────┐
START → resolve_reference → classify_intent → route            │
                 │                                             │
   ┌─────────────┼───────────────┬───────────────┬─────────────┐
   │ 知识类       │ 业务数据类     │ 投诉          │ 闲聊         │
   ▼             ▼               ▼               ▼             │
retrieve      slot_gate       complaint       chitchat         │
   │             │               │               │             │
   ▼             │               │               │             │
confidence_gate  │               │               │             │
   │  pass       │ 有订单号       │               │             │
   ├─────────────┴───────────────┼───────────────┘             │
   │  reject                     ▼                             │
   ▼                        agent（ReAct 子图）                 │
refuse                         │                               │
   └──────────────┬─────────────┴───────────────┬──────────────┘
                  ▼                             │
                 log ← ─────────────────────────┘
                  ▼
                 END
```

分流表（代码常量，不交给模型）：

| 意图 | 出口 | 说明 |
|---|---|---|
| 商品咨询、退货退款 | 知识类 | 强制检索 → 置信度闸 → Agent；退货退款先检索政策，Agent 仍可调订单工具 |
| 物流、订单、售后 | 业务数据类 | 先过订单号插槽门，再进 Agent；不预检索、不走置信度闸 |
| 投诉 | 投诉出口 | 安抚话术 + 建议动作（转人工/建工单），不进 Agent |
| 闲聊 | 闲聊出口 | 固定话术 |

## 4. State（LangGraph）

```python
class WorkflowState(TypedDict, total=False):
    # 对话
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str                  # 本轮原始输入
    resolved_input: str              # 指代消解后的输入（本章等于 user_input）
    # 分流
    intent: str                      # 7 类之一
    intent_confidence: float | None
    route: str                       # knowledge|business|complaint|chitchat
    # 知识类
    evidence: list[dict]             # [{chunk_id, questions, answer, score}]
    evidence_coverage: float | None
    gate_passed: bool | None
    refusal_reason: str | None       # low_evidence | no_results | ...
    # 业务数据类插槽
    awaiting_slot: str | None        # 本章只有 "order_id"
    pending_intent: str | None       # 等待插槽期间记住原意图
    order_id: str | None
    # 输出控制
    actions: list[dict]              # [{type: handoff|ticket, ...}]
    suggestions_sent: bool
    # ReAct 控制
    steps: int
    tokens_used: int
    max_steps: int
    token_budget: int
    # 观测
    node_trace: Annotated[list[str], operator.add]  # 走过的节点名
    turn_started_at: float
```

约束：

- **只有可序列化数据进 State**。ORM 行、AsyncSession、registry 不进 State，通过 LangGraph `config["configurable"]` 传入运行时依赖（不参与 checkpointer 序列化）。
- `messages` 用 `add_messages` 合并；每轮用户输入由 API 以 `{"messages":[HumanMessage(user_input)], ...}` 注入。
- **turn 级字段每轮重置**：图入口第一个节点 `begin_turn` 清空 `intent/route/evidence/evidence_coverage/gate_passed/refusal_reason/actions/suggestions_sent/steps/tokens_used/node_trace`。跨轮只保留 `messages`、`awaiting_slot`、`pending_intent`、`order_id`。
- `node_trace` 用普通 `list[str]`（覆盖式），节点返回 `state["node_trace"] + ["node"]`；不用 `operator.add`，否则 `begin_turn` 无法清空，验收 1 会看到上一轮的节点。
- **失败轮的悬空工具调用**：checkpointer 每个 super-step 都落盘，若流在 `agent_model` 产出 tool_calls 之后、`agent_tools` 写回 ToolMessage 之前中断，线程里会留下没有结果的 assistant tool_calls，下一轮上游会拒绝该消息序列。图入口先跑 `sanitize_messages`：若末尾 assistant 消息的 tool_calls 没有一一对应的 ToolMessage，丢弃这组未完成的消息（及其后的孤立 ToolMessage），再注入本轮 HumanMessage。

## 5. 节点定义

### 5.1 `resolve_reference`（最简版）

原样透传：`resolved_input = user_input`。记一条 debug 日志，留出后续接指代消解的接口。

### 5.2 `classify_intent`

- 单次 LLM 调用（`temperature=0`，复用 ch04 的 `get_chat_model` 缓存模式），prompt 要求只输出 JSON：
  ```json
  {"intent": "物流", "confidence": 0.0}
  ```
- 7 类：`物流 | 订单 | 商品咨询 | 退货退款 | 售后 | 投诉 | 闲聊`。
- 解析失败/未知类：落到 `售后`（业务数据类，Agent 能兜住），记 warning，`intent_confidence=None`。
- 等待态判定**只放在本节点**，图不为此加回边：
  - `awaiting_slot=="order_id"` 且本轮消息能匹配订单号 → 沿用 `pending_intent`（用户只是在补号）。
  - `awaiting_slot=="order_id"` 但本轮消息匹配不到订单号 → 清空 `awaiting_slot/pending_intent`，按新消息正常分类（用户换了问题，不该再被追着要号）。
- `slot_gate` 只负责提取/放行/追问，不再承担重新分类。

### 5.3 `route`

纯函数，按 §3 表映射 `intent → route`，不调用模型。

### 5.4 `retrieve`（仅知识类）

- 调用
  ```python
  retrieval.search_with_metadata(
      session, resolved_input,
      top_k=settings.retrieval_top_k,
      strategy=settings.retrieval_strategy,
      session_id=str(conversation_id),
  )
  ```
  必须显式传 `strategy` 与 `session_id`：`search_with_metadata` 的默认策略是 `dense`，不传会悄悄改掉 ch04 的混合检索与重排分支，并让池里的 `session_id` 为空。
- 把 `chunks` 压成可序列化的 `evidence`：`{index, chunk_id, questions, answer, score}`，`index` 从 1 开始，与交给 Agent 的引用编号一致；`score` 来自 `meta["scores"]`，没有则 null。
- 检索为空：沿用 ch04 的 `no_results` 入池；`gate_passed=False`，`refusal_reason="no_results"`。
- 日志：`node=retrieve chunks=N strategy=...`（验收 1 的断言点）。

### 5.5 `confidence_gate`（仅知识类，在 Agent 之前）

最简版两条判据，满足其一即可进 Agent：

1. 若本轮启用了重排（`meta["rerank_applied"]` 为真）：`meta["top_score"] >= settings.retrieval_rerank_threshold`。
2. 否则用**词面覆盖率**：把查询切成字符二元组（去空白/标点、去重），统计有多少二元组出现在召回文本（`questions + answer` 拼接）里，`coverage = 命中数 / 总二元组数`；`coverage >= settings.confidence_min_coverage`（默认 0.30）。

- 通过：`gate_passed=True`，进入 Agent；把 `evidence` 一并注入 Agent 的上下文（作为一条 system/context 消息，**不写入审计流水**）。
- 不通过：`gate_passed=False`，`refusal_reason="low_evidence"`，输出固定兜底话术，**不进 Agent**；把原问题写入 `low_confidence_questions`，`reason="low_evidence"`。
- **入池只写一次**：若 `retrieve` 已经因为空结果写过 `no_results`（`refusal_reason` 非空），闸门不再写 `low_evidence`，避免同一问题两条记录。
- 业务数据类不经过本节点（无检索证据）。
- 覆盖率是本章的过渡判据，正式置信度检查留给可观测章节。

### 5.6 `slot_gate`（物流/订单/售后）

- 提取订单号：正则 `订单号?\s*[:：]?\s*([A-Za-z0-9-]{4,20})` 或独立数字串 `\b\d{4,20}\b`；也接受 State 里已存的 `order_id`。
- 有号：写入 `order_id`，清空 `awaiting_slot/pending_intent`，放行 Agent。
- 无号：`awaiting_slot="order_id"`、`pending_intent=intent`，输出固定追问（例：“请提供订单号（在「我的订单」页可查看），我帮您查。”），结束本轮，不进 Agent。
- 下一轮：`classify_intent` 先判断新消息是否匹配订单号；匹配则沿用 `pending_intent`，由 `slot_gate` 提取并放行；不匹配则清空等待态、正常分类新问题（图不需要回边）。
- 三类业务意图都拦（用户确认）；售后若确实不需要订单号，用户可继续描述，Agent 也能在 ReAct 里追问。

### 5.7 `agent`（ReAct 子图）

编译成子图，作为主图的一个节点：

```
agent_model ──(有 tool_calls 且在预算内)──> agent_tools ──> agent_model
     └──────(无 tool_calls / 超预算)──────> agent_done
```

- `agent_model`：`model.bind_tools(ALL_TOOLS).astream(messages)`；逐 token 通过 `StreamWriter` 发 `delta` 事件；累积出完整 AIMessage；累加 `steps`；从响应元数据读取 usage 累加 `tokens_used`。
- `agent_tools`：遍历 `tool_calls`，经现有 `ToolRegistry.execute()` 执行（保留超时/重试/上下文注入），发 `tool start/end` 事件，把 `ToolMessage` 追加回 messages。
- 停止条件：
  - 无 `tool_calls` → `agent_done`；
  - `steps >= settings.agent_max_steps`（默认 6）→ 发一条“我需要更多信息/请稍后再试”的收敛话术；
  - `tokens_used >= settings.agent_token_budget`（默认 6000）→ 同样收敛。
- 知识类：进 Agent 前把 `evidence` 作为一条上下文消息插入（标注“以下是检索到的知识，回答须以此为准并标注引用编号”，编号用 `evidence.index`）。
- **工具集按路由裁剪，避免两套引用编号打架**：
  - 知识类：`query_order`、`query_logistics`、`query_product`、`create_ticket`（**不含 `query_faq`**，检索已强制完成）。`done.citations` / 消息 metadata 的引用来自 `evidence`。
  - 业务数据类：五个工具全给（含 `query_faq`）。引用来自最后一次 `query_faq` 工具结果（沿用 ch04 行为）。
- `<INSUFFICIENT_KNOWLEDGE>` 行为保留：仍由 API 流式适配层过滤、替换为标准拒答话术、写 `low_confidence_questions(reason="model_insufficient")`。知识类已过闸，正常不会触发；业务类 Agent 调 `query_faq` 后仍可能触发。
- 缺信息追问：模型可直接输出追问文本（无工具调用即收敛），不强制走插槽门。
- **动作建议**：模型输出中的 `<SUGGEST_HANDOFF>` / `<SUGGEST_TICKET>` 标记由后端流式过滤器剥离，并转成 `actions`；不新增工具、不让模型直接执行动作。

### 5.8 `complaint`

- 不发 LLM 调用，输出固定安抚话术。
- `actions = [{type:"handoff",label:"转人工"}, {type:"ticket",label:"建工单",ticket_type:"投诉",description:user_input}]`。
- 不写 tickets、不改会话状态。

### 5.9 `chitchat`

固定话术（例：“您好，我是客服小助手，可以帮您查询订单、物流和售后政策。请问有什么可以帮您？”）。无动作。

### 5.10 `refuse`

输出置信度闸的兜底话术（例：“抱歉，我暂时没有找到足够的信息回答这个问题。您可以换个说法，或联系人工客服。”），并把 `actions` 置为 `[handoff]`（可选，便于用户转人工）。

### 5.11 `log`

- 结构化日志一行：`workflow_turn session=... intent=... route=... nodes=[...] chunks=N coverage=... gate=... steps=N tokens=N elapsed_ms=... actions=[...]`。
- 不写库；为后续可观测章节留字段。

## 6. 流式与前端协议

现有事件保留：`meta` / `tool` / `delta` / `done` / `error`。新增/扩展：

- `delta`：由 Agent 子图的 `StreamWriter` 产出，保持 `{"text": "..."}`。
- `tool`：start/end 与现有格式一致；end 帧继续带 `citations`（ch04 行为）。
- `done`：新增
  ```json
  {
    "finish_reason": "stop",
    "usage": {...},
    "message_id": 123,
    "citations": [...],
    "actions": [{"type":"handoff","label":"转人工"},
                {"type":"ticket","label":"建工单","ticket_type":"投诉","description":"..."}]
  }
  ```
- `actions` 也可在逐轮中间帧出现（Agent 提前建议时），前端按 message 聚合。

### 6.1 前端交互

- 收到 `actions` 后，在该条助手消息底部渲染独立按钮；两个按钮互不绑定。
- **转人工**：纯前端。点击后立即插入两条本地消息：
  1. “已转接人工客服”
  2. “您好，我是客服小猫，请问有什么可以帮您的”（客服小猫为模拟人设）
  然后按钮锁定/隐藏，不做任何网络请求。
- **建工单**：`POST /api/tickets`，body `{session_id, description, ticket_type}`；成功后在消息流里插入“已为您创建工单，工单号 T…”；失败显示错误并保留按钮。
- 用户不点任何按钮、继续发消息：正常进入下一轮图执行，不产生任何动作。
- 按钮只属于产生它的那条消息；历史消息上的按钮保持锁定状态。

### 6.2 动作建议的流式剥离

Agent 的文本里可能出现 `<SUGGEST_HANDOFF>` / `<SUGGEST_TICKET>`。沿用 ch04 的 marker 过滤思路：

- 在 Agent 子图的流式输出处过滤，保证标记不出现在 `delta` 帧。
- 检测到即写入 `actions`；同一轮去重。
- 大小写敏感、全大写，降低误伤。
- **动作只从 `done` 帧下发**（不新增中间事件）：前端拿到 `done.actions` 后渲染按钮。
- Agent 建议的工单参数映射：`ticket_type` 按路由取——知识类 → `咨询`，业务数据类 → `售后`，投诉 → `投诉`；`description` 用本轮用户原话（截断到 500 字）。`handoff` 无参数。

## 7. 工单端点（确定性，不经过 LLM）

`POST /api/tickets`

- 请求：`{"session_id": "...", "description": "...", "ticket_type": "投诉|售后|咨询"}`。
- 逻辑：按 `session_uuid` 找 `conversations.id`；用 `tool_context(conversation_id, session)` 包住 `create_ticket` 工具调用（复用第 2 章写 `tickets` 表 + 会话状态置「已转人工」的逻辑）；commit。
- 返回：`{"status": "ok", "ticket_no": "T..."}`；找不到会话 404。
- 幂等：前端点击后锁定；后端不做去重（与 ch02 工具语义一致）。

## 8. Checkpointer 与会话记忆

- 选用 LangGraph 官方 `AsyncSqliteSaver`（`langgraph-checkpoint-sqlite`，异步、文件落盘），路径由 `APP_CHECKPOINT_DB` 配置，默认 `./data/langgraph_checkpoints.sqlite`。
- 生命周期：FastAPI lifespan 启动时进入 `AsyncSqliteSaver.from_conn_string(...)` 上下文，编译一次图并挂在 `app.state`；关闭时退出。
- `thread_id = session_id`（`conversations.session_uuid`）。首次会话由 API 生成 UUID。
- **记忆归属（用户确认 A）**：模型上下文只来自 checkpointer 里的 `State.messages`；MySQL `messages` 表继续写，作为可 SQL 查询的审计流水，图不读它。
- **审计写入契约**：每轮把本轮的完整序列写 MySQL——
  - 正常 Agent 轮：`HumanMessage` → 每组 `AIMessage(tool_calls)` + 对应 `ToolMessage`（可多组）→ 最终 `AIMessage`；
  - 固定话术轮（投诉/闲聊/拒答/追问订单号）：`HumanMessage` + 一条 `AIMessage`；
  - citations/actions 写入最终 `AIMessage` 的 `metadata`；
  - `done.message_id` = 最终 assistant 行的 id；
  - `evidence` 注入消息只在 State 中，不写审计流水（`messages_io` 无需支持 SystemMessage）。
  - 客户端中断或异常：本轮不写任何消息（与 ch04 的语义一致），checkpointer 里的半轮由下一轮的 `sanitize_messages` 清理。
- 旧会话兼容：没有 checkpoint 的 `session_id` 视为全新会话，不尝试从 MySQL 回填历史。

## 9. 配置项（新增）

| 字段 | 默认 | 说明 |
|---|---|---|
| `checkpoint_db` | `./data/langgraph_checkpoints.sqlite` | checkpointer 文件 |
| `confidence_min_coverage` | `0.30` | 词面覆盖率阈值 |
| `agent_max_steps` | `6` | ReAct 最大模型轮次 |
| `agent_token_budget` | `6000` | 单轮 token 预算 |
| `intent_model_temperature` | `0.0` | 意图分类温度 |
| `workflow_log_enabled` | `true` | 结构化日志开关 |

## 10. 祛魅热身：裸 Agent 循环

- 新增 `scripts/bare_agent_loop.py`：不依赖 LangGraph，手写
  ```python
  while steps < max_steps:
      resp = await model.ainvoke(messages)
      if not resp.tool_calls:
          return resp.content
      for call in resp.tool_calls:
          result = await registry.execute(...)
          messages += [resp, ToolMessage(result)]
  ```
- 带 `--query` CLI 参数，用真实模型和工具跑一次，打印每一步的 tool_calls 与结果，作为“它就是个循环”的证据。
- 单测用脚本化 fake model 验证：一次工具调用后收敛；超过 max_steps 时停止。
- 该脚本保留为教学/诊断工具，不接入生产 API。

## 11. 测试策略

- **可单测代码照常 TDD**：路由表、插槽提取、覆盖率计算、动作解析、状态机（等待订单号跨轮）、停止条件。
- **Prompt/分类类任务**（意图识别、安抚话术、固定闲聊话术）：用标注样例跑一遍（≥20 条覆盖 7 类），准确率写进报告；不要求 100%。
- **图级测试**：用 fake model + 真 State/checkpointer（临时 sqlite 文件）验证分支、跨轮等待态、checkpointer 恢复。
- **边界测试**（对应 §4/§5 裁定）：`sanitize_messages` 丢弃悬空 tool_calls；`begin_turn` 重置 turn 字段（第二轮不会继承上一轮 step/token）；等待态收到非订单号消息时重新分类；检索已入池时闸门不重复写；知识类不挂 `query_faq` 工具。
- **API/SSE**：沿用 `tests/test_api*.py` 的 TestClient 模式，断言事件序列与 actions。
- **验收测试**（`tests/acceptance/`）：
  1. 政策类问题 → `node_trace`/日志包含 `retrieve`；
  2. “订单 1001 的物流到哪了” → Agent 工具调用事件含 `query_logistics`（可能先 `query_order`）；
  3. “我要投诉” → done.actions 同时含 handoff/ticket；模拟点击两个按钮分别验证（前端行为用 JS 单测/手工验收）；
  4. 闲聊 → 固定话术；
  5. 复杂问题“订单 1001 的物流到哪了”若需先查订单 → `steps >= 2` 且两次工具调用。
- 所有既有 342 个测试保持通过（允许为适配新流程调整 API 测试，但不得降低断言强度）。

## 12. 验收标准映射

| # | 标准 | 实现点 | 验证 |
|---|---|---|---|
| 1 | 政策类问题日志能看到强制检索 | `retrieve` 节点日志 `node=retrieve chunks>0` | caplog / 演示命令 |
| 2 | “订单 1001 的物流到哪了” Agent 调工具 | 业务类 → slot_gate 放行 → ReAct | tool 事件 + 回答 |
| 3 | 投诉出现两个独立按钮；转人工纯前端；建工单写表 | complaint 节点 + `/api/tickets` + 前端 | 验收测试 + 浏览器 |
| 4 | 闲聊固定话术 | chitchat 节点 | 测试断言固定字符串 |
| 5 | 复杂问题 ReAct 多步 | `steps>=2`、两次工具 | 测试断言 node_trace/steps |

## 13. 风险与取舍

| 风险 | 影响 | 缓解 |
|---|---|---|
| 词面覆盖率误判 | 该进 Agent 的被拒 | 阈值可配；先跑评估样例调；正式版留给可观测章节 |
| 意图分类不稳 | 走错出口 | 失败回落到售后（Agent 兜底）；20 条标注样例验证 |
| checkpointer 与 MySQL 两份消息 | 审计与图状态不一致 | 明确约定图不读 MySQL；审计写入失败只记日志不阻塞 |
| SQLite 并发写 | 多请求写锁 | 单 worker 部署；`AsyncSqliteSaver` 串行写；后续可换 Postgres |
| 标记式动作建议 | 模型不输出标记 | prompt 明确；投诉节点不依赖模型；测试覆盖解析 |
| 图内运行时依赖注入 | 串话/泄漏 | 依赖走 `config["configurable"]`，不落 State；每请求独立 session |

## 14. 留给实现者的判断

- 意图 prompt 的具体措辞与 few-shot 数量。
- 覆盖率分词细节（二元组 vs 三元组）与阈值调参。
- Agent 上下文消息的组织方式（system 注入 vs 独立 HumanMessage）。
- 前端按钮样式与本地消息的插入位置。
- checkpointer 文件目录的创建时机与权限。

## 15. Builder-check 澄清（已裁定）

| # | 问题 | 裁定 |
|---|---|---|
| 1 | 失败轮留下悬空 tool_calls | 图入口加 `sanitize_messages`，丢弃未闭合的工具调用组；失败轮不写审计 |
| 2 | 等待态下用户换问题 | `classify_intent` 判断新消息是否匹配订单号：匹配沿用原意图，不匹配则清空等待态并正常分类；不加回边 |
| 3 | State 字段跨轮残留 | `begin_turn` 每轮重置 turn 级字段；`node_trace` 改覆盖式列表；跨轮只留 `messages/awaiting_slot/pending_intent/order_id` |
| 4 | ch04 `<INSUFFICIENT_KNOWLEDGE>` 是否保留 | 保留，仍在 API 流式适配层过滤/替换/入池 `model_insufficient` |
| 5 | retrieve 参数与入池归属 | 显式传 `strategy` 与 `session_id`；闸门在已有 `refusal_reason` 时不再重复入池 |
| 6 | 预检索证据与 query_faq 引用编号冲突 | 知识类工具集移除 `query_faq`，引用只用 `evidence.index`；业务类保留五工具，沿用 ch04 引用 |
| 7 | 审计写入什么 | 完整 ReAct 序列 + 最终 assistant（metadata 带 citations/actions）；固定话术轮只写一问一答；evidence 注入消息不落库 |
| 8 | 动作传输与工单类型 | 动作只在 `done.actions`；工单类型按路由映射（知识→咨询/业务→售后/投诉→投诉），描述用用户原话（≤500 字） |

**依赖版本（本次 Context7/uv 解析确认）**：`langgraph==1.2.16`、`langgraph-checkpoint-sqlite==3.1.2`；需同步改掉 `pyproject.toml` 里“langgraph 为非目标”的注释。

## 附录 A：备选方案与被否原因

- **订单号逻辑放工具内**：需改第 2 章工具签名，且模型仍要先决定调工具；与“本章不新写/不改业务工具”冲突。
- **LangGraph `interrupt` 追问**：前端只有聊天框，恢复协议成本高；本章用 State 标记即可。
- **ReAct 写成单节点 while 循环**：能做，但每个 ReAct 步不落 checkpoint，且看不出图结构；改子图后每步可观测、可恢复。
- **MySQL 作历史来源（方案 B）**：每轮手动拼历史，和图内 `add_messages` 双轨，易漂移；用户选择 A。
