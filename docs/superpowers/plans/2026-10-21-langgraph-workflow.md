# Workflow 确定性编排 + 主力 Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangGraph 把客服系统改造成“确定性 Workflow 骨架 + ReAct Agent 节点”，并保持 ch04 的检索、引用、拒答能力。

**Architecture:** 主图按 7 类意图硬编码分流；知识类强制检索并过置信度闸，业务类先过订单号插槽门，投诉/闲聊走固定出口；所有出口汇聚日志节点。主力 Agent 是编译后的 ReAct 子图，工具沿用第 2 章。State 由官方 `AsyncSqliteSaver` 持久化，模型上下文以 checkpointer 为准，MySQL messages 只做审计。

**Tech Stack:** Python 3.12、FastAPI、LangGraph 1.2.16、langgraph-checkpoint-sqlite 3.1.2、langchain-core 1.6.3（既有）、MySQL、Milvus、原生 HTML/CSS/JS。

**Spec:** `docs/superpowers/specs/2026-10-21-langgraph-workflow-design.md`

## Global Constraints

- 依赖固定：`langgraph==1.2.16`、`langgraph-checkpoint-sqlite==3.1.2`；不引入 `langchain` 编排层，不引入新的业务工具。
- 所有新增 Python 函数带类型标注；测试命名 `test_*`；pytest `asyncio_mode=auto`。
- 现有 342 个测试必须继续通过；为适配新流程可改 `/api/chat` 相关测试，但不得降低断言强度。
- 固定文案（逐字实现，测试直接断言）：
  - `SLOT_QUESTION = "请提供订单号（在「我的订单」页面可以查看），我来帮您查询。"`
  - `GATE_REFUSAL = "抱歉，我暂时没有找到足够的信息回答这个问题。您可以换个说法再问一次，或联系人工客服。"`
  - `COMPLAINT_TEXT = "非常抱歉给您带来不好的体验，我们很重视您的反馈。您可以选择转人工客服详细说明，或留下一个问题让我们为您创建工单跟进。"`
  - `CHITCHAT_TEXT = "您好，我是客服小助手，可以帮您查询订单、物流和售后政策。请问有什么可以帮您？"`
  - `STEP_LIMIT_TEXT = "这个问题需要更多信息才能继续处理，请补充订单号或具体问题，或联系人工客服。"`
  - `TOKEN_LIMIT_TEXT = "这次查询的内容较多，我先停在这里。您可以补充更具体的问题，或联系人工客服。"`
- 意图标签固定为：`物流`、`订单`、`商品咨询`、`退货退款`、`售后`、`投诉`、`闲聊`。
- 路由值固定为：`knowledge`、`business`、`complaint`、`chitchat`。
- 动作类型固定为：`handoff`、`ticket`；`ticket_type` ∈ `售后|投诉|咨询`。
- 流式标记固定为：`<INSUFFICIENT_KNOWLEDGE>`、`<SUGGEST_HANDOFF>`、`<SUGGEST_TICKET>`。
- 不得修改第 2 章业务工具的函数签名与行为。
- `prompt.md` 与 `tests/probes/conftest.py` 是用户未提交改动，任何任务的 commit 都不得包含它们。

## Review Focus

1. **等待订单号期间用户换话题**：用户没回订单号而是问了别的（如“怎么开发票”），系统必须回答新问题，不能反复索要订单号。
2. **ReAct 中途断流/异常**：checkpoint 里留下无结果的 `tool_calls`，下一轮不能因此报错（必须被 `sanitize_messages` 清理）。
3. **动作按钮相互独立**：只点转人工不建单、只点建工单不转人工、两个都不点继续聊天；重复点击不能重复建单。
4. **引用编号跨来源一致**：预检索块与 `query_faq` 工具块共用全局编号，重复块复用原号，`done.citations` 与模型看到的 `[n]` 必须一一对应。
5. **停止条件与空输入**：达到步数/token 上限要给出收敛话术；空字符串/纯空白输入不能崩，也不该进入 Agent。

---

### Task 0: LangGraph 集成探测（阻塞性）

**Files:**
- Create: `scripts/probe_langgraph.py`

**Interfaces:**
- Produces: 探测结论（写入 `dev-notes/ch05.md`），确认 `AsyncSqliteSaver` 的构造/恢复、`astream(stream_mode="custom")` 与子图 `subgraphs=True` 在 1.2.16 的实际行为。

- [ ] **Step 1: 写探测脚本**

`scripts/probe_langgraph.py` 用一个两节点小图（`step_a` 写自定义事件，`step_b` 结束）验证：
1. `AsyncSqliteSaver.from_conn_string(<临时文件>)` 可直接使用（是否需要 `setup()` 以实测为准）；
2. `graph.astream(inputs, {"configurable":{"thread_id":"t1"}}, stream_mode="custom")` 能收到节点内 `get_stream_writer()` 发出的事件；
3. 同一 `thread_id` 第二次 `astream(None, ...)` 能读到上一次的 State（checkpoint 恢复）；
4. 子图（`StateGraph` 编译后作为节点）在 `subgraphs=True` 时自定义事件能冒泡；
5. `AIMessage.tool_calls` 在节点间传递后，`add_messages` 能正确合并。
脚本用 `print` 输出每步 PASS/FAIL，临时文件在 `finally` 删除。

- [ ] **Step 2: 运行探测**

Run: `uv run python -m scripts.probe_langgraph`
Expected: 5 项全部 PASS。

- [ ] **Step 3: 记录结论**

把实际 API 形态（是否需要 `setup()`、事件元组结构、子图事件路径）写入 `dev-notes/ch05.md` 的“阶段 1”段。若与 spec §15 的版本假设冲突，停止并询问用户，不自行换方案。

- [ ] **Step 4: Commit**

```bash
git add scripts/probe_langgraph.py dev-notes/ch05.md
git commit -m "test: LangGraph 1.2 checkpointer/streaming 探测"
```

---

### Task 1: 依赖与配置

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`
- Modify: `.env.example`
- Test: `tests/config/test_settings_ch05.py`

**Interfaces:**
- Produces: `Settings.checkpoint_db: str`、`Settings.confidence_min_coverage: float`、`Settings.agent_max_steps: int`、`Settings.agent_token_budget: int`、`Settings.intent_model_temperature: float`、`Settings.workflow_log_enabled: bool`。

- [ ] **Step 1: 添加依赖**

Run: `uv add "langgraph==1.2.16" "langgraph-checkpoint-sqlite==3.1.2"`
同时删掉 `pyproject.toml` 依赖注释里“spec §2 明确把 langgraph* 编排栈列为非目标”的句子，改为一句“ch05 起用 LangGraph 做确定性编排”。

- [ ] **Step 2: 写失败测试**

`tests/config/test_settings_ch05.py`：

```python
def test_workflow_settings_defaults():
    from app.config import Settings
    s = Settings()
    assert s.checkpoint_db == "./data/langgraph_checkpoints.sqlite"
    assert s.confidence_min_coverage == 0.30
    assert s.agent_max_steps == 6
    assert s.agent_token_budget == 6000
    assert s.intent_model_temperature == 0.0
    assert s.workflow_log_enabled is True


def test_workflow_settings_validation():
    import pytest
    from pydantic import ValidationError
    from app.config import Settings
    with pytest.raises(ValidationError):
        Settings(confidence_min_coverage=1.5)
    with pytest.raises(ValidationError):
        Settings(agent_max_steps=0)
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/config/test_settings_ch05.py -q`
Expected: FAIL（字段不存在）。

- [ ] **Step 4: 添加配置字段**

在 `app/config.py` 的 `Settings` 中新增（沿用现有 `Field` 风格）：

```python
checkpoint_db: str = Field(default="./data/langgraph_checkpoints.sqlite")
confidence_min_coverage: float = Field(default=0.30, ge=0, le=1)
agent_max_steps: int = Field(default=6, gt=0)
agent_token_budget: int = Field(default=6000, gt=0)
intent_model_temperature: float = Field(default=0.0, ge=0, le=2)
workflow_log_enabled: bool = Field(default=True)
```

`.env.example` 增加对应的 `APP_*` 六项及一行注释。

- [ ] **Step 5: 运行测试**

Run: `uv run pytest tests/config/test_settings_ch05.py -q && uv run pytest -q`
Expected: 新测试 PASS；全套件 342+ passed。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock app/config.py .env.example tests/config/test_settings_ch05.py
git commit -m "feat: ch05 依赖与工作流配置"
```

---

### Task 2: 裸 Agent 循环（祛魅热身）

**Files:**
- Create: `scripts/bare_agent_loop.py`
- Test: `tests/workflow/test_bare_loop.py`

**Interfaces:**
- Produces: `async def run_bare_agent(model: Any, registry: Any, messages: list[BaseMessage], *, max_steps: int = 6) -> list[BaseMessage]`——手写循环，返回追加了 assistant/tool 消息的完整序列。

- [ ] **Step 1: 写失败测试**

```python
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class ScriptedModel:
    """按脚本依次返回 AIMessage；记录调用次数。"""
    def __init__(self, replies): self.replies = list(replies); self.calls = 0
    def bind_tools(self, tools): return self
    async def ainvoke(self, messages):
        self.calls += 1
        return self.replies.pop(0)


class FakeRegistry:
    def __init__(self, result="工具结果"): self.result = result; self.calls = []
    async def execute(self, name, args, call_id, conversation_id=None):
        self.calls.append((name, args))
        return type("O", (), {"ok": True, "content": self.result})()


@pytest.mark.asyncio
async def test_bare_loop_converges_after_one_tool_call():
    from scripts.bare_agent_loop import run_bare_agent
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]),
        AIMessage(content="订单 1001 已发货。"),
    ])
    registry = FakeRegistry("订单号 1001：已发货")
    out = await run_bare_agent(model, registry, [HumanMessage("查订单 1001")])
    assert model.calls == 2
    assert registry.calls == [("query_order", {"order_id": "1001"})]
    assert isinstance(out[-1], AIMessage) and out[-1].content == "订单 1001 已发货。"


@pytest.mark.asyncio
async def test_bare_loop_stops_at_max_steps():
    from scripts.bare_agent_loop import run_bare_agent
    replies = [
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": f"c{i}"}])
        for i in range(3)
    ]
    model = ScriptedModel(replies)
    out = await run_bare_agent(model, FakeRegistry(), [HumanMessage("x")], max_steps=2)
    assert model.calls == 2
    assert out[-1].content  # 收敛话术非空
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/workflow/test_bare_loop.py -q`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

`run_bare_agent` 逻辑：循环 `for step in range(max_steps)`，`resp = await model.ainvoke(messages)`；无 `tool_calls` 则 append 并 return；有则 append resp，对每个 call 调 `registry.execute(name, args, id, conversation_id=1)`，append `ToolMessage(content=outcome.content, tool_call_id=id)`。循环耗尽时 append 一条 `AIMessage(content=STEP_LIMIT_TEXT)`（从 `app.workflow.texts` 导入，先建最小 `app/workflow/texts.py`）。
CLI：`argparse --query`，构造真实 `get_chat_model()`、`build_registry()`，打印每步 `step/tool_calls/content`。

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/workflow/test_bare_loop.py -q`
Expected: 2 passed。

- [ ] **Step 5: Commit**

```bash
git add scripts/bare_agent_loop.py app/workflow/__init__.py app/workflow/texts.py tests/workflow/test_bare_loop.py
git commit -m "feat: 裸 Agent 循环（祛魅热身）"
```

---

### Task 3: State、路由表与骨架节点

**Files:**
- Create: `app/workflow/__init__.py`
- Create: `app/workflow/state.py`
- Create: `app/workflow/nodes.py`
- Test: `tests/workflow/test_state_nodes.py`

**Interfaces:**
- Produces:
  - `WorkflowState`（TypedDict，字段见 spec §4）
  - `INTENTS: tuple[str, ...]`、`ROUTE_BY_INTENT: dict[str, str]`
  - `async def sanitize_messages(state: WorkflowState) -> dict`
  - `async def begin_turn(state: WorkflowState) -> dict`
  - `async def resolve_reference(state: WorkflowState) -> dict`
  - `def route_for(intent: str) -> str`
  - `TURN_RESET_FIELDS: tuple[str, ...]`

- [ ] **Step 1: 写失败测试**

```python
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.workflow.state import ROUTE_BY_INTENT, WorkflowState
from app.workflow.nodes import begin_turn, route_for, sanitize_messages


def test_route_table_is_exhaustive():
    assert set(ROUTE_BY_INTENT) == {"物流", "订单", "商品咨询", "退货退款", "售后", "投诉", "闲聊"}
    assert ROUTE_BY_INTENT["商品咨询"] == ROUTE_BY_INTENT["退货退款"] == "knowledge"
    assert ROUTE_BY_INTENT["物流"] == ROUTE_BY_INTENT["订单"] == ROUTE_BY_INTENT["售后"] == "business"
    assert ROUTE_BY_INTENT["投诉"] == "complaint"
    assert ROUTE_BY_INTENT["闲聊"] == "chitchat"
    assert route_for("未知") == "business"   # 兜底：售后/业务类
    assert route_for("投诉") == "complaint"


async def test_begin_turn_resets_turn_fields_and_sets_cursor():
    state = {"steps": 5, "tokens_used": 9000, "actions": [{"type": "handoff"}],
             "citations": [{"index": 1}], "node_trace": ["retrieve"],
             "awaiting_slot": "order_id", "pending_intent": "物流", "order_id": "1001"}
    out = await begin_turn(state)
    assert out["steps"] == 0 and out["tokens_used"] == 0
    assert out["actions"] == [] and out["citations"] == [] and out["node_trace"] == []
    assert out["citation_cursor"] == 1 and out["citation_index"] == {}
    assert out["awaiting_slot"] == "order_id" and out["order_id"] == "1001"  # 跨轮保留


async def test_sanitize_drops_dangling_tool_calls():
    msgs = [
        HumanMessage("查物流", id="h1"),
        AIMessage(content="", id="a1", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]),
    ]
    out = await sanitize_messages({"messages": msgs})
    # add_messages 的删除必须用 RemoveMessage；返回被删消息的 RemoveMessage 列表
    assert [type(m).__name__ for m in out["messages"]] == ["RemoveMessage"]
    assert out["messages"][0].id == "a1"


async def test_sanitize_keeps_complete_tool_round():
    msgs = [
        HumanMessage("查物流", id="h1"),
        AIMessage(content="", id="a1", tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}]),
        ToolMessage(content="运输中", tool_call_id="c1", id="t1"),
        AIMessage(content="在运输中。", id="a2"),
    ]
    out = await sanitize_messages({"messages": msgs})
    assert out.get("messages", []) == []  # 没有可删的
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/workflow/test_state_nodes.py -q`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

- `state.py`：TypedDict 字段与 spec §4 一致；`messages: Annotated[list[AnyMessage], add_messages]`；`INTENTS`、`ROUTE_BY_INTENT`；`TURN_RESET_FIELDS` 覆盖 spec §4 的 turn 级字段。
- `nodes.py`：
  - `sanitize_messages`：从后往前找最后一个 assistant 的 tool_calls；若其 id 集合不能全部在后随 ToolMessage 中找到，返回 **`RemoveMessage` 列表**（`from langchain_core.messages import RemoveMessage`）删除该 assistant 及其后的孤立 ToolMessage；没有可删的返回 `{}`。注意 `add_messages` 是追加语义，直接返回裁剪后的消息列表会重复而不是替换。
  - `begin_turn`：返回重置字段字典（不含 messages/awaiting_slot/pending_intent/order_id）。
  - `resolve_reference`：返回 `{"resolved_input": state.get("user_input", "")}`。
  - `route_for(intent)`：`ROUTE_BY_INTENT.get(intent, "business")`。

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/workflow/test_state_nodes.py -q && uv run pytest -q`
Expected: PASS；全套件不回归。

- [ ] **Step 5: Commit**

```bash
git add app/workflow/state.py app/workflow/nodes.py tests/workflow/test_state_nodes.py
git commit -m "feat: 工作流 State、路由表与骨架节点"
```

---

### Task 4: 意图识别节点与标注集验证

**Files:**
- Create: `app/workflow/intent.py`
- Create: `evals/intent_cases.yaml`
- Create: `evals/run_intent_eval.py`
- Test: `tests/workflow/test_intent.py`

**Interfaces:**
- Consumes: `Settings.intent_model_temperature`、`app.providers.get_chat_model`。
- Produces:
  - `INTENT_PROMPT: str`
  - `def parse_intent_json(text: str) -> tuple[str, float | None]`（未知/坏 JSON 返回 `("售后", None)`）
  - `async def classify_intent_node(state: WorkflowState, config: RunnableConfig) -> dict`：返回 `{"intent": ..., "intent_confidence": ..., "route": route_for(intent), "awaiting_slot": ..., "pending_intent": ...}`。
  - `async def classify_text(text: str, *, model) -> tuple[str, float | None]`（供评估脚本复用）。

- [ ] **Step 1: 写失败测试**

```python
def test_parse_intent_json_valid():
    from app.workflow.intent import parse_intent_json
    assert parse_intent_json('{"intent": "物流", "confidence": 0.9}') == ("物流", 0.9)


def test_parse_intent_json_unknown_or_broken_falls_back():
    from app.workflow.intent import parse_intent_json
    assert parse_intent_json('{"intent": "退款", "confidence": 0.9}') == ("售后", None)
    assert parse_intent_json("not json") == ("售后", None)
    assert parse_intent_json('{"intent": "投诉"}') == ("投诉", None)


async def test_classify_reuses_pending_intent_only_when_order_id_present():
    from app.workflow.intent import classify_intent_node
    state = {"user_input": "1001", "awaiting_slot": "order_id", "pending_intent": "物流"}
    out = await classify_intent_node(state, {"configurable": {"model": ExplodingModel()}})
    assert out["intent"] == "物流" and out["route"] == "business"
    assert out["awaiting_slot"] == "order_id"


async def test_classify_new_question_clears_waiting_state():
    from app.workflow.intent import classify_intent_node
    state = {"user_input": "怎么开发票", "awaiting_slot": "order_id", "pending_intent": "物流"}
    out = await classify_intent_node(state, {"configurable": {"model": FakeIntentModel("退货退款", 0.8)}})
    assert out["intent"] == "退货退款" and out["route"] == "knowledge"
    assert out["awaiting_slot"] is None and out["pending_intent"] is None


async def test_classify_uses_model_result():
    state = {"user_input": "我要投诉", "awaiting_slot": None, "pending_intent": None}
    out = await classify_intent_node(state, {"configurable": {"model": FakeIntentModel("投诉", 0.95)}})
    assert out["intent"] == "投诉" and out["route"] == "complaint"
```

`ExplodingModel` 的任何调用都抛 `AssertionError`（证明等待态+订单号时不调模型）；`FakeIntentModel` 的 `ainvoke` 返回 `AIMessage(content='{"intent": "...", "confidence": ...}')`。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/workflow/test_intent.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现意图识别**

- `INTENT_PROMPT` 要求“只输出 JSON，两个键 intent/confidence”，列出 7 个标签与各一个例句。
- `classify_text`：`model.ainvoke([SystemMessage(INTENT_PROMPT), HumanMessage(text)])`，取 `resp.content` 解析。
- `classify_intent_node`：
  - `awaiting_slot == "order_id"` 且 `extract_order_id(user_input)` 非空 → 不调模型，沿用 `pending_intent`；
  - `awaiting_slot == "order_id"` 且没提取到 → 清空等待态，正常分类；
  - 其他情况正常分类。

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/workflow/test_intent.py -q`
Expected: 5 passed。

- [ ] **Step 5: 建标注集并写评估脚本**

`evals/intent_cases.yaml`：20 条，覆盖 7 类（每类至少 2 条，投诉/闲聊各 2），格式：

```yaml
- {query: "订单 1001 的物流到哪了", intent: 物流}
- {query: "我要投诉你们客服", intent: 投诉}
# ... 共 20 条
```

`evals/run_intent_eval.py`：读 yaml，用真实 `get_chat_model(temperature=0)` 逐条分类，打印混淆矩阵与准确率；准确率 < 0.80 时非 0 退出。

- [ ] **Step 6: 跑标注集**

Run: `uv run python -m evals.run_intent_eval`
Expected: 准确率 ≥ 0.80；把实际数字与错例写进报告/`dev-notes`。若明显低于阈值，只调 prompt 措辞后重跑，不改判定规则。

- [ ] **Step 7: Commit**

```bash
git add app/workflow/intent.py tests/workflow/test_intent.py evals/intent_cases.yaml evals/run_intent_eval.py
git commit -m "feat: 意图识别节点与标注集评估"
```

---

### Task 5: 订单号插槽门

**Files:**
- Create: `app/workflow/slots.py`
- Test: `tests/workflow/test_slots.py`

**Interfaces:**
- Consumes: `WorkflowState`、`SLOT_QUESTION`。
- Produces:
  - `ORDER_ID_RE: re.Pattern[str]`
  - `def extract_order_id(text: str) -> str | None`
  - `async def slot_gate_node(state: WorkflowState) -> dict`：返回 `{"order_id": ..., "awaiting_slot": ..., "pending_intent": ..., "messages": [AIMessage(SLOT_QUESTION)]}` 之一。

- [ ] **Step 1: 写失败测试**

```python
def test_extract_order_id_formats():
    from app.workflow.slots import extract_order_id
    assert extract_order_id("订单 1001 的物流") == "1001"
    assert extract_order_id("订单号：A12345") == "A12345"
    assert extract_order_id("单号 20260920-001") == "20260920-001"
    assert extract_order_id("1001") == "1001"
    assert extract_order_id("怎么开发票") is None


async def test_slot_gate_passes_with_order_id():
    from app.workflow.slots import slot_gate_node
    out = await slot_gate_node({"user_input": "订单 1001 的物流", "intent": "物流",
                                "awaiting_slot": None, "pending_intent": None})
    assert out["order_id"] == "1001" and out["awaiting_slot"] is None
    assert "messages" not in out


async def test_slot_gate_asks_and_remembers():
    from app.workflow.slots import SLOT_QUESTION, slot_gate_node
    out = await slot_gate_node({"user_input": "我的物流到哪了", "intent": "物流",
                                "awaiting_slot": None, "pending_intent": None})
    assert out["awaiting_slot"] == "order_id" and out["pending_intent"] == "物流"
    assert out["messages"][0].content == SLOT_QUESTION


async def test_slot_gate_uses_existing_order_id_from_state():
    from app.workflow.slots import slot_gate_node
    out = await slot_gate_node({"user_input": "那物流呢", "intent": "物流",
                                "order_id": "1001", "awaiting_slot": None})
    assert out["order_id"] == "1001"
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/workflow/test_slots.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现**

正则两条：`(?:订单号?|单号)\s*[:：]?\s*([A-Za-z0-9-]{4,20})` 与 `\b(\d{4,20})\b`（第二条只在整个字符串很短、或前面没有明显非订单语义时使用；依次尝试）。`slot_gate_node` 优先用 state 里已有的 `order_id`；否则从 `user_input` 提取；失败则设等待态并返回 `messages=[AIMessage(SLOT_QUESTION)]`。裸数字规则可能把年份/金额误当订单号，这是本章可接受的启发式，误判时用户会得到“查无此单”，不会写坏数据。

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/workflow/test_slots.py -q`
Expected: 4 passed。

- [ ] **Step 5: Commit**

```bash
git add app/workflow/slots.py tests/workflow/test_slots.py
git commit -m "feat: 订单号插槽门"
```

---

### Task 6: 知识检索、置信度闸与全局编号

**Files:**
- Create: `app/workflow/citations.py`
- Create: `app/workflow/retrieval_nodes.py`
- Test: `tests/workflow/test_retrieval_gate.py`

**Interfaces:**
- Consumes: `retrieval.search_with_metadata`、`Settings.retrieval_*`、`Settings.confidence_min_coverage`。
- Produces:
  - `def char_bigrams(text: str) -> set[str]`
  - `def evidence_coverage(query: str, texts: list[str]) -> float`
  - `def assign_citations(hits: list[KnowledgeChunk], citations: list[dict], index_map: dict[str, int], cursor: int) -> tuple[str, list[dict], dict[str, int], int]`——返回（带 `[i]` 前缀的文本、更新后的 citations/index_map/cursor）；重复 `chunk_id` 复用原号。
  - `async def retrieve_node(state: WorkflowState, config: RunnableConfig) -> dict`
  - `async def confidence_gate_node(state: WorkflowState, config: RunnableConfig) -> dict`

- [ ] **Step 1: 写失败测试**

```python
def test_evidence_coverage_basic():
    from app.workflow.retrieval_nodes import evidence_coverage
    assert evidence_coverage("退货运费", ["退货运费由平台承担"]) == 1.0
    assert evidence_coverage("退货运费", ["发票怎么开"]) == 0.0
    assert 0 < evidence_coverage("退货运费谁承担", ["退货运费由平台承担，个人原因自理"]) < 1
    assert evidence_coverage("", ["任何文本"]) == 0.0


def test_assign_citations_numbers_and_dedupes():
    from app.workflow.citations import assign_citations
    hits = [FakeChunk("a", "q1", "a1"), FakeChunk("b", "q2", "a2")]
    text, citations, index_map, cursor = assign_citations(hits, [], {}, 1)
    assert text.startswith("[1] 问：q1")
    assert "[2] 问：q2" in text
    assert [c["chunk_id"] for c in citations] == ["a", "b"]
    assert cursor == 3
    # 再次召回 a：复用编号 1，不新增
    text2, citations2, index_map2, cursor2 = assign_citations(
        [FakeChunk("a", "q1", "a1")], citations, index_map, cursor)
    assert text2.startswith("[1] ")
    assert len(citations2) == 2 and cursor2 == 3


async def test_gate_passes_on_high_coverage():
    from app.workflow.retrieval_nodes import confidence_gate_node
    state = {"resolved_input": "退货运费", "evidence": [{"index": 1, "chunk_id": "a", "questions": "退货运费", "answer": "平台承担", "score": None}],
             "refusal_reason": None}
    out = await confidence_gate_node(state, {})
    assert out["gate_passed"] is True


async def test_gate_refuses_and_pools_once():
    from app.workflow.retrieval_nodes import confidence_gate_node
    state = {"resolved_input": "火星移民", "evidence": [{"index": 1, "chunk_id": "a", "questions": "退货", "answer": "政策", "score": None}],
             "refusal_reason": None}
    out = await confidence_gate_node(state, {"configurable": {"session": FakeSession(), "conversation_id": 7}})
    assert out["gate_passed"] is False and out["refusal_reason"] == "low_evidence"
    assert out["messages"][0].content == GATE_REFUSAL
```

`FakeSession` 记录 `add`；测试断言只调用一次 `lcq_repo.create`（若实现直接调用 repo，可用 monkeypatch 捕获）。另加一条：`refusal_reason="no_results"` 时不再写 `low_evidence`。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/workflow/test_retrieval_gate.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现**

- `char_bigrams`：去掉空白与标点后，取相邻两字符的集合（长度 <2 返回空集）。
- `evidence_coverage`：查询二元组在拼接文本中的命中比例；空查询/空文本返回 0.0。
- `assign_citations`：遍历 hits，`chunk_id in index_map` 复用，否则 `cursor` 分配并追加 `{"index": i, "chunk_id": ..., "score": None}`；文本每块 `f"[{i}] 问：{questions}\n答：{answer}"`，块间 `\n\n`。
- `retrieve_node`：按 spec §5.4 显式传 `strategy`/`session_id`；把 chunks 转为 evidence（先用 `assign_citations` 分配编号，再把 index 写回 evidence）；`meta["refusal_reason"]` 透传；日志 `node=retrieve chunks=N strategy=...`。
- `confidence_gate_node`：
  - 若 `evidence` 为空 → `gate_passed=False`、`refusal_reason` 保持 `no_results`（若无则 `"no_results"`），不发话术（refuse 节点发）。
  - 若有重排分（evidence 里任一块 `score` 非 null）：`max(score) >= settings.retrieval_rerank_threshold` 才通过。
  - 否则覆盖率 ≥ `settings.confidence_min_coverage` 才通过。
  - 不通过且 `refusal_reason` 为空 → 写 `low_confidence_questions(reason="low_evidence", session_id=str(conversation_id))`。

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/workflow/test_retrieval_gate.py -q`
Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add app/workflow/citations.py app/workflow/retrieval_nodes.py tests/workflow/test_retrieval_gate.py
git commit -m "feat: 知识检索、置信度闸与全局引用编号"
```

---

### Task 7: ReAct 子图与工具执行

**Files:**
- Create: `app/workflow/agent.py`
- Test: `tests/workflow/test_agent.py`

**Interfaces:**
- Consumes: `assign_citations`、`intent`/`route` state、`ToolRegistry`、`Settings.agent_max_steps/agent_token_budget`。
- Produces:
  - `def parse_action_markers(text: str) -> tuple[str, list[str]]`——返回（剥离标记后的文本、动作类型列表）。
  - `def build_agent_subgraph() -> StateGraph`（节点从 `config["configurable"]` 取 model/registry/session）
  - 子图节点：`agent_model`、`agent_tools`、`agent_done`。

- [ ] **Step 1: 写失败测试**

```python
def test_parse_action_markers_strips_and_dedupes():
    from app.workflow.agent import parse_action_markers
    text, actions = parse_action_markers("帮您转人工 <SUGGEST_HANDOFF><SUGGEST_TICKET><SUGGEST_HANDOFF>")
    assert text == "帮您转人工 "
    assert actions == ["handoff", "ticket"]
    assert parse_action_markers("普通回答") == ("普通回答", [])


async def test_agent_converges_without_tools():
    graph = build_test_agent_graph(ScriptedModel([AIMessage(content="直接回答")]))
    out = await graph.ainvoke({"messages": [HumanMessage("你好")], "steps": 0, "tokens_used": 0}, config)
    assert out["messages"][-1].content == "直接回答"
    assert out["steps"] == 1


async def test_agent_calls_tool_then_converges():
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}]),
        AIMessage(content="已发货"),
    ])
    registry = FakeRegistry("订单 1001 已发货")
    out = await run_agent(model, registry, {"messages": [HumanMessage("查订单")]})
    assert [m.type for m in out["messages"]] == ["human", "ai", "tool", "ai"]
    assert out["steps"] == 2


async def test_agent_stops_at_max_steps():
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "1"}, "id": f"c{i}"}])
        for i in range(5)
    ])
    out = await run_agent(model, FakeRegistry(), {"messages": [HumanMessage("x")]}, max_steps=2)
    assert out["steps"] == 2
    assert out["messages"][-1].content == STEP_LIMIT_TEXT


async def test_query_faq_uses_global_citation_numbers():
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[{"name": "query_faq", "args": {"keyword": "运费"}, "id": "c1"}]),
        AIMessage(content="见 [1] 和 [2]"),
    ])
    state = {"messages": [HumanMessage("运费")], "evidence": [], "citations": [], "citation_index": {}, "citation_cursor": 1}
    out = await run_agent(model, FakeRegistry(), state, faq_hits=[FakeChunk("faq:1", "运费", "满 99 包邮")])
    tool_msg = next(m for m in out["messages"] if m.type == "tool")
    assert tool_msg.content.startswith("[1] 问：运费")
    assert out["citations"] == [{"index": 1, "chunk_id": "faq:1", "score": None}]
```

`run_agent(...)` 是测试辅助：用 `build_agent_subgraph()` 编译并 `ainvoke`，`config["configurable"]` 带 model/registry/session/faq hits 注入点。`faq_hits` 通过 fake session/retrieval 注入：实现时让 `agent_tools` 调 `search_with_metadata`，测试 monkeypatch 该函数返回固定 hits。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/workflow/test_agent.py -q`
Expected: FAIL。

- [ ] **Step 3: 实现子图**

- `agent_model`：取 `messages`（知识类时前面已插入 evidence 上下文消息），`model.bind_tools(ALL_TOOLS)`，`astream` 逐 token 用 `get_stream_writer()` 发 `{"type":"delta","text":...}`，累积 `AIMessage`；`steps += 1`，`tokens_used += usage`。
- `parse_action_markers`：删除标记、按出现顺序去重。
- `agent_tools`：遍历 tool_calls；`query_faq` 特殊处理（`search_with_metadata` + `assign_citations` + 带编号 ToolMessage，发 `tool start/end` 自定义事件并带累计 citations）；`search_with_metadata` 抛 `KnowledgeBackendUnavailable` 时，ToolMessage 内容用 ch04 工具的降级文案“知识库暂时不可用，请稍后再试或转人工客服。”，不中断循环；其他工具走 `registry.execute`，发同样的 tool 事件；失败时 ToolMessage 内容为 `outcome.content`。
- 条件边：无 tool_calls 或 `steps >= max_steps` 或 `tokens_used >= token_budget` → `agent_done`；否则 → `agent_tools` → `agent_model`。
- `agent_done`：若因步数/预算触发且最后一条是带 tool_calls 的 assistant，追加 `AIMessage(STEP_LIMIT_TEXT 或 TOKEN_LIMIT_TEXT)`；若正常收敛，把最后 assistant 文本经 `parse_action_markers` 清洗后写回 `messages` 并设置 `actions`。
- 知识类 evidence 注入在子图入口完成：若 `state["evidence"]` 非空，在 messages 前插入一条 `SystemMessage`（内容列出 `[i]` 块），该消息不写审计（由 Task 9 过滤）。

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/workflow/test_agent.py -q`
Expected: 6 passed。

- [ ] **Step 5: Commit**

```bash
git add app/workflow/agent.py tests/workflow/test_agent.py
git commit -m "feat: ReAct 子图、全局编号工具执行与动作标记"
```

---

### Task 8: 图装配、Checkpointer 与生命周期

**Files:**
- Create: `app/workflow/graph.py`
- Modify: `app/main.py`
- Test: `tests/workflow/test_graph.py`

**Interfaces:**
- Consumes: 全部节点、`AsyncSqliteSaver`。
- Produces:
  - `def build_workflow(checkpointer) -> CompiledStateGraph`
  - `def workflow_deps(model, registry, session, conversation_id) -> dict`（放进 `config["configurable"]`）
  - `app.state.workflow` 与 `app.state.checkpointer`。

- [ ] **Step 1: 写失败测试**

测试内定义 `build_test_graph(tmp_path, model)` 辅助：`AsyncSqliteSaver` 指到 `tmp_path/checkpoints.sqlite`、`build_workflow(checkpointer)` 编译；`deps` 里放 fake model、fake registry、`db_session`、`conversation_id=1`。`astream(..., stream_mode="custom", subgraphs=True)` 产出 `(namespace_tuple, event)`，断言时先解包。验证：

```python
async def test_graph_routes_knowledge_and_runs_retrieve(tmp_path):
    graph = build_test_graph(tmp_path, model=ScriptedModel([AIMessage(content="按政策…")]))
    out = await graph.ainvoke({"messages": [HumanMessage("退货政策是什么")], "user_input": "退货政策是什么"},
                              {"configurable": {"thread_id": "t1", **deps}})
    assert "retrieve" in out["node_trace"]
    assert out["route"] == "knowledge"


async def test_graph_waits_for_order_id_across_turns(tmp_path):
    graph = build_test_graph(tmp_path, model=ScriptedModel([AIMessage(content="…")]))
    first = await graph.ainvoke({"messages": [HumanMessage("我的物流到哪了")], "user_input": "我的物流到哪了"}, cfg)
    assert first["awaiting_slot"] == "order_id"
    second = await graph.ainvoke({"messages": [HumanMessage("1001")], "user_input": "1001"}, cfg)
    assert second["order_id"] == "1001" and second["awaiting_slot"] is None


async def test_graph_recovers_from_dangling_tool_call(tmp_path):
    # 先手工写入一条悬空 assistant(tool_calls) 的 checkpoint 状态
    ...
    out = await graph.ainvoke({"messages": [HumanMessage("你好")], "user_input": "你好"}, cfg)
    assert out["messages"][-1].content  # 不抛异常、能正常回答
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/workflow/test_graph.py -q`
Expected: FAIL。

- [ ] **Step 3: 装配图**

节点与边：

```
START → sanitize_messages → begin_turn → resolve_reference → classify_intent
classify_intent --conditional(route)-->
  knowledge: retrieve → confidence_gate --(pass)--> agent
                                       --(fail)--> refuse
  business:  slot_gate --(有号)--> agent
                       --(无号)--> log
  complaint: complaint → log
  chitchat:  chitchat → log
agent → log → END
refuse → log → END
```

- 用 `StateGraph(WorkflowState)`；`add_conditional_edges` 用 `lambda s: s["route"]` 与 path map。
- `confidence_gate` 后的条件边用 `lambda s: s.get("gate_passed", False)`。
- `slot_gate` 后：`lambda s: "agent" if s.get("order_id") else "log"`。
- `complaint`/`chitchat`/`refuse` 节点直接返回固定文案与 actions。
- `log` 节点：若 `settings.workflow_log_enabled`，`logger.info("workflow_turn %s", json.dumps({...}, ensure_ascii=False))`，字段含 `intent/route/node_trace/steps/tokens/actions`。
- `build_workflow(checkpointer)` 只装配一次；`app/main.py` 增加 lifespan：
  ```python
  @asynccontextmanager
  async def lifespan(app: FastAPI):
      Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
      async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db) as cp:
          app.state.checkpointer = cp
          app.state.workflow = build_workflow(cp)
          yield
  app = FastAPI(..., lifespan=lifespan)
  ```

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/workflow/test_graph.py -q && uv run pytest -q`
Expected: 图测试 PASS；全套件不回归（此步 `/api/chat` 尚未接图，旧测试仍走旧路径）。

- [ ] **Step 5: Commit**

```bash
git add app/workflow/graph.py app/main.py tests/workflow/test_graph.py
git commit -m "feat: 工作流装配与 AsyncSqliteSaver 生命周期"
```

---

### Task 9: `/api/chat` 接入工作流与审计写入

**Files:**
- Modify: `app/api.py`
- Test: `tests/test_api_workflow.py`（新增）
- Test: 调整 `tests/test_api.py` / `tests/test_api_tools.py` / `tests/test_api_citations.py` 中依赖旧单轮流程的断言

**Interfaces:**
- Consumes: `app.state.workflow`、`workflow_deps(...)`。
- Produces: 新版 SSE 事件序列（`meta/tool/delta/done/error`），`done.actions`、`done.message_id`、`done.citations`。

- [ ] **Step 1: 写失败测试**

`tests/test_api_workflow.py` 用 TestClient + `app.dependency_overrides` 注入 fake model，覆盖：
1. 知识类：事件中有 `tool`? 不——知识类是预检索，无 tool 事件；有 `delta`，`done.actions == []`；审计写入 user + final assistant；`done.message_id` 非空。
2. 业务类：tool start/end 事件存在；`done.citations` 来自 query_faq（若调了）。
3. 投诉：`done.actions` 含 handoff 与 ticket；没有 tool 事件。
4. 闲聊：delta 拼起来等于 `CHITCHAT_TEXT`。
5. 客户端断流：不写消息（用现有测试的取消手法）。

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_api_workflow.py -q`
Expected: FAIL。

- [ ] **Step 3: 改造 event_stream**

- 用 `app.state.workflow.astream(initial_state, config, stream_mode="custom", subgraphs=True)`；`config["configurable"]` 带 `thread_id=session_id` 与 `workflow_deps(model, registry, db, conv.id)`。注意 `subgraphs=True` 时每项是 `(namespace_tuple, event)`，适配层先解包（主图事件的 namespace 为空元组，Agent 子图事件带子图前缀）。
- `StreamWriter` 事件约定（dict）：`{"type":"delta","text":...}`、`{"type":"tool","phase":"start|end",...}`。适配层翻译成现有 SSE 帧。
- 测试隔离：新增 fixture 在创建 `TestClient` 前把 `Settings.checkpoint_db` 指到 `tmp_path`（monkeypatch env + `get_settings.cache_clear()`），避免测试共用 `./data` 下的 checkpoint 文件；每个测试用独立 `session_id`。
- 保留 ch04 的 `<INSUFFICIENT_KNOWLEDGE>` 流式过滤器与替换/入池逻辑（放在适配层，作用于最终回答）。
- 审计：请求结束时（`completed` 且有最终回答），从最终 State 取本轮新增消息，过滤掉 `SystemMessage`（evidence 注入），写 MySQL；metadata 带 `citations/actions/retrieval_strategy`；`done.message_id` = 最后 assistant 行 id。
- 失败/断流：不写消息；checkpointer 半轮由下一轮 sanitize 处理。
- `done.actions` 与 `done.citations` 从最终 State 取。

- [ ] **Step 4: 移除旧单轮循环并改造其测试**

`stream_reply_with_tools` 及 `chat.py` 里只服务于它的私有函数已无调用者。**删除**它们（不留死代码）；把 `tests/test_chat.py`、`tests/test_chat_tools.py` 中仍有价值的覆盖（工具轮事件顺序、错误分类、二次不绑工具收敛）改写为对工作流适配层的断言，或迁到 `tests/test_api_workflow.py`。事件 dataclass（`DeltaEvent/ToolCallEvent/ToolResultEvent/DoneEvent/ErrorEvent`）保留并按需扩展。旧测试若断言“第一次模型调用即工具”或依赖旧函数的调用形状，改为断言新的 SSE 契约；不得删掉对工具结果、引用、拒答的断言。

- [ ] **Step 5: 运行测试**

Run: `uv run pytest tests/test_api_workflow.py tests/test_api.py tests/test_api_tools.py tests/test_api_citations.py -q && uv run pytest -q`
Expected: 全部 PASS。

- [ ] **Step 6: Commit**

```bash
git add app/api.py app/chat.py tests/test_api_workflow.py tests/test_api.py tests/test_api_tools.py tests/test_api_citations.py tests/test_chat.py tests/test_chat_tools.py
git commit -m "feat: /api/chat 接入 LangGraph 工作流并移除旧单轮循环"
```

---

### Task 10: `POST /api/tickets` 端点

**Files:**
- Create: `app/api_tickets.py`
- Modify: `app/api.py`（注册路由）
- Test: `tests/test_api_tickets.py`

**Interfaces:**
- Produces: `POST /api/tickets`，请求 `{"session_id","description","ticket_type"}`，响应 `{"status":"ok","ticket_no":"T..."}`。

- [ ] **Step 1: 写失败测试**

```python
def test_post_ticket_creates_row_and_marks_handoff(client, db_session):
    # 先建会话
    ...
    r = client.post("/api/tickets", json={"session_id": sid, "description": "投诉物流太慢", "ticket_type": "投诉"})
    assert r.status_code == 200
    assert r.json()["ticket_no"].startswith("T")
    # 查 tickets 表与 conversations.status
    ...


def test_post_ticket_unknown_session_404(client):
    r = client.post("/api/tickets", json={"session_id": "nope", "description": "x", "ticket_type": "咨询"})
    assert r.status_code == 404


def test_post_ticket_rejects_bad_type(client):
    r = client.post("/api/tickets", json={"session_id": "x", "description": "x", "ticket_type": "其它"})
    assert r.status_code == 422
```

- [ ] **Step 2: 运行确认失败**

Run: `uv run pytest tests/test_api_tickets.py -q`
Expected: FAIL（404 Not Found）。

- [ ] **Step 3: 实现**

`app/api_tickets.py`：pydantic `TicketRequest`（`ticket_type: Literal["售后","投诉","咨询"]`）；按 `session_uuid` 查 `Conversation`；`with tool_context(conv.id, db): result = await create_ticket.ainvoke({"description": ..., "ticket_type": ...})`；`await db.commit()`；返回 `{"status":"ok","ticket_no": ...}`（从结果文本中用正则 `T\d+` 提取）。在 `app/api.py` 注册 `router.include_router(tickets_router)`。

- [ ] **Step 4: 运行测试**

Run: `uv run pytest tests/test_api_tickets.py -q`
Expected: 3 passed。

- [ ] **Step 5: Commit**

```bash
git add app/api_tickets.py app/api.py tests/test_api_tickets.py
git commit -m "feat: 确定性建工单端点"
```

---

### Task 11: 前端动作按钮与确认交互

**Files:**
- Modify: `web/index.html`

**Interfaces:**
- Consumes: `done.actions`、`POST /api/tickets`。

- [ ] **Step 1: 实现按钮渲染**

在 `done` 处理里读取 `data.actions`，为每条助手消息渲染独立按钮（`data-action`）：
- `handoff`：点击后本地插入两条消息：“已转接人工客服”、“您好，我是客服小猫，请问有什么可以帮您的”；锁定按钮；不发请求。
- `ticket`：点击后 `fetch("/api/tickets", {method:"POST", body: JSON.stringify({session_id: sessionId, description: action.description, ticket_type: action.ticket_type})})`；成功后插入“已为您创建工单，工单号 T…”；锁定；失败提示并保留按钮。
- 两按钮互不绑定；不点击继续聊天不受影响。
- 历史消息的按钮点击后同样锁定，且不重复建单（点击即 disable）。

- [ ] **Step 2: 验证**

Run: `node --check`（把内联脚本抽到 `/tmp/ch05.js` 后检查）；
启动 `uv run uvicorn app.main:app --port 8000`，`curl -s http://127.0.0.1:8000/chat | grep -c "data-action"`（应 ≥1，确认代码已上页）。
浏览器手工验收步骤写入报告：投诉 → 两个按钮；分别点击；不点继续聊天。

- [ ] **Step 3: Commit**

```bash
git add web/index.html
git commit -m "feat: 前端转人工/建工单按钮与确认交互"
```

---

### Task 12: 五条验收测试

**Files:**
- Create: `tests/acceptance/test_workflow_acceptance.py`

**Interfaces:**
- Consumes: 全链路（图、API、工具、DB）。

- [ ] **Step 1: 写验收测试**

1. `test_policy_question_hits_retrieve_node`：POST “退货政策是什么”，断言 `caplog` 含 `node=retrieve`（或响应里通过调试元数据暴露 `node_trace`；优先 caplog，不改 API）。
2. `test_logistics_question_uses_agent_tool`：fake model 脚本先 `query_logistics("1001")` 再回答；断言 tool start/end 事件含 `query_logistics`。
3. `test_complaint_actions_are_independent`：
   - POST “我要投诉”，断言 `done.actions` 同时含 `handoff`/`ticket`，且 tickets 表仍为空；
   - POST `/api/tickets`（模拟点建工单）后 tickets 表有且仅有一行；
   - “转人工”不调后端（本测试不调任何接口即证明表未变）。
4. `test_chitchat_fixed_text`：POST “今天天气怎么样”，delta 拼接等于 `CHITCHAT_TEXT`。
5. `test_react_takes_multiple_steps`：fake model 依次 `query_order` → `query_logistics` → 回答；断言两次工具事件、`steps >= 2`（通过 caplog 或响应调试字段）。

- [ ] **Step 2: 运行**

Run: `uv run pytest tests/acceptance/test_workflow_acceptance.py -v`
Expected: 5 passed。

- [ ] **Step 3: 完整套件**

Run: `uv run pytest -q`
Expected: 全部 PASS（342 + 新增）。

- [ ] **Step 4: Commit**

```bash
git add tests/acceptance/test_workflow_acceptance.py
git commit -m "test: ch05 五条验收测试"
```

---

### Task 13: 文档、留痕与最终评审

**Files:**
- Modify: `README.md`
- Modify: `dev-notes/ch05.md`

- [ ] **Step 1: 更新 README**

新增 ch05 段：架构图、启动方式（新增依赖、checkpoint 文件目录）、意图/路由说明、动作按钮说明、验收命令、已知限制（意图最简版、置信度词面覆盖率、SQLite 单 worker）。

- [ ] **Step 2: 补 dev-notes**

按阶段补记：计划评审通过、每个任务完成、评审结论、纠偏、翻车与返工；结尾放交付命令、测试结果、留痕路径。

- [ ] **Step 3: 完整套件与演示**

Run: `uv run pytest -q`；`uv run python -m evals.run_intent_eval`；`uv run python -m evals.run_hybrid_evals`（确认 ch04 不回归）。
Expected: 套件全绿；意图准确率 ≥0.80；四策略仍能出数字。

- [ ] **Step 4: Commit**

```bash
git add README.md dev-notes/ch05.md
git commit -m "docs: ch05 文档、留痕与交付说明"
```

---

## Self-Review

**1. Spec 覆盖**

| Spec 章节 | 任务 |
|---|---|
| §3 图骨架/分流表 | Task 3（路由表）、Task 8（装配） |
| §4 State/重置/sanitize | Task 3 |
| §5.1 指代消解 | Task 3 |
| §5.2 意图识别 | Task 4 |
| §5.3 路由 | Task 3 |
| §5.4 检索 | Task 6 |
| §5.5 置信度闸 | Task 6 |
| §5.6 插槽门 | Task 5（+Task 4 等待态） |
| §5.7 ReAct/全局编号/标记 | Task 7 |
| §5.8-5.10 投诉/闲聊/拒答 | Task 8（节点）+ Task 7（文案在 texts.py） |
| §5.11 日志 | Task 8 |
| §6 SSE/前端 | Task 9、Task 11 |
| §7 工单端点 | Task 10 |
| §8 checkpointer/审计 | Task 8、Task 9 |
| §9 配置 | Task 1 |
| §10 裸循环 | Task 2 |
| §11 测试策略 | Task 4（标注集）、Task 12（验收）、各任务单测 |
| §12 验收映射 | Task 12 |
| §13 风险 | 见各任务边界测试 |

**2. 步骤扫描**：每个任务都有失败测试 → 实现 → 通过 → commit；Prompt 类（Task 4）用标注集替代 TDD。无“TBD/适当处理”空话。

**3. 类型一致性**：`WorkflowState` 字段、`assign_citations` 四元组返回、`parse_action_markers` 二元组返回、`ROUTE_BY_INTENT` 键值与 spec 一致；`search_with_metadata` 签名沿用 ch04。

**4. Review Focus 对应测试**：
1. 等待中换话题 → Task 4 `test_classify_new_question_clears_waiting_state` + Task 8 跨轮测试。
2. 断流悬空 → Task 3 `test_sanitize_drops_dangling_tool_calls` + Task 8 恢复测试。
3. 动作独立/重复点击 → Task 10 表断言 + Task 11 前端 disable + Task 12 验收 3。
4. 引用编号一致 → Task 6 `test_assign_citations_numbers_and_dedupes` + Task 7 `test_query_faq_uses_global_citation_numbers`。
5. 停止条件/空输入 → Task 7 `test_agent_stops_at_max_steps`；空输入在 Task 9 适配层加一条 `test_blank_message_returns_graceful_reply`（422 或固定提示），实现时补。

**5. 比例**：本计划约 600 行，spec 约 360 行；代码块以测试断言和签名为准，函数体留给实现者。
