from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.workflow.nodes import begin_turn, route_for, sanitize_messages
from app.workflow.state import ROUTE_BY_INTENT, WorkflowState


def test_route_table_is_exhaustive():
    assert set(ROUTE_BY_INTENT) == {"物流", "订单", "商品咨询", "退货退款", "售后", "投诉", "闲聊"}
    assert ROUTE_BY_INTENT["商品咨询"] == ROUTE_BY_INTENT["退货退款"] == "knowledge"
    assert ROUTE_BY_INTENT["物流"] == ROUTE_BY_INTENT["订单"] == ROUTE_BY_INTENT["售后"] == "business"
    assert ROUTE_BY_INTENT["投诉"] == "complaint"
    assert ROUTE_BY_INTENT["闲聊"] == "chitchat"
    assert route_for("未知") == "business"  # 兜底：售后/业务类
    assert route_for("投诉") == "complaint"


async def test_begin_turn_resets_turn_fields_and_sets_cursor():
    state = {
        "steps": 5,
        "tokens_used": 9000,
        "finish_reason": "length",
        "usage": {"total_tokens": 9000},
        "insufficient": True,
        "actions": [{"type": "handoff"}],
        "citations": [{"index": 1}],
        "node_trace": ["retrieve"],
        "awaiting_slot": "order_id",
        "pending_intent": "物流",
        "order_id": "1001",
    }
    out = await begin_turn(state)
    assert out["steps"] == 0 and out["tokens_used"] == 0
    assert out["actions"] == [] and out["citations"] == [] and out["node_trace"] == []
    assert out["finish_reason"] is None and out["usage"] is None
    assert out["insufficient"] is False
    assert out["citation_cursor"] == 1 and out["citation_index"] == {}
    assert out["awaiting_slot"] == "order_id" and out["order_id"] == "1001"  # 跨轮保留


async def test_sanitize_drops_dangling_tool_calls():
    msgs = [
        HumanMessage("查物流", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
        ),
    ]
    out = await sanitize_messages({"messages": msgs})
    # add_messages 的删除必须用 RemoveMessage；返回被删消息的 RemoveMessage 列表
    assert [type(m).__name__ for m in out["messages"]] == ["RemoveMessage"]
    assert out["messages"][0].id == "a1"


async def test_sanitize_keeps_complete_tool_round():
    msgs = [
        HumanMessage("查物流", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[{"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"}],
        ),
        ToolMessage(content="运输中", tool_call_id="c1", id="t1"),
        AIMessage(content="在运输中。", id="a2"),
    ]
    out = await sanitize_messages({"messages": msgs})
    assert out.get("messages", []) == []  # 没有可删的


async def test_begin_turn_keeps_cross_turn_fields_and_state_shape():
    state: WorkflowState = {
        "messages": [HumanMessage("你好", id="h1")],
        "intent": "物流",
        "route": "business",
        "evidence": [{"index": 1}],
        "evidence_coverage": 0.9,
        "gate_passed": True,
        "refusal_reason": "low_evidence",
        "pending_intent": "物流",
        "awaiting_slot": "order_id",
        "order_id": "1001",
    }
    out = await begin_turn(state)
    assert "messages" not in out
    # 跨轮字段原样回传，不重置
    assert out["pending_intent"] == "物流" and out["awaiting_slot"] == "order_id"
    assert out["intent"] == "" and out["route"] == ""
    assert out["evidence"] == [] and out["evidence_coverage"] is None
    assert out["gate_passed"] is None and out["refusal_reason"] is None


async def test_sanitize_removes_orphan_tool_message_but_keeps_new_human():
    msgs = [
        HumanMessage("查物流", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[
                {"name": "query_logistics", "args": {"order_id": "1001"}, "id": "c1"},
                {"name": "query_order", "args": {"order_id": "1001"}, "id": "c2"},
            ],
        ),
        ToolMessage(content="运输中", tool_call_id="c1", id="t1"),
        HumanMessage("你好", id="h2"),
    ]
    out = await sanitize_messages({"messages": msgs})
    assert [m.id for m in out["messages"]] == ["a1", "t1"]


async def test_sanitize_without_messages_is_noop():
    assert await sanitize_messages({}) == {}
