from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.db.messages_io import from_rows, to_rows


class FakeRow:
    """模拟 app.db.models.Message 的行对象。"""

    def __init__(self, role, content, tool_calls=None, tool_call_id=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id


def test_human_message_roundtrip():
    rows = to_rows([HumanMessage("你好")])
    assert rows == [{"role": "user", "content": "你好", "tool_calls": None, "tool_call_id": None}]
    back = from_rows([FakeRow(**rows[0])])
    assert isinstance(back[0], HumanMessage)
    assert back[0].content == "你好"


def test_plain_ai_message_roundtrip():
    rows = to_rows([AIMessage("您好")])
    assert rows[0]["role"] == "assistant"
    back = from_rows([FakeRow(**rows[0])])
    assert isinstance(back[0], AIMessage)
    assert back[0].content == "您好"
    assert back[0].tool_calls == []


def test_ai_message_with_tool_calls_roundtrip():
    # langchain-core 1.x 会把 tool_calls 规范化成 ToolCall typed dict
    # （补上 type="tool_call"），故期望值按规范化后的形状写。
    calls = [
        {"id": "call_1", "name": "query_order", "args": {"order_id": "1001"}, "type": "tool_call"}
    ]
    msg = AIMessage(content="", tool_calls=calls)
    rows = to_rows([msg])
    assert rows[0]["role"] == "assistant"
    assert rows[0]["tool_calls"] == calls

    back = from_rows([FakeRow(**rows[0])])
    assert isinstance(back[0], AIMessage)
    assert back[0].tool_calls == calls


def test_tool_message_roundtrip():
    msg = ToolMessage(content="查到 3 条轨迹", tool_call_id="call_1")
    rows = to_rows([msg])
    assert rows[0] == {
        "role": "tool", "content": "查到 3 条轨迹",
        "tool_calls": None, "tool_call_id": "call_1",
    }
    back = from_rows([FakeRow(**rows[0])])
    assert isinstance(back[0], ToolMessage)
    assert back[0].content == "查到 3 条轨迹"
    assert back[0].tool_call_id == "call_1"


def test_full_conversation_order_preserved():
    msgs = [
        HumanMessage("订单 1001 的物流到哪了"),
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "query_logistics", "args": {"order_id": "1001"}}]),
        ToolMessage(content="已到达杭州中转站", tool_call_id="c1"),
        AIMessage("您的包裹已到达杭州中转站。"),
    ]
    rows = to_rows(msgs)
    back = from_rows([FakeRow(**r) for r in rows])
    assert [type(m).__name__ for m in back] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]
    assert back[2].tool_call_id == "c1"
