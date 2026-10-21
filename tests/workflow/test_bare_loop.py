import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class ScriptedModel:
    """按脚本依次返回 AIMessage；记录调用次数。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        return self.replies.pop(0)


class FakeRegistry:
    def __init__(self, result="工具结果"):
        self.result = result
        self.calls = []

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
