from langchain_core.messages import AIMessageChunk, HumanMessage

from app.chat import (
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
    stream_reply_with_tools,
)
from app.tools.registry import ToolOutcome, ToolSpec
from app.tools.registry import ToolRegistry


class ScriptedTool:
    """一个假的 LangChain 工具，只为让 registry 能注册它。"""

    def __init__(self, name):
        self.name = name
        from pydantic import BaseModel

        class Args(BaseModel):
            order_id: str = "1001"

        self.args_schema = Args

    async def ainvoke(self, args):
        return f"{self.name} 的结果：已到达杭州中转站"


def make_registry(names=("query_logistics",)):
    return ToolRegistry([ToolSpec(tool=ScriptedTool(n), timeout_s=1.0, max_retries=0)
                         for n in names])


class NoToolModel:
    """第一个增量就是正文 —— 模型不需要工具。"""

    def __init__(self):
        self.bound_tools = None
        self.calls = 0

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def astream(self, messages):
        self.calls += 1
        for t in ["您", "好"]:
            yield AIMessageChunk(content=t)


class ToolThenAnswerModel:
    """第一次吐工具申请，第二次吐正文。记录每次调用绑了哪些工具。"""

    def __init__(self):
        self.bound_history: list[list] = []
        self.seen_messages: list[list] = []
        self._bound: list = []

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        self.bound_history.append(list(self._bound))
        self.seen_messages.append(list(messages))
        if len(self.bound_history) == 1:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": "query_logistics", "args": '{"order_id": "1001"}',
                     "id": "call_1", "index": 0, "type": "tool_call_chunk"}
                ],
            )
        else:
            for t in ["已", "到", "杭", "州"]:
                yield AIMessageChunk(content=t)


async def collect(gen):
    return [e async for e in gen]


async def test_no_tool_path_streams_directly():
    model = NoToolModel()
    events = await collect(stream_reply_with_tools(
        "你好", [], model=model, registry=make_registry(), token_budget=4096, conversation_id=1
    ))
    assert [type(e).__name__ for e in events] == ["DeltaEvent", "DeltaEvent", "DoneEvent"]
    assert model.calls == 1, "不需要工具时只调一次模型"


async def test_tool_path_emits_tool_events_then_deltas():
    model = ToolThenAnswerModel()
    events = await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=model, registry=make_registry(),
        token_budget=4096, conversation_id=1,
    ))
    names = [type(e).__name__ for e in events]
    assert names == [
        "ToolCallEvent", "ToolResultEvent", "DeltaEvent", "DeltaEvent", "DeltaEvent",
        "DeltaEvent", "DoneEvent",
    ]
    call = next(e for e in events if isinstance(e, ToolCallEvent))
    assert call.name == "query_logistics"
    assert call.arguments == {"order_id": "1001"}
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.ok is True
    assert "杭州" in result.summary


async def test_second_call_has_no_tools_bound():
    """★ 单轮收敛的结构性保证：第二次调用不绑工具，模型不可能再申请工具。"""
    model = ToolThenAnswerModel()
    await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=model, registry=make_registry(),
        token_budget=4096, conversation_id=1,
    ))
    assert len(model.bound_history) == 2
    assert len(model.bound_history[0]) == 1, "第一次要绑工具"
    assert model.bound_history[1] == [], "第二次绝不能绑工具"


async def test_second_call_receives_tool_result_message():
    from langchain_core.messages import ToolMessage

    model = ToolThenAnswerModel()
    await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=model, registry=make_registry(),
        token_budget=4096, conversation_id=1,
    ))
    second = model.seen_messages[1]
    assert isinstance(second[-1], ToolMessage)
    assert second[-1].tool_call_id == "call_1"
    assert "杭州" in second[-1].content


async def test_failing_tool_is_fed_back_not_raised():
    """工具失败回灌成 tool 结果，让模型说人话——不产生 ErrorEvent。"""

    class Boom:
        name = "query_logistics"

        def __init__(self):
            from pydantic import BaseModel

            class Args(BaseModel):
                order_id: str = "1001"

            self.args_schema = Args

        async def ainvoke(self, args):
            raise RuntimeError("上游炸了")

    reg = ToolRegistry([ToolSpec(tool=Boom(), timeout_s=1.0, max_retries=0)])
    model = ToolThenAnswerModel()
    events = await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=model, registry=reg,
        token_budget=4096, conversation_id=1,
    ))
    assert not any(isinstance(e, ErrorEvent) for e in events)
    result = next(e for e in events if isinstance(e, ToolResultEvent))
    assert result.ok is False
    assert any(isinstance(e, DeltaEvent) for e in events), "回灌后模型仍要作答"


class PreambleThenToolModel:
    """★ 实测出来的真实行为：模型先吐一段正文，然后才发工具申请。

    第一版设计按「第一个增量是正文 → 判定不需要工具」来分叉，遇到这个模型
    会走错分支——工具根本不执行，用户只看到"我帮您查一下"然后没有下文。
    这条测试就是防那个回归的。
    """

    def __init__(self):
        self.bound_history: list[list] = []

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        self.bound_history.append(list(self._bound))
        if len(self.bound_history) == 1:
            yield AIMessageChunk(content="好的，我帮您查一下")
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "query_logistics", "args": '{"order_id": "1001"}',
                    "id": "call_p", "index": 0, "type": "tool_call_chunk",
                }],
            )
        else:
            yield AIMessageChunk(content="您的包裹已到达杭州中转站。")


async def test_preamble_then_tool_still_executes_tool():
    """★ 回归测试：先说后调，工具也必须执行。

    这是本章设计改动的原因（spec §7.2/§7.7）。如果哪天有人把实现改回
    「看第一个增量就分叉」，这条测试会红。
    """
    model = PreambleThenToolModel()
    events = await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=model, registry=make_registry(),
        token_budget=4096, conversation_id=1,
    ))
    kinds = [type(e).__name__ for e in events]

    # 前言要转发出去（用户看得到），工具要执行，最终回答要有
    assert kinds[0] == "DeltaEvent", "前言应该被实时转发"
    assert "ToolCallEvent" in kinds, "先说后调时工具仍必须执行"
    assert "ToolResultEvent" in kinds
    assert kinds[-1] == "DoneEvent"

    # 工具在第二次请求之前执行完
    assert kinds.index("ToolResultEvent") < kinds.index("DeltaEvent", kinds.index("ToolResultEvent"))

    # 第二次仍然不绑工具
    assert model.bound_history[1] == []


async def test_no_tool_round_means_no_tool_events():
    """不需要工具时不该出现任何 tool 事件。"""
    model = NoToolModel()
    events = await collect(stream_reply_with_tools(
        "你好", [], model=model, registry=make_registry(), token_budget=4096, conversation_id=1
    ))
    kinds = [type(e).__name__ for e in events]
    assert "ToolCallEvent" not in kinds
    assert "ToolResultEvent" not in kinds


async def test_current_message_is_last_in_rendered_prompt():
    """当前用户消息必须排在渲染结果最后，历史在它前面。"""

    class RecordingModel:
        def __init__(self):
            self.seen = None

        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            self.seen = list(messages)
            yield AIMessageChunk(content="好")

    model = RecordingModel()
    await collect(stream_reply_with_tools(
        "最新提问", [HumanMessage("之前的话")], model=model,
        registry=make_registry(), token_budget=4096, conversation_id=1,
    ))
    contents = [m.content for m in model.seen]
    assert contents[0].startswith("你是「小助手」"), "第一条应是 System Prompt"
    assert contents[1:] == ["之前的话", "最新提问"]


class MetadataModel:
    """带真实元数据的桩：第一次申请工具、第二次作答，两次都带
    response_metadata / usage —— 裸 chunk 的桩看不到这些值。"""

    USAGE = {"input_tokens": 31, "output_tokens": 7, "total_tokens": 38}

    def __init__(self, *, tool_call: bool = True):
        self._bound = []
        self._tool_call = tool_call

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        if self._bound and self._tool_call:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "query_logistics", "args": '{"order_id": "1001"}',
                    "id": "call_m", "index": 0, "type": "tool_call_chunk",
                }],
                response_metadata={"finish_reason": "tool_calls"},
            )
        else:
            yield AIMessageChunk(
                content="已到达杭州。",
                response_metadata={"finish_reason": "length"},
                usage_metadata=MetadataModel.USAGE,
            )


async def test_tool_path_done_event_carries_real_metadata():
    """★ 工具轮的 DoneEvent 不得写死 "stop"：被上游截断（length）的轮次必须
    如实上报，否则客户端以为这是一次干净结束。"""
    events = await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=MetadataModel(), registry=make_registry(),
        token_budget=4096, conversation_id=1,
    ))
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.finish_reason == "length"
    assert done.usage == MetadataModel.USAGE


async def test_no_tool_path_done_event_carries_real_metadata():
    events = await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=MetadataModel(tool_call=False),
        registry=make_registry(), token_budget=4096, conversation_id=1,
    ))
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.finish_reason == "length"
    assert done.usage == MetadataModel.USAGE


async def test_done_event_metadata_is_none_when_chunks_are_bare():
    """裸 chunk（无元数据）时两个字段是 None —— 不编造一个 "stop"。

    与 ch01 `stream_reply` 的行为一致（tests/test_chat.py::test_done_event_
    carries_finish_reason）。
    """
    model = ToolThenAnswerModel()
    events = await collect(stream_reply_with_tools(
        "订单 1001 的物流到哪了", [], model=model, registry=make_registry(),
        token_budget=4096, conversation_id=1,
    ))
    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.finish_reason is None
    assert done.usage is None

