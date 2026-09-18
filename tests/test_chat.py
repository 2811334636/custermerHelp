import openai
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.chat import DeltaEvent, DoneEvent, ErrorEvent, classify_exception, stream_reply
from app.prompts import SYSTEM_PROMPT


def _resp(status):
    import httpx

    return httpx.Response(status, request=httpx.Request("POST", "https://x/"))


def _boom():
    return ValueError("boom")


class FakeModel:
    """打桩模型：按预设脚本吐 chunk，或按预设抛异常。

    - raises: 流开始前就抛（首个 chunk 之前）
    - raises_after: 吐完整个 script 之后再抛（模拟真实上游的**流中失败**）
    - chunks: 直接指定 chunk 对象，用来携带 response_metadata / usage_metadata
    """

    def __init__(self, script=None, raises=None, raises_after=None, chunks=None):
        self._script = script or []
        self._raises = raises
        self._raises_after = raises_after
        self._chunks = chunks

    async def astream(self, messages):
        self.seen_messages = messages
        if self._raises:
            raise self._raises
        for chunk in self._chunks or [AIMessageChunk(content=t) for t in self._script]:
            yield chunk
        if self._raises_after:
            raise self._raises_after


async def collect(gen):
    return [e async for e in gen]


async def test_streams_deltas_then_done():
    model = FakeModel(script=["您", "好", "，", "很高兴为您服务"])
    events = await collect(
        stream_reply("你好", [], model=model, token_budget=4096)
    )
    assert [type(e).__name__ for e in events] == [
        "DeltaEvent", "DeltaEvent", "DeltaEvent", "DeltaEvent", "DoneEvent",
    ]
    assert "".join(e.text for e in events if isinstance(e, DeltaEvent)) == "您好，很高兴为您服务"
    assert isinstance(events[-1], DoneEvent)


async def test_current_message_is_appended_to_prompt():
    """模型收到的应是 [system, ...历史, 当前消息]。"""
    model = FakeModel(script=["好"])
    await collect(
        stream_reply("我的订单号是多少", [HumanMessage("我叫张三")], model=model, token_budget=4096)
    )
    contents = [m.content for m in model.seen_messages]
    assert contents[0] == SYSTEM_PROMPT
    assert contents[1:] == ["我叫张三", "我的订单号是多少"]


async def test_system_prompt_is_first_message():
    model = FakeModel(script=["好"])
    await collect(stream_reply("你好", [], model=model, token_budget=4096))
    assert model.seen_messages[0].type == "system"


async def test_done_event_carries_finish_reason():
    """裸 chunk（无 metadata）时 DoneEvent 的字段保持 None，不崩。"""
    model = FakeModel(script=["好"])
    events = await collect(stream_reply("你好", [], model=model, token_budget=4096))
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].finish_reason is None
    assert events[-1].usage is None


async def test_done_event_extracts_finish_reason_and_usage():
    """I4.1：带 metadata 的 chunk 必须把 finish_reason / usage 送到 DoneEvent。

    否则这段抽取逻辑（app/chat.py）永远没被真正跑过 —— 旧断言写成
    `in (None, "stop")`，在裸 chunk 上恰好也通过，等于没测。
    """
    usage = {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17}
    chunks = [
        AIMessageChunk(content="您"),
        AIMessageChunk(content="好"),
        AIMessageChunk(
            content="",
            response_metadata={"finish_reason": "stop"},
            usage_metadata=usage,
        ),
    ]
    model = FakeModel(chunks=chunks)
    events = await collect(stream_reply("你好", [], model=model, token_budget=4096))

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.finish_reason == "stop"
    assert done.usage == usage


async def test_error_after_partial_deltas_is_midstream_failure():
    """I4.2：流中失败（已吐部分 delta 再抛）也必须终结为 ErrorEvent。

    此前只测过"首个 chunk 之前就抛"，流中途失败是未覆盖的路径。
    """
    events = await collect(
        stream_reply(
            "你好",
            [],
            model=FakeModel(script=["您", "好"], raises_after=_boom()),
            token_budget=4096,
        )
    )
    assert [type(e).__name__ for e in events] == ["DeltaEvent", "DeltaEvent", "ErrorEvent"]
    assert events[-1].code == "internal_error"
    assert not any(isinstance(e, DoneEvent) for e in events)


async def test_upstream_error_yields_error_event_not_exception():
    err = openai.AuthenticationError(
        "bad key", response=_resp(401), body=None
    )
    events = await collect(stream_reply("你好", [], model=FakeModel(raises=err), token_budget=4096))
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].code == "upstream_auth"


async def test_error_event_is_terminal():
    """出错后不得再发 DoneEvent（spec §5.1）。"""
    events = await collect(
        stream_reply("你好", [], model=FakeModel(raises=_boom()), token_budget=4096)
    )
    assert not any(isinstance(e, DoneEvent) for e in events)


@pytest.mark.parametrize(
    "exc,expected",
    [
        (openai.AuthenticationError("x", response=_resp(401), body=None), "upstream_auth"),
        (openai.PermissionDeniedError("x", response=_resp(403), body=None), "upstream_auth"),
        (openai.RateLimitError("x", response=_resp(429), body=None), "upstream_rate_limit"),
        (openai.InternalServerError("x", response=_resp(500), body=None), "upstream_unavailable"),
        (openai.APIConnectionError(request=None), "upstream_unavailable"),
        (ValueError("随便什么"), "internal_error"),
    ],
)
def test_classify_exception(exc, expected):
    assert classify_exception(exc) == expected
