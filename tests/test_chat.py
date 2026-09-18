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
    """打桩模型：按预设脚本吐 chunk，或按预设抛异常。"""

    def __init__(self, script=None, raises=None):
        self._script = script or []
        self._raises = raises

    async def astream(self, messages):
        self.seen_messages = messages
        if self._raises:
            raise self._raises
        for text in self._script:
            yield AIMessageChunk(content=text)


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
    model = FakeModel(script=["好"])
    events = await collect(stream_reply("你好", [], model=model, token_budget=4096))
    assert isinstance(events[-1], DoneEvent)
    assert events[-1].finish_reason in (None, "stop")


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
