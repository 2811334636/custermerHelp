"""领域事件层的行为（ch01 那批断言，ch02 修复轮改驱动生产函数）。

★ 为什么这里调 `stream_reply_with_tools` 而不是已经删掉的 ch01 `stream_reply`
（修复轮 · Important 3）：两者是同一份 delta 循环 / 异常分类 / 元数据吸收的两份
副本。`app/api.py` 只 import `stream_reply_with_tools`，ch01 那份**生产上从不
执行**，却还留着 12 条绿灯——将来改活路径改错了对象，这 12 条会替作者背书。
删掉副本后本文件改为驱动生产函数 + 一个**空 registry**：不绑任何工具时它走的
就是同一条直通路径（`tests/test_chat_tools.py::test_no_tool_path_streams_directly`
覆盖同一件事）。**每一条断言原样保留，只是换了入口**。
"""

import openai
import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from app.chat import (
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    classify_exception,
    stream_reply_with_tools,
)
from app.prompts import SYSTEM_PROMPT
from app.tools.registry import ToolRegistry

# 空 registry：不绑工具 → stream_reply_with_tools 走"第一次流结束后没有
# tool_calls"的直通分支，与 ch01 `stream_reply` 的行为逐字一致。
EMPTY_REGISTRY = ToolRegistry([])


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

    def bind_tools(self, tools):
        """生产函数第一次请求会 bind_tools（这里是空列表）；返回 self 即可。"""
        return self

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


async def _run(message, history=None, *, model):
    return await collect(
        stream_reply_with_tools(
            message,
            history or [],
            model=model,
            registry=EMPTY_REGISTRY,
            token_budget=4096,
            conversation_id=1,
        )
    )


async def test_streams_deltas_then_done():
    model = FakeModel(script=["您", "好", "，", "很高兴为您服务"])
    events = await _run("你好", model=model)
    assert [type(e).__name__ for e in events] == [
        "DeltaEvent", "DeltaEvent", "DeltaEvent", "DeltaEvent", "DoneEvent",
    ]
    assert "".join(e.text for e in events if isinstance(e, DeltaEvent)) == "您好，很高兴为您服务"
    assert isinstance(events[-1], DoneEvent)


async def test_current_message_is_appended_to_prompt():
    """模型收到的应是 [system, ...历史, 当前消息]。"""
    model = FakeModel(script=["好"])
    await _run("我的订单号是多少", [HumanMessage("我叫张三")], model=model)
    contents = [m.content for m in model.seen_messages]
    assert contents[0] == SYSTEM_PROMPT
    assert contents[1:] == ["我叫张三", "我的订单号是多少"]


async def test_system_prompt_is_first_message():
    model = FakeModel(script=["好"])
    await _run("你好", model=model)
    assert model.seen_messages[0].type == "system"


async def test_done_event_carries_finish_reason():
    """裸 chunk（无 metadata）时 DoneEvent 的字段保持 None，不崩。"""
    model = FakeModel(script=["好"])
    events = await _run("你好", model=model)
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
    events = await _run("你好", model=model)

    done = events[-1]
    assert isinstance(done, DoneEvent)
    assert done.finish_reason == "stop"
    assert done.usage == usage


async def test_error_after_partial_deltas_is_midstream_failure():
    """I4.2：流中失败（已吐部分 delta 再抛）也必须终结为 ErrorEvent。

    此前只测过"首个 chunk 之前就抛"，流中途失败是未覆盖的路径。
    """
    events = await _run(
        "你好", model=FakeModel(script=["您", "好"], raises_after=_boom())
    )
    assert [type(e).__name__ for e in events] == ["DeltaEvent", "DeltaEvent", "ErrorEvent"]
    assert events[-1].code == "internal_error"
    assert not any(isinstance(e, DoneEvent) for e in events)


async def test_upstream_error_yields_error_event_not_exception():
    err = openai.AuthenticationError(
        "bad key", response=_resp(401), body=None
    )
    events = await _run("你好", model=FakeModel(raises=err))
    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert events[0].code == "upstream_auth"


async def test_error_event_is_terminal():
    """出错后不得再发 DoneEvent（spec §5.1）。"""
    events = await _run("你好", model=FakeModel(raises=_boom()))
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
