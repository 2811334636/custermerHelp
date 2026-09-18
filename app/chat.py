from collections.abc import AsyncIterator
from dataclasses import dataclass

import openai
from langchain_core.messages import BaseMessage, HumanMessage

from app.context import prepare_messages
from app.prompts import build_chat_prompt
from app.providers import get_chat_model


@dataclass(frozen=True)
class DeltaEvent:
    text: str


@dataclass(frozen=True)
class DoneEvent:
    finish_reason: str | None = None
    usage: dict | None = None


@dataclass(frozen=True)
class ErrorEvent:
    code: str
    message: str


DomainEvent = DeltaEvent | DoneEvent | ErrorEvent

_AUTH_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError)
_UNAVAILABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def classify_exception(exc: BaseException) -> str:
    """把上游异常映射成对外的错误码（spec §9）。"""
    if isinstance(exc, _AUTH_ERRORS):
        return "upstream_auth"
    if isinstance(exc, openai.RateLimitError):
        return "upstream_rate_limit"
    if isinstance(exc, _UNAVAILABLE_ERRORS):
        return "upstream_unavailable"
    if isinstance(exc, openai.APIStatusError):
        return "upstream_unavailable"
    return "internal_error"


async def stream_reply(
    message: str,
    history: list[BaseMessage],
    *,
    model=None,
    token_budget: int,
) -> AsyncIterator[DomainEvent]:
    """跑一轮对话，产出领域事件。

    只吐事件，不碰 HTTP —— SSE 帧由 api.py 负责（spec §4.2）。
    出错时只发 ErrorEvent 且不补发 DoneEvent（spec §5.1）。
    """
    model = model or get_chat_model()
    messages = prepare_messages(
        history, HumanMessage(message), token_budget=token_budget
    )
    # 展开为 [SystemMessage, *剪裁后历史, 当前用户消息]：system 文案的唯一来源
    # 是 app/prompts.py 的模板，这里不自己拼 SystemMessage。
    rendered = build_chat_prompt().format_messages(history=messages)

    finish_reason: str | None = None
    usage: dict | None = None

    try:
        async for chunk in model.astream(rendered):
            if chunk.text:
                yield DeltaEvent(text=chunk.text)
            meta = getattr(chunk, "response_metadata", None) or {}
            if meta.get("finish_reason"):
                finish_reason = meta["finish_reason"]
            if getattr(chunk, "usage_metadata", None):
                usage = dict(chunk.usage_metadata)
    except Exception as exc:  # noqa: BLE001 — 需要对上游任意异常做分类
        yield ErrorEvent(code=classify_exception(exc), message=str(exc)[:500])
        return

    yield DoneEvent(finish_reason=finish_reason, usage=usage)
