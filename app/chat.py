import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

# 例外说明：spec §4.2 要求除 providers.py 外其余模块只依赖 BaseChatModel 接口，
# 这里却直接 import 了 openai SDK。这是**有意为之的已知泄漏**：classify_exception
# 需要按上游异常的具体类型（AuthenticationError / RateLimitError / ...）分错误码，
# 而 BaseChatModel.astream 的契约只保证抛 BaseException，不暴露类型层级。
# 若以后换 provider，这个文件是除 providers.py 外唯一需要跟着改的地方。
import openai
from langchain_core.messages import BaseMessage, HumanMessage

from app.context import prepare_messages
from app.prompts import build_chat_prompt
from app.providers import get_chat_model

logger = logging.getLogger(__name__)


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
        code = classify_exception(exc)
        # spec §9：失败轮次必须在服务端留痕（错误码 + 原始异常类型），
        # 否则每一个失败对服务端都是隐形的，只剩客户端能看见。
        logger.warning("chat turn failed: code=%s exc=%s", code, type(exc).__name__)
        yield ErrorEvent(code=code, message=str(exc)[:500])
        return

    yield DoneEvent(finish_reason=finish_reason, usage=usage)
