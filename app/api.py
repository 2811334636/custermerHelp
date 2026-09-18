import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.chat import (
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    stream_reply,
)
from app.config import get_settings
from app.extract import ExtractionFailed, extract_ticket
from app.providers import get_default_chat_model
from app.schemas import AfterSalesTicket, ChatRequest, ExtractRequest
from app.sessions import InMemorySessionStore, SessionStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_store = InMemorySessionStore(
    max_sessions=get_settings().session_max_sessions,
    max_messages=get_settings().session_max_messages,
)


def get_store() -> SessionStore:
    """依赖注入点：测试通过 app.dependency_overrides 替换它。"""
    return _store


def sse_frame(name: str, data: dict) -> str:
    """把领域事件封装成一个 SSE 帧。这是本项目唯一产出 SSE 文本的地方。"""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(
    req: ChatRequest,
    store: SessionStore = Depends(get_store),
    model=Depends(get_default_chat_model),
) -> StreamingResponse:
    # is_new_session 必须和 session_id 同源，否则空串 "" 会走到
    # "服务端生成了 id 却不发 meta" 的岔路：客户端永远学不到 id，
    # 会话被静默孤立（spec §5.1）。这里以最终 id 为准，一个表达式派生两者。
    session_id = req.session_id or str(uuid.uuid4())
    is_new_session = req.session_id != session_id
    settings = get_settings()

    # 先读出历史（不含当前消息），剪裁由 stream_reply 内部完成
    history = store.get(session_id)

    async def event_stream() -> AsyncIterator[str]:
        if is_new_session:
            yield sse_frame("meta", {"session_id": session_id})

        collected: list[str] = []
        completed = False
        try:
            async for event in stream_reply(
                req.message,
                history,
                model=model,
                token_budget=settings.history_token_budget,
            ):
                if isinstance(event, DeltaEvent):
                    collected.append(event.text)
                    yield sse_frame("delta", {"text": event.text})
                elif isinstance(event, DoneEvent):
                    completed = True
                    yield sse_frame(
                        "done",
                        {"finish_reason": event.finish_reason, "usage": event.usage},
                    )
                elif isinstance(event, ErrorEvent):
                    yield sse_frame("error", {"code": event.code, "message": event.message})
        except (GeneratorExit, asyncio.CancelledError):
            # 客户端中途断开（spec §9）：半截回复不写入历史，但要留痕，
            # 否则"回复莫名少了半截"在服务端完全不可见。
            #
            # 这里必须同时捕获 CancelledError：实测（真实 uvicorn + 客户端提前
            # 关流）starlette 的 StreamingResponse 是靠取消 task 来中断的，
            # 抛进生成器的是 CancelledError 而非 GeneratorExit；只捕获后者
            # 在真实断线下什么都不会记。GeneratorExit 是 asyncio 关闭生成器
            # 时的另一条路径，一并兜住。
            logger.info(
                "chat client disconnected mid-stream: session_id=%s", session_id
            )
            raise
        finally:
            # 只在整轮正常结束时写入历史。客户端断开或上游出错时整轮丢弃，
            # 否则会留下"两个 human 连排"的畸形历史（spec §7.2.1）。
            if completed:
                store.append(
                    session_id,
                    HumanMessage(req.message),
                    AIMessage("".join(collected)),
                )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 阻止反向代理缓冲，否则流式失效
        },
    )


@router.post("/extract", response_model=AfterSalesTicket)
async def extract(
    req: ExtractRequest,
    model=Depends(get_default_chat_model),
) -> AfterSalesTicket:
    try:
        return await extract_ticket(req.text, model=model)
    except ExtractionFailed as exc:
        logger.warning("extraction failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail={"code": "extraction_failed", "message": str(exc)},
        ) from exc
