import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat import (
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    ToolCallEvent,
    ToolResultEvent,
    stream_reply_with_tools,
)
from app.config import get_settings
from app.db.base import get_db
from app.extract import ExtractionFailed, extract_ticket
from app.providers import get_default_chat_model
from app.repositories import conversation as conv_repo
from app.schemas import AfterSalesTicket, ChatRequest, ExtractRequest
from app.sessions import MySQLSessionStore, SessionStore
from app.tools import build_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


def get_store(db: AsyncSession = Depends(get_db)) -> SessionStore:
    """依赖注入点：测试通过 app.dependency_overrides 替换它。"""
    return MySQLSessionStore(session=db)


def sse_frame(name: str, data: dict) -> str:
    """把领域事件封装成一个 SSE 帧。这是本项目唯一产出 SSE 文本的地方。"""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(
    req: ChatRequest,
    store: SessionStore = Depends(get_store),
    model=Depends(get_default_chat_model),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    # is_new_session 必须和 session_id 同源，否则空串 "" 会走到
    # "服务端生成了 id 却不发 meta" 的岔路：客户端永远学不到 id，
    # 会话被静默孤立（spec §5.1）。这里以最终 id 为准，一个表达式派生两者。
    session_id = req.session_id or str(uuid.uuid4())
    is_new_session = req.session_id != session_id
    settings = get_settings()

    # 先读出历史（不含当前消息），剪裁由 stream_reply_with_tools 内部完成
    history = await store.get(session_id)

    # 会话行在这里就建出来：create_ticket 要拿 conversation_id 当外键，
    # 工具执行时它必须已经存在（flush 出 id 即可，提交与否见 finally）。
    conv = await conv_repo.get_or_create(db, session_id, settings.default_user_id)

    # ★ build_registry 必须在请求处理函数里构造：它绑定的是**当前请求**的库
    # 会话。放到模块级会持有一个早已随上个请求关闭的 session，第二个请求起
    # 工具全部报错。
    registry = build_registry(session=db)

    async def event_stream() -> AsyncIterator[str]:
        if is_new_session:
            yield sse_frame("meta", {"session_id": session_id})

        # 落库用的累积器（spec §7.1 步骤 ⑧）
        # 工具轮之前的正文 → preamble；工具轮之后的正文 → final_answer。
        # 分开放是因为它们分属两条 assistant 消息：带 tool_calls 的那条，
        # 和最终回答那条。混成一条会让历史回放失真。
        preamble: list[str] = []
        final_answer: list[str] = []
        tool_calls_seen: list[dict] = []
        tool_messages: list[ToolMessage] = []
        in_tool_round = False
        completed = False

        try:
            async for event in stream_reply_with_tools(
                req.message,
                history,
                model=model,
                registry=registry,
                token_budget=settings.history_token_budget,
                conversation_id=conv.id,
            ):
                if isinstance(event, ToolCallEvent):
                    in_tool_round = True
                    tool_calls_seen.append({
                        "id": event.call_id, "name": event.name,
                        "args": event.arguments, "type": "tool_call",
                    })
                    yield sse_frame("tool", {
                        "phase": "start", "call_id": event.call_id,
                        "name": event.name, "arguments": event.arguments,
                    })
                elif isinstance(event, ToolResultEvent):
                    # ★ 两个文本字段各走各的路：帧里只放给前端看的摘要，
                    # 库里存回灌给模型的全文（存摘要 = 下一轮模型只看到截断结果）。
                    tool_messages.append(
                        ToolMessage(content=event.content, tool_call_id=event.call_id)
                    )
                    yield sse_frame("tool", {
                        "phase": "end", "call_id": event.call_id,
                        "ok": event.ok, "summary": event.summary,
                        "elapsed_ms": event.elapsed_ms,
                    })
                elif isinstance(event, DeltaEvent):
                    (final_answer if in_tool_round else preamble).append(event.text)
                    yield sse_frame("delta", {"text": event.text})
                elif isinstance(event, DoneEvent):
                    completed = True
                    yield sse_frame("done", {
                        "finish_reason": event.finish_reason, "usage": event.usage,
                    })
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
                # ★ 完整序列：user → assistant(带 tool_calls) → tool 结果* → 最终 assistant
                # 只存 user + 最终 assistant 会让下一轮回放历史时 tool_call_id
                # 指向一条不存在的申请单，上游会直接报错——而且只在**第二轮**
                # 才现形，单轮用例看不出来（见 tests/test_api_tools.py）。
                to_store: list[BaseMessage] = [HumanMessage(req.message)]
                if tool_calls_seen:
                    to_store.append(AIMessage(
                        content="".join(preamble), tool_calls=tool_calls_seen
                    ))
                    to_store.extend(tool_messages)
                    to_store.append(AIMessage("".join(final_answer)))
                else:
                    to_store.append(AIMessage("".join(preamble)))
                await store.append(session_id, *to_store)
            # ★ 无条件提交，两件事分开看：get_or_create 产生的**会话行**即使
            # 这一轮失败也要保留（否则下次请求又要重建，工具的外键还得先有），
            # 而业务消息只在 completed 时才写。get_db 是 `async with` 形式，
            # 不提交则未提交的写入在退出时被回滚——消息永远不落库。
            await db.commit()

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
