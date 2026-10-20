import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.api_chunks import router as chunks_router
from app.api_feedback import router as feedback_router
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
from app.repositories import low_confidence_question as lcq_repo
from app.schemas import AfterSalesTicket, ChatRequest, ExtractRequest
from app.sessions import MySQLSessionStore, SessionStore
from app.tools import build_registry

logger = logging.getLogger(__name__)

# spec §3.5.2：模型自评知识不足时输出该标记，后端替换为标准拒答话术。
INSUFFICIENT_MARKER = "<INSUFFICIENT_KNOWLEDGE>"
REFUSAL_TEXT = (
    "抱歉，我暂时没有找到相关信息。您可以：\n"
    "1. 换个说法再问一次\n"
    '2. 联系人工客服（输入"转人工"）'
)
REASON_MODEL_INSUFFICIENT = "model_insufficient"


class _MarkerStreamFilter:
    """在 SSE 增量里吞掉模型自评标记，并按需换成标准拒答话术。

    约束：
    - 只扣留「可能是标记前缀」的尾巴（最长 len(marker)-1 个字符），
      其余文本立刻放行——不能为了过滤牺牲流式延迟。
    - 标记可能被切在两个 delta 帧之间，所以必须跨帧累积判断。
    - 一旦看到完整标记，后续文本一律丢弃（标记后的内容按噪声处理），
      并在 flush 时返回一次拒答话术，保证浏览器看到的文本与落库一致。
    """

    def __init__(self, marker: str, replacement: str) -> None:
        self._marker = marker
        self._replacement = replacement
        self._buf = ""
        self.seen = False

    def feed(self, text: str) -> str:
        if self.seen:
            return ""

        self._buf += text
        idx = self._buf.find(self._marker)
        if idx != -1:
            prefix = self._buf[:idx]
            self._buf = ""
            self.seen = True
            return prefix

        # 扣住最长的「标记真前缀」尾巴，等下一帧补齐后再判断。
        hold = 0
        max_hold = min(len(self._marker) - 1, len(self._buf))
        for n in range(max_hold, 0, -1):
            if self._buf.endswith(self._marker[:n]):
                hold = n
                break
        if hold:
            emit, self._buf = self._buf[:-hold], self._buf[-hold:]
        else:
            emit, self._buf = self._buf, ""
        return emit

    def flush(self) -> str:
        """本轮文本结束时调用：标记出现过则补发拒答话术，否则放行残留尾巴。"""
        if self.seen:
            self._buf = ""
            return self._replacement
        emit, self._buf = self._buf, ""
        return emit


router = APIRouter(prefix="/api")
router.include_router(chunks_router)
router.include_router(feedback_router)


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
        done_event: DoneEvent | None = None
        # 请求级可变容器：query_faq 原地写入 citations，registry 把同一个
        # dict 挂到 ToolOutcome.metadata，工具帧与最终落库都从这里取。
        retrieval_meta: dict = {}
        citations: list[dict] = []
        # 浏览器只看 delta，不读落库正文：标记必须在流里就被替换，否则用户
        # 会看到 "<INSUFFICIENT_KNOWLEDGE>"。工具调用会开启新的 assistant
        # 消息段，届时 flush + 重建过滤器。
        delta_filter = _MarkerStreamFilter(INSUFFICIENT_MARKER, REFUSAL_TEXT)

        try:
            async for event in stream_reply_with_tools(
                req.message,
                history,
                model=model,
                registry=registry,
                token_budget=settings.history_token_budget,
                conversation_id=conv.id,
                retrieval_meta=retrieval_meta,
            ):
                if isinstance(event, ToolCallEvent):
                    # 前一段正文到此结束：把过滤器扣留的尾巴放出去，再为新
                    # 的最终回答段建一个干净的状态。
                    tail = delta_filter.flush()
                    if tail:
                        yield sse_frame("delta", {"text": tail})
                    delta_filter = _MarkerStreamFilter(
                        INSUFFICIENT_MARKER, REFUSAL_TEXT
                    )
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
                    if event.metadata is not None and "citations" in event.metadata:
                        citations = list(event.metadata.get("citations") or [])
                    # 前端没有 message-fetch 接口：citations 只能搭在工具帧
                    # 与 done 帧上送到浏览器（spec §3.5.1）。
                    yield sse_frame("tool", {
                        "phase": "end", "call_id": event.call_id,
                        "ok": event.ok, "summary": event.summary,
                        "elapsed_ms": event.elapsed_ms,
                        "citations": citations,
                    })
                elif isinstance(event, DeltaEvent):
                    # 累积原始文本用于落库；SSE 只发过滤后的文本。两者最终
                    # 都走同一条「标记 → 标准拒答」规则，保持一致。
                    (final_answer if in_tool_round else preamble).append(event.text)
                    filtered = delta_filter.feed(event.text)
                    if filtered:
                        yield sse_frame("delta", {"text": filtered})
                elif isinstance(event, DoneEvent):
                    completed = True
                    # 放行残留尾巴；若标记出现过，这里会补发 REFUSAL_TEXT，
                    # 使流里拼出来的正文与落库正文完全一致。
                    tail = delta_filter.flush()
                    if tail:
                        yield sse_frame("delta", {"text": tail})
                    # 先不 yield done：done 帧要带 message_id，而 id 只有在
                    # 落库 flush 之后才存在。持久化放到循环外统一做。
                    done_event = event
                elif isinstance(event, ErrorEvent):
                    yield sse_frame("error", {"code": event.code, "message": event.message})

            # ★ 持久化必须在 done 帧之前完成：message_id 是前端反馈
            # （POST /api/feedback）的唯一凭据，只能在 flush 后拿到。
            if completed and done_event is not None:
                answer_parts = final_answer if tool_calls_seen else preamble
                answer = "".join(answer_parts)
                insufficient = INSUFFICIENT_MARKER in answer
                if insufficient:
                    # spec §3.5.2：模型自评不足 → 替换为标准拒答话术 + 入池。
                    # 只保留标记之前的正文、丢弃标记后的噪声，与 SSE 过滤器
                    # 的行为严格一致。
                    answer = (
                        answer.split(INSUFFICIENT_MARKER, 1)[0] + REFUSAL_TEXT
                    )
                    if tool_calls_seen:
                        final_answer = [answer]
                    else:
                        preamble = [answer]

                # 工具轮前言里若也出现标记（罕见），同样规范化，保持落库
                # 文本与浏览器看到的流式文本一致。
                if tool_calls_seen:
                    preamble_text = "".join(preamble)
                    if INSUFFICIENT_MARKER in preamble_text:
                        preamble = [
                            preamble_text.split(INSUFFICIENT_MARKER, 1)[0]
                            + REFUSAL_TEXT
                        ]

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

                # ★ 入池与落库放进同一个 SAVEPOINT：append 可能已经 flush 了
                # user 行之后才抛错（或被取消）。若直接让 finally 提交，半截
                # 一轮会落库；回滚到保存点则只丢弃本轮业务写入，先前建好的
                # 会话行仍然保留。
                async with db.begin_nested():
                    # ★ 检索层已经拒答（no_results / low_score）时，同一问题
                    # 已经入过池；模型随后再吐 <INSUFFICIENT_KNOWLEDGE> 不能
                    # 再写一条 model_insufficient，否则同一轮出现两行。
                    if insufficient and not retrieval_meta.get("refusal_reason"):
                        await lcq_repo.create(
                            db,
                            req.message,
                            str(conv.id),
                            REASON_MODEL_INSUFFICIENT,
                        )
                    ids = await store.append(
                        session_id,
                        *to_store,
                        metadata=retrieval_meta or None,
                    )
                    citations = list(retrieval_meta.get("citations") or [])
                assistant_message_id = ids[-1] if ids else None

                # message_id 必须在客户端收到 done 时已经持久化（反馈接口
                # 的唯一凭据）。finally 的 commit 保留为无害的 no-op。
                await db.commit()

                yield sse_frame("done", {
                    "finish_reason": done_event.finish_reason,
                    "usage": done_event.usage,
                    "message_id": assistant_message_id,
                    "citations": citations,
                })
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
            # ★ 只提交、不再在 finally 里写消息：持久化已经挪到 done 帧
            # 之前（message_id 需要 flush 后的 id）。客户端断开或上游出错时
            # completed=False，上面的落库分支根本不会执行，历史仍然零痕迹。
            # 无条件提交是另一件事：get_or_create 产生的**会话行**即使这一轮
            # 失败也要保留（否则下次请求又要重建，工具的外键还得先有）。
            # get_db 是 `async with` 形式，不提交则未提交的写入在退出时被回滚。
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
