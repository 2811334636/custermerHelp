import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.api_chunks import router as chunks_router
from app.api_feedback import router as feedback_router
from app.api_tickets import router as tickets_router
from app.chat import classify_exception
from app.config import get_settings
from app.db.base import get_db
from app.extract import ExtractionFailed, extract_ticket
from app.providers import get_default_chat_model
from app.repositories import conversation as conv_repo
from app.repositories import low_confidence_question as lcq_repo
from app.schemas import AfterSalesTicket, ChatRequest, ExtractRequest
from app.sessions import MySQLSessionStore, SessionStore
from app.tools import build_registry
from app.workflow.graph import workflow_deps

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
router.include_router(tickets_router)


def get_store(db: AsyncSession = Depends(get_db)) -> SessionStore:
    """依赖注入点：测试通过 app.dependency_overrides 替换它。"""
    return MySQLSessionStore(session=db)


def sse_frame(name: str, data: dict) -> str:
    """把领域事件封装成一个 SSE 帧。这是本项目唯一产出 SSE 文本的地方。"""
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _message_text(message: BaseMessage) -> str:
    """把消息正文归一成字符串（本项目的模型/节点都产出纯文本）。"""
    content = message.content
    return content if isinstance(content, str) else str(content)


def _normalize_insufficient(message: BaseMessage) -> BaseMessage:
    """落库前把最后一轮的自评标记替换成标准拒答话术。

    与 SSE 过滤器共用同一条规则：标记之前的正文保留，标记之后的内容
    （按噪声处理）丢弃。只改副本，不碰 checkpointer 里的 State。
    """
    if not isinstance(message, AIMessage):
        return message
    text = _message_text(message)
    if INSUFFICIENT_MARKER not in text:
        return message
    cleaned = text.split(INSUFFICIENT_MARKER, 1)[0] + REFUSAL_TEXT
    return message.model_copy(update={"content": cleaned})


@router.post("/chat")
async def chat(
    req: ChatRequest,
    request: Request,
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
    # lifespan 里编译一次的图；checkpointer 才是模型上下文的唯一来源。
    workflow = request.app.state.workflow

    # 会话行在这里就建出来：create_ticket 要拿 conversation_id 当外键，
    # 工具执行时它必须已经存在（flush 出 id 即可，提交与否见 finally）。
    conv = await conv_repo.get_or_create(db, session_id, settings.default_user_id)

    # ★ build_registry 必须在请求处理函数里构造：它绑定的是**当前请求**的库
    # 会话。放到模块级会持有一个早已随上个请求关闭的 session，第二个请求起
    # 工具全部报错。
    registry = build_registry(session=db)

    # 请求级依赖只放 configurable，绝不进 State（见 app/workflow/graph.py）。
    config = {
        "configurable": {
            "thread_id": session_id,
            **workflow_deps(model, registry, db, conv.id),
        }
    }
    initial = {
        "messages": [HumanMessage(req.message)],
        "user_input": req.message,
        "max_steps": settings.agent_max_steps,
        "token_budget": settings.agent_token_budget,
    }

    async def event_stream() -> AsyncIterator[str]:
        if is_new_session:
            yield sse_frame("meta", {"session_id": session_id})

        # 浏览器只看 delta，不读落库正文：标记必须在流里就被替换，否则用户
        # 会看到 "<INSUFFICIENT_KNOWLEDGE>"。工具调用会开启新的 assistant
        # 消息段，届时 flush + 重建过滤器。
        delta_filter = _MarkerStreamFilter(INSUFFICIENT_MARKER, REFUSAL_TEXT)
        # 当前 assistant 消息段**过滤前**的原始增量。收敛节点可能在流之后
        # 追加/替换最终回答（步数/预算触顶的固定话术、非 Agent 出口的固定
        # 文案），也可能是模型自评标记被 State 规范化；结束时拿它与最终
        # State 对比，决定是否补发。
        segment_raw: list[str] = []

        try:
            # 审计切片用 **id 差集**，不能用 prior_len 下标：sanitize_messages
            # 会在本轮开头删掉上一轮悬空的 tool_calls 组，最终列表长度与
            # 请求前不再对齐；按下标切会截掉本轮消息或把旧消息算进来。
            prior = await workflow.aget_state(config)
            prior_ids = {
                message.id
                for message in (prior.values or {}).get("messages", [])
                if message.id
            }

            async for _namespace, payload in workflow.astream(
                initial,
                config,
                stream_mode="custom",
                subgraphs=True,
            ):
                # subgraphs=True 时每项是 (namespace_tuple, payload)；主图与
                # 子图事件统一在这里翻译，namespace 本身不参与语义。
                if not isinstance(payload, dict):
                    continue
                kind = payload.get("type")
                if kind == "delta":
                    text = payload.get("text") or ""
                    if not text:
                        continue
                    segment_raw.append(text)
                    filtered = delta_filter.feed(text)
                    if filtered:
                        yield sse_frame("delta", {"text": filtered})
                elif kind == "tool":
                    if payload.get("phase") == "start":
                        # 前一段正文到此结束：放行过滤器扣留的尾巴，再为新的
                        # 最终回答段建一个干净的状态。
                        tail = delta_filter.flush()
                        if tail:
                            yield sse_frame("delta", {"text": tail})
                        delta_filter = _MarkerStreamFilter(
                            INSUFFICIENT_MARKER, REFUSAL_TEXT
                        )
                        segment_raw = []
                    # 帧里保留 worker 发出的全部字段（去掉仅内部使用的 type）。
                    yield sse_frame(
                        "tool",
                        {key: value for key, value in payload.items() if key != "type"},
                    )
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
        except Exception as exc:  # noqa: BLE001 — 流已开始，只能补 error 帧
            # spec §11.3：出错必须补一个 error 帧；此时不写审计（半截一轮）。
            logger.exception("chat workflow failed: session_id=%s", session_id)
            yield sse_frame(
                "error",
                {"code": classify_exception(exc), "message": str(exc)[:500]},
            )
        else:
            final = await workflow.aget_state(config)
            values = final.values or {}
            new_messages = [
                message
                for message in values.get("messages", [])
                if message.id not in prior_ids
            ]
            final_ai = next(
                (
                    message
                    for message in reversed(new_messages)
                    if isinstance(message, AIMessage)
                ),
                None,
            )
            final_text = _message_text(final_ai) if final_ai is not None else ""
            raw_streamed = "".join(segment_raw)
            # State 里的标志是“最终回答来自模型自评不足”；API 仍负责把标记
            # 换成面向用户的 REFUSAL_TEXT，并据此入池。
            insufficient = bool(values.get("insufficient"))
            audit_final_text: str | None = None

            if insufficient and INSUFFICIENT_MARKER in raw_streamed:
                # 流式过滤器已经（在 flush 时）把标记换成了 REFUSAL_TEXT；
                # State 里是 agent_done 规范化后的 GATE_REFUSAL，不能再补发。
                # 审计采用与浏览器完全一致的那段文本。
                tail = delta_filter.flush()
                if tail:
                    yield sse_frame("delta", {"text": tail})
                audit_final_text = (
                    raw_streamed.split(INSUFFICIENT_MARKER, 1)[0] + REFUSAL_TEXT
                )
            elif final_text != raw_streamed:
                # 最终回答不是当前段流出来的：非 Agent 出口的固定文案，或
                # agent_done 在流之后追加/替换的触顶话术（例如 max_steps=1
                # 时前言已流出、最终答案却是 STEP_LIMIT_TEXT）。
                # 先收尾上一段，再为新段重建过滤器后补发。
                tail = delta_filter.flush()
                if tail:
                    yield sse_frame("delta", {"text": tail})
                delta_filter = _MarkerStreamFilter(
                    INSUFFICIENT_MARKER, REFUSAL_TEXT
                )
                if final_text:
                    filtered = delta_filter.feed(final_text)
                    if filtered:
                        yield sse_frame("delta", {"text": filtered})
                tail = delta_filter.flush()
                if tail:
                    yield sse_frame("delta", {"text": tail})
            else:
                tail = delta_filter.flush()
                if tail:
                    yield sse_frame("delta", {"text": tail})

            # evidence 注入是临时 SystemMessage，不属于对话历史（Task 7 约定）。
            audit_messages: list[BaseMessage] = [
                _normalize_insufficient(message)
                for message in new_messages
                if not isinstance(message, SystemMessage)
            ]
            if audit_final_text is not None and final_ai is not None:
                # 用浏览器实际看到的文本覆盖 State 里规范化后的拒答话术，
                # 保证“流式拼接 == 落库正文”。
                audit_messages = [
                    message.model_copy(update={"content": audit_final_text})
                    if message.id == final_ai.id
                    else message
                    for message in audit_messages
                ]

            citations = list(values.get("citations") or [])
            actions = list(values.get("actions") or [])
            # top_score 取预检索 evidence 与工具 citations 里非空分数的最大值。
            scores = [
                entry.get("score")
                for entry in values.get("evidence") or []
                if entry.get("score") is not None
            ] + [
                # 当前 assign_citations 固定写 score=None；保留这一半是为了
                # 将来工具/检索回填真实分数时 top_score 自动跟上。
                citation.get("score")
                for citation in citations
                if citation.get("score") is not None
            ]
            metadata = {
                "citations": citations,
                "actions": actions,
                "retrieval_strategy": settings.retrieval_strategy,
                "top_score": max(scores) if scores else None,
            }

            assistant_message_id: int | None = None
            if audit_messages:
                # ★ 入池与落库放进同一个 SAVEPOINT：append 可能已经 flush 了
                # user 行之后才抛错。若直接让 finally 提交，半截一轮会落库；
                # 回滚到保存点则只丢弃本轮业务写入，会话行仍然保留。
                async with db.begin_nested():
                    # spec §3.5.2：闸门/检索已拒答（refusal_reason 非空）时
                    # 同一问题已经入过池，不能再写第二条 model_insufficient。
                    if insufficient and not values.get("refusal_reason"):
                        await lcq_repo.create(
                            db,
                            req.message,
                            str(conv.id),
                            REASON_MODEL_INSUFFICIENT,
                        )
                    ids = await store.append(
                        session_id,
                        *audit_messages,
                        metadata=metadata,
                    )
                    assistant_message_id = ids[-1] if ids else None

                # message_id 必须在客户端收到 done 时已经持久化（反馈接口
                # 的唯一凭据）。finally 的 commit 保留为无害的 no-op。
                await db.commit()

            yield sse_frame("done", {
                # 真实终局由 agent_model 从上游 response_metadata 写入 State；
                # 固定话术出口没有上游元数据，回退为干净的 stop / None。
                "finish_reason": values.get("finish_reason") or "stop",
                "usage": values.get("usage"),
                "message_id": assistant_message_id,
                "citations": citations,
                "actions": actions,
            })
        finally:
            # ★ 只提交、不再在 finally 里写消息：持久化已经挪到 done 帧
            # 之前（message_id 需要 flush 后的 id）。客户端断开或上游出错时
            # 不会执行上面 else 里的落库分支，历史仍然零痕迹。
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
