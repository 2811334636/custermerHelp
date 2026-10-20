from collections.abc import AsyncIterator
from dataclasses import dataclass

# 例外说明：spec §4.2 要求除 providers.py 外其余模块只依赖 BaseChatModel 接口，
# 这里却直接 import 了 openai SDK。这是**有意为之的已知泄漏**：classify_exception
# 需要按上游异常的具体类型（AuthenticationError / RateLimitError / ...）分错误码，
# 而 BaseChatModel.astream 的契约只保证抛 BaseException，不暴露类型层级。
# 若以后换 provider，这个文件是除 providers.py 外唯一需要跟着改的地方。
import openai
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

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


def _absorb_metadata(
    chunk, finish_reason: str | None, usage: dict | None
) -> tuple[str | None, dict | None]:
    """从 chunk 的元数据里取真实的 finish_reason / usage。

    取法沿用 ch01 的三行（空值不覆盖，最后一个有值的赢）。工具轮有两个流
    各要取一次，抽成函数是为了让这两处只有一处真相。

    ★ ch01 的 `stream_reply` 已删除（修复轮 · Important 3）：它生产上无人调用，
    只剩 12 条测试在引用，而这 12 条测的正是 `stream_reply_with_tools` 的空
    工具路径（同一份 delta 循环 / 异常分类 / 元数据吸收）。留一份已分叉的副本
    的代价是：将来改活路径改错了对象，还有 12 条绿灯替你背书。
    """
    meta = getattr(chunk, "response_metadata", None) or {}
    if meta.get("finish_reason"):
        finish_reason = meta["finish_reason"]
    if getattr(chunk, "usage_metadata", None):
        usage = dict(chunk.usage_metadata)
    return finish_reason, usage


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


@dataclass(frozen=True)
class ToolCallEvent:
    name: str
    arguments: dict
    call_id: str


@dataclass(frozen=True)
class ToolResultEvent:
    call_id: str
    ok: bool
    summary: str
    content: str          # 回灌给模型的完整文本；落库时用它构造 ToolMessage
    elapsed_ms: int
    # 工具写入的引用元数据（query_faq 的 citations 等）；失败/无元数据为 None。
    metadata: dict | None = None


DomainEventWithTools = DeltaEvent | DoneEvent | ErrorEvent | ToolCallEvent | ToolResultEvent


async def stream_reply_with_tools(
    message: str,
    history: list[BaseMessage],
    *,
    model=None,
    registry,
    token_budget: int,
    conversation_id: int,
    retrieval_meta: dict | None = None,
) -> AsyncIterator[DomainEventWithTools]:
    """带工具轮的一轮对话。

    时序（spec §7.1）：
      ① 组装消息（system + 裁剪后历史 + 当前消息）
      ② 第一次请求模型，带工具 —— 正文实时转发，同时累积整条响应
      ③ 流结束后看累积结果的 tool_calls：
         为空 → 不需要工具，正文已转发完，结束（零额外延迟）
         非空 → 执行工具，再发第二次请求
      ④ 推 ToolCallEvent / ToolResultEvent
      ⑤ 第二次请求模型 —— ★ 不绑工具，结构性保证单轮收敛
      ⑥ 流式转发最终回答

    ★ 为什么不是「看第一个增量就分叉」（spec §7.2）：实测这个模型会
    **先吐正文再申请工具**。按第一个增量分叉会走错分支——用户看到机器人
    说"我去查一下"然后工具不执行、没有下文。改为流结束后看最终结果。
    """
    from langchain_core.messages import ToolMessage

    model = model or get_chat_model()
    messages = prepare_messages(history, HumanMessage(message), token_budget=token_budget)
    rendered = build_chat_prompt().format_messages(history=messages)

    # 真实元数据（不是写死的 "stop"/None）：最后一个流的取值即本轮终局，
    # 工具轮由第二次调用（最终回答那次）覆盖第一次的 "tool_calls"。
    finish_reason: str | None = None
    usage: dict | None = None

    # ---- ② 第一次请求：带工具。正文实时转发，同时累积整条响应 ----
    bound = model.bind_tools(registry.as_langchain_tools())
    gathered = None
    try:
        async for chunk in bound.astream(rendered):
            gathered = chunk if gathered is None else gathered + chunk
            if chunk.text:
                yield DeltaEvent(text=chunk.text)
            finish_reason, usage = _absorb_metadata(chunk, finish_reason, usage)
    except Exception as exc:  # noqa: BLE001
        yield ErrorEvent(code=classify_exception(exc), message=str(exc)[:500])
        return

    tool_calls = list(getattr(gathered, "tool_calls", None) or [])

    if not tool_calls:
        # 不需要工具：正文已逐块转发完毕
        yield DoneEvent(finish_reason=finish_reason, usage=usage)
        return

    # ---- ④ 执行工具（可能多个）----
    # ★ 错误边界（修复轮 · Important 4）：`registry.execute` 是**写着**不抛的，
    #   但这不由类型系统保证——`spec.tool.args_schema.model_validate` 只挡了
    #   ValidationError，别的意外异常会一路窜出去，变成"半截流 + 没有 error 帧"，
    #   违反 spec §11.3「出错必须补一个 error 帧」这条无条件规则。用与上面两个
    #   模型流完全相同的形状兜住它。
    tool_messages: list[ToolMessage] = []
    try:
        for call in tool_calls:
            yield ToolCallEvent(name=call["name"], arguments=call["args"], call_id=call["id"])

            execute_kwargs: dict = {"conversation_id": conversation_id}
            # 只在调用方真的带了容器时才传：保持对旧 registry 替身
            # （测试里只接受 conversation_id）的兼容。
            if retrieval_meta is not None:
                execute_kwargs["retrieval_meta"] = retrieval_meta
            outcome = await registry.execute(
                call["name"], call["args"], call["id"], **execute_kwargs
            )

            yield ToolResultEvent(
                call_id=outcome.call_id, ok=outcome.ok,
                summary=outcome.summary, content=outcome.content,
                elapsed_ms=outcome.elapsed_ms,
                metadata=outcome.metadata,
            )
            tool_messages.append(ToolMessage(content=outcome.content, tool_call_id=call["id"]))
    except Exception as exc:  # noqa: BLE001 — 与两个模型流的错误边界同形
        yield ErrorEvent(code=classify_exception(exc), message=str(exc)[:500])
        return

    # ---- ⑤⑥ 第二次请求：不绑工具 ----
    # followup 里放的是累积出来的 gathered 本身（AIMessageChunk），
    # 而不是新构造的 AIMessage——这样模型若说过前言，前言也如实带上。
    followup = [*rendered, gathered, *tool_messages]

    # ★ 这一行偏离了 brief 原样代码里的 `model.astream(followup)`，理由如下：
    #   「不绑工具」在真实 LangChain 里靠**对象身份**表达（bind_tools 返回新
    #   对象，原对象始终没绑）；本项目的打桩模型把绑定状态记在实例上
    #   （`self._bound`），所以"同一个 model 对象"观察不到"这一轮没有工具"。
    #   显式绑空列表把"无工具可用"变成**调用方式本身**携带的事实，测试与生产
    #   走的是同一条机制，而不是测试模拟一种、生产走另一种。
    #   实测（2026-09-20）：`bind_tools([])` 发出的 `tools: []` 被上游正常接受
    #   （真机 ainvoke 返回正常正文），且除该空键外与不绑时对模型而言等价：
    #   没有工具可用 → 模型不可能再申请工具 → 单轮收敛仍由调用形状保证。
    unbound = model.bind_tools([])

    try:
        async for chunk in unbound.astream(followup):
            if chunk.text:
                yield DeltaEvent(text=chunk.text)
            finish_reason, usage = _absorb_metadata(chunk, finish_reason, usage)
    except Exception as exc:  # noqa: BLE001
        yield ErrorEvent(code=classify_exception(exc), message=str(exc)[:500])
        return

    yield DoneEvent(finish_reason=finish_reason, usage=usage)
