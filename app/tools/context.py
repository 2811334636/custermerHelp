"""工具上下文：把模型看不到的信息（如当前会话 id、当前请求的库会话）传给工具。

为什么需要它：create_ticket 的签名里没有会话 id —— 模型不该知道、
也不该能指定往哪个会话建单（spec §6.3）。会话 id 由 registry 在
执行前注入，工具用 current_context() 取。

为什么库会话也走这里（而非模块级全局）：单 worker 下 SSE 流会持续数秒，
两个请求必然重叠；模块级 `_session` 会被后一个请求覆盖，A 请求的工具就会
读到 B 的数据库会话。ContextVar 的每个 asyncio 任务持有自己的上下文副本，
重叠执行也不会串号。
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class ToolContext:
    conversation_id: int
    # 只做类型标注，不引入 app.db —— AsyncSession 是个纯类型，
    # context.py 不该依赖数据库实现。
    session: AsyncSession | None = None
    # 本次请求的检索元数据（citations/scores/strategy）。dataclass 是 frozen
    # 的，但 dict 本身可变：调用方建一个空 dict 传进来，工具在里面原地更新，
    # registry 执行完后把同一个 dict 挂到 ToolOutcome.metadata 上。
    retrieval_meta: dict | None = None


_ctx: ContextVar[ToolContext] = ContextVar("tool_context")


@contextmanager
def tool_context(
    conversation_id: int,
    session: AsyncSession | None = None,
    retrieval_meta: dict | None = None,
) -> Iterator[None]:
    token = _ctx.set(
        ToolContext(
            conversation_id=conversation_id,
            session=session,
            retrieval_meta=retrieval_meta,
        )
    )
    try:
        yield
    finally:
        _ctx.reset(token)


def current_context() -> ToolContext:
    """取当前工具上下文。不在工具执行期间调用会抛 LookupError。"""
    try:
        return _ctx.get()
    except LookupError as exc:
        raise LookupError(
            "当前不在工具执行上下文里；工具只能由 ToolRegistry.execute() 调用"
        ) from exc
