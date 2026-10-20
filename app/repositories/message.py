from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Message


async def append(
    session: AsyncSession,
    conversation_id: int,
    role: str,
    content: str | None,
    tool_calls: list[dict] | None = None,
    tool_call_id: str | None = None,
    metadata: dict | None = None,
) -> int:
    """插入一条消息并返回主键（flush 后可用，事务边界仍归调用方）。"""
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=tool_call_id,
        metadata_=metadata,
    )
    session.add(msg)
    await session.flush()
    return msg.id


async def list_by_conversation(session: AsyncSession, conversation_id: int) -> list[Message]:
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    )
    return list((await session.execute(stmt)).scalars())
