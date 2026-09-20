from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation


async def get_or_create(session: AsyncSession, session_uuid: str, user_id: str) -> Conversation:
    stmt = select(Conversation).where(Conversation.session_uuid == session_uuid)
    found = (await session.execute(stmt)).scalar_one_or_none()
    if found is not None:
        return found

    conv = Conversation(session_uuid=session_uuid, user_id=user_id, status="进行中")
    session.add(conv)
    await session.flush()          # 拿到自增 id，但不提交（事务由调用方管）
    return conv


async def set_status(session: AsyncSession, conversation_id: int, status: str) -> None:
    conv = await session.get(Conversation, conversation_id)
    if conv is not None:
        conv.status = status
        await session.flush()
