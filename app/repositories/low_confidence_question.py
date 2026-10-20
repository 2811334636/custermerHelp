"""低置信度问题池仓储。"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import LowConfidenceQuestion


async def create(
    session: AsyncSession, question: str, session_id: str, reason: str
) -> LowConfidenceQuestion:
    """创建低置信度问题记录。

    只 flush 不 commit：事务边界归调用方（与其余仓储一致），
    这样测试 fixture 才能回滚、在线路径也能随本轮消息一起提交。
    """
    lcq = LowConfidenceQuestion(
        question=question, session_id=session_id, reason=reason
    )
    session.add(lcq)
    await session.flush()
    return lcq


async def get_recent(
    session: AsyncSession, limit: int = 100
) -> list[LowConfidenceQuestion]:
    """获取最近的低置信度问题。"""
    result = await session.execute(
        select(LowConfidenceQuestion)
        .order_by(LowConfidenceQuestion.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars())
