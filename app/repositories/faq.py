from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Faq


async def search_by_keyword(session: AsyncSession, keyword: str, limit: int = 3) -> list[Faq]:
    """按关键词检索 faq。

    ★ 只搜 question 列，不搜 answer。若连 answer 一起搜，某条答案里顺带
    提到"邮费自理"就会命中，会让验收标准 3 的结论依赖种子数据的措辞巧合，
    测的就不是检索能力而是数据运气（spec §5.5）。
    """
    stmt = (
        select(Faq)
        .where(Faq.question.like(f"%{keyword}%"))
        .order_by(Faq.id)
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars())
