from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MinedQaStaging
from app.knowledge.chunk import Chunk


async def load_mined_chunks(session: AsyncSession) -> list[Chunk]:
    stmt = select(MinedQaStaging).where(MinedQaStaging.status == "待入库").order_by(
        MinedQaStaging.id
    )
    rows = list((await session.execute(stmt)).scalars())
    return [
        Chunk(
            chunk_id=f"mined:{r.id}",
            category=r.category,
            questions=r.question,
            answer=r.answer,
            section_path=r.category,
            content_type="mined",
        )
        for r in rows
    ]
