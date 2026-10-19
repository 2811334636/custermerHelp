from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Faq
from app.knowledge.chunk import Chunk


async def load_faq_chunks(session: AsyncSession, *, content_type: str = "faq") -> list[Chunk]:
    rows = list((await session.execute(select(Faq).order_by(Faq.id))).scalars())
    return [
        Chunk(
            chunk_id=f"faq:{r.id}",
            category=r.category,
            questions=r.question,
            answer=r.answer,
            section_path=r.category,
            content_type=content_type,  # type: ignore[arg-type]
        )
        for r in rows
    ]
