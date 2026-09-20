import datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Ticket


async def create(
    session: AsyncSession,
    ticket_no: str,
    conversation_id: int,
    description: str,
    ticket_type: str,
) -> Ticket:
    t = Ticket(
        ticket_no=ticket_no,
        conversation_id=conversation_id,
        description=description,
        ticket_type=ticket_type,
        status="待处理",
    )
    session.add(t)
    await session.flush()
    return t


async def count_on_date(session: AsyncSession, day: datetime.date) -> int:
    stmt = select(func.count()).select_from(Ticket).where(
        func.date(Ticket.created_at) == day
    )
    return (await session.execute(stmt)).scalar_one()


async def get_by_no(session: AsyncSession, ticket_no: str) -> Ticket | None:
    return await session.get(Ticket, ticket_no)
