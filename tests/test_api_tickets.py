"""POST /api/tickets —— 确定性建工单（不经模型、不经工作流）。

转人工按钮是确定性的用户动作（spec §11.4），端点直接调用与 Agent 同一个
`create_ticket` 工具：建单与「已转人工」在同一个事务里提交。
"""

import re
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.base import get_db
from app.db.models import Conversation, Ticket
from app.main import app


class SessionSpy:
    """包住真实 db session，只为了数 commit 次数；其余调用一律透传。"""

    def __init__(self, inner):
        self._inner = inner
        self.commits = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def commit(self):
        self.commits += 1
        await self._inner.commit()


async def _rollback(session) -> None:
    await session.rollback()


@pytest.fixture
def client(db_session):
    spy = SessionSpy(db_session)
    app.dependency_overrides[get_db] = lambda: spy
    with TestClient(app) as c:
        c.db = spy
        yield c
        # 404 等不提交的路径会在请求里开一个只读事务；必须在 portal 循环
        # 还活着时收尾，否则 db_session fixture 会在测试外层循环里回滚它，
        # 撞上 "Future attached to a different loop"。
        c.portal.call(_rollback, db_session)
    app.dependency_overrides.clear()


def _maker():
    # ★ 每次另起引擎并在 portal 循环里用完即弃：db_session 属于测试外层
    # 事件循环，请求与这里都在 TestClient 的 portal 循环里跑；复用同一个
    # session 会把它的连接/事务带出 loop，teardown 时撞
    # "Future attached to a different loop"。
    engine = create_async_engine(get_settings().db_url)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _seed_conversation(sid: str) -> None:
    engine, maker = _maker()
    try:
        async with maker() as db:
            db.add(Conversation(session_uuid=sid, user_id="ticket-test", status="进行中"))
            await db.commit()
    finally:
        await engine.dispose()


async def _fetch_tickets(sid: str):
    """按会话取出工单行与 conversations.status（另起引擎只读已提交数据）。"""
    engine, maker = _maker()
    try:
        async with maker() as db:
            tickets = list(
                (
                    await db.scalars(
                        select(Ticket)
                        .join(Conversation, Ticket.conversation_id == Conversation.id)
                        .where(Conversation.session_uuid == sid)
                        .order_by(Ticket.ticket_no)
                    )
                ).all()
            )
            status = (
                await db.execute(
                    select(Conversation.status).where(
                        Conversation.session_uuid == sid
                    )
                )
            ).scalar_one()
            return tickets, status
    finally:
        await engine.dispose()


def test_post_ticket_creates_row_and_marks_handoff(client):
    sid = str(uuid.uuid4())
    client.portal.call(_seed_conversation, sid)

    r = client.post(
        "/api/tickets",
        json={
            "session_id": sid,
            "description": "投诉物流太慢",
            "ticket_type": "投诉",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert re.fullmatch(r"T\d+", body["ticket_no"])

    # 建单与转人工只提交一次：tickets + conversations.status 同一事务。
    assert client.db.commits == 1

    tickets, status = client.portal.call(_fetch_tickets, sid)
    assert len(tickets) == 1
    assert tickets[0].ticket_no == body["ticket_no"]
    assert tickets[0].description == "投诉物流太慢"
    assert tickets[0].ticket_type == "投诉"
    assert status == "已转人工"


def test_post_ticket_unknown_session_404(client):
    # 用随机 uuid：固定字符串可能与别的用例（如 /api/chat 会建会话）撞上。
    r = client.post(
        "/api/tickets",
        json={
            "session_id": f"nope-{uuid.uuid4()}",
            "description": "x",
            "ticket_type": "咨询",
        },
    )
    assert r.status_code == 404
    assert r.json()["code"] == "session_not_found"
    assert client.db.commits == 0  # 找不到会话就不该有任何写入


def test_post_ticket_rejects_blank_description(client):
    r = client.post(
        "/api/tickets",
        json={"session_id": "x", "description": "   ", "ticket_type": "咨询"},
    )
    assert r.status_code == 422


def test_post_ticket_rejects_bad_type(client):
    r = client.post(
        "/api/tickets",
        json={"session_id": "x", "description": "x", "ticket_type": "其它"},
    )
    assert r.status_code == 422


def test_post_ticket_does_not_dedupe(client):
    """连续两次调用 = 两张工单（端点按设计不去重）。"""
    sid = str(uuid.uuid4())
    client.portal.call(_seed_conversation, sid)
    payload = {"session_id": sid, "description": "要人工", "ticket_type": "售后"}

    r1 = client.post("/api/tickets", json=payload)
    r2 = client.post("/api/tickets", json=payload)

    assert r1.status_code == r2.status_code == 200
    assert r1.json()["ticket_no"] != r2.json()["ticket_no"]
    tickets, _ = client.portal.call(_fetch_tickets, sid)
    assert len(tickets) == 2
