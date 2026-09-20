import datetime

import pytest
from sqlalchemy import text

from app.repositories import conversation, faq, message, ticket


async def test_faq_hit(db_session):
    rows = await faq.search_by_keyword(db_session, "退货")
    assert len(rows) >= 3
    assert all("退货" in r.question for r in rows)


async def test_faq_miss(db_session):
    """★ 验收标准 3 的数据前提：『邮费』在 faq 表里查不到。

    种子数据故意不含邮费条目。这条测试锁住这个事实——
    如果哪天有人为了"让验收 3 通过"补了数据，这条测试会红。
    """
    rows = await faq.search_by_keyword(db_session, "邮费")
    assert rows == []


async def test_faq_limit(db_session):
    rows = await faq.search_by_keyword(db_session, "退", limit=2)
    assert len(rows) <= 2


async def test_conversation_get_or_create_is_idempotent(db_session):
    a = await conversation.get_or_create(db_session, "uuid-1", "anonymous")
    b = await conversation.get_or_create(db_session, "uuid-1", "anonymous")
    assert a.id == b.id


async def test_conversation_set_status(db_session):
    c = await conversation.get_or_create(db_session, "uuid-2", "anonymous")
    assert c.status == "进行中"
    await conversation.set_status(db_session, c.id, "已转人工")
    await db_session.refresh(c)
    assert c.status == "已转人工"


async def test_message_append_and_list_in_order(db_session):
    c = await conversation.get_or_create(db_session, "uuid-3", "anonymous")
    await message.append(db_session, c.id, "user", "第一条")
    await message.append(db_session, c.id, "assistant", "第二条")
    rows = await message.list_by_conversation(db_session, c.id)
    assert [r.content for r in rows] == ["第一条", "第二条"]


async def test_message_stores_tool_calls_json(db_session):
    c = await conversation.get_or_create(db_session, "uuid-4", "anonymous")
    calls = [{"id": "call_1", "name": "query_order", "args": {"order_id": "1001"}}]
    await message.append(db_session, c.id, "assistant", None, tool_calls=calls)
    rows = await message.list_by_conversation(db_session, c.id)
    assert rows[0].tool_calls == calls


async def test_message_stores_tool_call_id(db_session):
    c = await conversation.get_or_create(db_session, "uuid-5", "anonymous")
    await message.append(db_session, c.id, "tool", "查询结果", tool_call_id="call_1")
    rows = await message.list_by_conversation(db_session, c.id)
    assert rows[0].tool_call_id == "call_1"


async def test_ticket_create(db_session):
    c = await conversation.get_or_create(db_session, "uuid-6", "anonymous")
    t = await ticket.create(db_session, "T20260920001", c.id, "我要投诉", "投诉")
    assert t.ticket_no == "T20260920001"
    assert t.status == "待处理"


async def test_ticket_count_on_date(db_session):
    c = await conversation.get_or_create(db_session, "uuid-7", "anonymous")
    today = datetime.date.today()
    before = await ticket.count_on_date(db_session, today)
    await ticket.create(db_session, "T20260920999", c.id, "咨询", "咨询")
    after = await ticket.count_on_date(db_session, today)
    assert after == before + 1
