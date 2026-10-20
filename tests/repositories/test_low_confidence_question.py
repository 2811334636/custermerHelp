"""低置信度问题池仓储测试。"""

import datetime

from app.repositories import low_confidence_question as repo


async def test_create_persists_all_fields(db_session):
    row = await repo.create(
        db_session, "火星移民计划详情", "sess-1", "no_results"
    )

    assert row.id is not None
    assert row.question == "火星移民计划详情"
    assert row.session_id == "sess-1"
    assert row.reason == "no_results"

    # created_at 由 server_default 生成，flush 后属性处于过期状态；
    # 必须显式 refresh，不能在异步上下文里触发隐式 IO。
    await db_session.refresh(row)
    assert row.created_at is not None

    rows = await repo.get_recent(db_session, limit=1000)
    assert any(r.id == row.id for r in rows)


async def test_get_recent_orders_by_created_at_desc_and_limits(db_session):
    older = await repo.create(db_session, "旧问题", "sess-a", "no_results")
    newer = await repo.create(db_session, "新问题", "sess-b", "low_score")
    # 用远期时间戳避开库里可能存在的历史行，保证排序断言确定。
    older.created_at = datetime.datetime(2099, 1, 1, 0, 0, 0)
    newer.created_at = datetime.datetime(2099, 1, 2, 0, 0, 0)
    await db_session.flush()

    latest = await repo.get_recent(db_session, limit=1)
    assert [r.question for r in latest] == ["新问题"]

    all_recent = await repo.get_recent(db_session, limit=1000)
    assert [r.question for r in all_recent[:2]] == ["新问题", "旧问题"]
