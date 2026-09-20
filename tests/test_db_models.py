from sqlalchemy import inspect

from app.db.base import Base
from app.db import models  # noqa: F401  确保模型被注册进 metadata


def test_four_tables_registered():
    assert set(Base.metadata.tables) == {"conversations", "messages", "faq", "tickets"}


def test_conversations_has_session_uuid_unique():
    t = Base.metadata.tables["conversations"]
    assert "session_uuid" in t.columns
    assert t.columns["session_uuid"].unique is True


def test_messages_columns():
    t = Base.metadata.tables["messages"]
    assert {"conversation_id", "role", "content", "tool_calls", "tool_call_id"} <= set(t.columns.keys())


def test_tickets_primary_key_is_ticket_no():
    t = Base.metadata.tables["tickets"]
    assert [c.name for c in t.primary_key.columns] == ["ticket_no"]


def test_messages_has_fk_to_conversations():
    t = Base.metadata.tables["messages"]
    fks = {fk.target_fullname for fk in t.foreign_keys}
    assert "conversations.id" in fks


async def test_live_ddl_matches_orm_metadata(db_session):
    """活库 DDL 与 ORM 声明必须一致（修复轮 · Important 2）。

    建表的**唯一真相**是 `scripts/schema.sql`（docker init 与 scripts/seed.py
    都只跑它）；`app/db/models.py` 只是 ORM 侧的另一份声明。两份分家时没有
    任何东西会红——本文件其余用例只重述 ORM 自己的声明，一个与活库矛盾的
    models.py 照样全绿（它测的是"models.py 与自己一致"）。

    所以这里**反射活库**（`sqlalchemy.inspect`，走 db_session fixture 的 engine）
    逐表逐列比对：列名、可空性、长度。谁只在一边改了 length/nullable，这条即红。
    """
    engine = db_session.bind
    async with engine.connect() as conn:
        reflected = await conn.run_sync(
            lambda sync_conn: {
                name: {c["name"]: c for c in inspect(sync_conn).get_columns(name)}
                for name in Base.metadata.tables
            }
        )

    problems: list[str] = []
    for name, table in Base.metadata.tables.items():
        live = reflected[name]
        if set(live) != set(table.columns.keys()):
            problems.append(
                f"{name}: 列集合不一致 —— 活库 {sorted(live)} / ORM {sorted(table.columns)}"
            )
        for column in table.columns:
            got = live.get(column.name)
            if got is None:
                continue
            if bool(got["nullable"]) != bool(column.nullable):
                problems.append(
                    f"{name}.{column.name}: 可空性不一致 —— "
                    f"活库 nullable={got['nullable']} / ORM nullable={column.nullable}"
                )
            orm_len = getattr(column.type, "length", None)
            live_len = getattr(got["type"], "length", None)
            if orm_len != live_len:
                problems.append(
                    f"{name}.{column.name}: 长度不一致 —— "
                    f"活库 {live_len} ({got['type']}) / ORM {orm_len} ({column.type})"
                )

    assert not problems, "活库 DDL 与 ORM 声明分家了：\n  " + "\n  ".join(problems)
