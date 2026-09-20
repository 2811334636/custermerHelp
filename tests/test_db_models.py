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
