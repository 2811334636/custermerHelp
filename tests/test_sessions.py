import pytest

from app.sessions import MySQLSessionStore


async def test_get_unknown_session_returns_empty(db_session):
    store = MySQLSessionStore(session=db_session)
    assert await store.get("从没见过的-uuid") == []
    assert await store.exists("从没见过的-uuid") is False


async def test_append_then_get_roundtrip(db_session):
    from langchain_core.messages import AIMessage, HumanMessage

    store = MySQLSessionStore(session=db_session)
    await store.append("u1", HumanMessage("你好"), AIMessage("您好"))
    msgs = await store.get("u1")
    assert [m.content for m in msgs] == ["你好", "您好"]
    assert await store.exists("u1") is True


async def test_sessions_are_isolated(db_session):
    from langchain_core.messages import HumanMessage

    store = MySQLSessionStore(session=db_session)
    await store.append("u1", HumanMessage("A"))
    await store.append("u2", HumanMessage("B"))
    assert [m.content for m in await store.get("u1")] == ["A"]
    assert [m.content for m in await store.get("u2")] == ["B"]


async def test_get_returns_new_list_not_shared_state(db_session):
    """返回的列表被调用方改动，不应影响后续读取。"""
    from langchain_core.messages import HumanMessage

    store = MySQLSessionStore(session=db_session)
    await store.append("u1", HumanMessage("A"))
    got = await store.get("u1")
    got.append(HumanMessage("偷偷加的"))
    assert len(await store.get("u1")) == 1


async def test_tool_messages_survive_roundtrip(db_session):
    """带 tool_calls 的 assistant 与 tool 结果都要能还原——工具徽章依赖这个。"""
    from langchain_core.messages import AIMessage, ToolMessage

    store = MySQLSessionStore(session=db_session)
    # ★ tool_calls 里必须带 "type": "tool_call"。langchain-core 1.6.3 的
    # AIMessage 在构造时会把 tool_calls 规范化——自动注入 type 并重排键序，
    # 所以不带 type 的字面量与读出结果**永远不相等**，与本 store 的实现无关。
    # 这与 Task 4 修正过的 fixture 是同一件事（见 docs 计划里的说明）。
    calls = [{"id": "c1", "name": "query_order",
              "args": {"order_id": "1001"}, "type": "tool_call"}]
    await store.append(
        "u1",
        AIMessage(content="", tool_calls=calls),
        ToolMessage(content="查到订单", tool_call_id="c1"),
    )
    msgs = await store.get("u1")
    assert msgs[0].tool_calls == calls
    assert msgs[1].tool_call_id == "c1"
