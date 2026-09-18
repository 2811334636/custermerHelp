from langchain_core.messages import AIMessage, HumanMessage

from app.sessions import InMemorySessionStore


def test_get_unknown_session_returns_empty_list():
    store = InMemorySessionStore(max_sessions=10, max_messages=10)
    assert store.get("nope") == []
    assert store.exists("nope") is False


def test_append_and_get_roundtrip():
    store = InMemorySessionStore(max_sessions=10, max_messages=10)
    store.append("s1", HumanMessage("你好"), AIMessage("您好"))
    msgs = store.get("s1")
    assert [m.content for m in msgs] == ["你好", "您好"]
    assert store.exists("s1") is True


def test_sessions_are_isolated():
    store = InMemorySessionStore(max_sessions=10, max_messages=10)
    store.append("s1", HumanMessage("A"))
    store.append("s2", HumanMessage("B"))
    assert [m.content for m in store.get("s1")] == ["A"]
    assert [m.content for m in store.get("s2")] == ["B"]


def test_get_returns_a_copy_not_the_internal_list():
    """返回内部引用会让调用方的剪裁操作污染存储。"""
    store = InMemorySessionStore(max_sessions=10, max_messages=10)
    store.append("s1", HumanMessage("A"))
    got = store.get("s1")
    got.append(HumanMessage("偷偷加的"))
    assert len(store.get("s1")) == 1


def test_max_messages_drops_oldest():
    store = InMemorySessionStore(max_sessions=10, max_messages=3)
    for i in range(5):
        store.append("s1", HumanMessage(f"m{i}"))
    assert [m.content for m in store.get("s1")] == ["m2", "m3", "m4"]


def test_max_sessions_evicts_least_recently_used():
    store = InMemorySessionStore(max_sessions=2, max_messages=10)
    store.append("s1", HumanMessage("1"))
    store.append("s2", HumanMessage("2"))
    store.get("s1")              # s1 变成最近使用
    store.append("s3", HumanMessage("3"))
    assert store.exists("s1") is True
    assert store.exists("s2") is False   # s2 最久未用，被淘汰
    assert store.exists("s3") is True
