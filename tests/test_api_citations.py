"""引用溯源从检索一路走到 SSE 与 messages.metadata（ch04 Task 7 / ch05 Task 9）。

覆盖四件事：
1. query_faq 召回后，tool end / done 帧带 citations，message_id 可反馈；
2. 最终 assistant 行的 metadata 落库（citations / actions / retrieval_strategy）；
3. 模型输出 <INSUFFICIENT_KNOWLEDGE> 时替换话术并写入低置信度池；
4. 闸门拒答时只入池一次，适配层不再写 model_insufficient。
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import get_store
from app.config import get_settings
from app.db.base import get_db
from app.db.models import Conversation, LowConfidenceQuestion
from app.knowledge import retrieval as retrieval_module
from app.main import app
from app.providers import get_default_chat_model
from app.workflow.texts import GATE_REFUSAL

# 在 autouse fixture 替换模块属性之前保存真实入口，供"重复入池"用例恢复。
_REAL_SEARCH_WITH_METADATA = retrieval_module.search_with_metadata


class FakeStore:
    def __init__(self):
        self._data: dict[str, list] = {}
        self.metadata: dict[str, dict] = {}
        self._next_id = 1

    async def get(self, session_id):
        return list(self._data.get(session_id, []))

    async def append(self, session_id, *messages, metadata=None):
        msgs = self._data.setdefault(session_id, [])
        ids = []
        for message in messages:
            msgs.append(message)
            ids.append(self._next_id)
            self._next_id += 1
        if metadata is not None and messages:
            self.metadata[session_id] = metadata
        return ids

    async def exists(self, session_id):
        return session_id in self._data


class GraphModel:
    """按调用次序作答；意图分类走 `ainvoke`。"""

    def __init__(self, intent: str, replies=None):
        self.intent = intent
        self.replies = list(replies or [])
        self.agent_calls = 0

    async def ainvoke(self, messages):
        return AIMessage(
            content=json.dumps(
                {"intent": self.intent, "confidence": 0.9}, ensure_ascii=False
            )
        )

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.agent_calls += 1
        reply = self.replies[min(self.agent_calls - 1, len(self.replies) - 1)]
        for chunk in reply if isinstance(reply, list) else [reply]:
            if isinstance(chunk, AIMessageChunk):
                yield chunk
            else:
                yield AIMessageChunk(
                    content=chunk.content or "",
                    tool_calls=list(chunk.tool_calls or []),
                )


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


TOOL_THEN_ANSWER = [
    tool_call("query_faq", {"keyword": "退货"}, "call_cite"),
    [AIMessageChunk(content="签收后 7 天内可无理由退货 [1]。")],
]


class PlainModel(GraphModel):
    def __init__(self):
        super().__init__("闲聊", [])


class InsufficientModel(GraphModel):
    """模型自评知识不足：输出 spec §3.5.2 的特殊标记。"""

    def __init__(self):
        super().__init__("退货退款", [[AIMessageChunk(content="<INSUFFICIENT_KNOWLEDGE>")]])


class SplitMarkerModel(GraphModel):
    """标记被切在两个 delta 帧之间，且标记后还有噪声文本。"""

    def __init__(self):
        super().__init__(
            "退货退款",
            [
                [
                    AIMessageChunk(content="抱歉 <INSUFF"),
                    AIMessageChunk(content="ICIENT_KNOWLEDGE> 后记噪声"),
                ]
            ],
        )


def _knowledge_row():
    from app.db.models import KnowledgeChunk

    return KnowledgeChunk(
        chunk_id="policy.md:9",
        category="售后政策 > 退换货",
        questions="退货政策",
        answer="签收后 7 天内可无理由退货。",
        section_path="售后政策 > 退换货 > 退货政策",
        content_type="policy",
        status="已向量化",
        content_hash="x",
    )


@pytest.fixture(autouse=True)
def _stub_faq_retrieval(monkeypatch):
    """避免真实 Milvus/embedding；分数对齐交给策略层测试覆盖。"""
    from app.knowledge import retrieval

    async def fake_search_with_metadata(session, query, top_k=None, **kw):
        return (
            [_knowledge_row()],
            {"strategy": "hybrid_rerank", "scores": [0.88], "top_score": 0.88},
        )

    monkeypatch.setattr(retrieval, "search_with_metadata", fake_search_with_metadata)


@pytest.fixture
def client(db_session):
    store = FakeStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_default_chat_model] = lambda: GraphModel(
        "退货退款", TOOL_THEN_ANSWER
    )
    with TestClient(app) as c:
        c.store = store
        yield c
    app.dependency_overrides.clear()


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    out = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        out.append((name, data))
    return out


def use_model(model):
    app.dependency_overrides[get_default_chat_model] = lambda: model


EXPECTED_CITATIONS = [
    {"index": 1, "chunk_id": "policy.md:9", "score": None},
]


def test_tool_end_and_done_frames_carry_citations(client):
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    events = parse_sse(r.text)

    tool_frames = [data for name, data in events if name == "tool"]
    assert tool_frames[1]["phase"] == "end"
    assert tool_frames[1]["citations"] == EXPECTED_CITATIONS

    done = [data for name, data in events if name == "done"][-1]
    assert done["citations"] == EXPECTED_CITATIONS
    # done 帧的 message_id 必须是最后一条 assistant 行的真实 id（flush 后）。
    assert done["message_id"] == client.store._next_id - 1
    assert isinstance(done["message_id"], int)


def test_citations_are_persisted_on_last_assistant_message(client):
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    events = parse_sse(r.text)
    sid = events[0][1]["session_id"]

    stored = client.store._data[sid]
    assert [type(m).__name__ for m in stored] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]
    assert "[1]" in stored[-1].content

    metadata = client.store.metadata[sid]
    assert metadata["citations"] == EXPECTED_CITATIONS
    assert metadata["actions"] == []
    assert metadata["retrieval_strategy"] == get_settings().retrieval_strategy


def test_done_frame_has_message_id_and_empty_citations_without_retrieval(client):
    use_model(PlainModel())
    r = client.post("/api/chat", json={"message": "你好"})
    events = parse_sse(r.text)
    sid = events[0][1]["session_id"]

    done = [data for name, data in events if name == "done"][-1]
    assert done["citations"] == []
    assert done["message_id"] == client.store._next_id - 1
    # 审计 metadata 仍记录本轮事实：空引用、空动作与当前检索策略。
    assert client.store.metadata[sid]["citations"] == []
    assert client.store.metadata[sid]["actions"] == []


def test_done_message_id_points_at_persisted_assistant_row(client, db_session):
    """真实 MySQL 存取：done.message_id 就是最终 assistant 行的主键，metadata 同列落库。"""
    import uuid

    from app.db.models import Message
    from app.sessions import MySQLSessionStore

    app.dependency_overrides[get_store] = lambda: MySQLSessionStore(session=db_session)
    sid = str(uuid.uuid4())

    r = client.post(
        "/api/chat", json={"session_id": sid, "message": "退货政策是什么"}
    )
    done = [d for n, d in parse_sse(r.text) if n == "done"][-1]
    message_id = done["message_id"]
    assert isinstance(message_id, int)

    async def _fetch():
        # 路由 finally 已 commit；另起引擎才能看到已提交行。
        engine = create_async_engine(get_settings().db_url)
        try:
            async with async_sessionmaker(engine)() as other:
                conv = (
                    await other.execute(
                        select(Conversation).where(Conversation.session_uuid == sid)
                    )
                ).scalar_one()
                row = (
                    await other.execute(
                        select(Message).where(Message.id == message_id)
                    )
                ).scalar_one()
                return conv.id, row.role, row.content, row.metadata_
        finally:
            await engine.dispose()

    conv_id, role, content, metadata = client.portal.call(_fetch)
    assert role == "assistant"
    assert "[1]" in content
    assert metadata["citations"] == EXPECTED_CITATIONS
    assert metadata["actions"] == []
    assert metadata["retrieval_strategy"] == get_settings().retrieval_strategy

    # 同一会话的最后一条消息就是 done 帧指的那条
    async def _fetch_last_id():
        engine = create_async_engine(get_settings().db_url)
        try:
            async with async_sessionmaker(engine)() as other:
                rows = list(
                    (
                        await other.execute(
                            select(Message)
                            .where(Message.conversation_id == conv_id)
                            .order_by(Message.id)
                        )
                    ).scalars()
                )
                return rows[-1].id
        finally:
            await engine.dispose()

    assert client.portal.call(_fetch_last_id) == message_id


def test_model_insufficient_marker_is_replaced_and_pooled(client):
    from app.api import REFUSAL_TEXT

    use_model(InsufficientModel())
    r = client.post("/api/chat", json={"message": "火星天气怎么样"})
    events = parse_sse(r.text)
    sid = events[0][1]["session_id"]

    stored = client.store._data[sid]
    assert stored[-1].content == REFUSAL_TEXT
    assert "<INSUFFICIENT_KNOWLEDGE>" not in stored[-1].content

    # 浏览器只看 delta：流里不能出现原始标记，且最终拼出的正文必须是拒答话术。
    deltas = [d["text"] for n, d in events if n == "delta"]
    assert all("<INSUFFICIENT_KNOWLEDGE>" not in text for text in deltas)
    assert "".join(deltas) == REFUSAL_TEXT

    async def _fetch() -> tuple[int, list[tuple[str, str]]]:
        engine = create_async_engine(get_settings().db_url)
        try:
            async with async_sessionmaker(engine)() as other:
                conv = (
                    await other.execute(
                        select(Conversation).where(Conversation.session_uuid == sid)
                    )
                ).scalar_one()
                rows = list(
                    (
                        await other.execute(
                            select(LowConfidenceQuestion).where(
                                LowConfidenceQuestion.question == "火星天气怎么样"
                            )
                        )
                    ).scalars()
                )
                return conv.id, [(row.reason, row.session_id) for row in rows]
        finally:
            await engine.dispose()

    conv_id, pooled = client.portal.call(_fetch)
    assert pooled == [("model_insufficient", str(conv_id))]


def test_retrieval_refusal_pools_once_and_skips_model_pooling(
    client, monkeypatch
):
    """闸门已 no_results 入池后：只保留这一条，不出现 model_insufficient。

    ch05 的闸门在检索失败时直接去 log，Agent 根本不会被调用；适配层仍用
    State 里的 refusal_reason 兜住“模型自评不足”的二次入池。
    """
    question = "火星天气怎么样（重复入池用例）"

    class EmptyVectorStore:
        def search(self, vector, top_k, *, filter_expr=None):
            return []

    async def fake_embed(texts):
        return [[0.0] * 1024 for _ in texts]

    # 恢复真实检索入口，只把基础设施替换成"零命中"。
    monkeypatch.setattr(
        retrieval_module, "search_with_metadata", _REAL_SEARCH_WITH_METADATA
    )
    monkeypatch.setattr(retrieval_module, "VectorStore", EmptyVectorStore)
    monkeypatch.setattr(retrieval_module, "embed_texts", fake_embed)
    # 本用例只关心拒答→入池，不需要 hybrid/rerank；固定 dense 消除
    # 对 query 改写与重排上游的依赖。
    monkeypatch.setattr(get_settings(), "retrieval_strategy", "dense")

    # 模型即使吐标记也不会被调用（闸门在 Agent 之前拒答）。
    use_model(InsufficientModel())
    r = client.post("/api/chat", json={"message": question})
    assert r.status_code == 200
    events = parse_sse(r.text)
    sid = events[0][1]["session_id"]

    assert client.store._data[sid][-1].content == GATE_REFUSAL
    deltas = [d["text"] for n, d in events if n == "delta"]
    assert "".join(deltas) == GATE_REFUSAL

    async def _fetch() -> list[str]:
        engine = create_async_engine(get_settings().db_url)
        try:
            async with async_sessionmaker(engine)() as other:
                rows = list(
                    (
                        await other.execute(
                            select(LowConfidenceQuestion).where(
                                LowConfidenceQuestion.question == question
                            )
                        )
                    ).scalars()
                )
                return [row.reason for row in rows]
        finally:
            await engine.dispose()

    assert client.portal.call(_fetch) == ["no_results"], (
        "检索已入池时 api 不应再写 model_insufficient"
    )


def test_split_marker_is_filtered_across_delta_frames(client):
    """标记跨帧 + 标记后有噪声：流里既不能漏出标记，也不能保留后记。"""
    from app.api import REFUSAL_TEXT

    use_model(SplitMarkerModel())
    r = client.post("/api/chat", json={"message": "火星天气怎么样"})
    events = parse_sse(r.text)
    sid = events[0][1]["session_id"]

    deltas = [d["text"] for n, d in events if n == "delta"]
    assert all("<INSUFFICIENT_KNOWLEDGE>" not in text for text in deltas)
    assert "".join(deltas) == "抱歉 " + REFUSAL_TEXT
    assert client.store._data[sid][-1].content == "抱歉 " + REFUSAL_TEXT


class FlushThenFailStore:
    """模拟 append 已 flush user 行、随后抛错的存储。"""

    def __init__(self, session):
        self._session = session

    async def get(self, session_id):
        return []

    async def exists(self, session_id):
        return False

    async def append(self, session_id, *messages, metadata=None):
        from langchain_core.messages import HumanMessage

        from app.config import get_settings
        from app.repositories import conversation as conv_repo
        from app.repositories import message as message_repo

        conv = await conv_repo.get_or_create(
            self._session, session_id, get_settings().default_user_id
        )
        for message in messages:
            if isinstance(message, HumanMessage):
                await message_repo.append(
                    self._session, conv.id, "user", message.content
                )
                break
        raise RuntimeError("append failed after flush")


def test_failed_persistence_rolls_back_to_savepoint(client, db_session):
    """落库中途失败：已 flush 的 user 行必须随 SAVEPOINT 回滚，finally 不能提交半截一轮。"""
    import uuid

    from app.db.models import Message

    app.dependency_overrides[get_store] = lambda: FlushThenFailStore(db_session)
    use_model(PlainModel())
    sid = str(uuid.uuid4())
    msg = "这条消息不该落库"

    with pytest.raises(RuntimeError, match="append failed after flush"):
        client.post("/api/chat", json={"session_id": sid, "message": msg})

    async def _fetch():
        engine = create_async_engine(get_settings().db_url)
        try:
            async with async_sessionmaker(engine)() as other:
                conv = (
                    await other.execute(
                        select(Conversation).where(Conversation.session_uuid == sid)
                    )
                ).scalar_one_or_none()
                rows = list(
                    (
                        await other.execute(
                            select(Message).where(Message.content == msg)
                        )
                    ).scalars()
                )
                return conv is not None, rows
        finally:
            await engine.dispose()

    conv_exists, rows = client.portal.call(_fetch)
    assert conv_exists, "SAVEPOINT 之外的会话行仍应保留"
    assert rows == [], "append 已 flush 的 user 行必须被回滚"
