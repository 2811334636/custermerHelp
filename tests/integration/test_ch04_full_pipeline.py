"""ch04 Task 11：完整链路集成测试。

覆盖 ch04 的四条验收标准在**同一条生产链路**上的表现（聊天模型除外）：

    /api/chat（脚本化假模型）
      → 模型申请 query_faq（真实工具）
      → 真实 query 改写替身（identity，不调在线 LLM）
      → 真实 embedding API → 真实 Milvus hybrid(稠密 + BM25) → 真实 MySQL 回表
      → 真实重排替身（确定性分数，不调在线 reranker）
      → citations 进 tool end / done 帧，并随最终 assistant 消息落库
      → GET /api/chunks/{chunk_id} 可取回原文与章节路径

拒答路径复用 Task 10 的做法：临时 Milvus collection 里只放一条 MySQL 中
不存在的「孤儿向量」——真实检索能召回 id、回表为空 → ``no_results`` 入池。
真实 ``knowledge`` collection 对任何 query 都会返回最近邻，永远走不到
空结果分支，所以必须用临时 collection 才能忠实触发拒答。

确定性边界：聊天模型、query 改写、reranker 都是脚本/替身；MySQL、Milvus、
embedding API 仍是真实的（与 Task 9/10 验收一致，infra 需已启动）。
"""

import json
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import get_store
from app.config import get_settings
from app.db.models import Conversation, LowConfidenceQuestion
from app.knowledge.vector_store import VectorStore
from app.main import app
from app.providers import get_default_chat_model

NORMAL_QUESTION = "邮费咋算"
REFUSAL_QUESTION = "火星移民计划的详细时间表"


class FakeStore:
    """内存会话存储桩：跟随现有 API 用例，便于断言最终 assistant 的 metadata。"""

    def __init__(self):
        self._data: dict[str, list] = {}
        self.metadata: dict[str, dict] = {}
        self._next_id = 1

    async def get(self, session_id):
        return list(self._data.get(session_id, []))

    async def append(self, session_id, *messages, metadata=None):
        msgs = self._data.setdefault(session_id, [])
        ids = []
        for m in messages:
            msgs.append(m)
            ids.append(self._next_id)
            self._next_id += 1
        if metadata is not None and messages:
            self.metadata[session_id] = metadata
        return ids

    async def exists(self, session_id):
        return session_id in self._data


class ScriptedToolModel:
    """脚本化假模型：turn 1 申请 query_faq，turn 2 原样吐回工具内容。

    ``suffix`` 用于正常路径补上引用编号 ``[1]``；拒答路径保持空串，让
    最终正文完全由真实检索结果决定（与 Task 10 的 RefusalModel 一致）。
    """

    def __init__(self, keyword: str, suffix: str = ""):
        self._keyword = keyword
        self._suffix = suffix
        self._bound: list = []

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        if self._bound:
            yield AIMessageChunk(
                content="",
                tool_calls=[{
                    "name": "query_faq",
                    "args": {"keyword": self._keyword},
                    "id": "call-1",
                }],
            )
        else:
            # 工具轮之后，最后一条一定是 query_faq 回灌的 ToolMessage。
            yield AIMessageChunk(content=f"{messages[-1].content}{self._suffix}")


@pytest.fixture(autouse=True)
def _reset_embedding_client():
    """每个用例的 portal 事件循环各用一个 embedding 客户端。

    ``embedding._client`` 是 lru_cache 的 ``OpenAIEmbeddings``，内部 httpx
    连接池绑定创建它的事件循环。TestClient 每个用例一个新循环；不清缓存
    的话，第二个用例会复用上个循环的客户端，直接抛
    ``EmbeddingUnavailable``（表现为「知识库暂时不可用」）。
    """
    from app.knowledge.embedding import _client

    _client.cache_clear()
    yield
    _client.cache_clear()


@pytest.fixture(autouse=True)
def _identity_rewrite(monkeypatch):
    """改写是可选 LLM 增强；集成测试不依赖在线 LLM。"""
    from app.knowledge import query_rewrite

    async def _identity(query: str) -> str:
        return query

    monkeypatch.setattr(query_rewrite, "rewrite_query", _identity)


@pytest.fixture(autouse=True)
def _deterministic_rerank(monkeypatch):
    """重排打分替身：分数确定、全部高于阈值，且不访问在线 reranker。

    生产 ``_hybrid_rerank`` 通过 ``app.knowledge.reranker.rerank`` 取函数，
    所以补丁打在模块属性上即可；真实 Milvus 融合路径不受影响。
    """

    async def _fake_rerank(query: str, passages: list[str], top_k: int = 10):
        return [(i, 0.99 - i * 0.01) for i in range(min(top_k, len(passages)))]

    from app.knowledge import reranker

    monkeypatch.setattr(reranker, "rerank", _fake_rerank)


@pytest.fixture
def client():
    """只替换会话存储与聊天模型；``get_db`` 用真实依赖。

    ★ 不注入 ``db_session`` fixture：路由在 TestClient 的 portal 事件循环里
    执行，而 ``db_session`` 的连接属于外层测试循环。把它跨循环借给路由
    会在 teardown 时抛 "got Future attached to a different loop"，并让
    后续用例拿到一个坏 session。生产 ``get_db`` 每个请求自建 session，
    提交行为与线上一致，正适合本集成测试。
    """
    store = FakeStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_default_chat_model] = lambda: ScriptedToolModel(
        NORMAL_QUESTION, " [1]"
    )
    with TestClient(app) as c:
        c.store = store
        yield c
        # ★ 在本测试自己的 portal 循环里释放全局连接池：不释放的话，下一个
        # 用例的新事件循环会复用上个循环里的 aiomysql 连接，抛
        # "got Future attached to a different loop"（路由用的是真实 get_db）。
        c.portal.call(_dispose_app_engine)
    app.dependency_overrides.clear()


async def _dispose_app_engine() -> None:
    from app.db.base import engine

    await engine.dispose()


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


def test_hybrid_rerank_full_pipeline(client):
    """完整链路：混合检索 → 重排 → 引用编号 → SSE → 落库 → 原文回查。"""
    # 路由会真提交会话行：非 *_test 库上直接失败，绝不污染生产库。
    _assert_test_database()

    r = client.post("/api/chat", json={"message": NORMAL_QUESTION})
    assert r.status_code == 200
    events = parse_sse(r.text)
    session_id = events[0][1]["session_id"]

    # 1) 工具真的被调用，且真实检索成功。
    tool_frames = [d for n, d in events if n == "tool"]
    assert [f["phase"] for f in tool_frames] == ["start", "end"]
    assert tool_frames[0]["name"] == "query_faq"
    assert tool_frames[0]["arguments"] == {"keyword": NORMAL_QUESTION}
    assert tool_frames[1]["ok"] is True

    # 2) tool end 帧带非空 citations，首条是真实 chunk_id。
    citations = tool_frames[1]["citations"]
    assert citations, "正常召回必须有引用"
    # 引用编号必须与召回顺序一一对应（1-based），不能只看首条。
    for position, citation in enumerate(citations):
        assert citation["index"] == position + 1
        assert isinstance(citation["chunk_id"], str) and citation["chunk_id"]
    first = citations[0]

    # 3) 浏览器看到的流式正文包含引用编号与运费内容。
    deltas = [d["text"] for n, d in events if n == "delta"]
    streamed = "".join(deltas)
    assert "[1]" in streamed, f"流式正文缺少引用编号：{streamed!r}"
    assert "运费" in streamed, f"未召回运费相关内容：{streamed!r}"

    # 4) done 帧带 message_id 与同一组 citations。
    done = [d for n, d in events if n == "done"][-1]
    assert done["message_id"] is not None
    assert isinstance(done["message_id"], int)
    assert done["citations"] == citations

    # 5) 最终 assistant 消息的 metadata 落库了引用（会话存储桩）。
    stored = client.store._data[session_id]
    assert stored[-1].content == streamed
    assert client.store.metadata[session_id]["citations"] == citations
    assert client.store.metadata[session_id]["retrieval_strategy"] == "hybrid_rerank"
    assert client.store.metadata[session_id]["top_score"] is not None

    # 6) 引用编号可点：GET /api/chunks/{chunk_id} 返回原文与章节路径，
    #    且这条引用确实指向运费知识块（不是随便一个 chunk_id）。
    resp = client.get(f"/api/chunks/{first['chunk_id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunk_id"] == first["chunk_id"]
    assert body["content"]
    assert body["section_path"]
    assert "运费" in (body["content"] or "") or "运费" in (body["questions"] or "")


def _assert_test_database() -> str:
    """只允许清理 *_test 库（与 tests/conftest.py 的守卫一致）。"""
    db_name = make_url(get_settings().db_url).database or ""
    assert db_name.endswith("_test"), (
        f"拒绝在非 *_test 库上清理低置信度池（当前库：{db_name!r}）"
    )
    return db_name


async def _fetch_pool(session_id: str):
    """另起引擎读已提交行，并用会话 uuid 取出会话主键。"""
    engine = create_async_engine(get_settings().db_url)
    try:
        async with async_sessionmaker(engine)() as other:
            conv = (
                await other.execute(
                    select(Conversation).where(
                        Conversation.session_uuid == session_id
                    )
                )
            ).scalar_one()
            rows = list(
                (
                    await other.execute(
                        select(LowConfidenceQuestion).where(
                            LowConfidenceQuestion.question == REFUSAL_QUESTION,
                            LowConfidenceQuestion.session_id == str(conv.id),
                        )
                    )
                ).scalars()
            )
            return conv.id, [
                (row.question, row.reason, row.session_id) for row in rows
            ]
    finally:
        await engine.dispose()


@pytest.fixture
def empty_kb(monkeypatch):
    """真实 Milvus 临时 collection，只含一条 MySQL 不存在的孤儿向量。"""
    name = f"integration_refusal_{uuid4().hex[:8]}"
    store = VectorStore(collection=name, support_bm25=True)
    store.ensure_collection()
    try:
        dim = get_settings().embedding_dim
        # 单位向量：dense 检索对任意 query 都会召回这一条。
        store.upsert([{
            "chunk_id": "orphan:0",
            "content": REFUSAL_QUESTION,
            "vector": [1.0] + [0.0] * (dim - 1),
        }])
        # 与 Task 9/10 一致：等待索引可见。
        time.sleep(0.5)
        monkeypatch.setattr(get_settings(), "milvus_collection", name)
        yield name
    finally:
        store.drop_collection()


def test_refusal_path_pools_question(client, empty_kb):
    """拒答路径：空召回 → 标准拒答话术 → 恰好一行 no_results 入池。"""
    _assert_test_database()
    app.dependency_overrides[get_default_chat_model] = lambda: ScriptedToolModel(
        REFUSAL_QUESTION
    )

    # 只清本用例的 question，且 DELETE 在应用事件循环里执行。
    async def _clear_pool():
        # 自建引擎并在同一个 portal 循环里 dispose：复用全局 AsyncSessionLocal
        # 会拿到上个测试循环遗留的连接池连接（跨循环 Future 错误）。
        engine = create_async_engine(get_settings().db_url)
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "DELETE FROM low_confidence_questions"
                        " WHERE question = :question"
                    ),
                    {"question": REFUSAL_QUESTION},
                )
        finally:
            await engine.dispose()

    client.portal.call(_clear_pool)

    r = client.post("/api/chat", json={"message": REFUSAL_QUESTION})
    assert r.status_code == 200
    events = parse_sse(r.text)
    session_id = events[0][1]["session_id"]

    # 工具调用成功（「查不到」是业务结果），但没有引用。
    tools = [d for n, d in events if n == "tool"]
    assert tools[0]["phase"] == "start"
    assert tools[1]["phase"] == "end"
    assert tools[1]["ok"] is True
    assert tools[1]["citations"] == []

    # 浏览器看到的就是标准拒答话术。
    streamed = "".join(d["text"] for n, d in events if n == "delta")
    assert "没有找到" in streamed, f"流式正文缺少拒答措辞：{streamed!r}"
    assert "转人工" in streamed
    assert all(
        "<INSUFFICIENT_KNOWLEDGE>" not in d["text"]
        for n, d in events
        if n == "delta"
    )

    # 有且仅有一行入池，reason 与作用域都对得上。
    conv_id, pooled = client.portal.call(_fetch_pool, session_id)
    assert pooled == [(REFUSAL_QUESTION, "no_results", str(conv_id))]
