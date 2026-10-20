"""验收标准 4：问知识库没有的内容，得到明确拒答，且问题进入低置信度池。

端到端链路（除聊天模型外全部是真的）：

    /api/chat（脚本化假模型）
      → 模型申请 query_faq（真实工具）
      → 真实检索：embedding API → 真实 Milvus hybrid → MySQL 回表
      → 回表为空 → retrieval._refuse 写入 low_confidence_questions（reason=no_results）
      → 工具把「没有找到…转人工」回灌模型
      → 假模型原样吐回该文本 → SSE delta 帧 → 落库

为什么用临时 collection：真实 ``knowledge`` collection 里 dense ANN 对**任何**
query 都会返回 top-k 最近邻（向量检索没有“无结果”），所以直接问真实库只会
召回无关块或触发 low_score，永远走不到 ``no_results``。这里用真实 Milvus
建一个临时 collection，只写入一条 **MySQL 中不存在的** 向量（孤儿向量）：
检索真实执行，回表阶段按孤儿跳过 → 结果为空 → 空结果拒答入池。这既保住了
「真实 Milvus + 真实 MySQL」的要求，又让 ``no_results`` 分支被真正执行。
"""

import json
import time
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import get_store
from app.config import get_settings
from app.db.base import get_db
from app.db.models import Conversation, LowConfidenceQuestion
from app.knowledge.vector_store import VectorStore
from app.main import app
from app.providers import get_default_chat_model

QUESTION = "火星移民计划的详细时间表"


class FakeStore:
    """内存会话存储桩：本用例不关心历史回放，只关心流与问题池。"""

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


class RefusalModel:
    """脚本化假模型（替代真实 LLM）。

    第一次流：申请 query_faq，keyword 就是用户原问题；
    第二次流：把工具回灌的 ToolMessage 内容**原样**吐回。这样最终答案
    完全由真实的检索结果决定，模型不会自己编话术。
    """

    def __init__(self):
        self._bound: list = []

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        if self._bound:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "query_faq",
                    "args": json.dumps({"keyword": QUESTION}, ensure_ascii=False),
                    "id": "call_refusal",
                    "index": 0,
                    "type": "tool_call_chunk",
                }],
            )
        else:
            # 工具轮之后，最后一条一定是 query_faq 回灌的 ToolMessage。
            yield AIMessageChunk(content=messages[-1].content)


@pytest.fixture(autouse=True)
def _identity_rewrite(monkeypatch):
    """query 改写是可选的 LLM 增强；验收只关心检索与拒答，不依赖在线 LLM。

    改写与否都不影响本用例结论：临时 collection 里只有一条向量，dense 分支
    必然召回它，随后在 MySQL 回表阶段被当作孤儿跳过。
    """
    from app.knowledge import query_rewrite

    async def _identity(query: str) -> str:
        return query

    monkeypatch.setattr(query_rewrite, "rewrite_query", _identity)


@pytest.fixture
def empty_kb(monkeypatch):
    """把在线检索指向一个只含孤儿向量的真实 Milvus 临时 collection。"""
    name = f"acceptance_refusal_{uuid4().hex[:8]}"
    store = VectorStore(collection=name, support_bm25=True)
    store.ensure_collection()
    try:
        dim = get_settings().embedding_dim
        # 单位向量：dense 检索对任意 query 都会召回这一条。
        store.upsert([{
            "chunk_id": "orphan:0",
            "content": QUESTION,
            "vector": [1.0] + [0.0] * (dim - 1),
        }])
        # 与 tests/acceptance/test_bm25_literal_match.py 一致：等待索引可见。
        time.sleep(0.5)
        # 生产代码在请求时读取 get_settings().milvus_collection；改的是缓存
        # 实例上的字段，monkeypatch 结束自动还原，不污染其它用例。
        monkeypatch.setattr(get_settings(), "milvus_collection", name)
        yield name
    finally:
        store.drop_collection()


@pytest.fixture
def client(db_session):
    store = FakeStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_default_chat_model] = lambda: RefusalModel()
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


async def _fetch_pool(session_id: str):
    """另起引擎读已提交行，并顺带取会话主键。

    路由在 portal 循环里提交；外层测试循环复用原 session/连接会撞上
    “got Future attached to a different loop”（见 tests/test_api.py 注释）。
    """
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
                (await other.execute(select(LowConfidenceQuestion))).scalars()
            )
            return conv.id, [
                (row.question, row.reason, row.session_id) for row in rows
            ]
    finally:
        await engine.dispose()


async def test_refusal_and_low_confidence_pool(client, db_session, empty_kb):
    # 不用 TRUNCATE：TRUNCATE 隐式提交，会破坏 db_session fixture 的事务边界。
    # 通过 portal 在应用所在循环里执行，连接才不会跨事件循环串用。
    async def _clear_pool():
        await db_session.execute(text("DELETE FROM low_confidence_questions"))

    client.portal.call(_clear_pool)

    r = client.post("/api/chat", json={"message": QUESTION})
    assert r.status_code == 200
    events = parse_sse(r.text)
    session_id = events[0][1]["session_id"]

    # 1) 工具真的被调用，且真实检索返回空结果（成功返回的「查不到」）。
    tools = [d for n, d in events if n == "tool"]
    assert tools[0]["phase"] == "start"
    assert tools[0]["name"] == "query_faq"
    assert tools[0]["arguments"] == {"keyword": QUESTION}
    assert tools[1]["phase"] == "end"
    assert tools[1]["ok"] is True, "「查不到」是正常业务结果，不是工具故障"
    assert tools[1]["citations"] == [], "没有召回就不该有引用"

    # 2) 浏览器看到的流式正文就是拒答话术（不是只有落库行对）。
    deltas = [d["text"] for n, d in events if n == "delta"]
    streamed = "".join(deltas)
    assert "没有找到" in streamed, f"流式正文缺少拒答措辞：{streamed!r}"
    assert "转人工" in streamed, "拒答应提示转人工"
    assert all("<INSUFFICIENT_KNOWLEDGE>" not in t for t in deltas)

    # 3) 有且仅有一行入池，且与会话主键对得上。
    conv_id, pooled = client.portal.call(_fetch_pool, session_id)
    assert pooled == [(QUESTION, "no_results", str(conv_id))]
