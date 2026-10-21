"""工具轮在 `/api/chat` 工作流适配层上的契约（工具帧、回灌全文、错误边界、回放）。

ch05 起模型回合由 LangGraph 驱动：`agent_model` 每次都绑全部工具，收敛靠
“没有新 tool_calls”或步数/预算触顶；适配层只负责把 custom 事件翻译成 SSE、
把最终 State 落审计。这里锁住这些行为，避免迁走旧单轮循环时丢掉覆盖。
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk

from app.api import get_store
from app.db.base import get_db
from app.main import app
from app.providers import get_default_chat_model


@pytest.fixture(autouse=True)
def _stub_faq_retrieval(monkeypatch):
    """路由测试不碰外部 embedding/Milvus。"""
    from app.db.models import KnowledgeChunk
    from app.knowledge import retrieval

    async def fake_search_with_metadata(session, query, top_k=None, **kw):
        return (
            [
                KnowledgeChunk(
                    chunk_id="policy.md:9",
                    category="售后政策 > 退换货",
                    questions="退货政策",
                    answer=(
                        "签收后 7 天内可无理由退货。商品需保持吊牌完整、不影响二次销售。"
                        "定制类商品不支持无理由退货。因商品质量问题导致的退换货，"
                        "往返运费由平台承担。"
                    ),
                    section_path="售后政策 > 退换货 > 退货政策",
                    content_type="policy",
                    status="已向量化",
                    content_hash="x",
                )
            ],
            {"strategy": "dense", "scores": [0.88], "top_score": 0.88},
        )

    monkeypatch.setattr(retrieval, "search_with_metadata", fake_search_with_metadata)


class FakeStore:
    def __init__(self):
        self._data = {}
        self.metadata = {}
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
    """按调用次序作答的模型桩；意图分类走 `ainvoke`。"""

    def __init__(self, intent: str, replies):
        self.intent = intent
        self.replies = list(replies)
        self.agent_calls: list[list] = []

    async def ainvoke(self, messages):
        return AIMessage(
            content=json.dumps(
                {"intent": self.intent, "confidence": 0.9}, ensure_ascii=False
            )
        )

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.agent_calls.append(list(messages))
        reply = self.replies[min(len(self.agent_calls) - 1, len(self.replies) - 1)]
        for chunk in reply if isinstance(reply, list) else [reply]:
            if isinstance(chunk, BaseException):
                raise chunk
            if isinstance(chunk, AIMessageChunk):
                yield chunk
            else:
                yield AIMessageChunk(
                    content=chunk.content or "",
                    tool_calls=list(chunk.tool_calls or []),
                    usage_metadata=chunk.usage_metadata,
                    response_metadata=chunk.response_metadata,
                )


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


FAQ_TOOL = tool_call("query_faq", {"keyword": "退货"}, "call_x")


@pytest.fixture
def client(db_session):
    """★ 必须注入真实的 db session：路由要用它做 get_or_create。"""
    store = FakeStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_default_chat_model] = lambda: GraphModel(
        "物流", [AIMessage(content="您好，有什么可以帮您？")]
    )
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as c:
        c.store = store
        yield c
    app.dependency_overrides.clear()


def parse_sse(raw: str):
    out = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        out.append((name, data))
    return out


def use_model(model):
    app.dependency_overrides[get_default_chat_model] = lambda: model


def test_tool_events_appear_before_deltas(client):
    model = GraphModel(
        "退货退款", [FAQ_TOOL, [AIMessageChunk(content="退货政策是七天无理由。")]]
    )
    use_model(model)

    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    names = [name for name, _ in parse_sse(r.text)]

    assert names[0] == "meta"
    assert "tool" in names
    assert names.index("tool") < names.index("delta")
    assert names[-1] == "done"


def test_tool_start_and_end_frames(client):
    model = GraphModel(
        "退货退款", [FAQ_TOOL, [AIMessageChunk(content="退货政策是七天无理由。")]]
    )
    use_model(model)

    tools = [
        data
        for name, data in parse_sse(
            client.post("/api/chat", json={"message": "退货政策是什么"}).text
        )
        if name == "tool"
    ]

    assert tools[0]["phase"] == "start"
    assert tools[0]["name"] == "query_faq"
    assert tools[0]["arguments"] == {"keyword": "退货"}
    assert tools[1]["phase"] == "end"
    assert tools[1]["call_id"] == tools[0]["call_id"]
    assert tools[1]["name"] == "query_faq"
    assert tools[1]["ok"] is True
    assert "summary" in tools[1]
    assert "elapsed_ms" in tools[1]
    assert "content" not in tools[1], "SSE 帧只发 summary，不发回灌全文"


def test_full_message_sequence_is_stored(client):
    """★ 审计必须是完整序列：user → assistant(带 tool_calls) → tool → 最终 assistant。

    checkpointer 用同一份完整序列，下一轮才不会有悬空 tool_call_id。
    """
    model = GraphModel(
        "退货退款", [FAQ_TOOL, [AIMessageChunk(content="退货政策是七天无理由。")]]
    )
    use_model(model)

    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    sid = parse_sse(r.text)[0][1]["session_id"]

    msgs = client.store._data[sid]
    assert [type(m).__name__ for m in msgs] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]
    assert msgs[0].content == "退货政策是什么"
    assert msgs[1].tool_calls and msgs[1].tool_calls[0]["name"] == "query_faq"
    assert msgs[2].tool_call_id == msgs[1].tool_calls[0]["id"], "tool_call_id 必须对得上"
    assert msgs[3].content, "最终回答不能为空"


def test_no_tool_turn_stores_two_messages(client):
    """不需要工具的轮次：只存 user + assistant，没有 tool 消息。"""
    model = GraphModel(
        "物流", [AIMessage(content="您好，有什么可以帮您？")]
    )
    use_model(model)

    r = client.post(
        "/api/chat", json={"message": "订单 1001 的物流到哪了"}
    )
    sid = parse_sse(r.text)[0][1]["session_id"]

    msgs = client.store._data[sid]
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"]
    assert msgs[1].content == "您好，有什么可以帮您？"
    assert msgs[1].tool_calls == []


def test_stored_tool_message_holds_full_content_not_the_badge_summary(client):
    """两个文本字段各走各的路：summary 给前端，content 回灌模型并落库。"""
    model = GraphModel(
        "退货退款", [FAQ_TOOL, [AIMessageChunk(content="退货政策是七天无理由。")]]
    )
    use_model(model)

    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    sid = parse_sse(r.text)[0][1]["session_id"]
    tools = [data for name, data in parse_sse(r.text) if name == "tool"]

    assert "content" not in tools[1]
    # 工作流里的 query_faq 事件给的是“检索到 N 条知识”的事实摘要。
    assert tools[1]["summary"] == "检索到 1 条知识"

    stored = client.store._data[sid][2]
    assert type(stored).__name__ == "ToolMessage"
    assert stored.content != tools[1]["summary"], "库里存的不能是摘要"
    assert len(stored.content) > len(tools[1]["summary"])
    assert "退货" in stored.content, "存的应是 FAQ 工具的真实返回全文"


class MetadataModel(GraphModel):
    USAGE = {"input_tokens": 31, "output_tokens": 7, "total_tokens": 38}

    def __init__(self):
        super().__init__(
            "退货退款",
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "query_faq", "args": {"keyword": "退货"}, "id": "call_m"}
                    ],
                ),
                AIMessage(
                    content="退货政策是七天无理由。",
                    usage_metadata=self.USAGE,
                    response_metadata={"finish_reason": "length"},
                ),
            ],
        )


def test_done_frame_surfaces_real_finish_reason_and_usage(client):
    """上游截断（finish_reason="length"）与完整 usage 必须如实到达客户端。"""
    use_model(MetadataModel())

    done = [
        data
        for name, data in parse_sse(
            client.post("/api/chat", json={"message": "退货政策是什么"}).text
        )
        if name == "done"
    ][-1]

    assert done["finish_reason"] == "length", "截断不能伪装成 stop"
    assert done["usage"] == MetadataModel.USAGE, "usage 应保留完整三项"


class FailsOnSecondCallModel(GraphModel):
    def __init__(self):
        super().__init__(
            "退货退款",
            [
                FAQ_TOOL,
                RuntimeError("上游炸了"),
            ],
        )


def test_tool_round_failure_is_error_not_done_and_persists_nothing(client):
    use_model(FailsOnSecondCallModel())
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    names = [name for name, _ in parse_sse(r.text)]

    assert names[-1] == "error", "工具轮中途失败仍是 event: error"
    assert "done" not in names, "error 与 done 互斥"
    assert "tool" in names, "工具帧已经发过了"
    assert client.store._data == {}, "半截的一轮一句都不许落库"


def test_second_turn_replays_the_first_turns_tool_round(client):
    """★ 历史回放：第二轮模型消息里必须有第一轮完整的工具回合。

    上下文归 checkpointer；审计归 MySQL/FakeStore。这条用例锁住“删掉
    `store.get()` 之后模型上下文没有断”。
    """
    model = GraphModel(
        "退货退款",
        [
            FAQ_TOOL,
            [AIMessageChunk(content="退货政策是七天无理由。")],
            [AIMessageChunk(content="退款一般三个工作日到账。")],
        ],
    )
    use_model(model)

    r1 = client.post("/api/chat", json={"message": "退货政策是什么"})
    sid = parse_sse(r1.text)[0][1]["session_id"]
    client.post("/api/chat", json={"session_id": sid, "message": "那退了多久能退款"})

    assert len(model.agent_calls) == 3, "第一轮两次调用 + 第二轮一次调用"
    seen = model.agent_calls[2]
    assert [type(m).__name__ for m in seen] == [
        "SystemMessage", "SystemMessage", "HumanMessage", "AIMessage",
        "ToolMessage", "AIMessage", "HumanMessage",
    ]
    assert seen[2].content == "退货政策是什么"
    assert seen[3].tool_calls and seen[3].tool_calls[0]["name"] == "query_faq"
    assert seen[4].tool_call_id == seen[3].tool_calls[0]["id"], "tool_call_id 必须对得上"
    assert seen[5].content == "退货政策是七天无理由。", "最终回答也要在历史里"
    assert seen[6].content == "那退了多久能退款"


def test_second_turn_replays_tool_round_from_mysql(client, db_session):
    """审计侧的 MySQL 往返：tool_calls / tool_call_id / 全文都能原样读回。"""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.config import get_settings
    from app.sessions import MySQLSessionStore

    model = GraphModel(
        "退货退款",
        [FAQ_TOOL, [AIMessageChunk(content="退货政策是七天无理由。")]],
    )
    use_model(model)
    app.dependency_overrides[get_store] = lambda: MySQLSessionStore(session=db_session)

    r1 = client.post("/api/chat", json={"message": "退货政策是什么"})
    sid = parse_sse(r1.text)[0][1]["session_id"]

    async def _read():
        # 另起引擎：只读已提交的行，也避免把 db_session 的事务带出 portal 循环。
        engine = create_async_engine(get_settings().db_url)
        try:
            async with async_sessionmaker(engine)() as other:
                return await MySQLSessionStore(session=other).get(sid)
        finally:
            await engine.dispose()

    stored = client.portal.call(_read)
    assert [type(m).__name__ for m in stored] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]
    assert stored[1].tool_calls[0]["name"] == "query_faq"
    assert stored[2].tool_call_id == stored[1].tool_calls[0]["id"]
    assert stored[2].content.startswith("[1] 问：")
    assert len(stored[2].content) > 60, "库里存的不是 60 字摘要，而是回灌给模型的全文"
