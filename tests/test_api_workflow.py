"""Task 9：`/api/chat` 接入 LangGraph 工作流的端到端契约。

覆盖 brief 的五条主路径（知识/业务/投诉/闲聊/断流）+ 审计切片恢复路径，
并承接从 `tests/test_chat.py` / `tests/test_chat_tools.py` 迁来的有价值断言：
工具事件顺序（含“先说后调”）、错误分类、收敛（工具轮后不再调工具）。
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import REFUSAL_TEXT, get_store
from app.config import get_settings
from app.db.base import get_db
from app.db.models import Conversation, LowConfidenceQuestion
from app.main import app
from app.providers import get_default_chat_model
from app.repositories import low_confidence_question as lcq_repo
from app.workflow.graph import workflow_deps
from app.workflow.texts import (
    CHITCHAT_TEXT,
    COMPLAINT_TEXT,
    GATE_REFUSAL,
    STEP_LIMIT_TEXT,
)


# ------------------------------------------------------------------ 测试替身


class FakeStore:
    """审计写入的内存桩（ch05 起上下文归 checkpointer，这里只记审计）。"""

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
    """按**调用次序**切换行为的模型桩。

    意图分类走 `ainvoke`；Agent 子图走 `bind_tools().astream`。每个 reply 可以是
    单条 `AIMessage`、一串 `AIMessageChunk`，或一个要抛出的异常。
    """

    def __init__(self, intent: str, replies=None):
        self.intent = intent
        self.replies = list(replies or [])
        self.agent_calls: list[list] = []
        self.bound_tools: list[list] = []

    async def ainvoke(self, messages):
        return AIMessage(
            content=json.dumps(
                {"intent": self.intent, "confidence": 0.9}, ensure_ascii=False
            )
        )

    def bind_tools(self, tools):
        self.bound_tools.append(list(tools))
        return self

    async def astream(self, messages):
        self.agent_calls.append(list(messages))
        index = min(len(self.agent_calls) - 1, len(self.replies) - 1)
        reply = self.replies[index]
        if isinstance(reply, BaseException):
            raise reply
        steps = reply if isinstance(reply, list) else [reply]
        for step in steps:
            if isinstance(step, BaseException):
                # 列表里的异常必须真的从 astream 抛出去，而不是被当成消息
                # 解引用（那样只会得到 AttributeError，测错错误路径）。
                raise step
            if isinstance(step, AIMessageChunk):
                yield step
            else:
                yield AIMessageChunk(
                    content=step.content or "",
                    tool_calls=list(step.tool_calls or []),
                    usage_metadata=step.usage_metadata,
                    response_metadata=step.response_metadata,
                )


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
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
def _stub_retrieval(monkeypatch):
    """工作流测试不碰 Milvus / embedding；检索质量另有专门用例。"""
    from app.knowledge import retrieval

    async def fake_search_with_metadata(session, query, top_k=None, **kwargs):
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
        "闲聊", [AIMessage(content=CHITCHAT_TEXT)]
    )
    with TestClient(app) as c:
        c.store = store
        yield c
    app.dependency_overrides.clear()


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    events = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        events.append((name, data))
    return events


def use_model(model) -> None:
    app.dependency_overrides[get_default_chat_model] = lambda: model


# ------------------------------------------------------------------ 1. 知识类


def test_knowledge_turn_has_no_tool_and_audits_user_plus_answer(client):
    sid = str(uuid.uuid4())
    model = GraphModel(
        "退货退款",
        [[AIMessageChunk(content="签收后"), AIMessageChunk(content="七天可退。")]],
    )
    use_model(model)

    r = client.post(
        "/api/chat", json={"session_id": sid, "message": "退货政策是什么"}
    )
    events = parse_sse(r.text)
    names = [name for name, _ in events]

    assert names[0] == "delta"  # 自带 session_id：不发 meta
    assert "tool" not in names, "知识类是预检索，不经过工具事件"
    assert names[-1] == "done"
    assert "".join(d["text"] for name, d in events if name == "delta") == (
        "签收后七天可退。"
    )

    done = events[-1][1]
    assert done["finish_reason"] == "stop"
    assert done["usage"] is None
    assert done["actions"] == []
    assert done["citations"] == [
        {"index": 1, "chunk_id": "policy.md:9", "score": None}
    ]
    assert isinstance(done["message_id"], int)

    stored = client.store._data[sid]
    assert [type(m).__name__ for m in stored] == ["HumanMessage", "AIMessage"]
    assert stored[0].content == "退货政策是什么"
    assert stored[1].content == "签收后七天可退。"
    # evidence 注入的 SystemMessage 已被过滤，不进审计。
    assert all(type(m).__name__ != "SystemMessage" for m in stored)

    metadata = client.store.metadata[sid]
    assert metadata["citations"] == done["citations"]
    assert metadata["actions"] == []
    assert metadata["retrieval_strategy"] == get_settings().retrieval_strategy
    assert metadata["top_score"] == 0.88, "top_score 取预检索 evidence 的最高分"


def test_second_turn_uses_checkpoint_history_not_mysql(client):
    """上下文归 checkpointer：第二轮模型能看到第一轮消息。

    MySQL 只做审计；删掉 `store.get()` 后，这个断言保证模型上下文没有断。
    """
    sid = str(uuid.uuid4())
    model = GraphModel(
        "物流",
        [
            [AIMessageChunk(content="第一次回答")],
            [AIMessageChunk(content="第二次回答")],
        ],
    )
    use_model(model)

    first = "订单 1001 的物流到哪了"
    second = "订单 1002 的物流到哪了"
    client.post("/api/chat", json={"session_id": sid, "message": first})

    # ★ 证明上下文真的来自 checkpointer：清空审计存储并让 get() 直接爆炸。
    # 若实现暗地里回退到 `store.get()`，第二轮会立即失败。
    client.store._data.pop(sid, None)

    async def _forbidden(_sid):
        raise AssertionError("store 不能作为模型上下文来源")

    client.store.get = _forbidden

    client.post("/api/chat", json={"session_id": sid, "message": second})

    second_turn_messages = model.agent_calls[1]
    contents = [message.content for message in second_turn_messages]
    assert first in contents
    assert "第一次回答" in contents
    assert contents[-1] == second
    # 第二轮审计只有第二轮的两条（第一轮已被手动清掉）。
    assert [m.content for m in client.store._data[sid]] == [
        second,
        "第二次回答",
    ]


# ------------------------------------------------------------------ 2. 业务类


def test_business_tool_frames_and_citations(client):
    model = GraphModel(
        "物流",
        [
            tool_call("query_faq", {"keyword": "退货"}, "call_1"),
            [AIMessageChunk(content="签收后 7 天内可退 [1]。")],
        ],
    )
    use_model(model)

    r = client.post(
        "/api/chat", json={"message": "订单 1001 的物流到哪了"}
    )
    events = parse_sse(r.text)
    sid = events[0][1]["session_id"]
    names = [name for name, _ in events]
    tools = [data for name, data in events if name == "tool"]

    assert names.index("tool") < names.index("delta")
    assert tools[0]["phase"] == "start"
    assert tools[0]["name"] == "query_faq"
    assert tools[0]["arguments"] == {"keyword": "退货"}
    assert tools[1]["phase"] == "end"
    assert tools[1]["call_id"] == tools[0]["call_id"]
    assert tools[1]["ok"] is True
    assert tools[1]["summary"] == "检索到 1 条知识"
    assert "content" not in tools[1], "SSE 只发摘要，不发回灌全文"
    assert tools[1]["citations"] == [
        {"index": 1, "chunk_id": "policy.md:9", "score": None}
    ]

    done = events[-1][1]
    assert done["citations"] == tools[1]["citations"]
    assert done["actions"] == []

    # 完整序列：user → assistant(tool_calls) → tool → 最终 assistant。
    stored = client.store._data[sid]
    assert [type(m).__name__ for m in stored] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    assert stored[1].tool_calls[0]["name"] == "query_faq"
    assert stored[2].tool_call_id == stored[1].tool_calls[0]["id"]
    assert "问：退货政策" in stored[2].content, "库里存回灌全文，不是摘要"
    assert "签收后 7 天" in stored[2].content
    assert stored[3].content == "签收后 7 天内可退 [1]。"


def test_tool_round_converges_without_another_tool_call(client):
    """收敛：第一次模型调用申请工具，第二次直接作答（图不再绑空工具，
    但 agent_done 的条件边保证没有新 tool_calls 就结束）。"""
    model = GraphModel(
        "物流",
        [
            tool_call("query_faq", {"keyword": "退货"}, "call_1"),
            [AIMessageChunk(content="已查到。")],
        ],
    )
    use_model(model)
    client.post("/api/chat", json={"message": "订单 1001 的物流到哪了"})

    assert len(model.agent_calls) == 2, "一次工具轮 + 一次最终作答"


def test_preamble_before_tool_is_streamed_then_tool_runs(client):
    """迁移自 test_chat_tools：模型先说后调，工具仍必须执行。"""
    model = GraphModel(
        "物流",
        [
            [
                AIMessageChunk(content="好的，我帮您查一下"),
                AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_faq",
                            "args": {"keyword": "退货"},
                            "id": "call_p",
                        }
                    ],
                ),
            ],
            [AIMessageChunk(content="您的包裹已发出。")],
        ],
    )
    use_model(model)

    events = parse_sse(
        client.post(
            "/api/chat", json={"message": "订单 1001 的物流到哪了"}
        ).text
    )
    names = [name for name, _ in events]

    first_tool = names.index("tool")
    assert names.index("delta") < first_tool, "前言要实时转发"
    assert "".join(
        data["text"]
        for position, (name, data) in enumerate(events)
        if name == "delta" and position < first_tool
    ) == "好的，我帮您查一下"
    assert names[-1] == "done"


# ------------------------------------------------------------------ 3. 投诉


def test_complaint_returns_handoff_and_ticket_without_tool(client):
    user_input = "我要投诉你们客服"
    model = GraphModel("投诉", [])
    use_model(model)

    events = parse_sse(
        client.post("/api/chat", json={"message": user_input}).text
    )
    names = [name for name, _ in events]
    sid = events[0][1]["session_id"]

    assert "tool" not in names
    assert "".join(d["text"] for name, d in events if name == "delta") == COMPLAINT_TEXT
    assert events[-1][1]["actions"] == [
        {"type": "handoff", "label": "转人工"},
        {
            "type": "ticket",
            "label": "建工单",
            "ticket_type": "投诉",
            "description": user_input,
        },
    ]
    assert model.agent_calls == [], "投诉出口不调模型"

    stored = client.store._data[sid]
    assert [type(m).__name__ for m in stored] == ["HumanMessage", "AIMessage"]
    assert stored[1].content == COMPLAINT_TEXT
    assert client.store.metadata[sid]["actions"] == events[-1][1]["actions"]


# ------------------------------------------------------------------ 4. 闲聊


def test_chitchat_deltas_join_to_fixed_text(client):
    model = GraphModel("闲聊", [])
    use_model(model)

    events = parse_sse(client.post("/api/chat", json={"message": "你好呀"}).text)
    deltas = [data["text"] for name, data in events if name == "delta"]

    assert "".join(deltas) == CHITCHAT_TEXT
    assert events[-1][1]["actions"] == []
    assert model.agent_calls == []
    sid = events[0][1]["session_id"]
    assert client.store._data[sid][-1].content == CHITCHAT_TEXT


# ------------------------------------------------------------------ 5. 断流


def test_client_disconnect_writes_no_messages(client):
    """客户端在 delta 之后断开：半截一轮不写审计。"""
    from app.api import chat
    from app.schemas import ChatRequest

    sid = str(uuid.uuid4())
    model = GraphModel(
        "退货退款",
        [[AIMessageChunk(content="第一段"), AIMessageChunk(content="第二段")]],
    )
    req = ChatRequest(session_id=sid, message="退货政策是什么")

    async def _drive():
        engine = create_async_engine(get_settings().db_url)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with maker() as db:
                store = FakeStore()
                response = await chat(
                    req,
                    request=SimpleNamespace(app=app),
                    store=store,
                    model=model,
                    db=db,
                )
                iterator = response.body_iterator
                seen_delta = False
                while not seen_delta:
                    frame = await iterator.__anext__()
                    seen_delta = "event: delta" in frame
                await iterator.aclose()
                return store._data
        finally:
            await engine.dispose()

    data = client.portal.call(_drive)

    assert data == {}, "断流时一句都不许落库"


# ------------------------------------------------------------------ 6. 恢复路径（审计 id 差集）


def test_audit_after_sanitize_keeps_only_the_new_turn(client):
    """上一轮悬空 tool_calls 被 sanitize 删掉后，审计仍是本轮完整消息。

    这是 ruling 4 修正的钉子：用 `prior_len` 下标切片会因 sanitize 缩短列表
    而错位（丢掉本轮 Human 或多算旧消息），必须用 id 差集。
    """
    sid = str(uuid.uuid4())
    model = GraphModel("闲聊", [])
    use_model(model)

    async def _seed_dangling():
        config = {
            "configurable": {
                "thread_id": sid,
                **workflow_deps(None, None, None, 0),
            }
        }
        await app.state.workflow.aupdate_state(
            config,
            {
                "messages": [
                    AIMessage(
                        content="",
                        id="ai-dangling",
                        tool_calls=[
                            {
                                "name": "query_order",
                                "args": {"order_id": "1001"},
                                "id": "call-dangling",
                            }
                        ],
                    )
                ]
            },
            as_node="agent",
        )

    client.portal.call(_seed_dangling)

    r = client.post("/api/chat", json={"session_id": sid, "message": "你好"})
    assert [name for name, _ in parse_sse(r.text)][-1] == "done"

    stored = client.store._data[sid]
    assert [type(m).__name__ for m in stored] == ["HumanMessage", "AIMessage"]
    assert stored[0].content == "你好"
    assert stored[1].content == CHITCHAT_TEXT
    assert all(getattr(m, "id", None) != "ai-dangling" for m in stored)


# ------------------------------------------------------------------ 7. 错误分类


def test_upstream_error_becomes_error_frame_and_no_audit(client):
    import httpx
    import openai

    err = openai.RateLimitError(
        "slow down",
        response=httpx.Response(429, request=httpx.Request("POST", "https://x/")),
        body=None,
    )
    model = GraphModel("物流", [err])
    use_model(model)
    sid = str(uuid.uuid4())

    r = client.post(
        "/api/chat", json={"session_id": sid, "message": "订单 1001 的物流到哪了"}
    )
    events = parse_sse(r.text)

    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "upstream_rate_limit"
    assert "done" not in [name for name, _ in events]
    assert client.store._data == {}, "失败的一轮不写审计"


# ------------------------------------------------------------------ 8. 闸门拒答


def test_gate_refusal_streams_once_and_does_not_call_agent(client, monkeypatch):
    from app.knowledge import retrieval

    async def empty_search(session, query, top_k=None, **kwargs):
        return [], {"strategy": "dense", "scores": None, "refusal_reason": "no_results"}

    monkeypatch.setattr(retrieval, "search_with_metadata", empty_search)
    model = GraphModel("退货退款", [])
    use_model(model)

    events = parse_sse(
        client.post("/api/chat", json={"message": "火星移民怎么退"}).text
    )
    deltas = [data["text"] for name, data in events if name == "delta"]

    assert "".join(deltas) == GATE_REFUSAL
    assert events[-1][1]["actions"] == [{"type": "handoff", "label": "转人工"}]
    assert model.agent_calls == [], "闸门失败不进 Agent"


# ------------------------------------------------- 9. 触顶收敛话术必须到达浏览器


def test_step_limit_text_after_preamble_is_streamed_and_persisted(
    client, monkeypatch
):
    """Important 1：max_steps=1 时前言已流出，最终答案是 STEP_LIMIT_TEXT。

    适配层不能因为“本段有过 delta”就认为最终回答已经发完；必须对比原始
    流式文本与最终 State，把触顶话术补发出去，且与落库正文一致。
    """
    monkeypatch.setattr(get_settings(), "agent_max_steps", 1)
    model = GraphModel(
        "物流",
        [
            [
                AIMessageChunk(content="我先帮您查一下"),
                AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "name": "query_order",
                            "args": {"order_id": "1001"},
                            "id": "call_step",
                        }
                    ],
                ),
            ]
        ],
    )
    use_model(model)

    events = parse_sse(
        client.post(
            "/api/chat", json={"message": "订单 1001 的物流到哪了"}
        ).text
    )
    names = [name for name, _ in events]
    streamed = "".join(
        data["text"] for name, data in events if name == "delta"
    )
    sid = events[0][1]["session_id"]

    assert "tool" not in names, "max_steps=1 时工具还没执行就收敛了"
    assert streamed == "我先帮您查一下" + STEP_LIMIT_TEXT

    # 前言属于带 tool_calls 的 assistant，触顶话术是最后一条 assistant；
    # 浏览器看到的拼接正文必须等于这一轮所有 assistant 正文的拼接。
    persisted = "".join(
        message.content
        for message in client.store._data[sid]
        if isinstance(message, AIMessage)
    )
    assert persisted == streamed, "浏览器文本与落库 assistant 序列必须一致"
    assert events[-1][1]["finish_reason"] == "stop"


# ------------------------------------------------- 10. Agent 内拒答的重复入池防护


def test_agent_faq_refusal_pools_once_and_not_model_insufficient(
    client, monkeypatch
):
    """Important 2：业务类进入 Agent，query_faq 已 no_results 入池，模型再吐
    自评标记时只能保留一条 no_results，且浏览器只看到一次 REFUSAL_TEXT。"""
    from app.knowledge import retrieval

    question = "火星天气怎么样（agent 内拒答）"

    async def refusing_search(session, query, top_k=None, **kwargs):
        # 生产 search_with_metadata 在空结果时会入池，这里如实模拟。
        await lcq_repo.create(
            session, query, kwargs.get("session_id", ""), "no_results"
        )
        return [], {
            "strategy": "dense",
            "scores": None,
            "refusal_reason": "no_results",
        }

    monkeypatch.setattr(retrieval, "search_with_metadata", refusing_search)
    model = GraphModel(
        "物流",
        [
            tool_call("query_faq", {"keyword": question}, "call_refuse"),
            [AIMessageChunk(content="<INSUFFICIENT_KNOWLEDGE>")],
        ],
    )
    use_model(model)
    sid = str(uuid.uuid4())

    events = parse_sse(
        client.post(
            "/api/chat",
            json={"session_id": sid, "message": f"订单 1001 {question}"},
        ).text
    )
    deltas = [data["text"] for name, data in events if name == "delta"]
    streamed = "".join(deltas)

    assert streamed == REFUSAL_TEXT
    assert streamed.count(REFUSAL_TEXT) == 1, "拒答话术只能出现一次"
    assert "<INSUFFICIENT_KNOWLEDGE>" not in streamed
    assert client.store._data[sid][-1].content == REFUSAL_TEXT, (
        "落库正文必须与浏览器看到的文本一致"
    )

    async def _fetch_pool():
        engine = create_async_engine(get_settings().db_url)
        try:
            async with async_sessionmaker(engine)() as other:
                conv = (
                    await other.execute(
                        select(Conversation).where(Conversation.session_uuid == sid)
                    )
                ).scalar_one()
                # ★ 必须按会话主键过滤，不能按工具关键词过滤：model_insufficient
                # 行的 question 是 req.message（带“订单 1001”前缀），用关键词
                # 查会把这条多余的行藏起来，令“去重守卫”测试永远通过。
                rows = list(
                    (
                        await other.execute(
                            select(LowConfidenceQuestion).where(
                                LowConfidenceQuestion.session_id == str(conv.id)
                            )
                        )
                    ).scalars()
                )
                return conv.id, [(row.question, row.reason) for row in rows]
        finally:
            await engine.dispose()

    conv_id, pooled = client.portal.call(_fetch_pool)
    assert pooled == [(question, "no_results")], (
        f"同一会话只允许一条 no_results；实际：{pooled}"
    )

    # Minor 3：标记不能留在 checkpoint 里（否则下一轮会被自己的拒答标记污染）。
    async def _state():
        config = {
            "configurable": {
                "thread_id": sid,
                **workflow_deps(None, None, None, 0),
            }
        }
        snapshot = await app.state.workflow.aget_state(config)
        return snapshot.values

    values = client.portal.call(_state)
    assert values["insufficient"] is True
    assert values["messages"][-1].content == GATE_REFUSAL
    assert all(
        "<INSUFFICIENT_KNOWLEDGE>" not in str(message.content)
        for message in values["messages"]
    )


# ------------------------------------------------- 11. 工具失败与错误边界


def test_business_tool_failure_is_fed_back_not_raised(client):
    """非 query_faq 工具 ok=False：ToolMessage 回灌，模型继续作答，无 error 帧。"""
    model = GraphModel(
        "物流",
        [
            tool_call("query_order", {}, "call_bad"),
            [AIMessageChunk(content="我暂时查不到这个订单。")],
        ],
    )
    use_model(model)

    events = parse_sse(
        client.post(
            "/api/chat", json={"message": "订单 1001 的物流到哪了"}
        ).text
    )
    names = [name for name, _ in events]
    tools = [data for name, data in events if name == "tool"]
    sid = events[0][1]["session_id"]

    assert "error" not in names
    assert names[-1] == "done"
    assert tools[0]["phase"] == "start"
    assert tools[1]["phase"] == "end"
    assert tools[1]["ok"] is False

    stored = client.store._data[sid]
    assert [type(m).__name__ for m in stored] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]
    assert "参数不合法" in stored[2].content
    assert stored[-1].content == "我暂时查不到这个订单。"


class BoomRegistry:
    """registry.execute 意外抛异常的桩。"""

    async def as_langchain_tools(self):
        return []

    async def execute(self, name, args, call_id, *, conversation_id=None):
        raise RuntimeError("registry 意外炸了")


def test_registry_crash_ends_with_single_error_and_no_audit(
    client, monkeypatch
):
    """工具执行里的意外异常：恰好一个 error 帧，无 done，不写审计。"""
    monkeypatch.setattr(
        "app.api.build_registry", lambda session=None: BoomRegistry()
    )
    model = GraphModel(
        "物流",
        [
            tool_call("query_order", {"order_id": "1001"}, "call_boom"),
            [AIMessageChunk(content="不应到达")],
        ],
    )
    use_model(model)

    events = parse_sse(
        client.post(
            "/api/chat", json={"message": "订单 1001 的物流到哪了"}
        ).text
    )
    names = [name for name, _ in events]

    assert names.count("error") == 1
    assert names[-1] == "error"
    assert "done" not in names
    assert events[-1][1]["code"] == "internal_error"
    assert client.store._data == {}


def test_error_after_partial_deltas_is_terminal(client):
    """已经流出一段正文后再失败：delta 之后仍必须以 error 收尾，不写审计。"""
    model = GraphModel(
        "物流",
        [[AIMessageChunk(content="部分回答"), RuntimeError("上游炸了")]],
    )
    use_model(model)
    sid = str(uuid.uuid4())

    events = parse_sse(
        client.post(
            "/api/chat", json={"session_id": sid, "message": "订单 1001 的物流到哪了"}
        ).text
    )
    names = [name for name, _ in events]

    assert names[0] == "delta"
    assert names[-1] == "error"
    assert "done" not in names
    # 必须是上游的 RuntimeError，而不是把异常对象当消息解引用得到的
    # AttributeError（否则这条用例测的就不是预期路径）。
    assert events[-1][1]["message"] == "上游炸了"
    assert "".join(
        data["text"] for name, data in events if name == "delta"
    ) == "部分回答"
    assert client.store._data == {}
