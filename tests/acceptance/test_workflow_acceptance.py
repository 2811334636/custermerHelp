"""ch05 五条验收标准的端到端测试（真实工作流 + 真实 MySQL/Milvus）。

覆盖 spec §12 的五条验收：

1. 知识类问题真的经过 ``retrieve`` 节点（caplog 的 ``node=retrieve``）；
2. 物流类问题由 Agent 调用 ``query_logistics`` 工具后作答；
3. 投诉出口给出「转人工 / 建工单」两个**互相独立**的动作——聊天本身不建单，
   只有点击建工单（``POST /api/tickets``）才落一行 tickets；
4. 闲聊返回固定话术；
5. 复杂问题走 ReAct 多步（两次工具调用，``workflow_turn`` 日志 ``steps >= 2``）。

外部 AI 依赖（embedding / query 改写 / reranker）在本文件里用确定性替身：
它们都是可降级的可选增强，且在线 API 在本机偶发挂起（Task 1 报告）。
检索本身仍是真实的：真实 Milvus hybrid（dense + BM25）→ 真实 MySQL 回表。
出题模型是脚本化的 ``GraphModel``，不会请求真实 LLM。
"""

from __future__ import annotations

import json
import logging
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import get_store
from app.config import get_settings
from app.db.base import get_db
from app.db.models import Conversation, Ticket
from app.knowledge.errors import RerankerUnavailable
from app.main import app
from app.providers import get_default_chat_model
from app.workflow.texts import CHITCHAT_TEXT

POLICY_QUESTION = "退货政策是什么"
LOGISTICS_QUESTION = "订单 1001 的物流到哪了"
COMPLAINT_MESSAGE = "我要投诉"
CHITCHAT_MESSAGE = "今天天气怎么样"
COMPLEX_QUESTION = "帮我看下订单 1001 什么时候发货"


# ------------------------------------------------------------------ 测试替身


class FakeStore:
    """审计写入的内存桩；模型上下文归 checkpointer，这里只关心流与落库。"""

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
    """按调用次序切换行为的脚本化模型。

    意图分类走 ``ainvoke``（固定返回 ``intent``）；Agent 子图走
    ``bind_tools().astream``，每个 reply 可以是一条消息、一串 chunk，
    或一个要抛出的异常。不请求真实 LLM。
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
        steps = reply if isinstance(reply, list) else [reply]
        for step in steps:
            if isinstance(step, BaseException):
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


# --------------------------------------------------------------- 外部 AI 替身


@pytest.fixture(autouse=True)
def _deterministic_external_ai(monkeypatch):
    """把三个可降级的外部 AI 调用换成确定性替身，检索链路保持真实。

    - ``embed_texts``：固定 1024 维向量。在线 embedding 本机偶发挂起
      （Task 1 报告实测 >10min）；dense 只是 hybrid 的一路，BM25 仍按真实
      查询文本召回，最终仍由真实 Milvus + MySQL 回表。
    - ``rewrite_query``：恒等改写，避免为了归一化再打一次在线 LLM。
    - ``rerank``：抛 ``RerankerUnavailable``，走产品既有的降级路径；
      置信度闸随后用真实的词面覆盖率判定。
    """
    from app.knowledge import query_rewrite, reranker, retrieval

    dim = get_settings().embedding_dim

    async def _unit_embeddings(texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * (dim - 1) for _ in texts]

    async def _identity_rewrite(query: str) -> str:
        return query

    async def _rerank_unavailable(*args, **kwargs):
        raise RerankerUnavailable("acceptance: reranker stubbed")

    monkeypatch.setattr(retrieval, "embed_texts", _unit_embeddings)
    monkeypatch.setattr(query_rewrite, "rewrite_query", _identity_rewrite)
    monkeypatch.setattr(reranker, "rerank", _rerank_unavailable)


@pytest.fixture
def client(db_session):
    store = FakeStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_db] = lambda: db_session
    # 每个用例会通过 use_model() 换成自己的脚本模型；这里只保证依赖已注入。
    app.dependency_overrides[get_default_chat_model] = lambda: GraphModel("闲聊", [])
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


def _new_session_id() -> str:
    return str(uuid.uuid4())


async def _fetch_ticket_state(session_id: str):
    """另起引擎读已提交的工单行与会话状态（避免跨事件循环复用 session）。"""
    engine = create_async_engine(get_settings().db_url)
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as db:
            conv = (
                await db.execute(
                    select(Conversation).where(
                        Conversation.session_uuid == session_id
                    )
                )
            ).scalar_one_or_none()
            if conv is None:
                return [], None
            tickets = list(
                (
                    await db.scalars(
                        select(Ticket).where(Ticket.conversation_id == conv.id)
                    )
                ).all()
            )
            return tickets, conv.status
    finally:
        await engine.dispose()


def _tool_frames(events):
    return [data for name, data in events if name == "tool"]


def _deltas(events) -> str:
    return "".join(data["text"] for name, data in events if name == "delta")


# ------------------------------------------------------- 验收 1：知识类走检索


def test_policy_question_hits_retrieve_node(client, caplog):
    """验收 1：政策问题先经过真实的 retrieve 节点，再由 Agent 作答。"""
    sid = _new_session_id()
    answer = "签收后七天内可无理由退货，商品需保持吊牌完整。"
    model = GraphModel("退货退款", [[AIMessageChunk(content=answer)]])
    use_model(model)

    with caplog.at_level(
        logging.INFO, logger="app.workflow.retrieval_nodes"
    ):
        events = parse_sse(
            client.post(
                "/api/chat",
                json={"session_id": sid, "message": POLICY_QUESTION},
            ).text
        )

    retrieve_logs = [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.workflow.retrieval_nodes"
        and "node=retrieve" in record.getMessage()
    ]
    assert retrieve_logs, "知识类问题必须经过 retrieve 节点"
    # 真实 hybrid 检索（dense + BM25 + 真实回表）应至少召回一条；
    # chunks=0 说明基础设施没起来或检索链路被绕开。
    assert "chunks=0" not in retrieve_logs[0], retrieve_logs[0]

    # 闸门通过 → Agent 被调用一次 → 脚本答案原样流给浏览器。
    assert len(model.agent_calls) == 1
    assert _deltas(events) == answer
    done = events[-1][1]
    assert done["actions"] == []
    assert done["citations"], "真实检索结果必须体现在引用里"


# --------------------------------------------- 验收 2：物流类由 Agent 调工具


def test_logistics_question_uses_agent_tool(client):
    """验收 2：物流问题由 Agent 调用 query_logistics 后作答。"""
    sid = _new_session_id()
    model = GraphModel(
        "物流",
        [
            tool_call("query_logistics", {"order_id": "1001"}, "call_log"),
            [AIMessageChunk(content="您的包裹已发出，正在运输途中。")],
        ],
    )
    use_model(model)

    events = parse_sse(
        client.post(
            "/api/chat",
            json={"session_id": sid, "message": LOGISTICS_QUESTION},
        ).text
    )
    tools = _tool_frames(events)
    starts = [t for t in tools if t["phase"] == "start"]
    ends = [t for t in tools if t["phase"] == "end"]

    assert [t["name"] for t in starts] == ["query_logistics"]
    assert starts[0]["arguments"] == {"order_id": "1001"}
    assert len(ends) == 1
    assert ends[0]["name"] == "query_logistics"
    assert ends[0]["ok"] is True
    assert _deltas(events) == "您的包裹已发出，正在运输途中。"
    assert events[-1][0] == "done"


# ------------------------------------------- 验收 3：投诉动作互相独立


def test_complaint_actions_are_independent(client):
    """验收 3：聊天只给两个动作；建单是用户点击后的独立确定性写入。"""
    sid = _new_session_id()
    model = GraphModel("投诉", [])
    use_model(model)

    events = parse_sse(
        client.post(
            "/api/chat", json={"session_id": sid, "message": COMPLAINT_MESSAGE}
        ).text
    )
    done = events[-1][1]
    actions = done["actions"]
    assert [action["type"] for action in actions] == ["handoff", "ticket"]
    assert model.agent_calls == [], "投诉出口不调 Agent"

    # 「转人工」不写后端：聊天结束到点按钮之前，tickets 必须仍为空。
    tickets, status = client.portal.call(_fetch_ticket_state, sid)
    assert tickets == [], "投诉聊天本身不建工单"
    assert status == "进行中"

    # 模拟点击「建工单」：确定性端点，不经过模型或工作流。
    r = client.post(
        "/api/tickets",
        json={
            "session_id": sid,
            "description": COMPLAINT_MESSAGE,
            "ticket_type": "投诉",
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    tickets, status = client.portal.call(_fetch_ticket_state, sid)
    assert len(tickets) == 1, "点击一次建工单只落一行"
    assert tickets[0].ticket_type == "投诉"
    assert tickets[0].description == COMPLAINT_MESSAGE
    assert status == "已转人工"


# ------------------------------------------------------ 验收 4：闲聊固定话术


def test_chitchat_fixed_text(client):
    """验收 4：闲聊返回固定话术，不进 Agent。"""
    sid = _new_session_id()
    model = GraphModel("闲聊", [])
    use_model(model)

    events = parse_sse(
        client.post(
            "/api/chat", json={"session_id": sid, "message": CHITCHAT_MESSAGE}
        ).text
    )

    assert _deltas(events) == CHITCHAT_TEXT
    assert model.agent_calls == []
    assert events[-1][1]["actions"] == []


# ------------------------------------------------- 验收 5：ReAct 多步工具调用


def test_react_takes_multiple_steps(client, caplog):
    """验收 5：复杂问题连续调用 query_order → query_logistics → 作答。"""
    sid = _new_session_id()
    model = GraphModel(
        "订单",
        [
            tool_call("query_order", {"order_id": "1001"}, "call_order"),
            tool_call("query_logistics", {"order_id": "1001"}, "call_log"),
            [AIMessageChunk(content="订单已发货，物流正在运输途中。")],
        ],
    )
    use_model(model)

    with caplog.at_level(logging.INFO, logger="app.workflow.graph"):
        events = parse_sse(
            client.post(
                "/api/chat",
                json={"session_id": sid, "message": COMPLEX_QUESTION},
            ).text
        )

    ends = [t for t in _tool_frames(events) if t["phase"] == "end"]
    assert [t["name"] for t in ends] == ["query_order", "query_logistics"]
    assert all(t["ok"] is True for t in ends)
    assert len(model.agent_calls) == 3, "两次工具轮 + 一次最终作答"

    workflow_logs = [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.workflow.graph"
        and record.getMessage().startswith("workflow_turn ")
    ]
    assert workflow_logs, "回合收敛必须留下 workflow_turn 日志"
    payload = json.loads(workflow_logs[-1].split(" ", 1)[1])
    assert payload["intent"] == "订单"
    assert payload["steps"] >= 2, f"ReAct 至少两步，实际：{payload['steps']}"
    assert "slot_gate" in payload["node_trace"]
    assert payload["node_trace"][-1] == "log"
