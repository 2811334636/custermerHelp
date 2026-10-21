"""主图装配、checkpointer 生命周期与跨轮恢复（Task 8）。

每个用例用独立的临时 sqlite checkpoint 文件；模型/registry/session 都是
替身，不碰 MySQL 或真实模型。图级测试验证：路由、跨轮插槽、悬空
tool_calls 清理、固定话术分支、拒答唯一性与结构化日志。
"""

import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import Settings
from app.knowledge import retrieval as retrieval_module
from app.workflow import graph as graph_module
from app.workflow.graph import build_workflow, workflow_deps
from app.workflow.texts import (
    CHITCHAT_TEXT,
    COMPLAINT_TEXT,
    GATE_REFUSAL,
    SLOT_QUESTION,
)


class ScriptedModel:
    """一个替身同时服务两处模型调用：意图分类（ainvoke）与 Agent（astream）。

    意图分类节点用 ``ainvoke`` 拿 JSON；ReAct 子图用 ``bind_tools().astream``。
    ``replies`` 少于两次调用时重复最后一条，便于跨轮用例。
    """

    def __init__(self, intent: str, *, confidence: float = 0.9, replies=None):
        self.intent = intent
        self.confidence = confidence
        self.replies = list(replies or [AIMessage(content="默认回答")])
        self.intent_calls = 0
        self.agent_calls = 0

    async def ainvoke(self, messages):
        self.intent_calls += 1
        return AIMessage(
            content=json.dumps(
                {"intent": self.intent, "confidence": self.confidence},
                ensure_ascii=False,
            )
        )

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.agent_calls += 1
        reply = (
            self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        )
        chunks = reply if isinstance(reply, list) else [reply]
        for chunk in chunks:
            if isinstance(chunk, AIMessage):
                yield AIMessageChunk(
                    content=chunk.content,
                    tool_calls=chunk.tool_calls,
                    usage_metadata=chunk.usage_metadata,
                    id=chunk.id,
                )
            else:
                yield chunk


class FakeRegistry:
    """主图用例不触发业务工具；实现 execute 只为满足子图依赖形状。"""

    def __init__(self):
        self.calls = []

    async def execute(self, name, args, call_id, *, conversation_id=None):
        self.calls.append((name, args, call_id, conversation_id))
        return SimpleNamespace(
            ok=True,
            content="工具结果",
            summary="工具结果",
            elapsed_ms=1,
            metadata=None,
        )


class FakeSession:
    """只记录 add/flush（低置信度入池用），不碰真实数据库。"""

    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        pass


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    questions: str
    answer: str


@asynccontextmanager
async def build_test_graph(tmp_path, model):
    """每个用例一个临时 sqlite 文件；退出时关闭 saver 连接。"""
    path = tmp_path / "checkpoints.sqlite"
    async with AsyncSqliteSaver.from_conn_string(str(path)) as checkpointer:
        yield build_workflow(checkpointer)


def _config(
    thread_id: str,
    model,
    *,
    registry=None,
    session=None,
    conversation_id: int = 1,
) -> dict:
    return {
        "configurable": {
            "thread_id": thread_id,
            **workflow_deps(
                model,
                registry if registry is not None else FakeRegistry(),
                session if session is not None else FakeSession(),
                conversation_id,
            ),
        }
    }


def _patch_retrieval(monkeypatch, hits, *, meta=None):
    async def fake_search(session, query, top_k=None, **kwargs):
        return list(hits), {"strategy": "dense", "scores": None, **(meta or {})}

    monkeypatch.setattr(
        retrieval_module, "search_with_metadata", fake_search
    )


async def test_graph_routes_knowledge_and_runs_retrieve(tmp_path, monkeypatch):
    """知识类先预检索；node_trace 必须能证明 retrieve 真的跑过。"""
    _patch_retrieval(
        monkeypatch,
        [FakeChunk("faq:1", "退货政策是什么", "七天无理由退货，运费由平台承担")],
    )
    model = ScriptedModel(
        "退货退款", replies=[AIMessage(content="按政策可以七天无理由退货。")]
    )
    async with build_test_graph(tmp_path, model) as graph:
        out = await graph.ainvoke(
            {
                "messages": [HumanMessage("退货政策是什么")],
                "user_input": "退货政策是什么",
            },
            _config("t-knowledge", model),
        )

    assert out["route"] == "knowledge"
    assert "retrieve" in out["node_trace"]
    assert "confidence_gate" in out["node_trace"]
    assert out["messages"][-1].content == "按政策可以七天无理由退货。"


async def test_graph_waits_for_order_id_across_turns(tmp_path):
    """业务类缺订单号先追问；下一轮补号沿用 pending_intent 进入 Agent。"""
    model = ScriptedModel(
        "物流", replies=[AIMessage(content="订单 1001 正在运输中。")]
    )
    async with build_test_graph(tmp_path, model) as graph:
        config = _config("t-slot", model)
        first = await graph.ainvoke(
            {
                "messages": [HumanMessage("我的物流到哪了")],
                "user_input": "我的物流到哪了",
            },
            config,
        )
        assert first["awaiting_slot"] == "order_id"
        assert first["messages"][-1].content == SLOT_QUESTION
        assert model.agent_calls == 0  # 缺槽不进 Agent

        second = await graph.ainvoke(
            {"messages": [HumanMessage("1001")], "user_input": "1001"},
            config,
        )

    assert second["order_id"] == "1001"
    assert second["awaiting_slot"] is None
    assert second["messages"][-1].content == "订单 1001 正在运输中。"


async def test_graph_recovers_from_dangling_tool_call(tmp_path):
    """上一轮中断留下的 assistant(tool_calls) 会被 sanitize 后再正常回答。"""
    model = ScriptedModel("闲聊")
    async with build_test_graph(tmp_path, model) as graph:
        config = _config("t-dangling", model)
        await graph.aupdate_state(
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

        out = await graph.ainvoke(
            {"messages": [HumanMessage("你好")], "user_input": "你好"},
            config,
        )

    assert out["messages"][-1].content == CHITCHAT_TEXT
    assert all(message.id != "ai-dangling" for message in out["messages"])
    # 只剩本轮的 Human + 固定话术 AI：悬空 assistant 及其 tool_calls 被清掉。
    assert [message.type for message in out["messages"]] == ["human", "ai"]


async def test_graph_complaint_returns_handoff_and_ticket(tmp_path):
    """投诉出口给固定安抚话术 + handoff/ticket 两个动作，且不进 Agent。"""
    user_input = "我要投诉你们客服"
    model = ScriptedModel("投诉")
    async with build_test_graph(tmp_path, model) as graph:
        out = await graph.ainvoke(
            {"messages": [HumanMessage(user_input)], "user_input": user_input},
            _config("t-complaint", model),
        )

    assert out["messages"][-1].content == COMPLAINT_TEXT
    assert out["actions"] == [
        {"type": "handoff", "label": "转人工"},
        {
            "type": "ticket",
            "label": "建工单",
            "ticket_type": "投诉",
            "description": user_input,
        },
    ]
    assert model.agent_calls == 0


async def test_graph_chitchat_returns_fixed_text(tmp_path):
    """闲聊出口只回固定话术，无动作。"""
    model = ScriptedModel("闲聊")
    async with build_test_graph(tmp_path, model) as graph:
        out = await graph.ainvoke(
            {"messages": [HumanMessage("你好呀")], "user_input": "你好呀"},
            _config("t-chitchat", model),
        )

    assert out["messages"][-1].content == CHITCHAT_TEXT
    assert out["actions"] == []


async def test_graph_gate_failure_emits_exactly_one_refusal(tmp_path, monkeypatch):
    """闸门是唯一拒答所有者：失败边直接去 log，恰好一条拒答。"""
    _patch_retrieval(
        monkeypatch, [], meta={"refusal_reason": "no_results"}
    )
    model = ScriptedModel("退货退款")
    async with build_test_graph(tmp_path, model) as graph:
        out = await graph.ainvoke(
            {
                "messages": [HumanMessage("火星移民怎么退")],
                "user_input": "火星移民怎么退",
            },
            _config("t-gate-fail", model),
        )

    assert out["gate_passed"] is False
    assert out["refusal_reason"] == "no_results"
    refusals = [
        message
        for message in out["messages"]
        if isinstance(message, AIMessage) and message.content == GATE_REFUSAL
    ]
    assert len(refusals) == 1
    assert out["messages"][-1].content == GATE_REFUSAL
    assert model.agent_calls == 0  # 失败不进 Agent


async def test_graph_astream_subgraphs_yields_namespace_payload_tuples(
    tmp_path, monkeypatch
):
    """Task 9 的消费方式：subgraphs=True 时每项都是 (namespace, payload) 元组。"""
    _patch_retrieval(
        monkeypatch,
        [FakeChunk("faq:1", "退货政策是什么", "七天无理由退货")],
    )
    model = ScriptedModel(
        "退货退款", replies=[AIMessage(content="按政策可以退货。")]
    )
    async with build_test_graph(tmp_path, model) as graph:
        config = _config("t-stream", model)
        deltas: list[str] = []
        delta_namespaces: list[tuple] = []
        async for namespace, event in graph.astream(
            {
                "messages": [HumanMessage("退货政策是什么")],
                "user_input": "退货政策是什么",
            },
            config,
            stream_mode="custom",
            subgraphs=True,
        ):
            assert isinstance(namespace, tuple)
            if isinstance(event, dict) and event.get("type") == "delta":
                deltas.append(event["text"])
                delta_namespaces.append(namespace)
        state = await graph.aget_state(config)

    assert "".join(deltas) == "按政策可以退货。"
    # delta 来自 Agent 子图：namespace 非空（形如 ('agent:<uuid>',)）。
    assert delta_namespaces and all(ns for ns in delta_namespaces)
    assert "retrieve" in state.values["node_trace"]


async def test_log_node_emits_structured_turn_record(tmp_path, caplog, monkeypatch):
    """workflow_log_enabled=True 时恰好一行 workflow_turn JSON，字段齐全。"""
    _patch_retrieval(
        monkeypatch,
        [FakeChunk("faq:1", "退货政策是什么", "七天无理由退货")],
    )
    model = ScriptedModel("退货退款", replies=[AIMessage(content="可以退货。")])
    with caplog.at_level(logging.INFO, logger="app.workflow.graph"):
        async with build_test_graph(tmp_path, model) as graph:
            await graph.ainvoke(
                {
                    "messages": [HumanMessage("退货政策是什么")],
                    "user_input": "退货政策是什么",
                },
                _config("t-log", model),
            )

    records = [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.workflow.graph"
        and record.getMessage().startswith("workflow_turn ")
    ]
    assert len(records) == 1
    payload = json.loads(records[0].removeprefix("workflow_turn "))
    assert payload["session_id"] == "t-log"
    assert payload["intent"] == "退货退款"
    assert payload["route"] == "knowledge"
    assert payload["node_trace"][-1] == "log"
    assert "retrieve" in payload["node_trace"]
    assert payload["steps"] == 1
    assert payload["tokens_used"] == 0
    assert payload["actions"] == []
    assert payload["retrieval_chunks"] == 1


async def test_log_node_is_silent_when_disabled(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(
        graph_module,
        "get_settings",
        lambda: Settings(_env_file=None, workflow_log_enabled=False),
    )
    model = ScriptedModel("闲聊")
    with caplog.at_level(logging.INFO, logger="app.workflow.graph"):
        async with build_test_graph(tmp_path, model) as graph:
            await graph.ainvoke(
                {"messages": [HumanMessage("你好")], "user_input": "你好"},
                _config("t-log-off", model),
            )

    assert not any(
        record.getMessage().startswith("workflow_turn ")
        for record in caplog.records
    )
