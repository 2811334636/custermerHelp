"""主图装配、运行时依赖与回合日志（spec §3/§5.11）。

图拓扑（controller ruling：置信度闸是唯一拒答所有者，因此没有独立 refuse 节点）::

    START → sanitize_messages → begin_turn → resolve_reference → classify_intent
      knowledge → retrieve → confidence_gate ─(pass)→ agent
                                             └(fail)→ log
      business  → slot_gate ─(有订单号)→ agent
                           └(无订单号)→ log
      complaint → complaint → log
      chitchat  → chitchat  → log
    agent → log → END

运行时依赖（model/registry/session/conversation_id）只放
``config["configurable"]``，绝不写进 State——它们不可序列化，也不应被
checkpointer 落盘。``build_workflow`` 每个进程只调用一次（FastAPI
lifespan 启动时），把编译好的图挂在 ``app.state.workflow``。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import get_settings
from app.workflow.agent import TICKET_DESCRIPTION_LIMIT, build_agent_subgraph
from app.workflow.intent import classify_intent_node
from app.workflow.nodes import begin_turn, resolve_reference, sanitize_messages
from app.workflow.retrieval_nodes import confidence_gate_node, retrieve_node
from app.workflow.slots import slot_gate_node
from app.workflow.state import WorkflowState
from app.workflow.texts import CHITCHAT_TEXT, COMPLAINT_TEXT

logger = logging.getLogger(__name__)


def workflow_deps(
    model, registry, session, conversation_id
) -> dict[str, Any]:
    """构造 ``config["configurable"]`` 里的请求级运行时依赖。

    这些对象（模型、工具注册表、数据库会话、会话 id）只在当前请求内有效，
    **不参与 checkpointer 序列化**，也不能写进 ``WorkflowState``；跨轮恢复
    时由调用方重新提供。返回值直接展开到 configurable::

        config = {"configurable": {"thread_id": sid, **workflow_deps(...)}}
    """
    return {
        "model": model,
        "registry": registry,
        "session": session,
        "conversation_id": conversation_id,
    }


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    return (config or {}).get("configurable", {}) or {}


async def complaint_node(state: WorkflowState) -> dict[str, Any]:
    """投诉出口：固定安抚话术 + 两个互不绑定的建议动作（不调模型、不写库）。"""
    description = (state.get("user_input") or "")[:TICKET_DESCRIPTION_LIMIT]
    return {
        "messages": [AIMessage(content=COMPLAINT_TEXT)],
        "actions": [
            {"type": "handoff", "label": "转人工"},
            {
                "type": "ticket",
                "label": "建工单",
                "ticket_type": "投诉",
                "description": description,
            },
        ],
    }


async def chitchat_node(state: WorkflowState) -> dict[str, Any]:
    """闲聊出口：固定话术，无动作。"""
    return {"messages": [AIMessage(content=CHITCHAT_TEXT)]}


async def log_node(state: WorkflowState, config: RunnableConfig) -> dict[str, Any]:
    """回合收敛节点：只打一行结构化日志，不写数据库。"""
    configurable = _configurable(config)
    trace = list(state.get("node_trace") or []) + ["log"]
    if get_settings().workflow_log_enabled:
        citations = state.get("citations") or []
        # citations 覆盖预检索与 Agent 内 query_faq 的并集，按 chunk_id 去重；
        # 没有任何引用时退回 evidence 条数（防御性兜底）。
        retrieval_chunks = (
            len(
                {
                    citation["chunk_id"]
                    for citation in citations
                    if citation.get("chunk_id") is not None
                }
            )
            if citations
            else len(state.get("evidence") or [])
        )
        payload = {
            # thread_id 即 conversations.session_uuid（spec §8）；缺省时退回库内 id。
            "session_id": configurable.get(
                "thread_id", configurable.get("conversation_id")
            ),
            "intent": state.get("intent"),
            "route": state.get("route"),
            "node_trace": trace,
            "steps": state.get("steps") or 0,
            "tokens_used": state.get("tokens_used") or 0,
            "actions": state.get("actions") or [],
            "retrieval_chunks": retrieval_chunks,
        }
        logger.info("workflow_turn %s", json.dumps(payload, ensure_ascii=False))
    return {"node_trace": trace}


async def _sanitize_with_trace(state: WorkflowState) -> dict[str, Any]:
    """sanitize 是每轮第一个节点：从空 trace 开始，不带上一轮的节点。"""
    result = await sanitize_messages(state)
    return {**result, "node_trace": ["sanitize_messages"]}


async def _begin_turn_with_trace(state: WorkflowState) -> dict[str, Any]:
    """begin_turn 重置 turn 级字段（含 node_trace）；只保留 sanitize 这一跳。"""
    result = await begin_turn(state)
    return {
        **result,
        "node_trace": list(state.get("node_trace") or []) + ["begin_turn"],
    }


def _traced(name: str, node: Callable[..., Any]) -> Callable[..., Any]:
    """给只接收 state 的节点补一条 ``node_trace``（覆盖式列表）。

    Task 3-5 的骨架节点没有各自维护 trace；在图装配层统一补，避免为了观测
    去改已验证的节点实现。retrieve/confidence_gate 自己维护 trace，直接挂。
    """

    async def wrapped(state: WorkflowState) -> dict[str, Any]:
        result = await node(state)
        return {
            **result,
            "node_trace": list(state.get("node_trace") or []) + [name],
        }

    wrapped.__name__ = f"traced_{name}"
    return wrapped


def _traced_with_config(name: str, node: Callable[..., Any]) -> Callable[..., Any]:
    """同 ``_traced``，用于需要 RunnableConfig 的节点（如意图分类）。"""

    async def wrapped(
        state: WorkflowState, config: RunnableConfig
    ) -> dict[str, Any]:
        result = await node(state, config)
        return {
            **result,
            "node_trace": list(state.get("node_trace") or []) + [name],
        }

    wrapped.__name__ = f"traced_{name}"
    return wrapped


def _route_after_gate(state: WorkflowState) -> str:
    return "pass" if state.get("gate_passed") else "fail"


def _route_after_slot_gate(state: WorkflowState) -> str:
    return "agent" if state.get("order_id") else "log"


def build_workflow(
    checkpointer: BaseCheckpointSaver | None,
) -> CompiledStateGraph:
    """装配并编译主图；FastAPI lifespan 启动时调用一次。"""
    graph = StateGraph(WorkflowState)

    graph.add_node("sanitize_messages", _sanitize_with_trace)
    graph.add_node("begin_turn", _begin_turn_with_trace)
    graph.add_node(
        "resolve_reference", _traced("resolve_reference", resolve_reference)
    )
    graph.add_node(
        "classify_intent",
        _traced_with_config("classify_intent", classify_intent_node),
    )
    # retrieve / confidence_gate 自己往 node_trace 里追加名字，不能再包一层，
    # 否则同一个名字会出现两次。
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("confidence_gate", confidence_gate_node)
    graph.add_node("slot_gate", _traced("slot_gate", slot_gate_node))
    graph.add_node("complaint", _traced("complaint", complaint_node))
    graph.add_node("chitchat", _traced("chitchat", chitchat_node))
    graph.add_node("agent", build_agent_subgraph())
    graph.add_node("log", log_node)

    graph.add_edge(START, "sanitize_messages")
    graph.add_edge("sanitize_messages", "begin_turn")
    graph.add_edge("begin_turn", "resolve_reference")
    graph.add_edge("resolve_reference", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        lambda state: state["route"],
        {
            "knowledge": "retrieve",
            "business": "slot_gate",
            "complaint": "complaint",
            "chitchat": "chitchat",
        },
    )
    graph.add_edge("retrieve", "confidence_gate")
    graph.add_conditional_edges(
        "confidence_gate",
        _route_after_gate,
        {"pass": "agent", "fail": "log"},
    )
    graph.add_conditional_edges(
        "slot_gate",
        _route_after_slot_gate,
        {"agent": "agent", "log": "log"},
    )
    graph.add_edge("complaint", "log")
    graph.add_edge("chitchat", "log")
    graph.add_edge("agent", "log")
    graph.add_edge("log", END)

    return graph.compile(checkpointer=checkpointer)
