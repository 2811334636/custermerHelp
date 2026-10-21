"""工作流骨架节点：入口清理、每轮重置、指代消解与路由（spec §4/§5.1/§5.3）。"""

import logging

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from app.workflow.state import ROUTE_BY_INTENT, TURN_RESET_FIELDS, WorkflowState
from app.workflow.texts import EVIDENCE_PREFIX

logger = logging.getLogger(__name__)


def route_for(intent: str) -> str:
    """按硬编码分流表映射意图；未知意图落到 business（Agent 可兜住）。"""
    return ROUTE_BY_INTENT.get(intent, "business")


async def begin_turn(state: WorkflowState) -> dict:
    """清空 turn 级字段并把引用游标复位到 1；跨轮字段不动。"""
    reset = {
        "intent": "",
        "route": "",
        "evidence": [],
        "evidence_coverage": None,
        "gate_passed": None,
        "refusal_reason": None,
        "actions": [],
        "steps": 0,
        "tokens_used": 0,
        "finish_reason": None,
        "usage": None,
        "insufficient": False,
        "node_trace": [],
        "citations": [],
        "citation_index": {},
        "citation_cursor": 1,
    }
    # TURN_RESET_FIELDS 是上面的字段清单（不含单独复位的 citation_cursor）；
    # 用断言把两者钉在一起，避免以后加字段只改一处。
    assert set(TURN_RESET_FIELDS) | {"citation_cursor"} == set(reset)
    # 跨轮字段不改值；原样回传只是让节点输出里能读到它们（LangGraph 里是
    # 覆盖式空操作）。messages 绝不原样回传：add_messages 会追加造成重复。
    for field in ("awaiting_slot", "pending_intent", "order_id"):
        if field in state:
            reset[field] = state[field]
    # agent_model 每轮都会注入一条 evidence SystemMessage；不清理的话
    # checkpoint 会随轮数无界增长。add_messages 的删除语义只能走
    # RemoveMessage，这里只删上一轮注入的证据消息，对话历史原样保留。
    evidence_removals = [
        RemoveMessage(id=message.id)
        for message in state.get("messages", [])
        if isinstance(message, SystemMessage)
        and message.id is not None
        and str(message.content).startswith(EVIDENCE_PREFIX)
    ]
    if evidence_removals:
        reset["messages"] = evidence_removals
    return reset


async def resolve_reference(state: WorkflowState) -> dict:
    """本章原样透传，留出后续接指代消解的接口。"""
    resolved = state.get("user_input", "")
    logger.debug("node=resolve_reference input=%r", resolved)
    return {"resolved_input": resolved}


async def sanitize_messages(state: WorkflowState) -> dict:
    """删除上一轮中断留下的悬空 tool_calls 组。

    checkpointer 可能在上一个 assistant 产出 tool_calls 之后、ToolMessage
    落盘之前中断；这种消息序列会被模型上游拒绝。这里从后往前找到最后一条
    带 tool_calls 的 assistant，若其每个调用 id 都能在随后、下一条 Human
    之前的 ToolMessage 里找到，则视为完整回合；否则返回 RemoveMessage 列表，
    删掉该 assistant 及其后的孤立 ToolMessage（新注入的 Human 保留）。

    ``add_messages`` 是追加语义，直接返回裁剪后的消息列表只会重复消息，
    所以删除必须走 ``RemoveMessage``；没有可删的返回 ``{}``。
    """
    messages = state.get("messages", [])
    last_tool_call_index: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if isinstance(message, AIMessage) and message.tool_calls:
            last_tool_call_index = i
            break
    if last_tool_call_index is None:
        return {}

    assistant = messages[last_tool_call_index]
    expected_ids = {call["id"] for call in assistant.tool_calls}

    # 只看这条 assistant 之后、下一条 Human 之前；新 Human 属于本轮输入，
    # 不能因为上轮悬空而误删。
    tail_tools: list[ToolMessage] = []
    for message in messages[last_tool_call_index + 1 :]:
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, ToolMessage):
            tail_tools.append(message)

    answered_ids = {message.tool_call_id for message in tail_tools}
    if expected_ids <= answered_ids:
        return {}

    removals: list[RemoveMessage] = []
    if assistant.id is not None:
        removals.append(RemoveMessage(id=assistant.id))
    # 该 assistant 之后的 ToolMessage 都属于这组未闭合的调用（正常情况下
    # 早已在上面按完整回合返回），一并删掉，避免留下孤立 tool 消息。
    for message in tail_tools:
        if message.id is not None:
            removals.append(RemoveMessage(id=message.id))
    logger.warning(
        "sanitize_messages removed dangling tool call group: assistant=%s missing=%s",
        assistant.id,
        sorted(expected_ids - answered_ids),
    )
    return {"messages": removals}
