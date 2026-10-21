"""ReAct 子图：模型循环、工具执行、全局引用编号与动作标记（spec §5.7）。

子图只有三个节点：

    agent_model ──(有 tool_calls 且未触顶)──> agent_tools ──> agent_model
         └────────(无 tool_calls / 超步数 / 超预算)────────> agent_done

依赖（model/registry/session/conversation_id）全部从
``config["configurable"]`` 取，节点本身不持有请求级状态，便于 Task 8 把
子图直接挂到主图里。

引用编号由 ``assign_citations`` 统一分配：预检索先占 1..N，Agent 里再调
``query_faq`` 时续号；同一个 chunk 重复召回始终复用原编号，因此模型看到
的 ``[i]`` 与最终 ``citations`` 天然一致，不需要事后修正。
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.knowledge import retrieval
from app.knowledge.errors import KnowledgeBackendUnavailable
from app.tools import ALL_TOOLS
from app.workflow.citations import assign_citations
from app.workflow.state import WorkflowState
from app.workflow.texts import STEP_LIMIT_TEXT, TOKEN_LIMIT_TEXT

ACTION_HANDOFF = "handoff"
ACTION_TICKET = "ticket"

# 标记大小写敏感、全大写（spec §6.2），降低正文误伤。
ACTION_MARKERS: tuple[tuple[str, str], ...] = (
    ("<SUGGEST_HANDOFF>", ACTION_HANDOFF),
    ("<SUGGEST_TICKET>", ACTION_TICKET),
)
_MARKER_BY_TEXT = dict(ACTION_MARKERS)
_MARKER_PATTERN = re.compile(
    "|".join(re.escape(marker) for marker, _ in ACTION_MARKERS)
)
_MAX_MARKER_LEN = max(len(marker) for marker, _ in ACTION_MARKERS)

TICKET_TYPE_BY_ROUTE = {
    "knowledge": "咨询",
    "business": "售后",
    "complaint": "投诉",
}
DEFAULT_TICKET_TYPE = "咨询"
TICKET_DESCRIPTION_LIMIT = 500

EVIDENCE_HEADER = "以下是检索到的知识，回答须以此为准并标注引用编号："
_EVIDENCE_FLAG = "workflow_evidence_context"

KNOWLEDGE_FALLBACK = "知识库暂时不可用，请稍后再试或转人工客服。"
NO_FAQ_RESULT = (
    "没有找到与「{keyword}」相关的常见问题。可以换个说法，或帮你转人工客服。"
)


def parse_action_markers(text: str) -> tuple[str, list[str]]:
    """剥离 ``<SUGGEST_*>`` 标记，返回（干净文本、按首次出现去重的动作类型）。"""
    source = text or ""
    actions: list[str] = []
    for match in _MARKER_PATTERN.finditer(source):
        action = _MARKER_BY_TEXT[match.group(0)]
        if action not in actions:
            actions.append(action)
    return _MARKER_PATTERN.sub("", source), actions


class _ActionMarkerStripper:
    """跨 delta 剥离动作标记。

    标记可能被切在两个流式 chunk 之间，所以只扣留「可能是某个标记前缀」的
    尾巴（最长 ``len(marker) - 1`` 个字符），其余文本立即放行；完整标记直接
    丢掉，标记之后的内容照常保留（动作标记是独立标记，不像
    ``<INSUFFICIENT_KNOWLEDGE>`` 那样表示“后面全是噪声”）。
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> str:
        self._buf += text
        emitted: list[str] = []
        while True:
            match = _MARKER_PATTERN.search(self._buf)
            if match is None:
                break
            emitted.append(self._buf[: match.start()])
            self._buf = self._buf[match.end() :]

        hold = 0
        limit = min(_MAX_MARKER_LEN - 1, len(self._buf))
        for n in range(limit, 0, -1):
            if any(
                self._buf.endswith(marker[:n]) for marker, _ in ACTION_MARKERS
            ):
                hold = n
                break
        if hold:
            emitted.append(self._buf[:-hold])
            self._buf = self._buf[-hold:]
        else:
            emitted.append(self._buf)
            self._buf = ""
        return "".join(emitted)

    def flush(self) -> str:
        text, self._buf = self._buf, ""
        return text


def _is_evidence_message(message: BaseMessage) -> bool:
    return isinstance(message, SystemMessage) and bool(
        message.additional_kwargs.get(_EVIDENCE_FLAG)
    )


def _evidence_message(evidence: list[dict]) -> SystemMessage:
    """把 State 里的 evidence 压成一条临时 SystemMessage。

    ``additional_kwargs`` 里的私有标记让同一条消息可以被识别出来：同一轮
    的多次模型调用不重复注入，跨轮则用新 evidence 重新生成一条；Task 9 写
    审计时按 ``SystemMessage`` 类型整体过滤。
    """
    blocks = []
    for entry in evidence:
        index = entry.get("index")
        prefix = f"[{index}] " if index is not None else ""
        blocks.append(
            f"{prefix}问：{entry.get('questions') or ''}\n"
            f"答：{entry.get('answer') or ''}"
        )
    return SystemMessage(
        content=f"{EVIDENCE_HEADER}\n\n" + "\n\n".join(blocks),
        additional_kwargs={_EVIDENCE_FLAG: True},
    )


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    return (config or {}).get("configurable", {}) or {}


def _limits(state: WorkflowState) -> tuple[int, int]:
    settings = get_settings()
    max_steps = state.get("max_steps") or settings.agent_max_steps
    token_budget = state.get("token_budget") or settings.agent_token_budget
    return int(max_steps), int(token_budget)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content)


def _as_chunk(chunk: Any) -> AIMessageChunk:
    """把非流式替身返回的完整 AIMessage 归一成可累加的 chunk。"""
    if isinstance(chunk, AIMessageChunk):
        return chunk
    return AIMessageChunk(
        content=getattr(chunk, "content", "") or "",
        tool_calls=list(getattr(chunk, "tool_calls", None) or []),
        usage_metadata=getattr(chunk, "usage_metadata", None),
        id=getattr(chunk, "id", None),
    )


def _usage_tokens(usage: dict | None) -> int:
    if not usage:
        return 0
    total = usage.get("total_tokens")
    if total is not None:
        return int(total)
    return int(usage.get("input_tokens") or 0) + int(
        usage.get("output_tokens") or 0
    )


async def _consume_model(
    bound, messages: list[BaseMessage], writer, stripper: _ActionMarkerStripper
):
    """流式消费模型输出：逐 token 发 delta，返回累积后的完整响应。

    真实 provider 一定有 ``astream``；脚本化替身可能只实现 ``ainvoke``，
    这里保留一条兜底路径，避免子图只能被“会流式”的替身驱动。
    """
    if not hasattr(bound, "astream"):
        response = await bound.ainvoke(messages)
        text = getattr(response, "text", "") or ""
        if text:
            filtered = stripper.feed(text)
            if filtered:
                writer({"type": "delta", "text": filtered})
        tail = stripper.flush()
        if tail:
            writer({"type": "delta", "text": tail})
        return response

    accumulated: AIMessageChunk | None = None
    async for raw_chunk in bound.astream(messages):
        chunk = _as_chunk(raw_chunk)
        accumulated = chunk if accumulated is None else accumulated + chunk
        text = getattr(raw_chunk, "text", "") or ""
        if text:
            filtered = stripper.feed(text)
            if filtered:
                writer({"type": "delta", "text": filtered})

    tail = stripper.flush()
    if tail:
        writer({"type": "delta", "text": tail})
    return accumulated


def _final_message(accumulated: Any) -> AIMessage:
    if accumulated is None:
        return AIMessage(content="", id=f"ai-{uuid.uuid4().hex}")
    return AIMessage(
        content=getattr(accumulated, "text", "") or "",
        tool_calls=list(getattr(accumulated, "tool_calls", None) or []),
        usage_metadata=getattr(accumulated, "usage_metadata", None),
        id=getattr(accumulated, "id", None) or f"ai-{uuid.uuid4().hex}",
    )


async def agent_model(
    state: WorkflowState, config: RunnableConfig
) -> dict[str, Any]:
    """调用模型一轮：流式发 delta、累积 AIMessage、累加 steps/tokens。"""
    configurable = _configurable(config)
    model = configurable["model"]
    writer = get_stream_writer()

    messages = list(state.get("messages") or [])
    evidence = state.get("evidence") or []
    current_evidence = _evidence_message(evidence) if evidence else None
    # 旧轮的 evidence 只留在 State 里供审计过滤，绝不带回本轮模型上下文；
    # 本轮 evidence 永远放在最前面（system 上下文先于对话）。
    model_messages: list[BaseMessage] = (
        ([current_evidence] if current_evidence is not None else [])
        + [message for message in messages if not _is_evidence_message(message)]
    )

    stripper = _ActionMarkerStripper()
    accumulated = await _consume_model(
        model.bind_tools(ALL_TOOLS), model_messages, writer, stripper
    )
    ai_message = _final_message(accumulated)

    usage = getattr(accumulated, "usage_metadata", None)
    updates: dict[str, Any] = {
        "steps": (state.get("steps") or 0) + 1,
        "tokens_used": (state.get("tokens_used") or 0) + _usage_tokens(usage),
        "messages": [ai_message],
    }
    if current_evidence is not None and not any(
        _is_evidence_message(message)
        and message.content == current_evidence.content
        for message in messages
    ):
        # 本轮第一次进入 agent_model：把注入消息留在 State 里（Task 9 过滤）。
        updates["messages"] = [current_evidence, ai_message]
    return updates


def _sorted_citations(citations: list[dict]) -> list[dict]:
    return sorted(citations, key=lambda citation: citation.get("index", 0))


async def _run_query_faq(
    args: dict,
    *,
    settings,
    session,
    conversation_id: int | str | None,
    citations: list[dict],
    index_map: dict[str, int],
    cursor: int,
) -> tuple[str, bool, str, list[dict], dict[str, int], int]:
    """Agent 内的 query_faq：检索 + 全局编号，返回 ToolMessage 所需字段。

    ``query_faq`` 刻意不走 ``ToolRegistry.execute``：工具原始输出没有引用编号，
    而本图要求模型在**看到内容之前**就拿到全局 ``[i]``（spec §5.7）。
    """
    keyword = str(args.get("keyword") or "")
    try:
        chunks, _meta = await retrieval.search_with_metadata(
            session,
            keyword,
            top_k=settings.retrieval_top_k,
            strategy=settings.retrieval_strategy,
            session_id=(
                "" if conversation_id is None else str(conversation_id)
            ),
        )
    except KnowledgeBackendUnavailable:
        # 基础设施故障降级为一句正常工具结果，不让整轮对话中断。
        return KNOWLEDGE_FALLBACK, False, "知识库不可用", citations, index_map, cursor

    numbered, citations, index_map, cursor = assign_citations(
        chunks, citations, index_map, cursor
    )
    if numbered:
        return (
            numbered,
            True,
            f"检索到 {len(chunks)} 条知识",
            citations,
            index_map,
            cursor,
        )
    # 「查不到」是正常业务结果，不标记为工具故障。
    return (
        NO_FAQ_RESULT.format(keyword=keyword),
        True,
        "未找到相关知识",
        citations,
        index_map,
        cursor,
    )


async def agent_tools(
    state: WorkflowState, config: RunnableConfig
) -> dict[str, Any]:
    """执行上一条 assistant 消息里的全部 tool_calls，逐个发 start/end 事件。"""
    configurable = _configurable(config)
    registry = configurable["registry"]
    session = configurable.get("session")
    conversation_id = configurable.get("conversation_id")
    settings = get_settings()
    writer = get_stream_writer()

    messages = list(state.get("messages") or [])
    last = messages[-1] if messages else None
    tool_calls = list(getattr(last, "tool_calls", None) or [])

    citations = [dict(citation) for citation in state.get("citations") or []]
    index_map = dict(state.get("citation_index") or {})
    cursor = state.get("citation_cursor")
    if cursor is None:
        cursor = 1

    tool_messages: list[ToolMessage] = []
    for position, call in enumerate(tool_calls):
        name = call.get("name", "")
        args = call.get("args") or {}
        call_id = call.get("id") or f"call-{position}"

        # start 阶段结果未知，ok 用 None 占位；end 帧覆盖为真实值。
        writer(
            {
                "type": "tool",
                "phase": "start",
                "call_id": call_id,
                "name": name,
                "ok": None,
            }
        )

        if name == "query_faq":
            (
                content,
                ok,
                summary,
                citations,
                index_map,
                cursor,
            ) = await _run_query_faq(
                args,
                settings=settings,
                session=session,
                conversation_id=conversation_id,
                citations=citations,
                index_map=index_map,
                cursor=cursor,
            )
        else:
            outcome = await registry.execute(
                name, args, call_id, conversation_id=conversation_id
            )
            content = outcome.content
            ok = bool(outcome.ok)
            summary = outcome.summary

        writer(
            {
                "type": "tool",
                "phase": "end",
                "call_id": call_id,
                "name": name,
                "ok": ok,
                "summary": summary,
                "citations": _sorted_citations(citations),
            }
        )
        tool_messages.append(
            ToolMessage(
                content=content,
                tool_call_id=call_id,
                name=name or None,
            )
        )

    return {
        "messages": tool_messages,
        "citations": _sorted_citations(citations),
        "citation_index": index_map,
        "citation_cursor": cursor,
    }


def _build_actions(action_types: list[str], state: WorkflowState) -> list[dict]:
    route = state.get("route") or ""
    user_input = state.get("user_input") or ""
    actions: list[dict] = []
    for action_type in action_types:
        if action_type == ACTION_HANDOFF:
            actions.append({"type": "handoff", "label": "转人工"})
        elif action_type == ACTION_TICKET:
            actions.append(
                {
                    "type": "ticket",
                    "label": "建工单",
                    "ticket_type": TICKET_TYPE_BY_ROUTE.get(
                        route, DEFAULT_TICKET_TYPE
                    ),
                    "description": user_input[:TICKET_DESCRIPTION_LIMIT],
                }
            )
    return actions


def _route_after_model(state: WorkflowState) -> str:
    """条件边：仅在还有工具调用且未触顶时继续 ReAct。"""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return "agent_done"

    max_steps, token_budget = _limits(state)
    if (state.get("steps") or 0) >= max_steps:
        return "agent_done"
    if (state.get("tokens_used") or 0) >= token_budget:
        return "agent_done"
    return "agent_tools"


async def agent_done(state: WorkflowState) -> dict[str, Any]:
    """收敛节点：触顶时补固定话术，正常收敛时清标记并生成 actions。"""
    messages = list(state.get("messages") or [])
    last = messages[-1] if messages else None
    if not isinstance(last, AIMessage):
        return {}

    max_steps, token_budget = _limits(state)
    if last.tool_calls:
        # 走到这里说明条件边判定触顶；工具结果还没执行，直接给收敛话术，
        # 不能把“我还想调工具”的半截 assistant 当成最终回答。
        if (state.get("steps") or 0) >= max_steps:
            return {"messages": [AIMessage(content=STEP_LIMIT_TEXT)]}
        if (state.get("tokens_used") or 0) >= token_budget:
            return {"messages": [AIMessage(content=TOKEN_LIMIT_TEXT)]}
        return {}

    clean_text, action_types = parse_action_markers(_message_text(last))
    updates: dict[str, Any] = {
        "actions": _build_actions(action_types, state),
    }
    if clean_text != last.content:
        # 用同一个 id 覆盖原消息（add_messages 对同 id 是替换而非追加），
        # 保证审计里只有一条干净的最终 assistant。
        updates["messages"] = [last.model_copy(update={"content": clean_text})]
    return updates


def build_agent_subgraph():
    """装配并编译 ReAct 子图；Task 8 把它作为一个节点挂到主图。"""
    graph = StateGraph(WorkflowState)
    graph.add_node("agent_model", agent_model)
    graph.add_node("agent_tools", agent_tools)
    graph.add_node("agent_done", agent_done)

    graph.add_edge(START, "agent_model")
    graph.add_conditional_edges(
        "agent_model",
        _route_after_model,
        {"agent_tools": "agent_tools", "agent_done": "agent_done"},
    )
    graph.add_edge("agent_tools", "agent_model")
    graph.add_edge("agent_done", END)
    return graph.compile()
