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

import asyncio
import logging
import re
import time
import uuid
from dataclasses import dataclass
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
from langgraph.graph.state import CompiledStateGraph

from app.config import get_settings
from app.knowledge import retrieval
from app.knowledge.errors import KnowledgeBackendUnavailable
from app.prompts import SYSTEM_PROMPT
from app.tools import ALL_TOOLS
from app.tools.knowledge import (
    KNOWLEDGE_UNAVAILABLE_TEXT,
    NO_FAQ_RESULT_TEMPLATE,
    format_hits,
)
from app.workflow.citations import assign_citations
from app.workflow.state import WorkflowState
from app.workflow.texts import (
    EVIDENCE_PREFIX,
    REFUSAL_TEXT,
    STEP_LIMIT_TEXT,
    TOKEN_LIMIT_TEXT,
)

logger = logging.getLogger(__name__)

ACTION_HANDOFF = "handoff"
ACTION_TICKET = "ticket"

# spec §3.5.2：模型自评知识不足时输出该标记。与 app/api.py 的流式过滤器同字面量；
# 图这边负责在**收敛时**把它从 State 里替换掉，避免标记进入 checkpoint 后
# 下一轮继续作为模型上下文。
INSUFFICIENT_MARKER = "<INSUFFICIENT_KNOWLEDGE>"

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

EVIDENCE_HEADER = EVIDENCE_PREFIX + "，回答须以此为准并标注引用编号："
_EVIDENCE_FLAG = "workflow_evidence_context"

# 动作标记的用法在 prompts.SYSTEM_PROMPT 里已有完整说明；这里再补一句
# 面向本子图的短指令，保证即使调用方传入的是别的 system prompt 也能工作。
ACTION_GUIDANCE = (
    "\n\n## 动作建议\n"
    "用户在对话中明确要求转人工时，在回答末尾输出 <SUGGEST_HANDOFF>；"
    "需要为用户创建工单跟进时，在回答末尾输出 <SUGGEST_TICKET>。"
    "标记会被系统识别并转成按钮，不要在正文里解释或重复。"
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


def _system_message() -> SystemMessage:
    """每次模型调用都本地构造 system prompt（不写入 State，避免重复注入）。"""
    return SystemMessage(content=SYSTEM_PROMPT + ACTION_GUIDANCE)


def _action_types(actions: list[dict] | None) -> list[str]:
    """从 state.actions 里取出动作类型，保持原有顺序。"""
    return [
        action["type"]
        for action in actions or []
        if isinstance(action, dict) and action.get("type")
    ]


def _merge_action_types(existing: list[str], new: list[str]) -> list[str]:
    merged = list(existing)
    for action_type in new:
        if action_type not in merged:
            merged.append(action_type)
    return merged


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
) -> tuple[Any, dict | None, str | None]:
    """流式消费模型输出：逐 token 发 delta，返回（累积响应、usage、finish_reason）。

    usage 只取**最后一个**带值的 chunk：部分 provider 会在每个 chunk 上重复
    携带累计用量，若把 chunk 逐个相加会成倍放大 token 数。finish_reason 同理，
    最后一个有值的 chunk 即本次调用的终局（``stop`` / ``length`` /
    ``tool_calls`` ...）。真实 provider 一定有 ``astream``；脚本化替身可能只
    实现 ``ainvoke``，这里保留一条兜底路径，避免子图只能被“会流式”的替身驱动。
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
        meta = getattr(response, "response_metadata", None) or {}
        return (
            response,
            getattr(response, "usage_metadata", None),
            meta.get("finish_reason"),
        )

    accumulated: AIMessageChunk | None = None
    usage: dict | None = None
    finish_reason: str | None = None
    async for raw_chunk in bound.astream(messages):
        chunk = _as_chunk(raw_chunk)
        accumulated = chunk if accumulated is None else accumulated + chunk
        if getattr(raw_chunk, "usage_metadata", None):
            usage = dict(raw_chunk.usage_metadata)
        meta = getattr(raw_chunk, "response_metadata", None) or {}
        if meta.get("finish_reason"):
            finish_reason = meta["finish_reason"]
        text = getattr(raw_chunk, "text", "") or ""
        if text:
            filtered = stripper.feed(text)
            if filtered:
                writer({"type": "delta", "text": filtered})

    tail = stripper.flush()
    if tail:
        writer({"type": "delta", "text": tail})
    return accumulated, usage, finish_reason


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
    # system prompt 每次本地构造、不落 State；旧轮 evidence 只留在 State 里
    # 供审计过滤，绝不带回本轮模型上下文。顺序：system → evidence → 对话。
    model_messages: list[BaseMessage] = (
        [_system_message()]
        + ([current_evidence] if current_evidence is not None else [])
        + [message for message in messages if not _is_evidence_message(message)]
    )

    stripper = _ActionMarkerStripper()
    accumulated, usage, finish_reason = await _consume_model(
        model.bind_tools(ALL_TOOLS), model_messages, writer, stripper
    )
    ai_message = _final_message(accumulated)
    # 每个中间 assistant 也要剥标记：ReAct 循环里模型可能边调工具边说
    # “我帮您转人工”，不能让它把标记带进 ToolMessage 回合或审计。
    clean_text, new_action_types = parse_action_markers(ai_message.text or "")
    ai_message = ai_message.model_copy(update={"content": clean_text})

    action_types = _merge_action_types(
        _action_types(state.get("actions")), new_action_types
    )
    updates: dict[str, Any] = {
        "steps": (state.get("steps") or 0) + 1,
        "tokens_used": (state.get("tokens_used") or 0) + _usage_tokens(usage),
        "messages": [ai_message],
        "actions": _build_actions(action_types, state),
        # 最后一次模型调用的真实终局元数据（多步 ReAct 时后一次覆盖前一次）。
        "finish_reason": finish_reason,
        "usage": usage,
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


@dataclass(frozen=True)
class _FaqOutcome:
    content: str
    ok: bool
    summary: str
    citations: list[dict]
    index_map: dict[str, int]
    cursor: int
    refusal_reason: str | None = None


async def _run_query_faq(
    args: dict,
    *,
    settings,
    session,
    conversation_id: int | str | None,
    citations: list[dict],
    index_map: dict[str, int],
    cursor: int,
) -> _FaqOutcome:
    """Agent 内的 query_faq：检索 + 全局编号，返回 ToolMessage 所需字段。

    ``query_faq`` 刻意不走 ``ToolRegistry.execute``：工具原始输出没有引用编号，
    而本图要求模型在**看到内容之前**就拿到全局 ``[i]``（spec §5.7）。

    这里绕过了 registry 的超时/异常边界，所以必须自己补：``wait_for`` 用与
    业务工具相同的 ``tool_timeout_s``，任何异常都降级成一句工具结果，绝不让
    知识库故障打断整轮对话。
    """
    keyword = str(args.get("keyword") or "")
    try:
        chunks, meta = await asyncio.wait_for(
            retrieval.search_with_metadata(
                session,
                keyword,
                top_k=settings.retrieval_top_k,
                strategy=settings.retrieval_strategy,
                session_id=(
                    "" if conversation_id is None else str(conversation_id)
                ),
            ),
            timeout=settings.tool_timeout_s,
        )
    except KnowledgeBackendUnavailable as exc:
        # 预期内的基础设施降级：不打完整堆栈，只留一行 warning。
        logger.warning("query_faq 知识库不可用：%s", exc)
        return _FaqOutcome(
            KNOWLEDGE_UNAVAILABLE_TEXT, False, "知识库不可用",
            citations, index_map, cursor,
        )
    except Exception:
        # 超时、代码 bug、上游协议异常……一律按“查不到”降级；堆栈留日志。
        logger.exception("query_faq 检索失败，降级为不可用文案")
        return _FaqOutcome(
            KNOWLEDGE_UNAVAILABLE_TEXT, False, "知识库不可用",
            citations, index_map, cursor,
        )

    refusal_reason = (meta or {}).get("refusal_reason")
    _, citations, index_map, cursor = assign_citations(
        chunks, citations, index_map, cursor
    )
    # 重排策略下 meta["scores"] 与 chunks 一一对齐；assign_citations 只负责
    # 编号，这里把真实分数回填到对应引用（同一 chunk 复用时也更新为最新分数）。
    scores = (meta or {}).get("scores")
    if scores is not None:
        score_by_index = {
            index_map[chunk.chunk_id]: scores[i]
            for i, chunk in enumerate(chunks)
            if i < len(scores) and scores[i] is not None
        }
        citations = [
            {
                **citation,
                "score": score_by_index.get(
                    citation.get("index"), citation.get("score")
                ),
            }
            for citation in citations
        ]
    blocks = format_hits(chunks)
    if blocks:
        # 编号仍由 assign_citations 分配；正文格式复用工具的 shared helper，
        # 保证图内与工具路径交给模型的问答块逐字一致。
        numbered = "\n\n".join(
            f"[{index_map[chunk.chunk_id]}] {block}"
            for chunk, block in zip(chunks, blocks)
        )
        return _FaqOutcome(
            numbered, True, f"检索到 {len(chunks)} 条知识",
            citations, index_map, cursor, refusal_reason,
        )
    # 「查不到」是正常业务结果，不标记为工具故障。
    return _FaqOutcome(
        NO_FAQ_RESULT_TEMPLATE.format(keyword=keyword),
        True,
        "未找到相关知识",
        citations,
        index_map,
        cursor,
        refusal_reason,
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
    refusal_reason: str | None = None
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
                "arguments": args,
                "ok": None,
            }
        )

        if name == "query_faq":
            started = time.perf_counter()
            faq = await _run_query_faq(
                args,
                settings=settings,
                session=session,
                conversation_id=conversation_id,
                citations=citations,
                index_map=index_map,
                cursor=cursor,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            content = faq.content
            ok = faq.ok
            summary = faq.summary
            citations = faq.citations
            index_map = faq.index_map
            cursor = faq.cursor
            if faq.refusal_reason:
                # 透传给 Task 9：检索层已拒答，模型再吐
                # <INSUFFICIENT_KNOWLEDGE> 时不能重复入池。
                refusal_reason = faq.refusal_reason
        else:
            outcome = await registry.execute(
                name, args, call_id, conversation_id=conversation_id
            )
            content = outcome.content
            ok = bool(outcome.ok)
            summary = outcome.summary
            elapsed_ms = outcome.elapsed_ms

        writer(
            {
                "type": "tool",
                "phase": "end",
                "call_id": call_id,
                "name": name,
                "ok": ok,
                "summary": summary,
                "elapsed_ms": elapsed_ms,
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

    updates: dict[str, Any] = {
        "messages": tool_messages,
        "citations": _sorted_citations(citations),
        "citation_index": index_map,
        "citation_cursor": cursor,
    }
    if refusal_reason is not None:
        updates["refusal_reason"] = refusal_reason
    return updates


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
        # 收敛话术由系统生成，终局就是干净的 stop。
        if (state.get("steps") or 0) >= max_steps:
            return {
                "messages": [AIMessage(content=STEP_LIMIT_TEXT)],
                "finish_reason": "stop",
            }
        if (state.get("tokens_used") or 0) >= token_budget:
            return {
                "messages": [AIMessage(content=TOKEN_LIMIT_TEXT)],
                "finish_reason": "stop",
            }
        return {}

    clean_text, new_action_types = parse_action_markers(_message_text(last))
    # spec §3.5.2：自评标记不得进入 checkpoint——否则下一轮模型上下文里会
    # 出现一个“我不想回答”的标记，诱发后续轮次继续拒答。这里把它替换成
    # 标准拒答话术，并用 State 标志通知 API 做入池判定。
    insufficient = INSUFFICIENT_MARKER in clean_text
    if insufficient:
        clean_text = clean_text.split(INSUFFICIENT_MARKER, 1)[0] + REFUSAL_TEXT
    # agent_model 已逐条清过标记并把动作写进 state；这里做最后一次兜底解析，
    # 并与已有动作按首次出现顺序合并（不能把中间轮的动作覆盖成空）。
    action_types = _merge_action_types(
        _action_types(state.get("actions")), new_action_types
    )
    updates: dict[str, Any] = {
        "actions": _build_actions(action_types, state),
        "insufficient": insufficient,
    }
    if clean_text != last.content:
        # 用同一个 id 覆盖原消息（add_messages 对同 id 是替换而非追加），
        # 保证审计里只有一条干净的最终 assistant。
        updates["messages"] = [last.model_copy(update={"content": clean_text})]
    return updates


def build_agent_subgraph() -> CompiledStateGraph:
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
