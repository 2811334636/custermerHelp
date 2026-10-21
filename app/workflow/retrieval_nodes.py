"""知识类检索节点与置信度闸（spec §5.4/§5.5）。"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from app.config import get_settings
from app.knowledge import retrieval
from app.repositories import low_confidence_question as lcq_repo
from app.workflow.citations import assign_citations
from app.workflow.state import WorkflowState
from app.workflow.texts import GATE_REFUSAL

logger = logging.getLogger(__name__)

REASON_NO_RESULTS = "no_results"
REASON_LOW_EVIDENCE = "low_evidence"


def char_bigrams(text: str) -> set[str]:
    """去掉空白和标点后取相邻字符二元组；不足两个字符时返回空集。"""
    cleaned = "".join(ch for ch in text if ch.isalnum())
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def evidence_coverage(query: str, texts: list[str]) -> float:
    """查询字符二元组在召回文本中的命中比例；空查询/空文本返回 0.0。

    逐条文本取二元组后再求并集，不能先拼成一个字符串：否则上一块的尾字
    与下一块的首字会拼出一个现实中不存在的二元组（``["退", "货"]`` 会
    误命中查询 ``"退货"``）。
    """
    query_bigrams = char_bigrams(query)
    if not query_bigrams:
        return 0.0
    evidence_bigrams: set[str] = set().union(
        *(char_bigrams(text) for text in texts)
    )
    if not evidence_bigrams:
        return 0.0
    return len(query_bigrams & evidence_bigrams) / len(query_bigrams)


def _refusal_payload(
    reason: str, coverage: float, trace: list[str]
) -> dict[str, Any]:
    """统一的拒答输出：闸门是唯一话术所有者（Task 8 失败边直接到 log）。"""
    return {
        "gate_passed": False,
        "refusal_reason": reason,
        "evidence_coverage": coverage,
        "messages": [AIMessage(GATE_REFUSAL)],
        "actions": [{"type": "handoff", "label": "转人工"}],
        "node_trace": trace,
    }


def _configurable(config: RunnableConfig | None) -> dict[str, Any]:
    return (config or {}).get("configurable", {}) or {}


def _conversation_session_id(configurable: dict[str, Any]) -> str:
    conversation_id = configurable.get("conversation_id")
    return str(conversation_id) if conversation_id is not None else ""


async def retrieve_node(
    state: WorkflowState, config: RunnableConfig
) -> dict:
    """预检索知识并分配全局引用编号（spec §5.4）。

    ``strategy`` 与 ``session_id`` 必须显式传：``search_with_metadata`` 的
    默认值是 ``dense`` / ``""``，不传会悄悄绕过 ch04 配置的混合检索与重排，
    并让低置信度问题池里的 session_id 为空。
    """
    configurable = _configurable(config)
    settings = get_settings()
    query = state.get("resolved_input") or state.get("user_input", "")

    chunks, meta = await retrieval.search_with_metadata(
        configurable.get("session"),
        query,
        top_k=settings.retrieval_top_k,
        strategy=settings.retrieval_strategy,
        session_id=_conversation_session_id(configurable),
    )

    cursor = state.get("citation_cursor")
    if cursor is None:
        cursor = 1
    _, citations, index_map, cursor = assign_citations(
        chunks,
        state.get("citations") or [],
        state.get("citation_index") or {},
        cursor,
    )

    # meta["scores"] 与 chunks 一一对齐（仅重排策略有值），其余策略为 None。
    scores = meta.get("scores")
    evidence = [
        {
            "index": index_map[chunk.chunk_id],
            "chunk_id": chunk.chunk_id,
            "questions": chunk.questions,
            "answer": chunk.answer,
            "score": (
                scores[i] if scores is not None and i < len(scores) else None
            ),
        }
        for i, chunk in enumerate(chunks)
    ]
    # assign_citations 只负责编号；把重排分数回填到 citation，done 帧与审计
    # metadata 才能给出可解释的引用分数。
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

    logger.info(
        "node=retrieve chunks=%d strategy=%s",
        len(chunks),
        meta.get("strategy", settings.retrieval_strategy),
    )

    out: dict[str, Any] = {
        "evidence": evidence,
        "citations": citations,
        "citation_index": index_map,
        "citation_cursor": cursor,
        "node_trace": state.get("node_trace", []) + ["retrieve"],
    }
    refusal_reason = meta.get("refusal_reason")
    if refusal_reason:
        # 空结果/低分已在 search_with_metadata 里入池；这里只透传原因并拦住 Agent。
        out["refusal_reason"] = refusal_reason
        out["gate_passed"] = False
    return out


async def confidence_gate_node(
    state: WorkflowState, config: RunnableConfig
) -> dict:
    """置信度闸：重排分优先，否则用词面覆盖率；不通过时把问题入池（spec §5.5）。"""
    configurable = _configurable(config)
    settings = get_settings()
    query = state.get("resolved_input") or state.get("user_input", "")
    evidence = state.get("evidence") or []
    prior_reason = state.get("refusal_reason")
    trace = state.get("node_trace", []) + ["confidence_gate"]

    if not evidence:
        # 检索空结果的入池已由 search_with_metadata 完成，这里不重复写；
        # 但话术由闸门统一发出，失败边直接到 log（Task 8 无独立 refuse 节点）。
        return _refusal_payload(
            prior_reason or REASON_NO_RESULTS, 0.0, trace
        )

    coverage = evidence_coverage(
        query,
        [
            f"{entry.get('questions') or ''}\n{entry.get('answer') or ''}"
            for entry in evidence
        ],
    )
    scores = [
        float(entry["score"])
        for entry in evidence
        if entry.get("score") is not None
    ]
    if scores:
        passed = max(scores) >= settings.retrieval_rerank_threshold
    else:
        passed = coverage >= settings.confidence_min_coverage

    if passed:
        return {
            "gate_passed": True,
            "refusal_reason": None,
            "evidence_coverage": coverage,
            "node_trace": trace,
        }

    reason = prior_reason or REASON_LOW_EVIDENCE
    if prior_reason is None:
        # 入池只写一次：retrieval 已因 no_results/low_score 写过时不重复写。
        session = configurable.get("session")
        conversation_id = configurable.get("conversation_id")
        if session is None:
            logger.warning(
                "confidence_gate missing session; low_evidence not pooled question=%r",
                query,
            )
        elif conversation_id is None:
            # 没有会话 id 时宁可不写：空 session_id 会让问题池失去溯源能力。
            logger.warning(
                "confidence_gate missing conversation_id; "
                "low_evidence not pooled question=%r",
                query,
            )
        else:
            await lcq_repo.create(
                session,
                query,
                str(conversation_id),
                REASON_LOW_EVIDENCE,
            )
            logger.warning(
                "置信度闸拒答（reason=%s），问题已入低置信度问题池：%s",
                reason,
                query,
            )

    return _refusal_payload(reason, coverage, trace)
