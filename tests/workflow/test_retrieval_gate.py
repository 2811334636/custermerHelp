"""知识检索、置信度闸与全局引用编号（spec §5.4/§5.5）。"""

import logging
from dataclasses import dataclass

from app.config import get_settings
from app.knowledge import retrieval as retrieval_module
from app.workflow.citations import assign_citations
from app.workflow.retrieval_nodes import (
    confidence_gate_node,
    evidence_coverage,
    retrieve_node,
)
from app.workflow.texts import GATE_REFUSAL


@dataclass
class FakeChunk:
    chunk_id: str
    questions: str
    answer: str


class FakeSession:
    """只记录 add/flush：入池断言看 added，不碰真实数据库。"""

    def __init__(self):
        self.added = []
        self.flush_count = 0

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self.flush_count += 1


def test_evidence_coverage_basic():
    assert evidence_coverage("退货运费", ["退货运费由平台承担"]) == 1.0
    assert evidence_coverage("退货运费", ["发票怎么开"]) == 0.0
    assert 0 < evidence_coverage("退货运费谁承担", ["退货运费由平台承担，个人原因自理"]) < 1
    assert evidence_coverage("", ["任何文本"]) == 0.0


def test_evidence_coverage_ignores_whitespace_and_punctuation():
    assert evidence_coverage("退货运费！", [" 退货运费，由平台承担。"]) == 1.0
    assert evidence_coverage("退货运费", [""]) == 0.0


def test_assign_citations_numbers_and_dedupes():
    hits = [FakeChunk("a", "q1", "a1"), FakeChunk("b", "q2", "a2")]
    text, citations, index_map, cursor = assign_citations(hits, [], {}, 1)
    assert text.startswith("[1] 问：q1")
    assert "[2] 问：q2" in text
    assert [c["chunk_id"] for c in citations] == ["a", "b"]
    assert cursor == 3
    # 再次召回 a：复用编号 1，不新增
    text2, citations2, index_map2, cursor2 = assign_citations(
        [FakeChunk("a", "q1", "a1")], citations, index_map, cursor
    )
    assert text2.startswith("[1] ")
    assert len(citations2) == 2 and cursor2 == 3
    assert index_map2 == {"a": 1, "b": 2}


async def test_gate_passes_on_high_coverage():
    state = {
        "resolved_input": "退货运费",
        "evidence": [
            {
                "index": 1,
                "chunk_id": "a",
                "questions": "退货运费",
                "answer": "平台承担",
                "score": None,
            }
        ],
        "refusal_reason": None,
    }
    out = await confidence_gate_node(state, {})
    assert out["gate_passed"] is True
    assert out["evidence_coverage"] == 1.0


async def test_gate_refuses_and_pools_once():
    session = FakeSession()
    state = {
        "resolved_input": "火星移民",
        "evidence": [
            {
                "index": 1,
                "chunk_id": "a",
                "questions": "退货",
                "answer": "政策",
                "score": None,
            }
        ],
        "refusal_reason": None,
    }
    out = await confidence_gate_node(
        state, {"configurable": {"session": session, "conversation_id": 7}}
    )
    assert out["gate_passed"] is False
    assert out["refusal_reason"] == "low_evidence"
    assert out["messages"][0].content == GATE_REFUSAL
    assert len(session.added) == 1
    assert session.added[0].question == "火星移民"
    assert session.added[0].session_id == "7"
    assert session.added[0].reason == "low_evidence"


async def test_gate_does_not_pool_again_after_retrieval_refused():
    """retrieve 已写 no_results：闸门不再写第二条 low_evidence。"""
    session = FakeSession()
    state = {
        "resolved_input": "火星移民",
        "evidence": [
            {
                "index": 1,
                "chunk_id": "a",
                "questions": "退货",
                "answer": "政策",
                "score": None,
            }
        ],
        "refusal_reason": "no_results",
    }
    out = await confidence_gate_node(
        state, {"configurable": {"session": session, "conversation_id": 7}}
    )
    assert out["gate_passed"] is False
    assert out["refusal_reason"] == "no_results"
    assert session.added == []


async def test_gate_empty_evidence_keeps_no_results_without_extra_pool_row():
    session = FakeSession()
    out = await confidence_gate_node(
        {"resolved_input": "火星移民", "evidence": [], "refusal_reason": None},
        {"configurable": {"session": session, "conversation_id": 7}},
    )
    assert out["gate_passed"] is False
    assert out["refusal_reason"] == "no_results"
    assert "messages" not in out  # 话术由 refuse 节点发
    assert session.added == []


async def test_gate_uses_rerank_score_when_present():
    # 词面覆盖率极低，但重排分高 → 以重排分为准放行。
    state = {
        "resolved_input": "火星移民",
        "evidence": [
            {
                "index": 1,
                "chunk_id": "a",
                "questions": "退货",
                "answer": "政策",
                "score": 0.9,
            }
        ],
        "refusal_reason": None,
    }
    out = await confidence_gate_node(state, {})
    assert out["gate_passed"] is True


async def test_gate_refuses_on_low_rerank_score_even_if_coverage_high():
    session = FakeSession()
    state = {
        "resolved_input": "退货运费",
        "evidence": [
            {
                "index": 1,
                "chunk_id": "a",
                "questions": "退货运费",
                "answer": "平台承担",
                "score": 0.1,
            }
        ],
        "refusal_reason": None,
    }
    out = await confidence_gate_node(
        state, {"configurable": {"session": session, "conversation_id": 7}}
    )
    assert out["gate_passed"] is False
    assert out["refusal_reason"] == "low_evidence"
    assert len(session.added) == 1


async def test_retrieve_node_passes_explicit_settings_and_assigns_citations(
    monkeypatch, caplog
):
    chunks = [
        FakeChunk("a", "退货运费谁承担", "平台承担"),
        FakeChunk("b", "发票怎么开", "在线申请"),
    ]
    calls = {}

    async def fake_search(session, query, top_k=None, **kwargs):
        calls.update(session=session, query=query, top_k=top_k, **kwargs)
        return chunks, {
            "strategy": "hybrid_rerank",
            "rerank_applied": True,
            "top_score": 0.9,
            "scores": [0.9, 0.4],
        }

    monkeypatch.setattr(retrieval_module, "search_with_metadata", fake_search)
    settings = get_settings()
    session = FakeSession()
    state = {"resolved_input": "退货运费", "citation_cursor": 1}

    with caplog.at_level(logging.INFO, logger="app.workflow.retrieval_nodes"):
        out = await retrieve_node(
            state, {"configurable": {"session": session, "conversation_id": 7}}
        )

    assert calls["session"] is session
    assert calls["query"] == "退货运费"
    assert calls["top_k"] == settings.retrieval_top_k
    assert calls["strategy"] == settings.retrieval_strategy
    assert calls["session_id"] == "7"
    assert [e["index"] for e in out["evidence"]] == [1, 2]
    assert [e["score"] for e in out["evidence"]] == [0.9, 0.4]
    assert [c["chunk_id"] for c in out["citations"]] == ["a", "b"]
    assert out["citation_index"] == {"a": 1, "b": 2}
    assert out["citation_cursor"] == 3
    assert "node=retrieve chunks=2 strategy=" in caplog.text


async def test_retrieve_node_reuses_existing_citation_numbers(monkeypatch):
    async def fake_search(session, query, top_k=None, **kwargs):
        return [FakeChunk("a", "q1", "a1"), FakeChunk("c", "q3", "a3")], {
            "strategy": "dense",
            "scores": None,
        }

    monkeypatch.setattr(retrieval_module, "search_with_metadata", fake_search)
    state = {
        "resolved_input": "问题",
        "citations": [{"index": 1, "chunk_id": "a", "score": None}],
        "citation_index": {"a": 1},
        "citation_cursor": 2,
    }
    out = await retrieve_node(state, {"configurable": {"session": FakeSession(), "conversation_id": 1}})

    assert [e["index"] for e in out["evidence"]] == [1, 2]
    assert out["citation_index"] == {"a": 1, "c": 2}
    assert out["citation_cursor"] == 3
    assert [e["score"] for e in out["evidence"]] == [None, None]


async def test_retrieve_node_passes_through_refusal_and_fails_gate(monkeypatch):
    async def fake_search(session, query, top_k=None, **kwargs):
        return [], {"strategy": "hybrid_rerank", "refusal_reason": "no_results", "scores": None}

    monkeypatch.setattr(retrieval_module, "search_with_metadata", fake_search)
    out = await retrieve_node(
        {"resolved_input": "火星移民"},
        {"configurable": {"session": FakeSession(), "conversation_id": 3}},
    )

    assert out["evidence"] == []
    assert out["refusal_reason"] == "no_results"
    assert out["gate_passed"] is False
    assert out["citations"] == []
