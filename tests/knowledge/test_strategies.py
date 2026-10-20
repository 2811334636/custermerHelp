"""检索策略与拒答控制测试。

全部依赖都通过注入提供（store / embed / rerank_fn / rewrite_fn 或模块级
monkeypatch），不依赖真实 Milvus、embedding、LLM 与 reranker API。
"""

import logging

import pytest

from app.config import get_settings
from app.knowledge.chunk import Chunk
from app.knowledge.errors import RerankerUnavailable
from app.knowledge.retrieval import search
from app.knowledge.strategies import retrieve_with_strategy
from app.repositories import knowledge_chunk as repo
from app.repositories import low_confidence_question as lcq_repo


class FakeStore:
    """记录调用参数并可分别配置三路检索结果的替身。"""

    def __init__(
        self,
        *,
        dense_ids: list[str] | None = None,
        bm25_ids: list[str] | None = None,
        fused_ids: list[str] | None = None,
    ):
        self.dense_ids = list(dense_ids or [])
        self.bm25_ids = list(bm25_ids or [])
        self.fused_ids = list(fused_ids or [])
        self.search_calls: list[tuple[list[float], int]] = []
        self.bm25_calls: list[tuple[str, int]] = []
        self.hybrid_calls: list[tuple[str, int, int, int]] = []
        # 过滤表达式单独记录，避免改坏既有调用形态断言。
        self.search_filters: list[str | None] = []
        self.bm25_filters: list[str | None] = []
        self.hybrid_filters: list[str | None] = []

    def search(
        self, vector: list[float], top_k: int, *, filter_expr: str | None = None
    ) -> list[str]:
        self.search_calls.append((vector, top_k))
        self.search_filters.append(filter_expr)
        return self.dense_ids[:top_k]

    def bm25_search(
        self, query_text: str, top_k: int, *, filter_expr: str | None = None
    ) -> list[str]:
        self.bm25_calls.append((query_text, top_k))
        self.bm25_filters.append(filter_expr)
        return self.bm25_ids[:top_k]

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        dense_limit: int,
        bm25_limit: int,
        final_limit: int,
        *,
        filter_expr: str | None = None,
    ) -> list[str]:
        self.hybrid_calls.append(
            (query_text, dense_limit, bm25_limit, final_limit)
        )
        self.hybrid_filters.append(filter_expr)
        return self.fused_ids[:final_limit]


async def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[0.0] * 1024 for _ in texts]


def make_chunk(cid: str) -> Chunk:
    return Chunk(cid, "分类", f"问题{cid}", "答案。", "分类 > 问题", "policy")


async def seed_chunks(db_session, ids: list[str]) -> None:
    # 不 commit：同一事务内 SELECT 能看见 flush 过的行，fixture 也能回滚。
    await repo.upsert_chunks(db_session, [make_chunk(i) for i in ids])
    await repo.mark_vectorized(db_session, ids, ids)


def make_rewrite(text: str = "改写后的query"):
    async def _rewrite(query: str) -> str:
        return text

    return _rewrite


async def test_strategy_dense_only(db_session):
    """strategy=dense 只用向量检索（ch03 模式）。"""
    await seed_chunks(db_session, ["a:0", "a:1"])
    store = FakeStore(dense_ids=["a:1", "a:0"])

    chunks, meta = await retrieve_with_strategy(
        db_session,
        "测试query",
        "dense",
        top_k=3,
        store=store,
        embed=fake_embed,
    )

    assert meta["strategy"] == "dense"
    assert "bm25_count" not in meta
    assert meta["scores"] is None, "dense 没有可跨 chunk 比较的分数"
    assert len(chunks) <= 3
    assert [c.chunk_id for c in chunks] == ["a:1", "a:0"]
    assert store.search_calls[0][1] == 3
    assert store.hybrid_calls == [], "dense 不能走混合检索"


async def test_strategy_bm25_only(db_session):
    """strategy=bm25 只走 BM25 一路，并先做 query 改写。"""
    await seed_chunks(db_session, ["a:0"])
    store = FakeStore(bm25_ids=["a:0"])

    chunks, meta = await retrieve_with_strategy(
        db_session,
        "邮费咋算",
        "bm25",
        top_k=5,
        store=store,
        rewrite_fn=make_rewrite("邮费如何计算"),
    )

    assert meta["strategy"] == "bm25"
    assert meta["query_rewritten"] == "邮费如何计算"
    assert meta["scores"] is None
    assert [c.chunk_id for c in chunks] == ["a:0"]
    assert store.bm25_calls == [("邮费如何计算", 5)]
    assert store.search_calls == [] and store.hybrid_calls == []


async def test_strategy_hybrid_metadata_counts(db_session):
    """hybrid 的元数据用请求的候选上限计数，并记录融合结果数。"""
    await seed_chunks(db_session, ["a:0", "a:1"])
    store = FakeStore(fused_ids=["a:0", "a:1"])

    chunks, meta = await retrieve_with_strategy(
        db_session,
        "邮费咋算",
        "hybrid",
        top_k=3,
        store=store,
        embed=fake_embed,
        rewrite_fn=make_rewrite("邮费如何计算"),
    )

    s = get_settings()
    assert meta["strategy"] == "hybrid"
    assert meta["dense_count"] == s.retrieval_dense_top_k
    assert meta["bm25_count"] == s.retrieval_bm25_top_k
    assert meta["fused_count"] == 2
    assert meta["scores"] is None, "RRF 融合分不可比，不能作为 chunk 分数"
    assert "rerank_applied" not in meta
    assert store.hybrid_calls == [
        (
            "邮费如何计算",
            s.retrieval_dense_top_k,
            s.retrieval_bm25_top_k,
            3,
        )
    ]


async def test_strategy_hybrid_rerank(db_session):
    """strategy=hybrid_rerank 完整链路：改写 → 混合 → 重排 → 首尾排列。"""
    await seed_chunks(db_session, ["a:0", "a:1", "a:2"])
    store = FakeStore(fused_ids=["a:0", "a:1", "a:2"])

    async def fake_rerank(query, passages, top_k: int = 10):
        assert query == "邮费如何计算"
        assert len(passages) == 3
        # 上游顺序故意打乱：封装层契约是返回降序。
        return [(2, 0.9), (0, 0.7), (1, 0.3)][:top_k]

    chunks, meta = await retrieve_with_strategy(
        db_session,
        "邮费咋算",
        "hybrid_rerank",
        top_k=10,
        store=store,
        embed=fake_embed,
        rerank_fn=fake_rerank,
        rewrite_fn=make_rewrite("邮费如何计算"),
    )

    assert meta["strategy"] == "hybrid_rerank"
    assert meta["query_rewritten"] != "邮费咋算"  # 已改写
    assert meta["dense_count"] > 0
    assert meta["bm25_count"] > 0
    assert len(chunks) <= 10
    assert meta["rerank_applied"] is True
    assert meta["top_score"] == 0.9
    # 重排降序 [a:2, a:0, a:1]，first_last 排列后为 [a:2, a:1, a:0]
    assert [c.chunk_id for c in chunks] == ["a:2", "a:1", "a:0"]
    # scores 必须跟着同一套 first_last 排列，否则 citations 的分数会错位。
    assert meta["scores"] == [0.9, 0.3, 0.7]


async def test_first_last_arrangement_matches_spec_for_ten(db_session):
    """n=10 时排列必须是 0,2,4,6,8,9,7,5,3,1（spec 的 1,3,5,7,9,10,8,6,4,2）。"""
    ids = [f"c:{i}" for i in range(10)]
    await seed_chunks(db_session, ids)
    store = FakeStore(fused_ids=ids)

    async def fake_rerank(query, passages, top_k: int = 10):
        return [(i, 1.0 - i * 0.01) for i in range(10)][:top_k]

    chunks, meta = await retrieve_with_strategy(
        db_session,
        "问题",
        "hybrid_rerank",
        top_k=10,
        store=store,
        embed=fake_embed,
        rerank_fn=fake_rerank,
        rewrite_fn=make_rewrite(),
    )

    assert meta["rerank_applied"] is True
    assert [c.chunk_id for c in chunks] == [
        "c:0", "c:2", "c:4", "c:6", "c:8",
        "c:9", "c:7", "c:5", "c:3", "c:1",
    ]
    assert meta["scores"] == pytest.approx([
        1.0, 0.98, 0.96, 0.94, 0.92,
        0.91, 0.93, 0.95, 0.97, 0.99,
    ])
    assert len(meta["scores"]) == len(chunks), "scores 必须与 chunks 一一对齐"


async def test_hybrid_rerank_falls_back_when_reranker_unavailable(
    db_session, caplog
):
    """重排服务故障时退回未重排的混合结果，不能让整个检索失败。"""
    await seed_chunks(db_session, ["a:0", "a:1"])
    store = FakeStore(fused_ids=["a:0", "a:1"])

    async def boom(query, passages, top_k: int = 10):
        raise RerankerUnavailable("connection refused")

    with caplog.at_level(logging.WARNING, logger="app.knowledge.strategies"):
        chunks, meta = await retrieve_with_strategy(
            db_session,
            "问题",
            "hybrid_rerank",
            top_k=3,
            store=store,
            embed=fake_embed,
            rerank_fn=boom,
            rewrite_fn=make_rewrite(),
        )

    assert [c.chunk_id for c in chunks] == ["a:0", "a:1"]
    assert meta["rerank_applied"] is False
    assert meta["top_score"] is None
    assert meta["scores"] is None, "降级重排没有分数，不能编造"
    assert "重排不可用" in caplog.text


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid", "hybrid_rerank"])
async def test_filter_expr_is_threaded_to_every_strategy(db_session, strategy):
    """spec §3.3：四种策略都必须把 filter_expr 传到 store 调用。"""
    await seed_chunks(db_session, ["a:0"])
    store = FakeStore(
        dense_ids=["a:0"], bm25_ids=["a:0"], fused_ids=["a:0"]
    )
    expr = 'category == "售后政策"'

    async def fake_rerank(query, passages, top_k: int = 10):
        return [(0, 0.9)]

    chunks, _ = await retrieve_with_strategy(
        db_session,
        "邮费咋算",
        strategy,
        top_k=3,
        store=store,
        embed=fake_embed,
        rerank_fn=fake_rerank,
        rewrite_fn=make_rewrite(),
        filter_expr=expr,
    )

    assert [c.chunk_id for c in chunks] == ["a:0"]
    if strategy == "dense":
        assert store.search_filters == [expr]
    elif strategy == "bm25":
        assert store.bm25_filters == [expr]
    else:
        assert store.hybrid_filters == [expr]


async def test_hybrid_rerank_falls_back_when_rerank_returns_empty(
    db_session, caplog
):
    """重排返回 [] 是上游异常，应降级到混合结果而不是当作"没有知识"。"""
    await seed_chunks(db_session, ["a:0", "a:1"])
    store = FakeStore(fused_ids=["a:0", "a:1"])

    async def empty_rerank(query, passages, top_k: int = 10):
        return []

    with caplog.at_level(logging.WARNING, logger="app.knowledge.strategies"):
        chunks, meta = await retrieve_with_strategy(
            db_session,
            "问题",
            "hybrid_rerank",
            top_k=3,
            store=store,
            embed=fake_embed,
            rerank_fn=empty_rerank,
            rewrite_fn=make_rewrite(),
        )

    assert [c.chunk_id for c in chunks] == ["a:0", "a:1"]
    assert meta["rerank_applied"] is False
    assert meta["top_score"] is None
    assert meta["scores"] is None
    assert "重排返回空结果" in caplog.text


async def test_empty_after_filter_triggers_no_results_pooling(db_session):
    """过滤后为空 → no_results 入池，不放宽过滤（spec §3.3）。"""
    from app.knowledge.retrieval import search_with_metadata

    chunks, meta = await search_with_metadata(
        db_session,
        "过滤后为空的问题",
        top_k=3,
        store=FakeStore(),  # 模拟 Milvus 侧过滤后零命中
        embed=fake_embed,
        session_id="sess-filter-empty",
        filter_expr='category == "不存在的分类"',
    )

    assert chunks == []
    assert meta["refusal_reason"] == "no_results"

    questions = await lcq_repo.get_recent(db_session, limit=1)
    assert len(questions) == 1
    assert questions[0].question == "过滤后为空的问题"
    assert questions[0].reason == "no_results"


async def test_unknown_strategy_raises(db_session):
    with pytest.raises(ValueError, match="Unknown strategy"):
        await retrieve_with_strategy(db_session, "问题", "semantic", 3)


async def test_low_confidence_no_results(db_session):
    """检索为空触发拒答并入池（reason=no_results）。"""
    chunks = await search(
        db_session,
        "火星移民计划详情",
        top_k=3,
        store=FakeStore(),
        embed=fake_embed,
        session_id="sess-no-results",
    )

    assert len(chunks) == 0

    questions = await lcq_repo.get_recent(db_session, limit=1)
    assert len(questions) == 1
    assert questions[0].question == "火星移民计划详情"
    assert questions[0].reason == "no_results"
    assert questions[0].session_id == "sess-no-results"


async def test_low_confidence_low_score(db_session, monkeypatch):
    """重排最高分低于阈值触发拒答并入池（reason=low_score）。"""
    await seed_chunks(db_session, ["a:0", "a:1"])

    # 与 brief 一致：用同步 Mock 风格注入；策略层兼容同步替身。
    monkeypatch.setattr(
        "app.knowledge.query_rewrite.rewrite_query", lambda query: query
    )
    monkeypatch.setattr(
        "app.knowledge.reranker.rerank",
        lambda query, passages, top_k=10: [(0, 0.3), (1, 0.2)][:top_k],
    )

    chunks = await search(
        db_session,
        "测试低分query",
        top_k=3,
        store=FakeStore(fused_ids=["a:0", "a:1"]),
        embed=fake_embed,
        strategy="hybrid_rerank",
        session_id="sess-low-score",
    )

    assert chunks == []
    questions = await lcq_repo.get_recent(db_session, limit=1)
    assert questions[0].question == "测试低分query"
    assert questions[0].reason == "low_score"
