"""四种检索策略：dense / bm25 / hybrid / hybrid_rerank。

策略层只负责「取回候选并按策略排序」，不做拒答判定——拒答需要知道
阈值与问题池，统一放在 `app.knowledge.retrieval.search` 里。

依赖注入（store / embed / rerank_fn / rewrite_fn）既服务于测试，也允许
Task 6 的评估脚本在不改全局配置的前提下单独观测每一路。
"""

import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import KnowledgeChunk
from app.knowledge import query_rewrite, reranker
from app.knowledge.chunk import format_embedding_text
from app.knowledge.embedding import embed_texts
from app.knowledge.errors import RerankerUnavailable
from app.knowledge.retrieval import _load_chunks
from app.knowledge.vector_store import VectorStore

logger = logging.getLogger(__name__)

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank")

RerankFn = Callable[..., Any]
RewriteFn = Callable[[str], Any]


async def _maybe_await(value: Any) -> Any:
    """允许注入同步替身（如 unittest.mock.Mock），生产函数本身是 async。"""
    if inspect.isawaitable(value):
        return await value
    return value


def _arrange_first_last(
    chunks: list[KnowledgeChunk],
) -> list[KnowledgeChunk]:
    """首尾优先排列：偶数位（0-based）升序 + 奇数位降序。

    n=10 时得到 0,2,4,6,8,9,7,5,3,1，即
    [1st, 3rd, 5th, 7th, 9th, 10th, 8th, 6th, 4th, 2nd]。
    最相关的在首尾，避免全部堆在中间被模型忽略。
    """
    return chunks[0::2] + list(reversed(chunks[1::2]))


async def _default_store() -> Any:
    """构造 Milvus 客户端；同步阻塞调用必须离开事件循环。"""
    return await asyncio.to_thread(VectorStore)


async def _dense_only(
    session: AsyncSession,
    query: str,
    top_k: int,
    *,
    store: Any | None = None,
    embed: Any | None = None,
) -> tuple[list[KnowledgeChunk], dict[str, Any]]:
    """纯向量检索（ch03 模式）。"""
    if store is None:
        store = await _default_store()
    if embed is None:
        embed = embed_texts

    query_vector = (await embed([query]))[0]
    ids = await asyncio.to_thread(store.search, query_vector, top_k)
    chunks = await _load_chunks(session, ids)
    meta: dict[str, Any] = {
        "strategy": "dense",
        "query_rewritten": query,
        "dense_count": len(ids),
    }
    logger.info("dense 检索 dense_count=%d", len(ids))
    return chunks, meta


async def _bm25_only(
    session: AsyncSession,
    query: str,
    top_k: int,
    *,
    store: Any | None = None,
) -> tuple[list[KnowledgeChunk], dict[str, Any]]:
    """纯 BM25 全文检索。"""
    if store is None:
        store = await _default_store()

    ids = await asyncio.to_thread(store.bm25_search, query, top_k)
    chunks = await _load_chunks(session, ids)
    meta: dict[str, Any] = {
        "strategy": "bm25",
        "query_rewritten": query,
        "bm25_count": len(ids),
    }
    logger.info("bm25 检索 bm25_count=%d", len(ids))
    return chunks, meta


async def _hybrid_ids(
    session: AsyncSession,
    query: str,
    final_limit: int,
    *,
    store: Any | None,
    embed: Any | None,
) -> tuple[list[KnowledgeChunk], dict[str, Any]]:
    """混合召回公共部分：embedding → hybrid_search → MySQL 行 + 计数。"""
    if store is None:
        store = await _default_store()
    if embed is None:
        embed = embed_texts

    s = get_settings()
    dense_limit = s.retrieval_dense_top_k
    bm25_limit = s.retrieval_bm25_top_k

    query_vector = (await embed([query]))[0]
    ids = await asyncio.to_thread(
        store.hybrid_search,
        query_vector,
        query,
        dense_limit,
        bm25_limit,
        final_limit,
    )
    chunks = await _load_chunks(session, ids)

    # hybrid_search 只返回融合后的 id，拿不到两路各自的真实条数；
    # 用请求的候选上限如实记录，fused_count 记录融合结果条数。
    meta: dict[str, Any] = {
        "strategy": "hybrid",
        "query_rewritten": query,
        "dense_count": dense_limit,
        "bm25_count": bm25_limit,
        "fused_count": len(ids),
    }
    logger.info(
        "混合检索 dense_count=%d bm25_count=%d fused_count=%d",
        dense_limit,
        bm25_limit,
        len(ids),
    )
    return chunks, meta


async def _hybrid(
    session: AsyncSession,
    query: str,
    top_k: int,
    *,
    store: Any | None = None,
    embed: Any | None = None,
) -> tuple[list[KnowledgeChunk], dict[str, Any]]:
    """混合检索 + RRF 融合，不重排。"""
    return await _hybrid_ids(
        session, query, top_k, store=store, embed=embed
    )


async def _hybrid_rerank(
    session: AsyncSession,
    query: str,
    top_k: int,
    *,
    store: Any | None = None,
    embed: Any | None = None,
    rerank_fn: RerankFn | None = None,
) -> tuple[list[KnowledgeChunk], dict[str, Any]]:
    """混合检索 → 重排 → 按配置排列。"""
    s = get_settings()
    # 重排输入取融合后的完整候选池（dense/bm25 各 Top-N），
    # 输出保留 retrieval_rerank_top_k 条且不超过调用方要求。
    candidate_limit = max(top_k, s.retrieval_dense_top_k, s.retrieval_bm25_top_k)
    chunks, meta = await _hybrid_ids(
        session, query, candidate_limit, store=store, embed=embed
    )
    meta["strategy"] = "hybrid_rerank"

    keep = min(top_k, s.retrieval_rerank_top_k)
    if not chunks:
        meta["rerank_applied"] = False
        meta["top_score"] = None
        return [], meta

    fn = rerank_fn or reranker.rerank
    passages = [
        format_embedding_text(c.category, c.questions, c.answer) for c in chunks
    ]
    try:
        ranked = await _maybe_await(fn(query, passages, top_k=keep))
    except RerankerUnavailable as exc:
        # 重排是可选增强：上游故障时退回未重排的混合结果，
        # 不能把整次检索（乃至 query_faq 工具）拖垮。
        logger.warning("重排不可用，退回混合检索结果：%s", exc)
        meta["rerank_applied"] = False
        meta["top_score"] = None
        return chunks[:keep], meta

    # 不信任上游索引范围：越界索引直接丢弃，避免一个坏响应打断整次检索。
    valid = [
        (idx, float(score))
        for idx, score in ranked
        if 0 <= idx < len(chunks)
    ]
    reranked = [chunks[idx] for idx, _ in valid][:keep]
    scores = [score for _, score in valid][:keep]
    top_score = scores[0] if scores else None

    if s.rerank_arrangement_strategy == "first_last":
        reranked = _arrange_first_last(reranked)

    meta["rerank_applied"] = True
    meta["top_score"] = top_score
    logger.info(
        "重排完成 rerank_applied=True rerank_top_score=%s 保留=%d 条",
        top_score,
        len(reranked),
    )
    return reranked, meta


async def retrieve_with_strategy(
    session: AsyncSession,
    query: str,
    strategy: str,
    top_k: int,
    *,
    store: Any | None = None,
    embed: Any | None = None,
    rerank_fn: RerankFn | None = None,
    rewrite_fn: RewriteFn | None = None,
) -> tuple[list[KnowledgeChunk], dict[str, Any]]:
    """按策略检索，返回 (chunks, metadata)。

    除 dense 外都会先做 query 改写（失败时 rewrite_query 自身回退原 query）。
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")

    if strategy == "dense":
        return await _dense_only(
            session, query, top_k, store=store, embed=embed
        )

    rewrite = rewrite_fn or query_rewrite.rewrite_query
    rewritten = await _maybe_await(rewrite(query))

    if strategy == "bm25":
        return await _bm25_only(
            session, rewritten, top_k, store=store
        )
    if strategy == "hybrid":
        return await _hybrid(
            session, rewritten, top_k, store=store, embed=embed
        )
    return await _hybrid_rerank(
        session,
        rewritten,
        top_k,
        store=store,
        embed=embed,
        rerank_fn=rerank_fn,
    )
