"""在线检索入口：query → 检索策略 → Milvus ids → MySQL 行 + 拒答控制。

策略本身在 `app.knowledge.strategies`；本模块负责：
- 构造/注入基础设施（VectorStore、embedding）；
- 把 Milvus 返回的 id 还原成 MySQL 行（跳过孤儿/过期行并留痕）；
- 拒答机制（空结果 / 重排低分）与低置信度问题入池。
"""

import asyncio
import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import KnowledgeChunk
from app.knowledge.embedding import embed_texts
from app.knowledge.vector_store import VectorStore
from app.repositories import knowledge_chunk as repo
from app.repositories import low_confidence_question as lcq_repo

logger = logging.getLogger(__name__)

# 拒答原因：与 low_confidence_questions.reason 的取值一致。
REASON_NO_RESULTS = "no_results"
REASON_LOW_SCORE = "low_score"


async def _load_chunks(
    session: AsyncSession, ids: list[str]
) -> list[KnowledgeChunk]:
    """把 Milvus 的 chunk id 还原成 MySQL 行，严格保持传入顺序。

    孤儿向量（Milvus 有、MySQL 无）会被跳过并记一条警告——它意味着双写
    不一致，静默吞掉会让缺口永远发现不了。
    """
    if not ids:
        return []

    rows = await repo.get_by_ids(session, ids, status=repo.VECTORIZED)
    by_id = {r.chunk_id: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        # 区分两种缺失：MySQL 根本没有这行（孤儿向量），或行还在但状态不是
        # 【已向量化】（内容变更后、重跑补齐前的窗口）。前者是双写不一致，
        # 后者只是暂时不可用；两者都要留痕，不能静默跳过。
        stale_rows = await repo.get_by_ids(session, missing)
        stale_ids = [r.chunk_id for r in stale_rows]
        orphan_ids = [i for i in missing if i not in set(stale_ids)]
        if stale_ids:
            logger.warning(
                "检索到 %d 个未向量化的知识块（内容已变、等待重跑），已跳过：%s",
                len(stale_ids),
                ", ".join(stale_ids),
            )
        if orphan_ids:
            logger.warning(
                "检索到 %d 个孤儿向量（MySQL 无对应行），已跳过：%s",
                len(orphan_ids),
                ", ".join(orphan_ids),
            )
    # 严格保持检索顺序，不能用 get_by_ids 的 chunk_id 排序。
    return [by_id[i] for i in ids if i in by_id]


async def _refuse(
    session: AsyncSession, query: str, session_id: str, reason: str
) -> None:
    """拒答：问题入低置信度池，供后续知识盲区分析。"""
    await lcq_repo.create(session, query, session_id, reason)
    logger.warning(
        "检索拒答（reason=%s），问题已入低置信度问题池：%s", reason, query
    )


async def search_with_metadata(
    session: AsyncSession,
    query: str,
    top_k: int | None = None,
    *,
    store: Any | None = None,
    embed: Any | None = None,
    strategy: str = "dense",
    session_id: str = "",
) -> tuple[list[KnowledgeChunk], dict[str, Any]]:
    """按策略召回知识块，并执行拒答控制，返回 ``(chunks, meta)``。

    `strategy` 刻意不默认读配置：ch03 的调用方与测试默认期望 dense 行为；
    生产路径（query_faq）显式传 ``settings.retrieval_strategy``。

    store / embed 仅用于注入测试替身；生产路径用真实 Milvus 与 embedding。
    拒答时返回空列表 + 本次策略元数据——面向用户的拒答话术由上层负责。

    ★ 拒答判定只在这里做一次；`search()` 是 ch03 兼容薄包装，委托本函数，
      避免"两条路径各判一次阈值"的分叉。
    """
    s = get_settings()
    if store is None:
        # 构造 MilvusClient 是同步阻塞调用（不可达时会等约 10s）。放进线程，
        # 才能让 ToolRegistry 的 asyncio.wait_for(3s) 真正打断它（final review C1）。
        store = await asyncio.to_thread(VectorStore)
    if embed is None:
        embed = embed_texts

    limit = top_k if top_k is not None else s.retrieval_top_k

    # 延迟导入：strategies 复用本模块的 _load_chunks，模块级互相导入会成环。
    from app.knowledge.strategies import retrieve_with_strategy

    chunks, meta = await retrieve_with_strategy(
        session, query, strategy, limit, store=store, embed=embed
    )

    if not chunks:
        await _refuse(session, query, session_id, REASON_NO_RESULTS)
        return [], meta

    # 只有重排策略有可比较的分数；dense/bm25/hybrid 的 RRF/相似度不可比，
    # 不能拿阈值误杀。
    if meta.get("strategy") == "hybrid_rerank" and meta.get("rerank_applied"):
        top_score = meta.get("top_score")
        if top_score is not None and top_score < s.retrieval_rerank_threshold:
            await _refuse(session, query, session_id, REASON_LOW_SCORE)
            return [], meta

    logger.info(
        "检索完成 strategy=%s 返回=%d 条 top_score=%s",
        meta.get("strategy"),
        len(chunks),
        meta.get("top_score"),
    )
    return chunks, meta


async def search(
    session: AsyncSession,
    query: str,
    top_k: int | None = None,
    *,
    store: Any | None = None,
    embed: Any | None = None,
    strategy: str = "dense",
    session_id: str = "",
) -> list[KnowledgeChunk]:
    """ch03 兼容入口：只返回 chunks，丢弃策略元数据。

    需要 citations / scores 的调用方（query_faq）改用 `search_with_metadata`。
    """
    chunks, _ = await search_with_metadata(
        session,
        query,
        top_k,
        store=store,
        embed=embed,
        strategy=strategy,
        session_id=session_id,
    )
    return chunks
