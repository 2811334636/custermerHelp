"""在线语义检索：query → embedding → Milvus ids → MySQL 行。"""

import asyncio
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import KnowledgeChunk
from app.knowledge.embedding import embed_texts
from app.knowledge.vector_store import VectorStore
from app.repositories import knowledge_chunk as repo

logger = logging.getLogger(__name__)


async def search(
    session: AsyncSession,
    query: str,
    top_k: int | None = None,
    *,
    store=None,
    embed=None,
) -> list[KnowledgeChunk]:
    """按语义召回知识块，顺序严格跟随 Milvus 的相似度排名。

    store / embed 仅用于注入测试替身；生产路径用真实 Milvus 与 embedding。
    孤儿向量（Milvus 有、MySQL 无）会被跳过并记一条警告——它意味着双写
    不一致，静默吞掉会让缺口永远发现不了。
    """
    s = get_settings()
    if store is None:
        # 构造 MilvusClient 是同步阻塞调用（不可达时会等约 10s）。放进线程，
        # 才能让 ToolRegistry 的 asyncio.wait_for(3s) 真正打断它（final review C1）。
        store = await asyncio.to_thread(VectorStore)
    if embed is None:
        embed = embed_texts

    query_vector = (await embed([query]))[0]
    limit = top_k if top_k is not None else s.retrieval_top_k
    # 同理：pymilvus 的 search 也是同步网络调用，必须离开事件循环。
    ids = await asyncio.to_thread(store.search, query_vector, limit)
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
    # 严格保持 Milvus 顺序，不能用 get_by_ids 的 chunk_id 排序。
    return [by_id[i] for i in ids if i in by_id]
