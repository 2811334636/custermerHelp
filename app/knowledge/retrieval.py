"""在线语义检索：query → embedding → Milvus ids → MySQL 行。"""

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
        store = VectorStore()
    if embed is None:
        embed = embed_texts

    query_vector = (await embed([query]))[0]
    limit = top_k if top_k is not None else s.retrieval_top_k
    ids = store.search(query_vector, limit)
    if not ids:
        return []

    rows = await repo.get_by_ids(session, ids)
    by_id = {r.chunk_id: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        logger.warning(
            "检索到 %d 个孤儿向量（MySQL 无对应行），已跳过：%s",
            len(missing),
            ", ".join(missing),
        )
    # 严格保持 Milvus 顺序，不能用 get_by_ids 的 chunk_id 排序。
    return [by_id[i] for i in ids if i in by_id]
