"""验收标准 2：BM25 能字面命中包含具体型号的 chunk。

真实依赖只有 Milvus（临时 BM25 collection）与 MySQL 测试库。LLM 不参与：
query 改写通过 ``rewrite_fn`` 注入恒等函数，保证测试确定且快速。

不写真实 ``knowledge`` collection：验收数据只进本用例专属的临时 collection，
``finally`` 中删除，避免污染线上知识库、也避免 collection 泄漏。
"""

import time
from uuid import uuid4

from app.config import get_settings
from app.knowledge.chunk import Chunk
from app.knowledge.strategies import retrieve_with_strategy
from app.knowledge.vector_store import VectorStore
from app.repositories import knowledge_chunk as repo


async def _identity(query: str) -> str:
    """替代真实 query 改写：验收只关心 BM25 的字面匹配能力。"""
    return query


async def test_bm25_matches_model_number(db_session):
    """问带具体型号的问题，BM25 召回包含该字符串的 chunk。"""
    chunk = Chunk(
        "product:abc123",
        "商品参数",
        "型号 ABC-123 支持 5G",
        "本产品型号为 ABC-123，支持 5G 网络。",
        "商品参数 > 型号",
        "faq",
    )

    # MySQL 侧：写入并标为已向量化，_load_chunks 只认 VECTORIZED 行。
    # 不 commit：fixture 负责回滚，同 session 内 flush 的行已可见。
    await repo.upsert_chunks(db_session, [chunk])
    await repo.mark_vectorized(
        db_session, [chunk.chunk_id], [chunk.chunk_id]
    )

    settings = get_settings()
    store = VectorStore(
        collection=f"acceptance_bm25_{uuid4().hex[:8]}",
        support_bm25=True,
    )
    try:
        store.ensure_collection()
        # 零向量即可：本用例只走 BM25 的 sparse_vector 一路。
        store.upsert(
            [
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.embedding_text,
                    "vector": [0.0] * settings.embedding_dim,
                }
            ]
        )
        # 与 tests/test_vector_store.py / 探测用例一致：等待索引可见。
        time.sleep(0.5)

        chunks, meta = await retrieve_with_strategy(
            db_session,
            "型号 ABC-123",
            "bm25",
            top_k=3,
            store=store,
            rewrite_fn=_identity,
        )

        # 对照：无共同词元的 query 不应召回该 chunk，证明不是“全量返回”。
        unrelated_ids = store.bm25_search("完全不相关的退货政策", top_k=3)
    finally:
        store.drop_collection()

    assert meta["strategy"] == "bm25"
    assert meta["query_rewritten"] == "型号 ABC-123"
    ids = [c.chunk_id for c in chunks]
    assert chunk.chunk_id in ids, "BM25 应精确匹配型号字符串"
    assert chunk.chunk_id not in unrelated_ids, "无关 query 不应命中该 chunk"
