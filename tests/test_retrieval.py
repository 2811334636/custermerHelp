import logging

import pytest

from app.knowledge.chunk import Chunk
from app.knowledge.retrieval import search
from app.repositories import knowledge_chunk as repo


class FakeStore:
    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def search(self, vector, top_k):
        self.calls.append((vector, top_k))
        return self.ids[:top_k]


async def fake_embed(texts):
    return [[0.0] * 1024 for _ in texts]


def c(cid, answer="内容。"):
    return Chunk(cid, "分类", "标题", answer, "分类 > 标题", "policy")


async def test_search_returns_chunks_in_milvus_order(db_session):
    # 不 commit：upsert_chunks 只 flush，同一 session 内 search 的 SELECT
    # 能看见未提交行；commit 会击穿 db_session 的回滚夹具并泄漏数据。
    await repo.upsert_chunks(db_session, [c("a:0"), c("a:1")])

    out = await search(db_session, "问题", store=FakeStore(["a:1", "a:0"]), embed=fake_embed)
    assert [x.chunk_id for x in out] == ["a:1", "a:0"]


async def test_empty_hits_returns_empty_list(db_session):
    out = await search(db_session, "问题", store=FakeStore([]), embed=fake_embed)
    assert out == []


async def test_vector_store_unavailable_propagates(db_session):
    from app.knowledge.vector_store import VectorStoreUnavailable

    class Boom:
        def search(self, v, k):
            raise VectorStoreUnavailable("down")

    with pytest.raises(VectorStoreUnavailable):
        await search(db_session, "问题", store=Boom(), embed=fake_embed)


async def test_orphan_vector_ids_are_skipped_with_a_warning(db_session, caplog):
    """Milvus 有、MySQL 没有的 id（孤儿向量）跳过，但必须留下警告日志。"""
    await repo.upsert_chunks(db_session, [c("a:0")])

    with caplog.at_level(logging.WARNING, logger="app.knowledge.retrieval"):
        out = await search(
            db_session, "问题", store=FakeStore(["orphan:1", "a:0"]), embed=fake_embed
        )

    assert [x.chunk_id for x in out] == ["a:0"]
    assert "orphan:1" in caplog.text
