import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.db.models import KnowledgeChunk
from app.knowledge.builder import build
from app.knowledge.chunk import Chunk, format_embedding_text
from app.repositories import knowledge_chunk as repo

PREFIX = "testbuilder:"


class FakeStore:
    """非 BM25 替身：ch04 起 builder 也会传字典行（携带元数据）。"""

    def __init__(self):
        self.data: dict[str, list[float]] = {}
        self.upserted_data: list[dict] | None = None

    def ensure_collection(self):
        pass

    def upsert(self, ids_or_data, vectors=None):
        self.upserted_data = ids_or_data
        if ids_or_data and isinstance(ids_or_data[0], dict):
            for row in ids_or_data:
                self.data[row["chunk_id"]] = row["vector"]
            return
        for i, v in zip(ids_or_data, vectors):
            self.data[i] = v

    def count(self):
        return len(self.data)


class FakeBm25Store(FakeStore):
    """BM25 store：builder 必须走 dict 形态 upsert，content 非空。"""

    support_bm25 = True

    def __init__(self):
        super().__init__()
        self.upserted_data: list[dict] | None = None
        self.vectors_arg: object = "not-called"

    def upsert(self, ids_or_data, vectors=None):
        self.vectors_arg = vectors
        self.upserted_data = ids_or_data
        for row in ids_or_data:
            self.data[row["chunk_id"]] = row["vector"]


async def fake_embed(texts):
    return [[float(len(t))] * 1024 for t in texts]


def chunk(cid, answer="内容。"):
    return Chunk(cid, "分类", "标题", answer, "分类 > 标题", "policy")


def test_format_embedding_text_matches_chunk_property():
    c = chunk("fmt:0")
    assert format_embedding_text(c.category, c.questions, c.answer) == c.embedding_text


@pytest.fixture
async def builder_session():
    """builder 会 commit（为了真实的重跑语义），所以不能用会回滚的 db_session。

    用独立 session + 唯一 id 前缀，结束后只清自己造的行，不碰其他测试的数据。
    """
    # 每个用例重建 engine：pytest-asyncio 每个用例一个新 loop，复用模块级
    # AsyncSessionLocal 会把上一个 loop 的连接带进来（Event loop is closed）。
    get_settings.cache_clear()
    engine = create_async_engine(get_settings().db_url, echo=False)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with maker() as session:
            yield session
            await session.execute(
                delete(KnowledgeChunk).where(
                    KnowledgeChunk.chunk_id.like(f"{PREFIX}%")
                )
            )
            await session.commit()
    finally:
        await engine.dispose()


async def _vectorized(session) -> int:
    stmt = (
        select(func.count())
        .select_from(KnowledgeChunk)
        .where(
            KnowledgeChunk.status == "已向量化",
            KnowledgeChunk.chunk_id.like(f"{PREFIX}%"),
        )
    )
    return (await session.execute(stmt)).scalar_one()


def ids(chunks) -> set[str]:
    return {c.chunk_id for c in chunks}


async def test_build_writes_mysql_then_vectors(builder_session):
    store = FakeStore()
    chunks = [chunk(f"{PREFIX}0"), chunk(f"{PREFIX}1")]
    report = await build(
        builder_session, chunks, store=store, embed=fake_embed,
        only_ids=ids(chunks),
    )
    assert report.chunks_total == 2
    assert report.chunks_pending == 2
    assert report.vectors_written == 2
    assert store.count() == 2
    assert await _vectorized(builder_session) == 2


async def test_build_writes_dict_rows_with_content_for_bm25_store(builder_session):
    """BM25 模式下必须写 dict + content，否则 sparse_vector 永远为空。"""
    store = FakeBm25Store()
    chunks = [
        chunk(f"{PREFIX}b0", answer="满 99 元包邮。"),
        chunk(f"{PREFIX}b1", answer="7 天无理由退货。"),
    ]
    report = await build(
        builder_session,
        chunks,
        store=store,
        embed=fake_embed,
        only_ids=ids(chunks),
    )

    assert report.vectors_written == 2
    assert store.vectors_arg is None, "BM25 模式不能走旧 upsert(ids, vectors) 签名"
    assert store.upserted_data is not None
    by_id = {row["chunk_id"]: row for row in store.upserted_data}
    assert set(by_id) == ids(chunks)
    for c in chunks:
        row = by_id[c.chunk_id]
        assert row["content"] == format_embedding_text(
            c.category, c.questions, c.answer
        )
        assert row["content"], "content 不能为空——BM25 Function 靠它生成 sparse_vector"
        assert len(row["vector"]) == 1024
        # spec §3.3：元数据必须进 Milvus，过滤才能下推。
        assert row["category"] == c.category
        assert row["section_path"] == c.section_path


async def test_build_writes_dict_rows_with_metadata_for_dense_store(builder_session):
    """非 BM25 模式也要写字典行（含元数据），但不能带 content 字段。"""
    store = FakeStore()
    c = chunk(f"{PREFIX}d0")
    await build(
        builder_session, [c], store=store, embed=fake_embed, only_ids=ids([c])
    )

    assert store.upserted_data == [
        {
            "chunk_id": c.chunk_id,
            "vector": store.data[c.chunk_id],
            "category": c.category,
            "section_path": c.section_path,
        }
    ]


async def test_rerun_is_idempotent(builder_session):
    store = FakeStore()
    calls = {"n": 0}

    async def counting_embed(texts):
        calls["n"] += 1
        return [[float(len(t))] * 1024 for t in texts]

    chunks = [chunk(f"{PREFIX}0"), chunk(f"{PREFIX}1")]
    await build(
        builder_session, chunks, store=store, embed=counting_embed,
        only_ids=ids(chunks),
    )
    first = calls["n"]
    assert first == 1  # 2 条在一个默认批次内
    report = await build(
        builder_session, chunks, store=store, embed=counting_embed,
        only_ids=ids(chunks),
    )  # 同样的输入再跑
    assert calls["n"] == first, "重跑不应再调用 embedding"
    assert report.chunks_total == 2
    assert report.chunks_pending == 0
    assert report.vectors_written == 0
    assert store.count() == 2  # 没有变 4
    assert await _vectorized(builder_session) == 2


async def test_interrupted_build_resumes(builder_session):
    store = FakeStore()
    chunks = [chunk(f"{PREFIX}{i}") for i in range(5)]

    # batch_size=2：第一批写完就触发中断，留下恰好 2 条已向量化。
    with pytest.raises(RuntimeError):
        await build(
            builder_session,
            chunks,
            store=store,
            embed=fake_embed,
            fail_after=2,
            batch_size=2,
            only_ids=ids(chunks),
        )

    assert store.count() == 2
    assert await _vectorized(builder_session) == 2

    report = await build(
        builder_session, chunks, store=store, embed=fake_embed, batch_size=2,
        only_ids=ids(chunks),
    )
    assert report.chunks_total == 5
    assert report.chunks_pending == 3
    assert report.vectors_written == 3  # 补了剩下 3 条（本次调用写入数，非累计）
    assert store.count() == 5  # 没有重复
    assert await _vectorized(builder_session) == 5


@pytest.mark.parametrize("bad", [0, -1])
async def test_non_positive_batch_size_is_rejected(builder_session, bad):
    with pytest.raises(ValueError, match="batch_size"):
        await build(
            builder_session,
            [chunk(f"{PREFIX}0")],
            store=FakeStore(),
            embed=fake_embed,
            batch_size=bad,
        )
    # 校验在任何写入之前完成：该块不应被落库。
    assert await repo.get_by_ids(builder_session, [f"{PREFIX}0"]) == []
