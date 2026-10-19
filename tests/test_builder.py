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
    def __init__(self):
        self.data: dict[str, list[float]] = {}

    def ensure_collection(self):
        pass

    def upsert(self, ids, vectors):
        for i, v in zip(ids, vectors):
            self.data[i] = v

    def count(self):
        return len(self.data)


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


async def test_build_writes_mysql_then_vectors(builder_session):
    store = FakeStore()
    chunks = [chunk(f"{PREFIX}0"), chunk(f"{PREFIX}1")]
    report = await build(builder_session, chunks, store=store, embed=fake_embed)
    assert report.chunks_total == 2
    assert report.chunks_pending == 2
    assert report.vectors_written == 2
    assert store.count() == 2
    assert await _vectorized(builder_session) == 2


async def test_rerun_is_idempotent(builder_session):
    store = FakeStore()
    calls = {"n": 0}

    async def counting_embed(texts):
        calls["n"] += 1
        return [[float(len(t))] * 1024 for t in texts]

    chunks = [chunk(f"{PREFIX}0"), chunk(f"{PREFIX}1")]
    await build(builder_session, chunks, store=store, embed=counting_embed)
    first = calls["n"]
    assert first == 1  # 2 条在一个默认批次内
    report = await build(
        builder_session, chunks, store=store, embed=counting_embed
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
        )

    assert store.count() == 2
    assert await _vectorized(builder_session) == 2

    report = await build(
        builder_session, chunks, store=store, embed=fake_embed, batch_size=2
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
