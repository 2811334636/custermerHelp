import pytest

from app.knowledge.chunk import Chunk
from app.repositories import knowledge_chunk as repo


def make_chunk(
    cid="policy.md:0",
    answer="内容。",
    content_type="policy",
    category="分类",
    questions="标题",
    section_path="分类 > 标题",
    is_key_clause=False,
    prev_chunk_id=None,
    next_chunk_id=None,
):
    return Chunk(
        chunk_id=cid,
        category=category,
        questions=questions,
        answer=answer,
        section_path=section_path,
        content_type=content_type,
        is_key_clause=is_key_clause,
        prev_chunk_id=prev_chunk_id,
        next_chunk_id=next_chunk_id,
    )


async def test_new_chunks_are_pending(db_session):
    await repo.upsert_chunks(db_session, [make_chunk()])
    pending = await repo.list_pending(db_session)
    assert [p.chunk_id for p in pending] == ["policy.md:0"]
    rows = await repo.get_by_ids(db_session, ["policy.md:0"])
    assert rows[0].status == "待向量化"
    assert rows[0].vector_id is None


async def test_empty_upsert_is_noop(db_session):
    before_all = await repo.count_all(db_session)
    await repo.upsert_chunks(db_session, [])
    assert await repo.count_all(db_session) == before_all


async def test_same_hash_keeps_vectorized_status(db_session):
    await repo.upsert_chunks(db_session, [make_chunk()])
    await repo.mark_vectorized(db_session, ["policy.md:0"], ["policy.md:0"])
    await repo.upsert_chunks(db_session, [make_chunk()])  # 内容未变
    assert await repo.list_pending(db_session) == []
    # 状态与 vector_id 一个都没被碰过：这才是重跑幂等的关键。
    row = (await repo.get_by_ids(db_session, ["policy.md:0"]))[0]
    assert (row.status, row.vector_id) == ("已向量化", "policy.md:0")


async def test_same_hash_refreshes_non_hashed_metadata(db_session):
    """hash 只覆盖 category/questions/answer；其余元数据变了也要落库。

    同 hash 时只保留状态机字段（status/vector_id），其余列照常刷新。
    """
    await repo.upsert_chunks(db_session, [make_chunk()])
    await repo.mark_vectorized(db_session, ["policy.md:0"], ["policy.md:0"])
    await repo.upsert_chunks(
        db_session,
        [
            make_chunk(
                section_path="分类 > 标题 > 新章节",
                content_type="faq",
                is_key_clause=True,
                prev_chunk_id="policy.md:-1",
                next_chunk_id="policy.md:1",
            )
        ],
    )
    row = (await repo.get_by_ids(db_session, ["policy.md:0"]))[0]
    assert row.section_path == "分类 > 标题 > 新章节"
    assert row.content_type == "faq"
    assert row.is_key_clause is True
    assert row.prev_chunk_id == "policy.md:-1"
    assert row.next_chunk_id == "policy.md:1"
    # 向量内容（category/questions/answer）没变，状态机字段必须原样保留
    assert (row.status, row.vector_id) == ("已向量化", "policy.md:0")


async def test_changed_hash_resets_to_pending(db_session):
    await repo.upsert_chunks(db_session, [make_chunk()])
    await repo.mark_vectorized(db_session, ["policy.md:0"], ["policy.md:0"])
    await repo.upsert_chunks(db_session, [make_chunk(answer="改过的内容。")])
    assert [p.chunk_id for p in await repo.list_pending(db_session)] == ["policy.md:0"]
    row = (await repo.get_by_ids(db_session, ["policy.md:0"]))[0]
    assert row.answer == "改过的内容。"
    assert row.vector_id is None


async def test_changed_hash_updates_all_mutable_metadata(db_session):
    await repo.upsert_chunks(db_session, [make_chunk()])
    await repo.mark_vectorized(db_session, ["policy.md:0"], ["policy.md:0"])
    changed = make_chunk(
        answer="新答案。",
        category="新分类",
        questions="新问题",
        section_path="新分类 > 新问题",
        content_type="faq",
        is_key_clause=True,
        prev_chunk_id="policy.md:-1",
        next_chunk_id="policy.md:1",
    )
    await repo.upsert_chunks(db_session, [changed])
    row = (await repo.get_by_ids(db_session, ["policy.md:0"]))[0]
    assert row.category == "新分类"
    assert row.questions == "新问题"
    assert row.answer == "新答案。"
    assert row.section_path == "新分类 > 新问题"
    assert row.content_type == "faq"
    assert row.is_key_clause is True
    assert row.prev_chunk_id == "policy.md:-1"
    assert row.next_chunk_id == "policy.md:1"
    assert row.status == "待向量化"
    assert row.vector_id is None


async def test_upsert_rerun_does_not_duplicate(db_session):
    await repo.upsert_chunks(db_session, [make_chunk("a:0")])
    await repo.upsert_chunks(db_session, [make_chunk("a:0")])
    assert [r.chunk_id for r in await repo.get_by_ids(db_session, ["a:0"])] == ["a:0"]


async def test_list_pending_is_ordered_by_chunk_id(db_session):
    await repo.upsert_chunks(
        db_session, [make_chunk("z:0"), make_chunk("a:0"), make_chunk("m:0")]
    )
    pending = await repo.list_pending(db_session)
    assert [p.chunk_id for p in pending] == ["a:0", "m:0", "z:0"]


async def test_mark_vectorized_sets_status_and_vector_id(db_session):
    await repo.upsert_chunks(db_session, [make_chunk("a:0"), make_chunk("a:1")])
    await repo.mark_vectorized(db_session, ["a:0", "a:1"], ["v0", "v1"])
    rows = {r.chunk_id: r for r in await repo.get_by_ids(db_session, ["a:0", "a:1"])}
    assert (rows["a:0"].status, rows["a:0"].vector_id) == ("已向量化", "v0")
    assert (rows["a:1"].status, rows["a:1"].vector_id) == ("已向量化", "v1")


async def test_mark_vectorized_rejects_mismatched_lengths(db_session):
    await repo.upsert_chunks(db_session, [make_chunk("a:0")])
    with pytest.raises(ValueError, match="长度"):
        await repo.mark_vectorized(db_session, ["a:0"], [])
    # 出错时不能留下半截写入
    row = (await repo.get_by_ids(db_session, ["a:0"]))[0]
    assert row.status == "待向量化"


async def test_mark_vectorized_rejects_unknown_chunk_id(db_session):
    await repo.upsert_chunks(db_session, [make_chunk("a:0")])
    with pytest.raises(ValueError, match="ghost:0"):
        await repo.mark_vectorized(db_session, ["a:0", "ghost:0"], ["v0", "v1"])
    # 整批都不写：未知 id 说明写入顺序有 bug，不能静默跳过
    row = (await repo.get_by_ids(db_session, ["a:0"]))[0]
    assert row.status == "待向量化"
    assert row.vector_id is None


async def test_mark_vectorized_empty_is_noop(db_session):
    await repo.upsert_chunks(db_session, [make_chunk("a:0")])
    await repo.mark_vectorized(db_session, [], [])
    assert [p.chunk_id for p in await repo.list_pending(db_session)] == ["a:0"]


async def test_counts(db_session):
    # 用增量断言，不依赖库里的绝对条数（其他测试可能 commit 过数据）
    before_all = await repo.count_all(db_session)
    before_vec = await repo.count_vectorized(db_session)
    await repo.upsert_chunks(db_session, [make_chunk("a:0"), make_chunk("a:1")])
    await repo.mark_vectorized(db_session, ["a:0"], ["a:0"])
    assert await repo.count_all(db_session) == before_all + 2
    assert await repo.count_vectorized(db_session) == before_vec + 1


async def test_counts_are_ints(db_session):
    assert isinstance(await repo.count_all(db_session), int)
    assert isinstance(await repo.count_vectorized(db_session), int)


async def test_get_by_ids_is_deterministic_by_chunk_id(db_session):
    await repo.upsert_chunks(db_session, [make_chunk("a:0"), make_chunk("a:1")])
    rows = await repo.get_by_ids(db_session, ["a:1", "a:0"])
    # 仓储显式按 chunk_id 排序，与传入顺序无关；调用方若要自定义顺序需自行排
    assert [r.chunk_id for r in rows] == ["a:0", "a:1"]


async def test_get_by_ids_empty_returns_empty(db_session):
    assert await repo.get_by_ids(db_session, []) == []


async def test_get_by_ids_unknown_ids_are_absent(db_session):
    await repo.upsert_chunks(db_session, [make_chunk("a:0")])
    rows = await repo.get_by_ids(db_session, ["a:0", "ghost:0"])
    assert [r.chunk_id for r in rows] == ["a:0"]
