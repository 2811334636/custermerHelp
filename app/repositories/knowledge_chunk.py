from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import KnowledgeChunk
from app.knowledge.chunk import Chunk

PENDING = "待向量化"
VECTORIZED = "已向量化"


async def upsert_chunks(session: AsyncSession, chunks: list[Chunk]) -> None:
    """按 chunk_id upsert。

    ★ 状态机只由 content_hash 驱动，元数据则每次都刷新：
      - 新 id：插入，状态待向量化；
      - 老 id 且 hash 相同：更新全部元数据列，但保留 status 与 vector_id，
        这样"中断后重跑"才不会把已经灌进 Milvus 的块重新排队；
      - 老 id 且 hash 变化：更新全部元数据，状态重置为待向量化并清空
        vector_id（旧向量对应旧内容，已经失效）。
      - prev/next/section_path/content_type/is_key_clause 不参与 hash，
        但会在重跑之间变化，所以即使 hash 相同也必须写回。

    一次 SELECT 取回所有已存在的行，避免 O(N) 次主键查询。
    本函数只 flush 不 commit：事务边界归调用方，测试 fixture 才能回滚。
    """
    if not chunks:
        return

    ids = [c.chunk_id for c in chunks]
    stmt = select(KnowledgeChunk).where(KnowledgeChunk.chunk_id.in_(ids))
    existing_by_id = {
        row.chunk_id: row for row in (await session.execute(stmt)).scalars()
    }

    for c in chunks:
        existing = existing_by_id.get(c.chunk_id)
        if existing is None:
            row = KnowledgeChunk(
                chunk_id=c.chunk_id,
                category=c.category,
                questions=c.questions,
                answer=c.answer,
                section_path=c.section_path,
                content_type=c.content_type,
                is_key_clause=c.is_key_clause,
                prev_chunk_id=c.prev_chunk_id,
                next_chunk_id=c.next_chunk_id,
                status=PENDING,
                vector_id=None,
                content_hash=c.content_hash,
            )
            session.add(row)
            # 同一批输入里重复的 chunk_id：第二次按"已存在"处理，避免重复 INSERT
            existing_by_id[c.chunk_id] = row
            continue

        hash_changed = existing.content_hash != c.content_hash
        existing.category = c.category
        existing.questions = c.questions
        existing.answer = c.answer
        existing.section_path = c.section_path
        existing.content_type = c.content_type
        existing.is_key_clause = c.is_key_clause
        existing.prev_chunk_id = c.prev_chunk_id
        existing.next_chunk_id = c.next_chunk_id
        existing.content_hash = c.content_hash
        if hash_changed:
            existing.status = PENDING
            existing.vector_id = None

    await session.flush()


async def list_pending(session: AsyncSession) -> list[KnowledgeChunk]:
    stmt = (
        select(KnowledgeChunk)
        .where(KnowledgeChunk.status == PENDING)
        .order_by(KnowledgeChunk.chunk_id)
    )
    return list((await session.execute(stmt)).scalars())


async def mark_vectorized(
    session: AsyncSession, chunk_ids: list[str], vector_ids: list[str]
) -> None:
    """把 chunk 标为已向量化并写入对应 vector_id。

    ★ 先整体校验再写：长度不一致或出现未知 chunk_id 都抛 ValueError。
      未知 id 意味着 builder 的写入顺序有问题，静默跳过会让缺口永远不被发现
      （重跑时该块仍是待向量化，但调用方以为已经完成）。
      一次 SELECT 取回全部行后再校验、再改内存对象。
    """
    if len(chunk_ids) != len(vector_ids):
        raise ValueError(
            f"chunk_ids 与 vector_ids 长度不一致：{len(chunk_ids)} != {len(vector_ids)}"
        )
    if not chunk_ids:
        return

    stmt = select(KnowledgeChunk).where(KnowledgeChunk.chunk_id.in_(chunk_ids))
    rows_by_id = {
        row.chunk_id: row for row in (await session.execute(stmt)).scalars()
    }
    missing = [cid for cid in chunk_ids if cid not in rows_by_id]
    if missing:
        raise ValueError(f"mark_vectorized 收到未知 chunk_id：{', '.join(missing)}")

    for chunk_id, vector_id in zip(chunk_ids, vector_ids):
        row = rows_by_id[chunk_id]
        row.status = VECTORIZED
        row.vector_id = vector_id

    await session.flush()


async def get_by_ids(
    session: AsyncSession, chunk_ids: list[str]
) -> list[KnowledgeChunk]:
    """按 id 批量取行；未知 id 直接缺席，不报错。结果按 chunk_id 排序。"""
    if not chunk_ids:
        return []
    stmt = (
        select(KnowledgeChunk)
        .where(KnowledgeChunk.chunk_id.in_(chunk_ids))
        .order_by(KnowledgeChunk.chunk_id)
    )
    return list((await session.execute(stmt)).scalars())


async def count_all(session: AsyncSession) -> int:
    stmt = select(func.count()).select_from(KnowledgeChunk)
    return (await session.execute(stmt)).scalar_one()


async def count_vectorized(session: AsyncSession) -> int:
    stmt = (
        select(func.count())
        .select_from(KnowledgeChunk)
        .where(KnowledgeChunk.status == VECTORIZED)
    )
    return (await session.execute(stmt)).scalar_one()
