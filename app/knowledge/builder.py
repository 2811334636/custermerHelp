from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.knowledge.chunk import Chunk
from app.knowledge.embedding import embed_texts
from app.knowledge.vector_store import VectorStore
from app.repositories import knowledge_chunk as repo


@dataclass(frozen=True)
class BuildReport:
    chunks_total: int
    chunks_pending: int
    vectors_written: int


async def build(
    session: AsyncSession,
    chunks: list[Chunk],
    *,
    store=None,
    embed=embed_texts,
    fail_after: int | None = None,
    batch_size: int | None = None,
) -> BuildReport:
    """双写：MySQL 记【待向量化】→ Milvus upsert → 回填状态。

    `batch_size` 覆盖 Settings.embed_batch_size（测试用，便于构造小批次）。
    `fail_after` 仅测试用：累计写入达到该数后抛异常，模拟进程中断。
    """
    # 先校验再动手：非法批大小不应留下任何已提交的 MySQL 行。
    if batch_size is not None and batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数：{batch_size}")
    effective_batch_size = (
        batch_size if batch_size is not None else get_settings().embed_batch_size
    )

    store = store or VectorStore()
    store.ensure_collection()

    # ① 写 MySQL：新块待向量化；hash 未变保持原状态；hash 变了重置。
    #    先提交，保证"已排队"这件事在进程中断前是持久的。
    await repo.upsert_chunks(session, chunks)
    await session.commit()

    # ② 取待办集（本进程启动时仍待向量化的行；已完成的不会重复处理）
    pending = await repo.list_pending(session)
    if not pending:
        return BuildReport(len(chunks), 0, 0)

    # ③④⑤ 分批向量化并写 Milvus，写完一批回填一批、提交一批。
    written = 0
    for i in range(0, len(pending), effective_batch_size):
        batch = pending[i : i + effective_batch_size]
        # list_pending 返回的是 ORM 行而非领域 Chunk；文本格式必须与
        # Chunk.embedding_text 一致（category/questions/answer 换行拼接），
        # 否则重排后算出的 content_hash 与这里送去向量化的内容会脱节。
        texts = [f"{c.category}\n{c.questions}\n{c.answer}" for c in batch]
        vectors = await embed(texts)

        ids = [c.chunk_id for c in batch]
        store.upsert(ids, vectors)
        # vector_id == chunk_id：Milvus 主键就是 chunk_id。
        await repo.mark_vectorized(session, ids, ids)
        await session.commit()
        written += len(batch)

        # 中断点放在提交之后：已完成的批次必须持久，重跑才只需补剩余。
        if fail_after is not None and written >= fail_after:
            raise RuntimeError(f"模拟中断：已写入 {written} 条")

    return BuildReport(len(chunks), len(pending), written)
