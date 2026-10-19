"""校验 MySQL 与 Milvus 的条数一致（spec §12.2，final review M9）。

用法：uv run python -m scripts.check_kb_consistency

断言 ``COUNT(status='已向量化')`` == Milvus 集合实体数，并打印两个数字。
不一致时 exit 1。这是每次建库后可以一键执行的轻量验收，不依赖人工。
"""

import asyncio
import sys

from app.config import get_settings
from app.db.base import AsyncSessionLocal, engine
from app.knowledge.vector_store import VectorStore, VectorStoreUnavailable
from app.repositories import knowledge_chunk as repo


async def main() -> int:
    s = get_settings()
    try:
        async with AsyncSessionLocal() as session:
            db_count = await repo.count_vectorized(session)
            total = await repo.count_all(session)
            pending = total - db_count
            try:
                store = await asyncio.to_thread(VectorStore)
                milvus_count = await asyncio.to_thread(store.count)
            except VectorStoreUnavailable as exc:
                print(f"错误：无法连接 Milvus（{s.milvus_uri}）：{exc}", file=sys.stderr)
                return 1
    finally:
        await engine.dispose()

    print(
        f"MySQL 已向量化：{db_count}（总数 {total}，待向量化 {pending}）\n"
        f"Milvus 集合 {s.milvus_collection!r} 实体数：{milvus_count}"
    )
    if db_count != milvus_count:
        print(
            "不一致：MySQL 已向量化条数与 Milvus 实体数不同。\n"
            "请重跑 jobs.build_knowledge；若集合/库配错，请检查 "
            "APP_MILVUS_COLLECTION 是否与该库建库时一致。",
            file=sys.stderr,
        )
        return 1
    print("一致 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
