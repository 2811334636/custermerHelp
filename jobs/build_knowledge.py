"""离线建库 CLI。

用法：uv run python -m jobs.build_knowledge
"""

import asyncio
from pathlib import Path

from app.config import get_settings
from app.db.base import AsyncSessionLocal, engine
from app.db.locks import mysql_named_lock
from app.knowledge.builder import build
from app.knowledge.sources.faq_table import load_faq_chunks
from app.knowledge.sources.markdown import load_markdown_dir
from app.knowledge.vector_store import VectorStore

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


async def main() -> None:
    s = get_settings()
    try:
        # cron 重叠/手动重跑互斥：拿不到锁就干净退出，不排队、不半途而废。
        async with mysql_named_lock("knowledge_build") as acquired:
            if not acquired:
                print("另一个建库任务正在运行（knowledge_build 锁被占用），本次退出。")
                return
            async with AsyncSessionLocal() as session:
                md = load_markdown_dir(
                    KNOWLEDGE_DIR,
                    content_type="policy",
                    max_chars=s.chunk_max_chars,
                    overlap_chars=s.chunk_overlap_chars,
                )
                faq = await load_faq_chunks(session)
                # ★ 不再读 mined_qa_staging：mined chunk 由 mine_knowledge 的
                # dedupe_and_import 负责去重入库；这里再读一次会绕过规范化
                # 去重，并让 staging 永远停在「待入库」（final review M7）。
                chunks = md + faq
                print(f"切分得到 {len(chunks)} 块（Markdown {len(md)} / FAQ {len(faq)}）")

                # 创建支持 BM25 的 VectorStore
                store = VectorStore(support_bm25=True)
                report = await build(session, chunks, store=store)
                print(
                    f"完成：总数 {report.chunks_total}，本次待办 {report.chunks_pending}，"
                    f"本次写入 {report.vectors_written}"
                )
    finally:
        # 与 scripts/seed.py 一致：显式释放全局异步引擎，避免解释器关闭时
        # aiomysql 连接被 GC 回收而在 stderr 打出 "Event loop is closed"。
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
