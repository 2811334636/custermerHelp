"""离线建库 CLI。

用法：uv run python -m jobs.build_knowledge
"""

import asyncio
from pathlib import Path

from app.config import get_settings
from app.db.base import AsyncSessionLocal
from app.knowledge.builder import build
from app.knowledge.sources.faq_table import load_faq_chunks
from app.knowledge.sources.markdown import load_markdown_dir
from app.knowledge.sources.mined import load_mined_chunks

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"


async def main() -> None:
    s = get_settings()
    async with AsyncSessionLocal() as session:
        md = load_markdown_dir(
            KNOWLEDGE_DIR,
            content_type="policy",
            max_chars=s.chunk_max_chars,
            overlap_chars=s.chunk_overlap_chars,
        )
        faq = await load_faq_chunks(session)
        mined = await load_mined_chunks(session)

        chunks = md + faq + mined
        print(f"切分得到 {len(chunks)} 块（Markdown {len(md)} / FAQ {len(faq)} / 挖掘 {len(mined)}）")

        report = await build(session, chunks)
        print(
            f"完成：总数 {report.chunks_total}，本次待办 {report.chunks_pending}，"
            f"本次写入 {report.vectors_written}"
        )


if __name__ == "__main__":
    asyncio.run(main())
