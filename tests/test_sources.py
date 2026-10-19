from sqlalchemy import func, select

from app.db.models import Faq, MinedQaStaging
from app.knowledge.sources.faq_table import load_faq_chunks
from app.knowledge.sources.markdown import load_markdown_dir
from app.knowledge.sources.mined import load_mined_chunks


def test_markdown_dir_scans_md_files(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n内容甲。\n", encoding="utf-8")
    (tmp_path / "b.md").write_text("# B\n\n内容乙。\n", encoding="utf-8")
    (tmp_path / "c.txt").write_text("忽略", encoding="utf-8")

    chunks = load_markdown_dir(tmp_path, max_chars=400, overlap_chars=80)
    answers = {c.answer for c in chunks}
    assert answers == {"内容甲。", "内容乙。"}
    assert {c.content_type for c in chunks} == {"policy"}


def test_markdown_source_key_is_filename(tmp_path):
    (tmp_path / "售后政策.md").write_text("# 政策\n\n七天无理由退货。\n", encoding="utf-8")
    chunks = load_markdown_dir(tmp_path, max_chars=400, overlap_chars=80)
    assert chunks[0].chunk_id.startswith("售后政策.md:")


async def test_faq_source_maps_fields(db_session):
    total = await db_session.scalar(select(func.count()).select_from(Faq))
    chunks = await load_faq_chunks(db_session)
    assert len(chunks) == total
    c = next(c for c in chunks if "退货政策" in c.questions)
    assert c.content_type == "faq"
    assert c.category == "售后"
    assert c.chunk_id.startswith("faq:")


async def test_mined_source_only_reads_pending(db_session):
    pending_before = await db_session.scalar(
        select(func.count()).select_from(MinedQaStaging).where(
            MinedQaStaging.status == "待入库"
        )
    )

    db_session.add(MinedQaStaging(question="能开专票吗", answer="可以。", category="咨询",
                                  status="待入库"))
    db_session.add(MinedQaStaging(question="已入库的", answer="x", category="咨询",
                                  status="已入库"))
    await db_session.flush()

    chunks = await load_mined_chunks(db_session)
    assert len(chunks) == pending_before + 1
    pending = next(c for c in chunks if c.questions == "能开专票吗")
    assert pending.content_type == "mined"
    assert all(c.questions != "已入库的" for c in chunks)
