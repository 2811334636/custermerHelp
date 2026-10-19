from sqlalchemy import select

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


def test_markdown_dir_skips_undecodable_file(tmp_path):
    (tmp_path / "good.md").write_text("# 好\n\n内容。\n", encoding="utf-8")
    (tmp_path / "broken.md").write_bytes(b"\xff\xfe\x00bad")
    chunks = load_markdown_dir(tmp_path, max_chars=400, overlap_chars=80)
    assert {c.answer for c in chunks} == {"内容。"}


def test_markdown_dir_is_deterministic_across_runs(tmp_path):
    (tmp_path / "b.md").write_text("# B\n\n乙。\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A\n\n甲。\n", encoding="utf-8")
    first = [c.chunk_id for c in load_markdown_dir(tmp_path, max_chars=400, overlap_chars=80)]
    second = [c.chunk_id for c in load_markdown_dir(tmp_path, max_chars=400, overlap_chars=80)]
    assert first == second
    assert first == sorted(first)


def test_markdown_dir_empty_returns_empty(tmp_path):
    assert load_markdown_dir(tmp_path, max_chars=400, overlap_chars=80) == []


async def test_faq_source_maps_fields(db_session):
    rows = list((await db_session.execute(select(Faq).order_by(Faq.id))).scalars())
    chunks = await load_faq_chunks(db_session)
    assert len(chunks) == len(rows)

    by_id = {c.chunk_id: c for c in chunks}
    for r in rows:
        c = by_id[f"faq:{r.id}"]
        assert c.questions == r.question
        assert c.category == r.category
        assert c.answer == r.answer
        assert c.content_type == "faq"
        assert c.chunk_id == f"faq:{r.id}"

    c = next(c for c in chunks if "退货政策" in c.questions)
    assert c.category == "售后"
    assert c.chunk_id.startswith("faq:")


async def test_mined_source_only_reads_pending(db_session):
    marker = "task4-mined-unique"
    pending = MinedQaStaging(
        question=f"{marker}-待入库", answer="可以。", category="咨询", status="待入库"
    )
    done = MinedQaStaging(
        question=f"{marker}-已入库", answer="x", category="咨询", status="已入库"
    )
    db_session.add_all([pending, done])
    try:
        await db_session.flush()

        chunks = await load_mined_chunks(db_session)
        c = next(c for c in chunks if c.questions == pending.question)
        assert c.questions == pending.question
        assert c.category == pending.category
        assert c.answer == pending.answer
        assert c.content_type == "mined"
        assert c.chunk_id == f"mined:{pending.id}"
        assert all(c.questions != done.question for c in chunks)
    finally:
        await db_session.delete(pending)
        await db_session.delete(done)
        await db_session.flush()
