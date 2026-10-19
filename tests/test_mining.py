from sqlalchemy import select

from app.db.models import Conversation, KnowledgeChunk, Message, MinedQaStaging
from app.knowledge.chunk import Chunk
from app.knowledge.dedupe import normalize
from app.repositories import knowledge_chunk as repo
from jobs.mine_knowledge import (
    MINING_BATCH_MESSAGES,
    chunk_messages,
    dedupe_and_import,
    load_unmined_conversations,
    parse_extraction,
)


def test_normalize_strips_punct_and_space():
    assert normalize(" 邮费 是多少？ ") == "邮费是多少"


def test_normalize_is_case_insensitive():
    assert normalize("REFUND Policy") == "refundpolicy"


async def test_dedupe_within_batch(db_session):
    first = MinedQaStaging(question="邮费是多少？", answer="A", category="咨询")
    duplicate = MinedQaStaging(question=" 邮费是多少 ", answer="A", category="咨询")
    db_session.add_all([first, duplicate])
    await db_session.flush()
    imported, dropped = await dedupe_and_import(db_session)
    assert (imported, dropped) == (1, 1)
    assert (first.status, duplicate.status) == ("已入库", "已丢弃")


async def test_dedupe_against_existing_chunks(db_session):
    await repo.upsert_chunks(db_session, [
        Chunk("faq:1", "咨询", "邮费是多少", "已有的答案。", "咨询", "faq"),
    ])
    db_session.add(MinedQaStaging(question="邮费是多少？", answer="新答案", category="咨询"))
    await db_session.flush()

    imported, dropped = await dedupe_and_import(db_session)
    assert (imported, dropped) == (0, 1)


async def test_imported_rows_become_pending_chunks(db_session):
    db_session.add(MinedQaStaging(question="能开专票吗", answer="可以。", category="咨询"))
    await db_session.flush()
    await dedupe_and_import(db_session)

    rows = list((await db_session.execute(select(KnowledgeChunk))).scalars())
    assert len(rows) == 1
    assert rows[0].content_type == "mined"
    assert rows[0].status == "待向量化"

    staging = list((await db_session.execute(select(MinedQaStaging))).scalars())
    assert staging[0].status == "已入库"


async def test_punctuation_only_question_is_discarded(db_session):
    db_session.add(MinedQaStaging(question="？！。", answer="无效", category="咨询"))
    await db_session.flush()
    imported, dropped = await dedupe_and_import(db_session)
    assert (imported, dropped) == (0, 1)

    rows = list((await db_session.execute(select(KnowledgeChunk))).scalars())
    assert rows == []


async def test_dedupe_import_reuses_existing_chunk_and_keeps_vectorized_status(db_session):
    """入库必须走 upsert_chunks：同 id 同 hash 的已向量化块不能被重置。

    构造：上一次运行已经写好 ``mined:{id}`` 并完成向量化，但 staging 行仍是
    「待入库」（模拟事务边界之间中断）。已有行的 questions 故意放一个不会与本次
    抽取撞车的历史值，好让去重检查放行、真正走到 upsert 分支；content_hash 与
    待入库内容一致。
    """
    row = MinedQaStaging(question="能开专票吗", answer="可以。", category="咨询")
    db_session.add(row)
    await db_session.flush()

    target = Chunk(
        chunk_id=f"mined:{row.id}",
        category="咨询",
        questions="能开专票吗",
        answer="可以。",
        section_path="咨询",
        content_type="mined",
    )
    await repo.upsert_chunks(db_session, [target])
    await repo.mark_vectorized(db_session, [target.chunk_id], ["vec-1"])

    existing = (await repo.get_by_ids(db_session, [target.chunk_id]))[0]
    existing.questions = "历史问题"
    await db_session.flush()

    imported, dropped = await dedupe_and_import(db_session)
    assert (imported, dropped) == (1, 0)

    after = (await repo.get_by_ids(db_session, [target.chunk_id]))[0]
    # 可观察行为：hash 没变 → 状态与 vector_id 原样保留，元数据刷新为本次内容。
    assert after.status == "已向量化"
    assert after.vector_id == "vec-1"
    assert after.questions == "能开专票吗"


async def test_rerun_guard_skips_conversation_with_staging_row(db_session):
    """任何 staging 行引用过的会话都不再喂给 LLM（重跑幂等的前提）。"""
    c1 = Conversation(session_uuid="task10-guard-1", user_id="u1")
    c2 = Conversation(session_uuid="task10-guard-2", user_id="u2")
    db_session.add_all([c1, c2])
    await db_session.flush()

    db_session.add_all([
        Message(conversation_id=c1.id, role="user", content="会话一的问题"),
        Message(conversation_id=c1.id, role="assistant", content="会话一的回答"),
        Message(conversation_id=c2.id, role="user", content="会话二的问题"),
        MinedQaStaging(
            question="旧问题", answer="旧答案", category="咨询", conversation_id=c1.id
        ),
    ])
    await db_session.flush()

    result = await load_unmined_conversations(db_session)

    assert c1.id not in result
    assert c2.id in result
    assert result[c2.id] == ["user: 会话二的问题"]


def test_chunk_messages_respects_limit_and_order():
    messages = [f"m{i}" for i in range(MINING_BATCH_MESSAGES + 5)]
    batches = list(chunk_messages(messages))

    assert [len(b) for b in batches] == [MINING_BATCH_MESSAGES, 5]
    assert batches[0][0] == "m0"
    assert batches[1][-1] == f"m{MINING_BATCH_MESSAGES + 4}"


def test_parse_extraction_strips_json_code_fence():
    raw = (
        '```json\n'
        '[{"question": "邮费是多少？", "answer": "包邮。", "category": "配送"}]\n'
        '```'
    )
    assert parse_extraction(raw) == [
        {"question": "邮费是多少？", "answer": "包邮。", "category": "配送"}
    ]


def test_parse_extraction_handles_fence_without_language():
    raw = '```\n[{"question": "q", "answer": "a"}]\n```'
    assert parse_extraction(raw) == [
        {"question": "q", "answer": "a", "category": "其他"}
    ]


def test_parse_extraction_bad_json_returns_empty_and_warns(caplog):
    assert parse_extraction("这不是 JSON") == []
    assert "这不是 JSON" in caplog.text


def test_parse_extraction_drops_incomplete_items():
    raw = '[{"question": "q"}, {"answer": "a"}, {"question": "q2", "answer": "a2"}]'
    assert parse_extraction(raw) == [
        {"question": "q2", "answer": "a2", "category": "其他"}
    ]
