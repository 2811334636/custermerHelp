"""从历史客服对话挖掘问答对。

用法：uv run python -m jobs.mine_knowledge

流程（spec §8）：
  ① 选未处理会话（已有 staging 行的会话跳过 ⇒ 重跑不重复挖）
  ② 单个会话内分批（每批至多 20 条消息）喂 LLM，抽 (question, answer, category)
  ③ 写 mined_qa_staging，status='待入库'，conversation_id 记真实会话
  ④ 整体去重入库：规范化文本 → 与 knowledge_chunks 现有记录精确比较
       存活者 → knowledge_chunks（content_type='mined'，chunk_id='mined:{staging_id}'）
       staging 行标 '已入库' / '已丢弃'
  ⑤ 新 chunk 保持 status='待向量化'，由下一次 build_knowledge 捡起
"""

import asyncio
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select

from app.db.base import AsyncSessionLocal, engine
from app.db.models import KnowledgeChunk, Message, MinedQaStaging
from app.knowledge.chunk import Chunk
from app.knowledge.dedupe import normalize
from app.providers import get_chat_model
from app.repositories import knowledge_chunk as repo

logger = logging.getLogger(__name__)

# 一次 LLM 调用最多喂多少条消息。★ 必须按会话内部切批：如果像原计划那样把
# 多个会话拼成一批，抽出的问答只能记到 batch[0]，其余会话永远不会被标记，
# 每次 cron 都会重挖、重复付费。
MINING_BATCH_MESSAGES = 20

EXTRACT_PROMPT = """你从客服对话里抽取可复用的问答知识。

只抽取**通用、可复用**的知识（政策、流程、规则）。
不要抽取一次性的个别订单信息（具体订单号、个人地址、单次物流）。
没有可抽取的内容就返回空数组。

输出 JSON 数组，每项形如：
{"question": "用户可能怎么问", "answer": "平台的标准答复", "category": "售后/售前/配送/发票"}

只输出 JSON，不要任何解释。"""

# 模型偶尔会用 Markdown 代码块包住 JSON；去掉围栏再解析。
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_extraction(raw: str) -> list[dict]:
    """把模型输出解析成问答对；坏 JSON / 非数组只记 warning 并丢弃本批。"""
    text = raw.strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        logger.warning("抽取结果不是合法 JSON，丢弃本批（前 200 字）：%s", text[:200])
        return []

    if not isinstance(data, list):
        logger.warning("抽取结果不是 JSON 数组，丢弃本批（前 200 字）：%s", text[:200])
        return []

    pairs: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        answer = item.get("answer")
        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(answer, str) or not answer.strip():
            continue
        category = item.get("category")
        if not isinstance(category, str) or not category.strip():
            category = "其他"
        pairs.append({"question": question, "answer": answer, "category": category})
    return pairs


def chunk_messages(
    messages: list[str], size: int = MINING_BATCH_MESSAGES
) -> list[list[str]]:
    """按原顺序把消息切成每批至多 ``size`` 条。"""
    return [messages[i : i + size] for i in range(0, len(messages), size)]


async def extract_from_conversation(text: str, *, model=None) -> list[dict]:
    """把一段对话文本交给模型抽取问答对。

    ``model`` 仅供测试注入；默认构造项目的 chat model。
    """
    if model is None:
        model = get_chat_model(temperature=0)
    resp = await model.ainvoke([
        SystemMessage(EXTRACT_PROMPT),
        HumanMessage(text),
    ])
    # langchain-core 1.x：`resp.text` 是 TextAccessor（str 子类），列表形式的
    # content blocks 会被拼接成纯文本，正好满足这里的需求。
    return parse_extraction(str(resp.text))


async def extract_conversation(messages: list[str], *, model) -> list[dict]:
    """按 ``MINING_BATCH_MESSAGES`` 分批抽取同一个会话的所有问答对。"""
    pairs: list[dict] = []
    for batch in chunk_messages(messages):
        pairs.extend(await extract_from_conversation("\n".join(batch), model=model))
    return pairs


async def load_unmined_conversations(session) -> dict[int, list[str]]:
    """按会话 id 顺序返回尚未挖掘过的会话消息（``role: content``）。

    重跑守卫：只要 ``mined_qa_staging`` 里有任何一行引用了该会话，就跳过。
    因为现在每条 staging 行都记录它真正来自的会话，这个判断是准确的。
    """
    already = set(
        (
            await session.execute(
                select(MinedQaStaging.conversation_id).where(
                    MinedQaStaging.conversation_id.is_not(None)
                )
            )
        ).scalars()
    )

    rows = (
        await session.execute(
            select(Message.conversation_id, Message.role, Message.content)
            .where(Message.role.in_(["user", "assistant"]))
            .order_by(Message.conversation_id, Message.id)
        )
    ).all()

    by_conversation: dict[int, list[str]] = {}
    for conversation_id, role, content in rows:
        if conversation_id in already or not content:
            continue
        by_conversation.setdefault(conversation_id, []).append(f"{role}: {content}")
    return by_conversation


async def dedupe_and_import(session) -> tuple[int, int]:
    """把 status='待入库' 的暂存问答去重后写入 knowledge_chunks。

    返回 (入库数, 丢弃数)。只 flush 不 commit：事务边界归调用方。
    """
    staged = list(
        (
            await session.execute(
                select(MinedQaStaging)
                .where(MinedQaStaging.status == "待入库")
                .order_by(MinedQaStaging.id)
            )
        ).scalars()
    )

    existing = list((await session.execute(select(KnowledgeChunk.questions))).scalars())
    seen = {normalize(q) for q in existing if q}

    imported = dropped = 0
    for row in staged:
        key = normalize(row.question)
        # 纯标点/空白问题规范化后是空串：不能当知识入库，否则第一条会变成
        # 真实 chunk、后续全被当成它的重复。直接丢弃。
        if not key:
            row.status = "已丢弃"
            dropped += 1
            continue
        if key in seen:
            row.status = "已丢弃"
            dropped += 1
            continue

        seen.add(key)
        # ★ 走仓储 upsert：hash/状态机/列映射只在 Task 6 那一个地方维护。
        # 同 id 同 hash 的已向量化块会原样保留 status/vector_id。
        chunk = Chunk(
            chunk_id=f"mined:{row.id}",
            category=row.category,
            questions=row.question,
            answer=row.answer,
            section_path=row.category,
            content_type="mined",
        )
        await repo.upsert_chunks(session, [chunk])
        row.status = "已入库"
        imported += 1

    await session.flush()
    return imported, dropped


async def main() -> None:
    try:
        async with AsyncSessionLocal() as session:
            by_conversation = await load_unmined_conversations(session)
            print(f"待挖掘会话：{len(by_conversation)} 个")

            model = get_chat_model(temperature=0)
            processed = 0
            for conversation_id, messages in by_conversation.items():
                try:
                    pairs = await extract_conversation(messages, model=model)
                except Exception:
                    # 单个会话失败不能拖垮整个 cron 任务；本会话不写 staging，
                    # 下次运行会重新尝试。
                    logger.exception("会话 %s 抽取失败，跳过", conversation_id)
                    await session.rollback()
                    continue

                for pair in pairs:
                    session.add(
                        MinedQaStaging(
                            question=pair["question"][:512],
                            answer=pair["answer"],
                            category=pair["category"][:64],
                            conversation_id=conversation_id,
                        )
                    )

                try:
                    # 逐会话提交：成功的会话立刻落库，失败的不影响其他会话。
                    await session.commit()
                    processed += 1
                except Exception:
                    logger.exception("会话 %s 写入暂存表失败，跳过", conversation_id)
                    await session.rollback()

            imported, dropped = await dedupe_and_import(session)
            await session.commit()
            print(
                f"挖掘完成：处理 {processed} 个会话，"
                f"入库 {imported}，去重丢弃 {dropped}"
            )
    finally:
        # 与 jobs/build_knowledge.py 一致：显式释放全局异步引擎，避免解释器
        # 关闭时 aiomysql 连接被 GC 回收而打出 "Event loop is closed"。
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
