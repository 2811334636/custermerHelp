"""ch02 工具选择 + 结果转述评估集。

用法：uv run python -m evals.run_tool_evals     ← 必须用 -m，不能用路径
判据两条，缺一不可：
  ① 工具选择：模型调了哪些工具、传给工具的参数
  ② 结果转述：拿到工具结果后，模型有没有**如实转述**，有没有补充工具没给的信息

★ 为什么必须有 ②（spec §6.5）：只测 ① 区分不了新旧 Prompt。旧 Prompt 下模型
**照调** query_logistics（工具选择满分），然后拿到真实轨迹**丢弃不用**，回一句
"我这边看不到订单的物流信息，没法帮您查"。所以只断言 tool_calls 的评估集对
本章的核心风险是**盲的**——它会给两个行为截然不同的 Prompt 打同样的满分，
等于一条恒绿的测试。本章真正要防的是"调了但不用"。

★ query_faq 已改为向量检索，本脚本的 FAQ 内容判据基于
  `knowledge_chunks` 的期望答案原文（不再查 `faq` 表的 LIKE），且要求
  该库与它的 Milvus 集合**显式配对**（测试库必须配独立的 *_test 集合）。
  评估库为空、集合为空或两者条数不一致时，脚本以退出码 2 明确失败，
  而不是把生产向量静默 join 到测试库的行上。检索质量仍以
  `evals.run_kb_evals` 为准。

★ 两处结构性改动（详见 evals/harness.py 顶部注释）：
  · 回合**直接驱动生产工作流** `app/workflow`，不再自己复刻一遍；
  · 工具接**真库**（spec §5.6 的 customer_service_test），faq/工单不再一律回
    "暂时不可用"——否则那 5 条用例全落在弱分支上，一条什么都没转述的回复也能过。
"""

import asyncio
import sys
from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.engine import make_url

from app.config import get_settings
from app.db.models import KnowledgeChunk
from app.knowledge.vector_store import VectorStore, VectorStoreUnavailable
from app.providers import get_chat_model
from app.repositories.knowledge_chunk import count_vectorized
from app.tools import build_registry
from evals.harness import (
    PROSE_MIN_RUN,
    apply_prompt_override,
    longest_common_run,
    open_test_db,
    relay_check,
    run_production_turn,
    test_db_url,
)


ROOT = Path(__file__).parent

JUDGE_TEMPLATE = """你在核对一个电商客服机器人有没有如实转述工具查询结果。

工具返回的内容（系统内部文本，可能含"数据库未绑定""暂时不可用"这类**内部故障说明**）：
{tool_output}

客服对用户的回复：
{reply}

请判断两点：
1. 客服有没有把工具返回的**结果如实**传达给用户？工具返回了数据（如运单号、状态、FAQ 条目）
   却说"我这边看不到/查不到"，即**不是**如实传达。
2. 客服有没有**添加工具没有给出的信息**？尤其是"已经帮您办好了/已为您转人工/已创建工单"
   这类**声称动作已完成**的话——工具说办不了时这么说就是编造。反之，工具说办成了而客服
   如实说办成了，不算添加。

注意：把内部故障说明（"数据库未绑定""暂时不可用"）翻译成用户听得懂的**口语说法**
（如"转人工通道暂时没接通""这边暂时查不了"）**不算**添加信息，那正是应该做的事。
只输出一个字：是 或 否。"""


async def check_faq_content(session, case: dict, turn) -> tuple[bool, str]:
    """FAQ 用例的加强判据（final review I2）：基于 `knowledge_chunks` 的期望原文。

    三层，都是确定性的：
      ① 用例标注的期望片段确实存在于当前库的 knowledge_chunks 中
         —— 否则是用例与知识库脱节，必须响亮失败；
      ② 工具输出里出现该片段 —— 证明语义检索真的召回了期望知识块，
         而不是答非所问；
      ③ 回复与该块 answer 的最长公共片段 ≥ PROSE_MIN_RUN —— 证明模型
         把召回内容如实转述给了用户。
    """
    needle = case.get("expect_faq_contains")
    if not needle:
        return True, ""
    stmt = (
        select(KnowledgeChunk)
        .where(KnowledgeChunk.answer.contains(needle))
        .order_by(KnowledgeChunk.chunk_id)
        .limit(1)
    )
    row = (await session.execute(stmt)).scalars().first()
    if row is None:
        return False, (
            f"评估期望「{needle}」在 knowledge_chunks 中不存在——"
            "用例与已建知识库已脱节，请先按 README 建库"
        )
    if needle not in turn.tool_output:
        return False, (
            f"检索没有返回含「{needle}」的知识块（期望 top-3 命中 {row.chunk_id}）"
            "——query_faq 的检索结果不对"
        )
    run = longest_common_run(row.answer, turn.final_text)
    if run < PROSE_MIN_RUN:
        return False, (
            f"回复没转述召回的知识内容（与答案原文最长公共片段仅 {run} 字，"
            f"下限 {PROSE_MIN_RUN}）：{row.answer[:30]}…"
        )
    return True, ""


async def main() -> int:
    # 负对照开关：EVAL_PROMPT_FILE=<旧 prompts.py> 时换掉 System Prompt 跑一遍。
    # 本评估的存在理由就是区分新旧 Prompt，不能真去区分就白搭——见 harness。
    override = apply_prompt_override()
    if override:
        print(f"★ 负对照：System Prompt 已被 {override} 覆盖（不是 app/prompts.py 的现行版本）")

    cases = yaml.safe_load((ROOT / "tool_cases.yaml").read_text(encoding="utf-8"))
    # model 传**未绑工具**的实例：绑工具是生产函数内部自己的事（app/chat.py:154）。
    # 评估若在外部先绑好，测的就不再是生产路径（修复轮 · Important 4）。
    model = get_chat_model(temperature=0)
    judge_model = get_chat_model(temperature=0)

    results = []
    async with open_test_db() as (session, conversation_id):
        # ★ 一库一集合，配置必须显式配对（Ruling 9）。测试库若仍指向生产
        # 集合 `knowledge`，条数可能碰巧相同，但向量与行来自不同库——
        # 这正是最隐蔽的错配，必须在跑用例前挡住。
        db_name = make_url(test_db_url()).database or ""
        collection = get_settings().milvus_collection
        if db_name.endswith("_test") and not collection.endswith("_test"):
            print(
                f"错误：评估库 {db_name} 必须配独立的 Milvus 集合，"
                f"当前 APP_MILVUS_COLLECTION={collection!r}。\n"
                "请用 APP_MILVUS_COLLECTION=knowledge_test 对该测试库建库后，"
                "在同一个环境变量下重跑本脚本（见 README）。",
                file=sys.stderr,
            )
            return 2

        # query_faq 已改为向量检索：若评估库为空或集合为空，检索会一律
        # 落空/变孤儿，内容判据被整体跳过、脚本假绿。宁可响亮失败。
        db_count = await count_vectorized(session)
        if db_count == 0:
            print(
                f"错误：评估库 {db_name} 的 knowledge_chunks 没有【已向量化】行，"
                "query_faq 的检索检查无法进行。\n"
                "请先对同一数据库建库（见 README 的测试库建库命令），或改用 "
                "evals.run_kb_evals 做检索质量评估。",
                file=sys.stderr,
            )
            return 2

        try:
            store = await asyncio.to_thread(VectorStore)
            vector_count = await asyncio.to_thread(store.count)
        except VectorStoreUnavailable as exc:
            print(
                f"错误：无法连接 Milvus 集合 {collection!r}：{exc}\n"
                "请确认 Milvus 已启动，且 APP_MILVUS_COLLECTION 与该库建库时一致。",
                file=sys.stderr,
            )
            return 2

        if vector_count != db_count:
            print(
                f"错误：数据库与 Milvus 集合不配对：库 {db_name} 已向量化 "
                f"{db_count} 行，集合 {collection!r} 有 {vector_count} 个实体。\n"
                "请用与建库相同的 APP_MILVUS_COLLECTION 重跑本脚本；"
                "不要拿生产向量 join 测试库的行。",
                file=sys.stderr,
            )
            return 2

        print(
            f"知识库配对检查：库 {db_name} 已向量化 {db_count} == "
            f"集合 {collection!r} 实体 {vector_count}"
        )

        registry = build_registry(session=session)
        for c in cases:
            turn = await run_production_turn(
                c["input"], model=model, registry=registry,
                conversation_id=conversation_id, session=session,
            )
            called = [tc["name"] for tc in turn.tool_calls]

            problems = []
            # 集合比较：重复调用同一工具不算"选对了"，但也不是"选错了"，
            # 单独点出来（去重后比较，重复进提示）。
            uniq = sorted(set(called))
            if uniq != sorted(c["expect_tools"]):
                problems.append(f"工具不符：期望 {c['expect_tools']}，实际 {called}")
            elif len(called) != len(uniq):
                problems.append(f"同一工具被重复调用：{called}")

            for tool_name, must_contain in (c.get("expect_args_contain") or {}).items():
                call = next((tc for tc in turn.tool_calls if tc["name"] == tool_name), None)
                if call is None:
                    problems.append(f"没调 {tool_name}，无法检查参数")
                elif must_contain not in str(call["args"]):
                    problems.append(f"{tool_name} 参数 {call['args']} 不含「{must_contain}」")

            # 工具自己执行失败（库没连上、外键撞了…）不是模型行为，不能算通过
            if any(not ok for ok in turn.tool_ok):
                problems.append(f"有工具执行失败（ok={turn.tool_ok}）：{turn.tool_output[:80]}")

            # ② 结果转述：只在**真的调了工具**的用例上测（没调工具无从转述）
            relay_passed: bool | None = None
            relay_why = ""
            judge_note = ""
            if called:
                verdict = relay_check(turn.tool_output, turn.final_text)
                relay_passed, relay_why = verdict.ok, verdict.why

                # FAQ 用例加一层「检索召回/回复转述」的判据。
                # ★ 只要用例标了 expect_faq_contains 就检查，不因
                # returned_data() 为假而跳过——检索没命中正是要抓的失败。
                if "expect_faq_contains" in c:
                    content_ok, content_why = await check_faq_content(session, c, turn)
                    if not content_ok:
                        relay_passed = False
                        relay_why = "；".join(
                            x for x in (relay_why, content_why) if x
                        )

                # judge 只作参考（见 harness 的说明：它系统性判错，不进闸门）
                judged = await judge_model.ainvoke(
                    JUDGE_TEMPLATE.format(
                        tool_output=turn.tool_output[:1500], reply=turn.final_text
                    )
                )
                judge_note = f"（judge 参考：{judged.text.strip()[:1] or '空'}）"

            results.append((c["id"], not problems, called, relay_passed))
            mark = "PASS" if not problems else "FAIL"
            detail = "" if not problems else "  " + "；".join(problems)
            print(f"  [{mark}] {c['id']:<14} 调用 {called}{detail}")
            if turn.first_round_text.strip():
                print(f"           首轮正文：{turn.first_round_text[:60]}")
            if problems:
                print(f"           模型改为回复：{turn.final_text[:60]}")
            if relay_passed is not None:
                rmark = "PASS" if relay_passed else "FAIL"
                print(f"         ↳ 转述 [{rmark}] {relay_why}{judge_note}")

    ok = sum(1 for _, p, _, _ in results if p)
    relay_tested = [r for r in results if r[3] is not None]
    relay_ok = sum(1 for r in relay_tested if r[3])
    print(f"\n工具选择通过率：{ok}/{len(results)}")
    print(f"结果转述通过率：{relay_ok}/{len(relay_tested)}（只统计真调了工具的用例）")
    return 0 if ok == len(results) and relay_ok == len(relay_tested) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
