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

★ 2026-09-20 修复轮起有两处结构性改动（详见 evals/harness.py 顶部注释）：
  · 回合**直接驱动 app/chat.py 的 stream_reply_with_tools**，不再自己复刻一遍；
  · 工具接**真库**（spec §5.6 的 customer_service_test），faq/工单不再一律回
    "暂时不可用"——否则那 5 条用例全落在弱分支上，一条什么都没转述的回复也能过。
"""

import asyncio
import sys
from pathlib import Path

import yaml

from app.providers import get_chat_model
from app.repositories import faq as faq_repo
from app.tools import build_registry
from evals.harness import (
    PROSE_MIN_RUN,
    apply_prompt_override,
    longest_common_run,
    open_test_db,
    relay_check,
    returned_data,
    run_production_turn,
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
    """faq 用例的加强判据（修复轮 · Important 3）：回复里要出现**库里答案**的内容。

    两层，都是确定性的：
      ① 工具返回的就是 faq 表里的答案原文 —— 证明确实查到了库，不是编的；
      ② 回复与答案原文的最长公共片段 ≥ PROSE_MIN_RUN —— 证明真的转述给了用户。
    只从用例的 `expect_args_contain` 取关键词，不额外写死数据。
    """
    keyword = (case.get("expect_args_contain") or {}).get("query_faq")
    if not keyword:
        return True, ""
    rows = await faq_repo.search_by_keyword(session, keyword)
    if not rows:
        return False, f"库里没有 question 含「{keyword}」的 FAQ 行，用例与种子数据已脱节"
    answer = rows[0].answer
    if answer not in turn.tool_output:
        return False, f"工具输出里没有 faq 表的答案原文（关键词「{keyword}」）——工具没真查库？"
    run = longest_common_run(answer, turn.final_text)
    if run < PROSE_MIN_RUN:
        return False, (
            f"回复没转述 faq 答案内容（与答案原文最长公共片段仅 {run} 字，"
            f"下限 {PROSE_MIN_RUN}）：{answer[:30]}…"
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
        registry = build_registry(session=session)
        for c in cases:
            turn = await run_production_turn(
                c["input"], model=model, registry=registry,
                conversation_id=conversation_id,
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

                # faq 用例加一层「回复里有库里答案」的判据。
                # ★ 只在**工具真的查到了数据**时用：faq-miss 这类用例库里本来
                # 就没有那一行，走的是"查不到"分支，不该拿"没有对应 FAQ 行"判它失败。
                if (
                    relay_passed
                    and returned_data(turn.tool_output)
                    and "query_faq" in (c.get("expect_args_contain") or {})
                ):
                    relay_passed, relay_why = await check_faq_content(session, c, turn)

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
