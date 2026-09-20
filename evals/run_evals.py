"""ch01 评估集：Prompt 行为 + 抽取准确率。

用法：uv run python -m evals.run_evals      ← 必须用 -m，不能用路径
产出：控制台通过率表格 + evals/out/report.json

行为用例有三层**确定性**断言（禁用词正则、回复长度上限 150 字、逐条 gate），
外加一层 LLM-judge——judge 只作参考打印，**不进闸门**。

★ 为什么 judge 不是闸门（修复轮 · Important 1）：实测它不是"偶尔抖"，是系统性
判错——给一条逐字转述了商品/规格/金额/状态的完美回复连判 4 次全是"否"。拿一个
不可靠的裁判当闸门，评估本身就变成随机的。现在闸门全部是确定性判据：
  · forbid / gate.forbid_extra 正则 —— 命中即 FAIL；
  · 长度上限 150 字（约束 4）；
  · gate.require_any —— 必须命中至少一条"明确拒绝/明确索取"的语义；
  · gate.relay —— 调了工具且工具返回了数据时，必须转述（evals/harness.relay_check）。
判据的强度有负对照背书：`EVAL_PROMPT_FILE=<旧 prompts.py> uv run python -m
evals.run_evals` 把修复前的 Prompt 接进来，它**必须**拿不到 15/15（见 report）。

★ 2026-09-20（ch02 Task 8）起，行为用例的模型**绑上与生产相同的工具**。
在此之前它不绑工具，测的是一个**永远不会真实运行的配置**，于是 ch02 改写
System Prompt 后 5 条用例稳定失败——失败的是"该明说查不到"这个已不存在的
期望，不是模型变差（spec §6.6）。抽取仍走未绑工具的模型。

★ 修复轮起另有两点：回合**直接驱动 app/chat.py 的 stream_reply_with_tools**，
不再自己复刻（Important 4）；工具接**真库** customer_service_test（Important 3）。

为什么必须用 `-m`：`python evals/run_evals.py` 会把 sys.path[0] 设成
`evals/` 目录，导致 `from app.extract import ...` 报 ModuleNotFoundError。
`-m` 会把 CWD 放进 sys.path[0]，`app` 才能被导入（Task 2 实测踩过同一个坑）。
"""

import asyncio
import json
import re
import sys
from pathlib import Path

import yaml

from app.extract import extract_ticket
from app.prompts import SYSTEM_PROMPT
from app.providers import get_chat_model
from app.tools import build_registry
from evals.harness import (
    apply_prompt_override,
    open_test_db,
    relay_check,
    run_production_turn,
)

ROOT = Path(__file__).parent
OUT = ROOT / "out"

# 约束 4（app/prompts.py）：单次回复不超过 150 字。断言按**用户实际看到的**整条
# 文本算（工具轮的"首轮前言 + 末轮回答"，见 harness.Turn.visible_text）。
MAX_REPLY_CHARS = 150

# 抽取字段级准确率的退出码下限，对齐 plan Step 5 的目标「≥ 90%」。
# 不写成 100%：36 个字段里错 1~3 个（即 35/36、33/36）在计划口径下是「通过」，
# 脚本若要求 wrong == 0 就会把「计划说 pass」的跑分报成失败。
MIN_EXTRACT_ACCURACY = 0.9

JUDGE_TEMPLATE = """你在给一个电商客服机器人的回复打分。

客服的行为约束：
{system}

用户说：{user}
客服回复：{reply}
本条期望的行为：{expect}

回复是否满足期望行为？只输出一个字：是 或 否。"""


async def run_prompt_case(
    case: dict, model, judge_model, registry, conversation_id: int
) -> dict:
    # 走**生产函数**（app/chat.py:stream_reply_with_tools），不自己拼回合：
    # 绑工具、先转发后决断、第二次不绑工具都只有那一份实现（Important 4）。
    # 评估若复刻一遍，迟早与生产走散——而评估的全部价值就是测真身。
    #
    # model 是**未绑工具**的：绑工具是生产函数内部自己的事（app/chat.py:154）。
    # judge_model 另起一个实例，判断"是/否"时不该带着工具。
    turn = await run_production_turn(
        case["input"], model=model, registry=registry, conversation_id=conversation_id
    )
    visible = turn.visible_text  # 用户实际看到的全部文本（工具轮是两段）
    reply = turn.final_text      # judge 与事实核对只看末轮回答

    gate = case.get("gate") or {}

    # 确定性断言 ①：禁用词（原有 forbid 原样保留 + gate 追加的 forbid_extra）
    forbid_patterns = list(case.get("forbid") or []) + list(gate.get("forbid_extra") or [])
    violations = [p for p in forbid_patterns if re.search(p, visible)]

    # 确定性断言 ②：语义要求（至少命中一条正则）
    require_any = gate.get("require_any") or []
    missing_semantics = bool(require_any) and not any(
        re.search(p, visible) for p in require_any
    )

    # 确定性断言 ③：转述。★ 先堵一个洞——relay 判据只在"真的调了工具"时才跑，
    # 于是一个**一次工具都不调**、直接回"我这边查不了"的 Prompt 会绕过它拿 PASS
    # （实测：ch01 时代那句"你没有订单系统访问权限"的 Prompt 正是这样）。
    # 用例的期望既然写着"调用工具查询"，就必须真调，否则本用例不成立。
    relay_ok: bool | None = None
    relay_why = ""
    missing_tools = bool(gate.get("require_tools")) and not turn.tool_calls
    if missing_tools:
        relay_ok = False
        relay_why = "该用例要求先调用工具，模型一次都没调（拒绝在正文里，无从转述）"
    elif gate.get("relay") and turn.tool_calls:
        verdict = relay_check(turn.tool_output, reply)
        relay_ok, relay_why = verdict.ok, verdict.why

    # LLM-judge：语义行为的参考信号，**不进闸门**（见模块 docstring）
    judge = await judge_model.ainvoke(
        JUDGE_TEMPLATE.format(
            system=SYSTEM_PROMPT, user=case["input"], reply=reply, expect=case["expect"]
        )
    )
    judged_ok = judge.text.strip().startswith("是")

    # 确定性断言 ④：长度上限。约束 4 写在 app/prompts.py，这里把它变成可执行的闸门。
    # 阈值就是约束写的那个数，不因模型做不到而下调。
    length = len(visible.strip())
    too_long = length > MAX_REPLY_CHARS

    passed = (
        not violations
        and not too_long
        and not missing_semantics
        and not missing_tools
        and relay_ok is not False
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "reply": reply,
        "first_round_text": turn.first_round_text,
        "tool_calls": [tc["name"] for tc in turn.tool_calls],
        "violations": violations,
        "missing_semantics": missing_semantics,
        "missing_tools": missing_tools,
        "require_any": require_any if missing_semantics else [],
        "relay_ok": relay_ok,
        "relay_why": relay_why,
        "judged_ok": judged_ok,
        "gated": bool(gate),
        "length": length,
        "too_long": too_long,
        "passed": passed,
    }


async def run_extract_case(case: dict, model) -> dict:
    ticket = await extract_ticket(case["text"], model=model)
    got = ticket.model_dump()
    mismatches = {
        k: {"expected": v, "got": got[k]}
        for k, v in case["expect"].items()
        if _norm(got[k]) != _norm(v)
    }
    return {"id": case["id"], "text": case["text"], "got": got, "mismatches": mismatches,
            "passed": not mismatches}


def _norm(v):
    return re.sub(r"\s+", "", str(v)).lower() if v is not None else None


def _why(r: dict) -> str:
    out = []
    if r["violations"]:
        out.append(f" 禁用词命中: {r['violations']}")
    if r["missing_semantics"]:
        out.append(f" 未命中必需语义: {r['require_any']}")
    if r["missing_tools"]:
        out.append(" 该用例要求调工具，模型没调")
    if r["relay_ok"] is False:
        out.append(f" 转述不通过: {r['relay_why']}")
    if r["too_long"]:
        out.append(f" 超长: {r['length']} 字（上限 {MAX_REPLY_CHARS}）")
    if not out and not r["judged_ok"]:
        out.append(" （judge 参考：否；judge 不进闸门）")
    return "".join(out)


async def main() -> int:
    override = apply_prompt_override()
    judge_model = get_chat_model(temperature=0)
    # model 传未绑工具的实例：绑工具是生产路径内部的行为（app/chat.py:154）
    model = get_chat_model(temperature=0)
    OUT.mkdir(exist_ok=True)

    prompt_cases = yaml.safe_load((ROOT / "cases.yaml").read_text(encoding="utf-8"))
    extract_cases = yaml.safe_load((ROOT / "extract_cases.yaml").read_text(encoding="utf-8"))

    print("=" * 68)
    print("A. System Prompt 行为评估")
    if override:
        print(f"   ★ 负对照：System Prompt 已被 {override} 覆盖（不是 app/prompts.py 的现行版本）")
    print("=" * 68)
    async with open_test_db() as (session, conversation_id):
        registry = build_registry(session=session)
        prompt_results = [
            await run_prompt_case(c, model, judge_model, registry, conversation_id)
            for c in prompt_cases
        ]

    for r in prompt_results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"  [{mark}] {r['id']:<16} {r['category']}{_why(r)}")
        if r["tool_calls"]:
            print(f"         ↳ 调用了 {r['tool_calls']}")
        if r["too_long"]:
            print(f"         ↳ 实测 {r['length']} 字，超出 {r['length'] - MAX_REPLY_CHARS} 字。回复原文：")
            for line in (r["first_round_text"] + "\n" + r["reply"]).strip().splitlines():
                print(f"           {line}")

    by_cat: dict[str, list[bool]] = {}
    for r in prompt_results:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    print("\n  分类通过率：")
    for cat, oks in by_cat.items():
        print(f"    {cat:<10} {sum(oks)}/{len(oks)}")

    gated = [r for r in prompt_results if r["gated"]]
    gated_pass = sum(r["passed"] for r in gated)
    print(f"\n  确定性闸门（forbid/语义/转述/长度）：FAIL {len([r for r in prompt_results if not r['passed']])}"
          f"/{len(prompt_results)}；带 gate 的用例 {gated_pass}/{len(gated)} 通过")
    for r in gated:
        print(f"    · {r['id']:<12} {'PASS' if r['passed'] else 'FAIL'}"
              f"  relay={r['relay_ok']}  语义={'OK' if not r['missing_semantics'] else 'MISS'}")
    over = [r for r in prompt_results if r["too_long"]]
    longest = max(prompt_results, key=lambda r: r["length"])
    print(f"  长度断言（约束 4：≤{MAX_REPLY_CHARS} 字）：超限 {len(over)}/{len(prompt_results)}；"
          f"最长 {longest['length']} 字（{longest['id']}）")

    print()
    print("=" * 68)
    print("B. 结构化抽取准确率（字段级）")
    print("=" * 68)
    # 抽取走未绑工具的 judge_model：with_structured_output 在 bind_tools 之后的
    # RunnableBinding 上不存在（会 AttributeError），且抽取本就不该带业务工具。
    extract_results = [await run_extract_case(c, judge_model) for c in extract_cases]
    for r in extract_results:
        mark = "PASS" if r["passed"] else "FAIL"
        why = f"  {r['mismatches']}" if r["mismatches"] else ""
        print(f"  [{mark}] {r['id']:<8}{why}")

    total = sum(len(c["expect"]) for c in extract_cases)
    wrong = sum(len(r["mismatches"]) for r in extract_results)
    print(f"\n  字段级准确率：{total - wrong}/{total} = {(total - wrong) / total:.1%}")

    report = {
        "prompt_override": override,
        "prompt": prompt_results,
        "extract": extract_results,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  明细已写入 {OUT / 'report.json'}")

    accuracy = (total - wrong) / total
    prompt_pass = sum(r["passed"] for r in prompt_results)
    print(f"\n总计：Prompt {prompt_pass}/{len(prompt_cases)}  "
          f"Extract 字段级 {total - wrong}/{total} = {accuracy:.1%}"
          f"（下限 {MIN_EXTRACT_ACCURACY:.0%}）")
    # 退出码口径：Prompt 必须 15/15；抽取字段级准确率 ≥ 90% 即可（plan Step 5）。
    extract_ok = accuracy >= MIN_EXTRACT_ACCURACY
    return 0 if prompt_pass == len(prompt_cases) and extract_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
