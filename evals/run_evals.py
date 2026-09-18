"""ch01 评估集：Prompt 行为 + 抽取准确率。

用法：uv run python -m evals.run_evals      ← 必须用 -m，不能用路径
产出：控制台通过率表格 + evals/out/report.json

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
from langchain_core.messages import HumanMessage

from app.extract import extract_ticket
from app.prompts import SYSTEM_PROMPT, build_chat_prompt
from app.providers import get_chat_model

ROOT = Path(__file__).parent
OUT = ROOT / "out"

JUDGE_TEMPLATE = """你在给一个电商客服机器人的回复打分。

客服的行为约束：
{system}

用户说：{user}
客服回复：{reply}
本条期望的行为：{expect}

回复是否满足期望行为？只输出一个字：是 或 否。"""


async def run_prompt_case(case: dict, model) -> dict:
    # 这里必须走 build_chat_prompt()，与 app/chat.py:70 的生产路径一致。
    # brief 原文是 `model.ainvoke([HumanMessage(case["input"])])` —— 那样
    # 模型根本收不到 SYSTEM_PROMPT，测的就不是本任务要测的东西（实测首轮
    # Prompt 3/15，且 outofscope-1 直接吐出了完整爬虫脚本）。此为对 brief
    # 脚本的必要修正，用例数据、judge 规则、输出格式均未改动。
    rendered = build_chat_prompt().format_messages(
        history=[HumanMessage(case["input"])]
    )
    reply = (await model.ainvoke(rendered)).text

    # 确定性断言：禁用词
    violations = []
    for pattern in case.get("forbid") or []:
        if re.search(pattern, reply):
            violations.append(pattern)

    # LLM-judge：语义行为
    judge = await model.ainvoke(
        JUDGE_TEMPLATE.format(
            system=SYSTEM_PROMPT, user=case["input"], reply=reply, expect=case["expect"]
        )
    )
    judged_ok = judge.text.strip().startswith("是")

    return {
        "id": case["id"],
        "category": case["category"],
        "reply": reply,
        "violations": violations,
        "judged_ok": judged_ok,
        "passed": judged_ok and not violations,
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


async def main() -> int:
    model = get_chat_model(temperature=0)
    OUT.mkdir(exist_ok=True)

    prompt_cases = yaml.safe_load((ROOT / "cases.yaml").read_text(encoding="utf-8"))
    extract_cases = yaml.safe_load((ROOT / "extract_cases.yaml").read_text(encoding="utf-8"))

    print("=" * 68)
    print("A. System Prompt 行为评估")
    print("=" * 68)
    prompt_results = [
        await run_prompt_case(c, model) for c in prompt_cases
    ]
    for r in prompt_results:
        mark = "PASS" if r["passed"] else "FAIL"
        why = ""
        if r["violations"]:
            why = f" 禁用词命中: {r['violations']}"
        elif not r["judged_ok"]:
            why = " judge 判定未满足期望"
        print(f"  [{mark}] {r['id']:<16} {r['category']}{why}")

    by_cat: dict[str, list[bool]] = {}
    for r in prompt_results:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    print("\n  分类通过率：")
    for cat, oks in by_cat.items():
        print(f"    {cat:<10} {sum(oks)}/{len(oks)}")

    print()
    print("=" * 68)
    print("B. 结构化抽取准确率（字段级）")
    print("=" * 68)
    extract_results = [await run_extract_case(c, model) for c in extract_cases]
    for r in extract_results:
        mark = "PASS" if r["passed"] else "FAIL"
        why = f"  {r['mismatches']}" if r["mismatches"] else ""
        print(f"  [{mark}] {r['id']:<8}{why}")

    total = sum(len(c["expect"]) for c in extract_cases)
    wrong = sum(len(r["mismatches"]) for r in extract_results)
    print(f"\n  字段级准确率：{total - wrong}/{total} = {(total - wrong) / total:.1%}")

    report = {"prompt": prompt_results, "extract": extract_results}
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n  明细已写入 {OUT / 'report.json'}")

    prompt_pass = sum(r["passed"] for r in prompt_results)
    print(f"\n总计：Prompt {prompt_pass}/{len(prompt_cases)}  "
          f"Extract 字段级 {total - wrong}/{total}")
    return 0 if prompt_pass == len(prompt_cases) and wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
