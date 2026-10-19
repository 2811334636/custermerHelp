"""ch03 · 对话挖知识抽取质量评估（spec §12.4，final review I4）。

用法：uv run python -m evals.run_mining_evals

不走数据库、不写 staging：直接把标注对话喂给 ``extract_from_conversation``
（与 jobs/mine_knowledge.py 使用的真实抽取路径相同），再用确定性的
关键词/事实包含判断抽取结果是否命中标注。

判据刻意不要求措辞一致（LLM 会改写问法），只要求：
  - 每个 expect 条目都能被某个抽出的问答对命中；
  - 一次性订单号/地址等细节没有被当成可复用知识抽出来。

任何一条失败都 exit 1，并打印原始抽取结果；这是 EXTRACT_PROMPT 的回归闸门。
"""

import asyncio
import json
import sys
from pathlib import Path

import yaml

from app.db.base import engine
from app.providers import get_chat_model
from jobs.mine_knowledge import extract_from_conversation

ROOT = Path(__file__).parent


def match_expectation(pairs: list[dict], expect: dict) -> bool:
    """是否存在一个抽出的问答对同时满足 question/answer 的关键词要求。"""
    q_any = expect.get("question_any") or []
    a_any = expect.get("answer_any") or []
    for pair in pairs:
        question = str(pair.get("question", ""))
        answer = str(pair.get("answer", ""))
        q_ok = not q_any or any(k in question for k in q_any)
        a_ok = not a_any or any(k in answer for k in a_any)
        if q_ok and a_ok:
            return True
    return False


def forbidden_hits(pairs: list[dict], forbidden: list[str]) -> list[str]:
    """一次性细节是否泄漏进了抽取结果。"""
    corpus = "\n".join(
        f"{p.get('question', '')}\n{p.get('answer', '')}" for p in pairs
    )
    return [item for item in forbidden if item in corpus]


async def main() -> int:
    cases = yaml.safe_load((ROOT / "mining_cases.yaml").read_text(encoding="utf-8"))
    model = get_chat_model(temperature=0)
    passed = 0
    try:
        for case in cases:
            pairs = await extract_from_conversation(
                case["conversation"], model=model
            )
            misses = [
                exp
                for exp in case.get("expect") or []
                if not match_expectation(pairs, exp)
            ]
            leaks = forbidden_hits(pairs, case.get("forbid_contains") or [])

            ok = not misses and not leaks
            passed += ok
            print(
                f"[{'PASS' if ok else 'FAIL'}] {case['id']} "
                f"({case.get('label', '')}) 抽出 {len(pairs)} 对"
            )
            # 原始结果留痕：失败时据此判断是 Prompt 问题还是标注问题。
            print("       抽取：" + json.dumps(pairs, ensure_ascii=False))
            if misses:
                print(f"       未命中期望：{json.dumps(misses, ensure_ascii=False)}")
            if leaks:
                print(f"       一次性细节泄漏：{leaks}")
    finally:
        # 与其它 CLI/评估一致：显式释放全局异步引擎，避免 aiomysql 在
        # 解释器关闭时打 "Event loop is closed" 噪声。
        await engine.dispose()

    total = len(cases)
    print(f"\n挖掘抽取通过率：{passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
