"""ch05 意图识别标注集评估。

用法：uv run python -m evals.run_intent_eval     ← 必须用 -m，不能用路径
用真实模型（temperature=0）逐条分类 20 条标注用例，打印混淆矩阵与准确率。
准确率 < 0.80 时以非 0 退出。

注意：这里只评估模型的标签判定；分流（route_for）是代码常量，不在评估范围。
"""

import asyncio
import sys
from pathlib import Path

import yaml

from app.providers import get_chat_model
from app.workflow.intent import classify_text
from app.workflow.state import INTENTS

CASES_PATH = Path(__file__).parent / "intent_cases.yaml"
THRESHOLD = 0.80


def _print_confusion(confusion: dict[str, dict[str, int]]) -> None:
    """行=标注，列=预测；只打印非零格，避免 7×7 全零噪声。"""
    print("\n混淆矩阵（行=标注，列=预测；只显示非零格）")
    for expected in INTENTS:
        preds = {
            predicted: count
            for predicted, count in confusion.get(expected, {}).items()
            if count
        }
        if not preds:
            continue
        detail = "  ".join(f"{predicted}:{count}" for predicted, count in preds.items())
        print(f"  {expected:<5} {detail}")


async def main() -> int:
    cases = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    model = get_chat_model(temperature=0)

    confusion: dict[str, dict[str, int]] = {intent: {} for intent in INTENTS}
    correct = 0
    wrong: list[tuple[str, str, str]] = []

    print(f"意图标注集：{len(cases)} 条，模型 temperature=0\n")
    for case in cases:
        query = str(case["query"])
        expected = str(case["intent"])
        predicted, confidence = await classify_text(query, model=model)

        confusion.setdefault(expected, {}).setdefault(predicted, 0)
        confusion[expected][predicted] += 1
        ok = predicted == expected
        correct += ok
        if not ok:
            wrong.append((query, expected, predicted))

        conf = "-" if confidence is None else f"{confidence:.2f}"
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] 标注 {expected:<4} 预测 {predicted:<4} conf={conf:<4} {query}")

    total = len(cases)
    accuracy = correct / total if total else 0.0
    print(f"\n准确率：{correct}/{total} = {accuracy:.2%}（阈值 {THRESHOLD:.0%}）")
    _print_confusion(confusion)

    if wrong:
        print("\n错例：")
        for query, expected, predicted in wrong:
            print(f"  「{query}」标注 {expected} → 预测 {predicted}")

    if accuracy < THRESHOLD:
        print(f"\n未达标：准确率低于 {THRESHOLD:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
