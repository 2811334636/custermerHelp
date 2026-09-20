"""ch02 工具选择评估集。

用法：uv run python -m evals.run_tool_evals
判据是「模型选了哪些工具 + 传给工具的参数」，不是最终回复的措辞。
"""

import asyncio
import sys
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage

from app.prompts import build_chat_prompt
from app.providers import get_chat_model
from app.tools import build_registry

ROOT = Path(__file__).parent


async def main() -> int:
    cases = yaml.safe_load((ROOT / "tool_cases.yaml").read_text(encoding="utf-8"))
    model = get_chat_model(temperature=0)
    # session=None：本评估只测「选不选工具、传什么参数」，不测工具的执行结果
    bound = model.bind_tools(build_registry(session=None).as_langchain_tools())

    results = []
    for c in cases:
        rendered = build_chat_prompt().format_messages(
            history=[HumanMessage(c["input"])]
        )
        reply = await bound.ainvoke(rendered)
        called = [tc["name"] for tc in (reply.tool_calls or [])]

        problems = []
        if sorted(called) != sorted(c["expect_tools"]):
            problems.append(f"工具不符：期望 {c['expect_tools']}，实际 {called}")

        for tool_name, must_contain in (c.get("expect_args_contain") or {}).items():
            call = next((tc for tc in (reply.tool_calls or []) if tc["name"] == tool_name), None)
            if call is None:
                problems.append(f"没调 {tool_name}，无法检查参数")
            elif must_contain not in str(call["args"]):
                problems.append(f"{tool_name} 参数 {call['args']} 不含「{must_contain}」")

        results.append((c["id"], not problems, problems, called))
        mark = "PASS" if not problems else "FAIL"
        detail = "" if not problems else "  " + "；".join(problems)
        print(f"  [{mark}] {c['id']:<14} 调用 {called}{detail}")
        if problems and reply.content:
            print(f"           模型改为回复：{reply.content[:60]}")

    ok = sum(1 for _, p, _, _ in results if p)
    print(f"\n工具选择通过率：{ok}/{len(results)}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
