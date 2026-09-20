"""ch02 工具选择 + 结果转述评估集。

用法：uv run python -m evals.run_tool_evals
判据两条，缺一不可：
  ① 工具选择：模型调了哪些工具、传给工具的参数
  ② 结果转述：拿到工具结果后，模型有没有**如实转述**，有没有补充工具没给的信息

★ 为什么必须有 ②（spec §6.5）：只测 ① 区分不了新旧 Prompt。旧 Prompt 下模型
**照调** query_logistics（工具选择满分），然后拿到真实轨迹**丢弃不用**，回一句
"我这边看不到订单的物流信息，没法帮您查"。所以只断言 tool_calls 的评估集对
本章的核心风险是**盲的**——它会给两个行为截然不同的 Prompt 打同样的满分，
等于一条恒绿的测试。本章真正要防的是"调了但不用"。
"""

import asyncio
import re
import sys
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage

from app.prompts import build_chat_prompt
from app.providers import get_chat_model
from app.tools import build_registry
from evals.turn import run_turn

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

# 工具明明返回了数据、客服却说"看不到/查不到"——旧 Prompt 的典型失效模式
REFUSAL_PATTERNS = ["看不到", "无法查询", "没法帮您查", "查询不到您的"]

# 工具"没查到"时的返回文案。★ 拒绝措辞只在**工具确实返回了数据**时才是罪证：
# 工具自己说"没有找到/暂时不可用"时，模型回一句"这边查不到"是**如实转述**，
# 不是拒绝。不设这道闸，faq 与工单用例（session=None，必然走"不可用"分支）
# 会把正确行为判成 FAIL。
NO_DATA_MARKERS = ("没有找到", "暂时不可用", "暂时查不到")

# 工具说办不到、客服却说办成了 —— 反向的失效模式（编造成功）。
SUCCESS_CLAIM_PATTERNS = [
    "已为您创建工单", "已创建工单", "已帮您建单", "已为您建单",
    "已为您转人工", "已转人工", "已帮您转接", "已经帮您转",
]

# ★ 从工具输出里抽出"具体事实"。这是转述维度的**确定性**判据（见 _relay_check）。
# 抽的都是可核对的硬事实：单号、金额、日期、时刻、轨迹里的城市与状态词。
# 不抽「整句相似度」——实测它分不开：旧 Prompt 的拒绝语里"看不到订单的物流信息"
# 与工具输出的"的物流信息"有 5 字公共子串，会把拒绝判成转述。
_FACT_PATTERNS = (
    r"[A-Za-z]{1,2}\d{6,}",                 # 运单号 / 工单号（如 Y767075701349、T20260920001）
    r"\d+\.\d{2}",                          # 金额（如 583.79）
    r"\d{4}-\d{2}-\d{2}",                   # 日期
    r"\d{2}:\d{2}",                         # 时刻
    r"\d{2}:\d{2}\s+([一-龥]{2,4})",  # 轨迹行里的城市
    r"[：:]([一-龥]{2,4})\s*$",       # 行尾的状态/名称词（已签收、运输中、缺货…）
)


def _facts(tool_output: str) -> set[str]:
    found: set[str] = set()
    for pattern in _FACT_PATTERNS:
        for m in re.finditer(pattern, tool_output, re.MULTILINE):
            found.add(m.group(1) if m.groups() else m.group(0))
    return found


def _returned_data(tool_output: str) -> bool:
    return not any(m in tool_output for m in NO_DATA_MARKERS)


def _relay_check(tool_output: str, reply: str) -> tuple[bool, str]:
    """确定性转述判据。返回 (是否如实转述, 失败原因)。

    ★ 这里**不用 LLM judge 把关**，虽然最初是那么写的。实测 judge 不是"偶尔抖"，
    是**系统性地判错**：给一条逐字转述了商品/规格/金额/状态的完美回复，连判 4 次
    全是"否"（同一份输入换次运行又会给"是"）。拿它当闸门，评估本身就变成随机的——
    而把判据交给一个不可靠的裁判，正是本章要避免的事。
    judge 仍会跑，但只作参考打印，不进通过率。

    分支：
    - 工具返回了数据 → 回复里必须出现至少一个工具给过的具体事实（单号/金额/日期/
      轨迹城市…），且不得使用"看不到"这类拒绝措辞；
    - 工具没返回数据（本轮 session=None 下 faq 与工单必然如此）→ 回复里不得
      **声称办成了**。工具说办不到而客服说"已为您转人工"，就是编造。
    """
    if _returned_data(tool_output):
        hit = [p for p in REFUSAL_PATTERNS if p in reply]
        if hit:
            return False, f"工具返回了数据却回拒绝措辞 {hit}"
        facts = _facts(tool_output)
        if not any(f in reply for f in facts):
            return False, f"回复里没有工具输出中的任何具体事实（工具给了 {sorted(facts)[:4]}…）"
        return True, ""

    claim = [p for p in SUCCESS_CLAIM_PATTERNS if p in reply]
    if claim:
        return False, f"工具没办成却声称办成了 {claim}"
    return True, ""


async def main() -> int:
    cases = yaml.safe_load((ROOT / "tool_cases.yaml").read_text(encoding="utf-8"))
    # session=None：本评估测「选不选工具、传什么参数、转不转述」，不测数据本身
    registry = build_registry(session=None)
    model = get_chat_model(temperature=0)
    bound = model.bind_tools(registry.as_langchain_tools())

    results = []
    for c in cases:
        rendered = build_chat_prompt().format_messages(
            history=[HumanMessage(c["input"])]
        )
        # 完整回合（流式 → 执行工具 → 第二次不绑工具），与生产一致
        turn = await run_turn(bound, model, rendered, registry)
        called = [tc["name"] for tc in turn.tool_calls]

        problems = []
        if sorted(called) != sorted(c["expect_tools"]):
            problems.append(f"工具不符：期望 {c['expect_tools']}，实际 {called}")

        for tool_name, must_contain in (c.get("expect_args_contain") or {}).items():
            call = next((tc for tc in turn.tool_calls if tc["name"] == tool_name), None)
            if call is None:
                problems.append(f"没调 {tool_name}，无法检查参数")
            elif must_contain not in str(call["args"]):
                problems.append(f"{tool_name} 参数 {call['args']} 不含「{must_contain}」")

        # ② 结果转述：只在**真的调了工具**的用例上测（没调工具无从转述）
        relay_passed: bool | None = None
        relay_why = ""
        judge_note = ""
        if called:
            relay_passed, relay_why = _relay_check(turn.tool_output, turn.final_text)

            # judge 只作参考（见 _relay_check 的说明：它系统性判错，不进闸门）
            judged = await model.ainvoke(
                JUDGE_TEMPLATE.format(
                    tool_output=turn.tool_output[:1500], reply=turn.final_text
                )
            )
            judge_note = f"（judge 参考：{judged.text.strip()[:1] or '空'}）"

        results.append((c["id"], not problems, called, relay_passed))
        mark = "PASS" if not problems else "FAIL"
        detail = "" if not problems else "  " + "；".join(problems)
        print(f"  [{mark}] {c['id']:<14} 调用 {called}{detail}")
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
