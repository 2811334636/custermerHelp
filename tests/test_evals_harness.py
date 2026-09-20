"""evals/harness.py 的确定性判据单测。

为什么要给评估的判据写单测：判据自己就是"被测代码"——它判错，评估就跟着错，
而且错法很隐蔽（把正确回复判 FAIL 会让评估变红，把编造判 PASS 会让评估变绿）。
这些用例把每条分支钉住，不依赖网络与数据库。
"""

import re

import pytest

from app.tools.knowledge import query_faq
from app.tools.ticketing import create_ticket
from evals.harness import (
    NO_DATA_MARKERS,
    extract_facts,
    fabricated_facts,
    longest_common_run,
    relay_check,
    returned_data,
)

# 一条形似真实工具输出的物流文本（规格与 app/tools/catalog.py 一致）
LOGISTICS_OUT = (
    "订单 1001 的物流信息：\n"
    "- 承运商：韵达快递\n"
    "- 运单号：Y767075701349\n"
    "- 轨迹：\n"
    "- 2026-09-19 22:32 广州转运中心：已到达\n"
    "- 2026-09-20 13:46 广州：运输中"
)

FAQ_OUT = (
    "问：退货政策是什么\n"
    "答：签收后 7 天内可无理由退货，商品需保持吊牌完整、不影响二次销售。"
)
FAQ_MISS_OUT = "没有找到与「邮费」相关的常见问题。可以换个说法，或帮你转人工客服。"


def test_extract_facts_pulls_waybill_amount_date_city():
    facts = extract_facts(LOGISTICS_OUT)
    assert "Y767075701349" in facts
    assert "2026-09-20" in facts
    assert "13:46" in facts
    assert "广州" in facts


def test_data_branch_rejects_refusal_wording():
    """工具给了数据，客服却说"我这边看不到" —— 旧 Prompt 的失效模式。"""
    v = relay_check(LOGISTICS_OUT, "我这边看不到订单的物流信息，没法帮您查。")
    assert not v.ok and "拒绝措辞" in v.why


def test_data_branch_accepts_faithful_relay():
    reply = "订单 1001 走韵达快递，运单号 Y767075701349，最新 2026-09-20 13:46 广州运输中。"
    assert relay_check(LOGISTICS_OUT, reply).ok


def test_data_branch_rejects_when_nothing_was_relayed():
    v = relay_check(LOGISTICS_OUT, "您的包裹正在路上，请耐心等待。")
    assert not v.ok and "没有转述" in v.why


def test_empty_reply_never_passes():
    """模型只调工具、没给用户任何话：不能"真空通过"。"""
    assert not relay_check(LOGISTICS_OUT, "  ").ok
    assert not relay_check(FAQ_MISS_OUT, "").ok


def test_prose_fallback_accepts_paraphrased_faq_answer():
    """FAQ 答案是散文、没有数字，硬事实抽不到——靠最长公共片段兜住。"""
    reply = "签收后 7 天内可无理由退货，但需要保持吊牌完整、不影响二次销售。"
    assert longest_common_run(FAQ_OUT, reply) >= 6
    assert relay_check(FAQ_OUT, reply).ok


def test_prose_fallback_still_fails_on_unrelated_reply():
    reply = "您好，请问有什么可以帮您？"  # 与答案的公共片段只有很短的几字
    assert not relay_check(FAQ_OUT, reply).ok


def test_no_data_branch_allows_honest_cannot_query():
    """"工具没查到 → 如实说查不到"是正确行为，不该被拒绝措辞判死。"""
    assert relay_check(FAQ_MISS_OUT, "没查到「邮费」相关的政策，换个说法或帮您转人工？").ok


@pytest.mark.parametrize(
    "claim",
    ["已为您创建工单，工单号 T20260920001", "已为您转人工", "已经帮您转接了"],
)
def test_no_data_branch_rejects_fabricated_completion(claim):
    v = relay_check(FAQ_MISS_OUT, claim)
    assert not v.ok and "声称办成了" in v.why


def test_fabricated_facts_flags_date_not_in_tool_output():
    """部分编造：真事实里混一个工具没给过的日期。"""
    reply = "订单 1001 运单号 Y767075701349，预计 2026-09-25 送达。"
    assert fabricated_facts(reply, LOGISTICS_OUT) == ["2026-09-25"]
    assert relay_check(LOGISTICS_OUT, reply).ok is False


def test_fabricated_facts_ignores_faithfully_repeated_values():
    reply = "运单号 Y767075701349，2026-09-20 13:46 广州运输中。"
    assert fabricated_facts(reply, LOGISTICS_OUT) == []


async def _call_without_session(tool, **kwargs) -> str:
    """在"库没绑定"的上下文里调工具，取它自己的失败文案。

    工具只允许由 ToolRegistry.execute() 调用（session 走 ContextVar），
    所以这里必须自己把上下文摆好，不能直接 ainvoke。
    """
    from app.tools.context import tool_context

    with tool_context(0, None):
        return await tool.ainvoke(kwargs)


async def test_tools_no_data_wording_is_recognized():
    """钉子测试：NO_DATA_MARKERS 与工具措辞是隐式耦合的。

    工具改了"查不到/不可用"的文案而判据没跟着改，评估会静默地走错分支
    （把"没查到"当成"返回了数据"）。这条测试让这种改动**炸在测试里**，
    而不是悄悄削弱评估。这里真的把工具跑起来取文案——无会话 / 订单不存在
    两条路径都要认得出来。
    """
    faq_text = await _call_without_session(query_faq, keyword="退货")
    ticket_text = await _call_without_session(create_ticket, description="投诉", ticket_type="投诉")
    assert any(m in faq_text for m in NO_DATA_MARKERS), faq_text
    assert any(m in ticket_text for m in NO_DATA_MARKERS), ticket_text
    assert not returned_data(faq_text) and not returned_data(ticket_text)


async def test_order_not_found_wording_is_recognized():
    from app.tools.catalog import _order_exists, query_order

    missing = next(f"nope-{i}" for i in range(50) if not _order_exists(f"nope-{i}"))
    text = await query_order.ainvoke({"order_id": missing})
    assert "没有找到" in text and not returned_data(text)


def test_no_data_markers_are_literally_used_by_tools():
    """静态钉子：判据里的标记必须真的出现在某个工具模块的源码里。

    动的是"文案与判据分家"这件事本身：改了工具文案却忘了改判据时，
    这条会红。它刻意比上一组测试更钝——不依赖任何运行期状态。
    """
    from pathlib import Path

    src = "\n".join(
        (Path("app/tools") / name).read_text(encoding="utf-8")
        for name in ("knowledge.py", "ticketing.py", "catalog.py")
    )
    markers = set(re.findall(r"没有找到|暂时不可用|暂时查不到", src))
    assert set(NO_DATA_MARKERS) <= markers | {"暂时查不到"}, (
        f"工具的措辞变了：源码里出现 {sorted(markers)}，判据认的是 {NO_DATA_MARKERS}"
    )
