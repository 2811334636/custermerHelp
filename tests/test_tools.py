import asyncio

import pytest

from app.tools import build_registry
from app.tools.catalog import query_logistics, query_order, query_product


async def test_registry_has_five_tools():
    reg = build_registry()
    names = sorted(t.name for t in reg.as_langchain_tools())
    assert names == [
        "create_ticket", "query_faq", "query_logistics", "query_order", "query_product",
    ]


@pytest.mark.parametrize("tool", [query_order, query_product, query_logistics])
async def test_fake_data_is_deterministic(tool):
    """★ 同一个 key 两次调用必须返回完全相同的数据。

    这是验收标准 1 能做对账的前提——模型说"已到杭州中转站"，
    你去工具结果里能查到同一句。纯随机会让验收退化成"看着像对"。
    """
    arg = {"order_id": "1001"} if tool is not query_product else {"product_id": "P1001"}
    a = await tool.ainvoke(arg)
    b = await tool.ainvoke(arg)
    assert a == b


@pytest.mark.parametrize("tool", [query_order, query_product, query_logistics])
async def test_different_keys_give_different_data(tool):
    if tool is query_product:
        a, b = await tool.ainvoke({"product_id": "P1"}), await tool.ainvoke({"product_id": "P2"})
    else:
        a, b = await tool.ainvoke({"order_id": "1"}), await tool.ainvoke({"order_id": "2"})
    assert a != b


async def test_query_faq_hit_via_registry(db_session):
    reg = build_registry(session=db_session)
    out = await reg.execute("query_faq", {"keyword": "退货"}, "c1", conversation_id=1)
    assert out.ok is True
    assert "退货" in out.content


async def test_query_faq_miss_via_registry(db_session):
    """★ 验收标准 3：『邮费』查不到。工具正常返回"没找到"，不是异常。"""
    reg = build_registry(session=db_session)
    out = await reg.execute("query_faq", {"keyword": "邮费"}, "c2", conversation_id=1)
    assert out.ok is True, "查不到是业务结果，不是工具故障"
    assert "没有找到" in out.content or "未找到" in out.content


async def test_create_ticket_writes_row_and_updates_conversation(db_session):
    from app.repositories import conversation as conv_repo

    conv = await conv_repo.get_or_create(db_session, "ticket-test-uuid", "anonymous")
    assert conv.status == "进行中"

    reg = build_registry(session=db_session)
    out = await reg.execute(
        "create_ticket",
        {"description": "我要投诉物流太慢", "ticket_type": "投诉"},
        "c3",
        conversation_id=conv.id,
    )
    assert out.ok is True

    await db_session.refresh(conv)
    assert conv.status == "已转人工", "建单即转人工，状态要跟着改"

    from app.db.models import Ticket
    from sqlalchemy import select

    rows = list((await db_session.execute(
        select(Ticket).where(Ticket.conversation_id == conv.id)
    )).scalars())
    assert len(rows) == 1
    assert rows[0].ticket_type == "投诉"
    assert rows[0].ticket_no.startswith("T")


async def test_create_ticket_rejects_bad_type(db_session):
    reg = build_registry(session=db_session)
    out = await reg.execute(
        "create_ticket", {"description": "x", "ticket_type": "退款"}, "c4", conversation_id=1
    )
    assert out.ok is False
    assert "参数不合法" in out.content


# ---------------------------------------------------------------------------
# Ruling 3 在业务工具上的落地：session 随 ContextVar 走，不落模块级全局
# ---------------------------------------------------------------------------


async def test_two_registries_built_with_different_sessions_do_not_cross(db_session):
    """★ 两个 registry 各带各的 session，重叠执行时不能互相看见。

    Task 6 证明了机制，这里是它开始真正起作用的地方（Task 9 的工具轮依赖它）。
    reg_real 带真库会话，reg_marker 带一个哨兵对象，并发跑同一句 query_faq：

    - 真会话那路必须命中 —— 说明它拿到的确实是自己的真会话；
    - 哨兵那路必须失败 —— 哨兵不是 AsyncSession，`session.execute` 会抛
      AttributeError，被 registry 包成 ToolOutcome(ok=False)。

    若 session 落在模块级全局（brief 原本的 bind_session 写法），后跑的
    registry 会把它覆盖掉，真会话那路就会拿到哨兵而失败——本测试即红。
    """
    reg_real = build_registry(session=db_session)
    reg_marker = build_registry(session=object())

    real, marker = await asyncio.gather(
        reg_real.execute("query_faq", {"keyword": "退货"}, "call_a", conversation_id=1),
        reg_marker.execute("query_faq", {"keyword": "退货"}, "call_b", conversation_id=1),
    )

    assert real.ok is True and "退货" in real.content, (
        f"带真库会话的 registry 必须命中；实际：{real.content}"
    )
    assert marker.ok is False, "哨兵不是 AsyncSession，那一路本就该失败——失败才说明它没串到真会话"


async def test_tool_session_is_scoped_to_the_registry_that_built_it(db_session):
    """不带 session 构造的 registry 里，工具看到的就是 None（未绑定），不会捡到别人的会话。"""
    bound = build_registry(session=db_session)
    unbound = build_registry()

    first = await bound.execute("query_faq", {"keyword": "退货"}, "call_c", conversation_id=1)
    second = await unbound.execute("query_faq", {"keyword": "退货"}, "call_d", conversation_id=1)

    assert first.ok is True and "退货" in first.content
    assert second.ok is True
    assert "数据库未绑定" in second.content or "不可用" in second.content
