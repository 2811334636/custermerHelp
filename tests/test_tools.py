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
    """★ 必须先把回显的 id 归一化掉再比，否则这条断言永远不会失败。

    每个工具输出的**第一行都会回显输入的 id**（"订单号 {order_id}：" 等），
    所以直接 `a != b` 光靠那一行就被满足了。把 _rng 换成"忽略 key、永远
    返回同一串数据"的实现，这条断言照样通过——Task 7 review 实测证实过。
    归一化之后，比的才是真正的派生数据。
    """
    if tool is query_product:
        args = ({"product_id": "P1"}, {"product_id": "P2"})
    else:
        args = ({"order_id": "1"}, {"order_id": "2"})

    def without_echo(text: str) -> str:
        # 丢掉首行（含回显的输入 id），只留派生出来的内容
        return text.partition("\n")[2]

    a = without_echo(await tool.ainvoke(args[0]))
    b = without_echo(await tool.ainvoke(args[1]))
    assert a != b


async def test_order_and_logistics_agree_on_the_same_order():
    """★ 同一个订单号，query_order 报的订单状态必须与 query_logistics 的最后一条一致。

    两者若各自从独立种子流判定，会出现订单说"已发货"、物流最后一条说"已签收"，
    或者订单"不存在"却有完整轨迹。**验收标准 1 要拿模型回答与工具输出逐字
    对账**，两个工具自相矛盾会直接毁掉这个验收。Task 7 review 实测到过这两类
    矛盾（1001 正是验收用的那个订单号）。这条测试就是锁住它们不再出现。
    """
    from app.tools.catalog import _STATUS_TO_LAST_NODE

    for oid in ["1001", "1002", "1006", "20260701001", "1", "2", "3"]:
        order = await query_order.ainvoke({"order_id": oid})
        logistics = await query_logistics.ainvoke({"order_id": oid})

        if "没有找到" in order:
            assert "没有找到" in logistics, f"{oid}: 订单说不存在，物流却给了完整轨迹"
            continue

        assert "没有找到" not in logistics, f"{oid}: 订单存在，物流却说找不到"
        status = order.split("订单状态：")[-1].strip()
        assert _STATUS_TO_LAST_NODE[status] in logistics, (
            f"{oid}: 订单状态「{status}」与物流最后节点不一致"
        )


def _knowledge_row(questions="运费说明", answer="因商品质量问题导致的退换货，往返运费由平台承担。"):
    """构造一条未落库的 KnowledgeChunk，供打桩的 retrieval.search 返回。"""
    from app.db.models import KnowledgeChunk

    return KnowledgeChunk(
        chunk_id="policy.md:9",
        category="售后政策 > 退换货",
        questions=questions,
        answer=answer,
        section_path="售后政策 > 退换货 > 运费",
        content_type="policy",
        status="已向量化",
        content_hash="x",
    )


async def test_query_faq_hit_via_registry(db_session, monkeypatch):
    """工具只负责取 session、调用 retrieval.search 并把结果排版成问答。"""
    from app.knowledge import retrieval

    async def fake_search(session, query, top_k=None, **kw):
        return [_knowledge_row()]

    monkeypatch.setattr(retrieval, "search", fake_search)
    reg = build_registry(session=db_session)
    out = await reg.execute("query_faq", {"keyword": "邮费是多少"}, "c1", conversation_id=1)
    assert out.ok is True
    assert "往返运费由平台承担" in out.content
    assert out.content.startswith("问：运费说明\n答：")


async def test_query_faq_miss_via_registry(db_session, monkeypatch):
    """★ 验收标准 3：空白检索是业务结果。工具正常返回"没找到"，不是异常。"""
    from app.knowledge import retrieval

    async def fake_search(session, query, top_k=None, **kw):
        return []

    monkeypatch.setattr(retrieval, "search", fake_search)
    reg = build_registry(session=db_session)
    out = await reg.execute("query_faq", {"keyword": "邮费"}, "c2", conversation_id=1)
    assert out.ok is True, "查不到是业务结果，不是工具故障"
    assert "没有找到" in out.content or "未找到" in out.content


async def test_query_faq_degrades_when_milvus_down(db_session, monkeypatch):
    """Milvus 故障在工具内降级成文案，不向 registry 抛异常。"""
    from app.knowledge import retrieval
    from app.knowledge.vector_store import VectorStoreUnavailable

    async def boom(session, query, top_k=None, **kw):
        raise VectorStoreUnavailable("down")

    monkeypatch.setattr(retrieval, "search", boom)
    reg = build_registry(session=db_session)
    out = await reg.execute("query_faq", {"keyword": "邮费"}, "c3", conversation_id=1)
    assert out.ok is True
    assert "暂时不可用" in out.content


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


async def test_two_registries_built_with_different_sessions_do_not_cross(
    db_session, monkeypatch
):
    """★ 两个 registry 各带各的 session，重叠执行时不能互相看见。

    Task 6 证明了机制，这里是它开始真正起作用的地方（Task 9 的工具轮依赖它）。
    reg_real 带真库会话，reg_marker 带一个哨兵对象，并发跑同一句 query_faq：

    - 真会话那路必须命中 —— 说明它拿到的确实是自己的真会话；
    - 哨兵那路必须失败 —— 哨兵不是 AsyncSession，`session.execute` 会抛
      AttributeError，被 registry 包成 ToolOutcome(ok=False)。

    若 session 落在模块级全局（brief 原本的 bind_session 写法），后跑的
    registry 会把它覆盖掉，真会话那路就会拿到哨兵而失败——本测试即红。
    """
    from sqlalchemy import select

    from app.knowledge import retrieval

    async def fake_search(session, query, top_k=None, **kw):
        # 打桩不碰 Milvus/embedding，但仍触碰 session：哨兵在这里 TypeError/AttributeError
        await session.execute(select(1))
        return [_knowledge_row(questions="退货政策", answer="签收后 7 天内可无理由退货。")]

    monkeypatch.setattr(retrieval, "search", fake_search)
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


async def test_tool_session_is_scoped_to_the_registry_that_built_it(
    db_session, monkeypatch
):
    """不带 session 构造的 registry 里，工具看到的就是 None（未绑定），不会捡到别人的会话。"""
    from app.knowledge import retrieval

    async def fake_search(session, query, top_k=None, **kw):
        return [_knowledge_row(questions="退货政策", answer="签收后 7 天内可无理由退货。")]

    monkeypatch.setattr(retrieval, "search", fake_search)
    bound = build_registry(session=db_session)
    unbound = build_registry()

    first = await bound.execute("query_faq", {"keyword": "退货"}, "call_c", conversation_id=1)
    second = await unbound.execute("query_faq", {"keyword": "退货"}, "call_d", conversation_id=1)

    assert first.ok is True and "退货" in first.content
    assert second.ok is True
    assert "数据库未绑定" in second.content or "不可用" in second.content
