def test_extract_order_id_formats():
    from app.workflow.slots import extract_order_id

    assert extract_order_id("订单 1001 的物流") == "1001"
    assert extract_order_id("订单号：A12345") == "A12345"
    assert extract_order_id("单号 20260920-001") == "20260920-001"
    assert extract_order_id("1001") == "1001"
    assert extract_order_id("怎么开发票") is None


async def test_slot_gate_passes_with_order_id():
    from app.workflow.slots import slot_gate_node

    out = await slot_gate_node(
        {
            "user_input": "订单 1001 的物流",
            "intent": "物流",
            "awaiting_slot": None,
            "pending_intent": None,
        }
    )
    assert out["order_id"] == "1001" and out["awaiting_slot"] is None
    assert "messages" not in out


async def test_slot_gate_asks_and_remembers():
    from app.workflow.slots import SLOT_QUESTION, slot_gate_node

    out = await slot_gate_node(
        {
            "user_input": "我的物流到哪了",
            "intent": "物流",
            "awaiting_slot": None,
            "pending_intent": None,
        }
    )
    assert out["awaiting_slot"] == "order_id" and out["pending_intent"] == "物流"
    assert out["messages"][0].content == SLOT_QUESTION


async def test_slot_gate_uses_existing_order_id_from_state():
    from app.workflow.slots import slot_gate_node

    out = await slot_gate_node(
        {
            "user_input": "那物流呢",
            "intent": "物流",
            "order_id": "1001",
            "awaiting_slot": None,
        }
    )
    assert out["order_id"] == "1001"
