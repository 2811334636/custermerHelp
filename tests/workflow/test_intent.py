"""意图识别节点：JSON 解析、等待态复用与正常分类（ch05 Task 4）。"""

import json

from langchain_core.messages import AIMessage


class ExplodingModel:
    """任何调用都抛错：证明等待态 + 订单号时节点根本没碰模型。"""

    async def ainvoke(self, *args, **kwargs):
        raise AssertionError("等待态已拿到订单号，不应调用模型")


class FakeIntentModel:
    def __init__(self, intent: str, confidence: float | None):
        payload = {"intent": intent}
        if confidence is not None:
            payload["confidence"] = confidence
        self._content = json.dumps(payload, ensure_ascii=False)

    async def ainvoke(self, *args, **kwargs):
        return AIMessage(content=self._content)


class FailingModel:
    """模拟上游超时/HTTP 异常：意图模型抖动不能让整轮失败。"""

    async def ainvoke(self, *args, **kwargs):
        raise RuntimeError("upstream timeout")


def test_parse_intent_json_valid():
    from app.workflow.intent import parse_intent_json

    assert parse_intent_json('{"intent": "物流", "confidence": 0.9}') == ("物流", 0.9)


def test_parse_intent_json_unknown_or_broken_falls_back():
    from app.workflow.intent import parse_intent_json

    assert parse_intent_json('{"intent": "退款", "confidence": 0.9}') == ("售后", None)
    assert parse_intent_json("not json") == ("售后", None)
    assert parse_intent_json('{"intent": "投诉"}') == ("投诉", None)


def test_parse_intent_json_rejects_non_finite_or_out_of_range_confidence():
    from app.workflow.intent import parse_intent_json

    # Python 的 json.loads 默认接受 NaN/Infinity，但置信度只接受 [0,1] 内的有限数。
    assert parse_intent_json('{"intent": "物流", "confidence": NaN}') == ("物流", None)
    assert parse_intent_json('{"intent": "物流", "confidence": Infinity}') == ("物流", None)
    assert parse_intent_json('{"intent": "物流", "confidence": 1.5}') == ("物流", None)
    assert parse_intent_json('{"intent": "物流", "confidence": -0.1}') == ("物流", None)
    assert parse_intent_json('{"intent": "物流", "confidence": "0.9"}') == ("物流", None)


async def test_classify_falls_back_when_model_raises():
    from app.workflow.intent import classify_intent_node, classify_text

    assert await classify_text("帮我查物流", model=FailingModel()) == ("售后", None)
    out = await classify_intent_node(
        {"user_input": "帮我查物流", "awaiting_slot": None},
        {"configurable": {"model": FailingModel()}},
    )
    assert out["intent"] == "售后" and out["route"] == "business"


async def test_classify_reuses_pending_intent_only_when_order_id_present():
    from app.workflow.intent import classify_intent_node

    state = {"user_input": "1001", "awaiting_slot": "order_id", "pending_intent": "物流"}
    out = await classify_intent_node(state, {"configurable": {"model": ExplodingModel()}})
    assert out["intent"] == "物流" and out["route"] == "business"
    assert out["awaiting_slot"] == "order_id"


async def test_classify_new_question_clears_waiting_state():
    from app.workflow.intent import classify_intent_node

    state = {"user_input": "怎么开发票", "awaiting_slot": "order_id", "pending_intent": "物流"}
    out = await classify_intent_node(state, {"configurable": {"model": FakeIntentModel("退货退款", 0.8)}})
    assert out["intent"] == "退货退款" and out["route"] == "knowledge"
    assert out["awaiting_slot"] is None and out["pending_intent"] is None


async def test_classify_uses_model_result():
    from app.workflow.intent import classify_intent_node

    state = {"user_input": "我要投诉", "awaiting_slot": None, "pending_intent": None}
    out = await classify_intent_node(state, {"configurable": {"model": FakeIntentModel("投诉", 0.95)}})
    assert out["intent"] == "投诉" and out["route"] == "complaint"
