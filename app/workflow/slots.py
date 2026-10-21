"""订单号插槽门：提取订单号，缺槽时追问并记住原意图（spec §5.6）。"""

import re

from langchain_core.messages import AIMessage

from app.workflow.state import WorkflowState
from app.workflow.texts import SLOT_QUESTION

# 带标签的写法优先（订单/订单号/单号），允许中英文冒号和空白。
ORDER_ID_RE = re.compile(r"(?:订单号?|单号)\s*[:：]?\s*([A-Za-z0-9-]{4,20})")
# 裸数字兜底：启发式，可能把年份/金额误当订单号；误判只会查无此单。
BARE_ORDER_ID_RE = re.compile(r"\b(\d{4,20})\b")


def extract_order_id(text: str) -> str | None:
    """从文本中提取订单号；先试带标签规则，再用裸数字兜底。"""
    if not text:
        return None
    match = ORDER_ID_RE.search(text)
    if match:
        return match.group(1)
    match = BARE_ORDER_ID_RE.search(text)
    return match.group(1) if match else None


async def slot_gate_node(state: WorkflowState) -> dict:
    """已有订单号或本轮能提取到就放行；否则追问并把原意图挂起。"""
    order_id = state.get("order_id") or extract_order_id(state.get("user_input", ""))
    if order_id:
        return {"order_id": order_id, "awaiting_slot": None, "pending_intent": None}
    return {
        "awaiting_slot": "order_id",
        "pending_intent": state.get("intent"),
        "messages": [AIMessage(SLOT_QUESTION)],
    }
