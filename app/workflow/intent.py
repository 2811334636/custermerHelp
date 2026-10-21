"""意图识别节点：模型只负责打标签，分流走硬编码路由表（spec §5.3）。"""

import json
import logging
import math
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.config import get_settings
from app.providers import get_chat_model
from app.workflow.nodes import route_for
from app.workflow.slots import extract_order_id
from app.workflow.state import INTENTS, WorkflowState

logger = logging.getLogger(__name__)

FALLBACK_INTENT = "售后"

INTENT_PROMPT = """你是电商客服的意图分类器。把用户这句话分到且仅分到下面 7 个标签之一：

- 物流：查询快递进度、配送时间、物流异常、修改收货地址。
  例：订单 1001 的物流到哪了
- 订单：查询或修改订单本身的状态、发货时间、取消订单、订单信息。
  例：帮我看下订单 1001 什么时候发货
- 商品咨询：询问商品参数、尺码、材质、库存、价格、使用方法。
  例：这件羽绒服有 L 码吗
- 退货退款：想退货、换货、退款，或询问退换货与退款的流程、进度。
  例：鞋子买大了想退货怎么申请
- 售后：商品破损、质量问题、保修维修、补发、开发票等售后服务。
  例：收到的杯子碎了，能补发吗
- 投诉：对客服态度、服务体验、商家行为表达不满或要求投诉。
  例：我要投诉你们客服
- 闲聊：问候、感谢、与业务无关的聊天。
  例：你好呀，今天天气不错

只输出一个 JSON 对象，不要输出任何解释，也不要使用 Markdown 代码块。格式严格为：
{"intent": "<上面 7 个标签之一>", "confidence": <0 到 1 之间的小数>}"""

# 模型偶尔仍会包一层 ```json 围栏，解析前先剥掉（这不是业务规则，只是容错）。
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _fallback(reason: str, text: Any) -> tuple[str, None]:
    """统一记录 warning 后返回兜底意图（spec §5.2：解析失败要记 warning）。"""
    logger.warning("intent_parse_fallback reason=%s raw=%r", reason, text)
    return FALLBACK_INTENT, None


def parse_intent_json(text: str) -> tuple[str, float | None]:
    """解析模型输出，任何异常都退回 ("售后", None)，绝不抛给调用方。

    - 合法 JSON + 已知标签 + confidence 合法 → (label, confidence)
    - 合法 JSON + 已知标签 + 缺/坏 confidence → (label, None)（记 warning）
    - 未知标签、非对象、坏 JSON → ("售后", None)（记 warning）
    """
    if not isinstance(text, str):
        return _fallback("not_text", text)
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1)
    try:
        payload: Any = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return _fallback("bad_json", text)
    if not isinstance(payload, dict):
        return _fallback("not_object", text)

    intent = payload.get("intent")
    if not isinstance(intent, str) or intent not in INTENTS:
        return _fallback("unknown_label", intent)

    confidence = payload.get("confidence")
    # bool 是 int 子类，必须显式排除：{"confidence": true} 不是合法置信度。
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        logger.warning("intent_confidence_invalid intent=%r confidence=%r", intent, confidence)
        return intent, None
    value = float(confidence)
    # NaN/Infinity 能通过 json.loads（Python 默认接受），但绝不能当置信度。
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        logger.warning("intent_confidence_out_of_range intent=%r confidence=%r", intent, confidence)
        return intent, None
    return intent, value


def _content_text(content: Any) -> str:
    """把 AIMessage.content 归一成文本：兼容部分 provider 的 content blocks。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "".join(parts)
    return str(content)


async def classify_text(text: str, *, model) -> tuple[str, float | None]:
    """用给定模型分类一句用户输入；评估脚本复用这条路径。

    意图模型只是分流的第一跳，超时/HTTP/provider 异常都不能让整轮失败：
    记 warning 后落到兜底意图，后续 Agent 仍可兜住。
    """
    try:
        response = await model.ainvoke(
            [SystemMessage(INTENT_PROMPT), HumanMessage(text)]
        )
    except Exception as exc:  # noqa: BLE001 - 外部服务不可控，必须兜住
        logger.warning("intent_model_failed input=%r error=%s", text, exc, exc_info=True)
        return FALLBACK_INTENT, None
    return parse_intent_json(_content_text(response.content))


async def classify_intent_node(state: WorkflowState, config: RunnableConfig) -> dict:
    """意图识别节点。

    等待订单号的状态优先：若本轮提供了订单号，说明用户是在回答追问，
    直接沿用 pending_intent，不再调模型；若用户改问别的事，则清空等待态，
    按正常流程重新分类。
    """
    user_input = state.get("user_input", "")
    awaiting_slot = state.get("awaiting_slot")
    pending_intent = state.get("pending_intent")

    if awaiting_slot == "order_id" and extract_order_id(user_input):
        intent = pending_intent if pending_intent in INTENTS else FALLBACK_INTENT
        logger.debug("intent reused pending=%r for input=%r", intent, user_input)
        return {
            "intent": intent,
            "intent_confidence": None,
            "route": route_for(intent),
            # 插槽门随后会消费订单号并清空等待态，这里保持原值。
            "awaiting_slot": awaiting_slot,
            "pending_intent": pending_intent,
        }

    model = (config or {}).get("configurable", {}).get("model")
    if model is None:
        model = get_chat_model(temperature=get_settings().intent_model_temperature)
    intent, confidence = await classify_text(user_input, model=model)

    # 非 order_id 的等待态不在本章范围内，原样保留，避免误清。
    clears_waiting = awaiting_slot == "order_id"
    return {
        "intent": intent,
        "intent_confidence": confidence,
        "route": route_for(intent),
        "awaiting_slot": None if clears_waiting else awaiting_slot,
        "pending_intent": None if clears_waiting else pending_intent,
    }
