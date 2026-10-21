"""LangGraph 工作流 State、意图路由表与每轮重置字段（spec §4）。"""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

INTENTS: tuple[str, ...] = ("物流", "订单", "商品咨询", "退货退款", "售后", "投诉", "闲聊")

# 分流表是代码常量，不交给模型（spec §3）。
ROUTE_BY_INTENT: dict[str, str] = {
    "商品咨询": "knowledge",
    "退货退款": "knowledge",
    "物流": "business",
    "订单": "business",
    "售后": "business",
    "投诉": "complaint",
    "闲聊": "chitchat",
}

# 每轮开始时清空的 turn 级字段；citation_cursor 单独置 1，messages /
# awaiting_slot / pending_intent / order_id 跨轮保留。
TURN_RESET_FIELDS: tuple[str, ...] = (
    "intent",
    "route",
    "evidence",
    "evidence_coverage",
    "gate_passed",
    "refusal_reason",
    "actions",
    "suggestions_sent",
    "steps",
    "tokens_used",
    "finish_reason",
    "usage",
    "insufficient",
    "node_trace",
    "citations",
    "citation_index",
)


class WorkflowState(TypedDict, total=False):
    # 对话
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str  # 本轮原始输入
    resolved_input: str  # 指代消解后的输入（本章等于 user_input）
    # 分流
    intent: str  # 7 类之一
    intent_confidence: float | None
    route: str  # knowledge|business|complaint|chitchat
    # 知识类
    evidence: list[dict]  # [{index, chunk_id, questions, answer, score}]
    evidence_coverage: float | None
    # 引用（全局唯一编号，跨预检索与工具检索）
    citations: list[dict]  # [{index, chunk_id, score}]，按 index 排序
    citation_index: dict[str, int]  # chunk_id -> index，重复召回复用同一编号
    citation_cursor: int  # 下一个可用编号（从 1 开始）
    gate_passed: bool | None
    refusal_reason: str | None  # low_evidence | no_results | ...
    # 业务数据类插槽
    awaiting_slot: str | None  # 本章只有 "order_id"
    pending_intent: str | None  # 等待插槽期间记住原意图
    order_id: str | None
    # 输出控制
    actions: list[dict]  # [{type: handoff|ticket, ...}]
    suggestions_sent: bool
    # ReAct 控制
    steps: int
    tokens_used: int
    max_steps: int
    token_budget: int
    # 上游终局元数据（最后一次模型调用的实际值，Task 9 的 done 帧使用）
    finish_reason: str | None
    usage: dict | None
    # 最终回答是否由模型的自评标记转化为拒答（Task 9 据此入池）
    insufficient: bool
    # 观测：覆盖式列表，节点返回 state["node_trace"] + ["node"]
    node_trace: list[str]
    turn_started_at: float
