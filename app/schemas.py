from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        min_length=1,
        description="会话 id；省略则服务端生成并经由 meta 事件返回",
    )
    message: str = Field(..., min_length=1, max_length=4000)


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class AfterSalesTicket(BaseModel):
    """售后工单的结构化抽取结果。

    三个字段全部可空：null 表示用户未提及。
    "用户没提" 与 "模型编造了一个值" 是本质区别，验收时肉眼可判。
    """

    order_id: str | None = Field(
        default=None, description="订单号。用户明确提到的订单号原样保留；未提及则为 null。"
    )
    demand: str | None = Field(
        default=None,
        description="诉求类型：退货/换货/退款/维修/催发货/咨询/投诉/其他。未提及则为 null。",
    )
    expected_solution: str | None = Field(
        default=None, description="用户期望的处理方案。用户未表达期望则为 null。"
    )
