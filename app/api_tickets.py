"""确定性建工单端点：``POST /api/tickets``。

“转人工”是按钮触发的确定性动作，不经过模型与工作流（spec §11.4）；但建单
语义必须与 Agent 调工具时一致，所以这里直接复用同一个 ``create_ticket``
工具函数，用 ``tool_context`` 注入会话 id 与当前请求的库会话。
"""

import re
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import Conversation
from app.tools.context import tool_context
from app.tools.ticketing import create_ticket

router = APIRouter()

_TICKET_NO_RE = re.compile(r"T\d+")


class TicketRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1, max_length=2000)
    ticket_type: Literal["售后", "投诉", "咨询"]

    @field_validator("description")
    @classmethod
    def _reject_blank_description(cls, value: str) -> str:
        """与 ChatRequest 一致：``min_length=1`` 挡不住纯空白，先 strip 再校验。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("description 不能为空")
        return stripped


@router.post("/tickets")
async def post_ticket(
    req: TicketRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    conv = (
        await db.execute(
            select(Conversation).where(Conversation.session_uuid == req.session_id)
        )
    ).scalar_one_or_none()
    if conv is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "session_not_found",
                "message": f"会话不存在：{req.session_id}",
            },
        )

    # 与 registry 执行工具时同一条路径：库会话与会话 id 只从上下文注入，
    # 工具签名保持“模型无权指定会话”（spec §6.3）。
    with tool_context(conv.id, db):
        result = await create_ticket.ainvoke(
            {"description": req.description, "ticket_type": req.ticket_type}
        )

    # 工单号在工具返回的文案里；契约为顶层直接返回 {"status","ticket_no"}，
    # 不让客户端去解析自然语言。
    match = _TICKET_NO_RE.search(result)
    if match is None:
        # 正常路径下 create_ticket 总会给出工单号；走到这里说明库未绑定等
        # 内部故障，宁可 500 也不能返回一个先成功后失败的成功响应。
        raise HTTPException(
            status_code=500,
            detail={"code": "ticket_creation_failed", "message": result},
        )

    # 建单与「已转人工」在工具内同一事务 flush，这里统一提交。
    await db.commit()

    return {"status": "ok", "ticket_no": match.group(0)}
