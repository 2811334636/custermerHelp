"""人工工单工具。成功建单后把会话状态改成「已转人工」——建单即转人工（spec §11.4）。"""

import datetime
from typing import Literal

from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories import conversation as conv_repo
from app.repositories import ticket as ticket_repo
from app.tools.context import current_context


async def _next_ticket_no(session: AsyncSession) -> str:
    """按「当天的建单序号」生成工单号，形如 T20260920001。"""
    today = datetime.date.today()
    n = await ticket_repo.count_on_date(session, today) + 1
    return f"T{today.strftime('%Y%m%d')}{n:03d}"


@tool
async def create_ticket(
    description: str, ticket_type: Literal["售后", "投诉", "咨询"]
) -> str:
    """创建人工工单并转接人工客服。仅当用户明确要求转人工、或问题无法自助解决时调用。

    ticket_type 只能是「售后」「投诉」「咨询」三者之一。
    """
    # 库会话从上下文取，不落模块级全局（见 context.py 顶部注释）。
    session = current_context().session
    if session is None:
        return "工单系统暂时不可用（数据库未绑定）。"

    # ★ 会话 id 也从上下文取，**不在函数签名里**——模型不该知道、也不该能
    # 指定往哪个会话建单（spec §6.3）。给模型加这个参数等于让它能替用户
    # 挑会话，凭空多出一个越权面。
    conversation_id = current_context().conversation_id

    ticket_no = await _next_ticket_no(session)
    # 建单与改状态在**同一个事务**里：都只 flush、不 commit，由调用方的
    # session 事务统一收口。「转人工」这个业务动作本就跨 tickets 与
    # conversations 两张表，中间提交会留下「有单但没转人工」的半截状态。
    await ticket_repo.create(
        session, ticket_no, conversation_id, description, ticket_type
    )
    await conv_repo.set_status(session, conversation_id, "已转人工")

    return f"已为您创建工单，工单号 {ticket_no}（类型：{ticket_type}）。人工客服会尽快跟进。"
