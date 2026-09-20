"""FAQ 检索工具。本章用 SQL LIKE，没有语义泛化能力——这正是验收标准 3 要暴露的问题。"""

from langchain_core.tools import tool

from app.repositories import faq as faq_repo
from app.tools.context import current_context


@tool
async def query_faq(keyword: str) -> str:
    """按关键词检索常见问答库（退货、换货、发票、发货时效、尺码等）。

    keyword 请传「核心业务词」——例如用户问「退货政策是什么」就传「退货」，
    问「怎么开发票」就传「发票」。不要传整句话。
    """
    # 库会话从上下文取，不落模块级全局：单 worker 下重叠请求会互相覆盖
    # （见 context.py 顶部注释），A 请求的查询就会打到 B 的会话上。
    session = current_context().session
    if session is None:
        return "FAQ 查询暂时不可用（数据库未绑定）。"

    rows = await faq_repo.search_by_keyword(session, keyword)
    if not rows:
        # ★ 「查不到」是正常业务结果，不是异常：走成功返回，不触发 registry 的
        # 故障重试。验收标准 3 要记录的检索空白就靠这条文案体现。
        return f"没有找到与「{keyword}」相关的常见问题。可以换个说法，或帮你转人工客服。"

    return "\n\n".join(f"问：{r.question}\n答：{r.answer}" for r in rows)
