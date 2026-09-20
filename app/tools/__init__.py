"""工具装配：把五个业务工具包成一个 ToolRegistry。

DB 会话交给 `ToolRegistry(session=...)`，由 execute() 塞进 ContextVar，
工具再用 `current_context().session` 取——**不给工具模块设模块级 _session**：
单 worker 下重叠请求会互相覆盖（见 context.py 顶部注释）。
"""

from collections.abc import Iterable

from langchain_core.tools import BaseTool
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.tools.catalog import query_logistics, query_order, query_product
from app.tools.knowledge import query_faq
from app.tools.registry import ToolRegistry, ToolSpec
from app.tools.ticketing import create_ticket

ALL_TOOLS: list[BaseTool] = [
    query_order, query_product, query_logistics, query_faq, create_ticket,
]


def build_registry(session: AsyncSession | None = None) -> ToolRegistry:
    """构造工具注册表，把当前请求的库会话交给它带到每次 execute()。

    超时与重试次数从配置读（不写死在 ToolSpec 的默认值里）——
    生产调参走 .env，不该为了改个超时动代码。
    """
    s = get_settings()
    specs: Iterable[ToolSpec] = [
        ToolSpec(tool=t, timeout_s=s.tool_timeout_s, max_retries=s.tool_max_retries)
        for t in ALL_TOOLS
    ]
    return ToolRegistry(specs, session=session)


__all__ = ["ALL_TOOLS", "build_registry"]
