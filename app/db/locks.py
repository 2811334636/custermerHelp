"""MySQL 命名锁（GET_LOCK / RELEASE_LOCK）。

★ 为什么用专用连接：GET_LOCK 是**连接级**的，连接一旦归还连接池，锁就
可能被别的会话意外持有/释放。这里用一个独立的 ``engine.connect()`` 覆盖
整个任务生命周期，并在同一个连接上 RELEASE_LOCK。

用途：cron 重叠（或手动重跑与 cron 撞车）时，建库/挖知识只允许一个进程
运行，避免重复付费与「MySQL 新文本 + Milvus 旧向量」的并发竞态。
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from sqlalchemy import text

from app.db.base import engine


@asynccontextmanager
async def mysql_named_lock(name: str, timeout: int = 0) -> AsyncIterator[bool]:
    """尝试拿命名锁，产出是否拿到；退出时释放（只释放自己拿到的）。"""
    async with engine.connect() as conn:
        acquired = (
            await conn.execute(
                text("SELECT GET_LOCK(:name, :timeout)"),
                {"name": name, "timeout": timeout},
            )
        ).scalar_one()
        try:
            yield acquired == 1
        finally:
            if acquired == 1:
                await conn.execute(
                    text("SELECT RELEASE_LOCK(:name)"), {"name": name}
                )
