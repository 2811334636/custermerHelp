"""MySQL 命名锁（final review I3）。

建库/挖知识两个 cron 任务靠它互斥；这里验证同一名字同时只有一个持有者，
且释放后可以再次获取。
"""

from app.db.locks import mysql_named_lock


async def test_named_lock_is_exclusive_and_released():
    name = "test_knowledge_named_lock"
    async with mysql_named_lock(name) as first:
        assert first is True
        # 第二次尝试必须立即失败（timeout=0），而不是排队。
        async with mysql_named_lock(name) as second:
            assert second is False

    # 第一个上下文退出时已 RELEASE_LOCK，可以再次拿到。
    async with mysql_named_lock(name) as again:
        assert again is True
