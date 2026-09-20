import os

import pytest

# 测试期间不读真实 .env，避免测试依赖本机密钥
os.environ.setdefault("APP_LLM_API_KEY", "test-key")
os.environ.setdefault(
    "APP_DB_URL",
    "mysql+aiomysql://app:apppw@127.0.0.1:3307/customer_service_test?charset=utf8mb4",
)

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine  # noqa: E402


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def db_session():
    """每个用例一个 session，跑完回滚。

    外层的 engine 每次测试重建，避免事件循环切换导致的连接池串用。

    ★ 唯一例外：**被测代码自己 commit 过**时不再回滚。`/api/chat` 路由就是
    这样——它必须提交本轮写入（Task 5 裁决 + ch02 的会话行），于是 fixture
    手里那笔事务在请求结束时就已 CLOSED，回滚会抛 ResourceClosedError。
    这不是被测代码的 bug，是"commit 不可撤销"的必然结果：那些行已经真落库了，
    回滚也回不来。拦在这里只会让每个驱动真路由的用例在 teardown 报错。

    代价要写清楚：tests/test_api.py 与 tests/test_api_tools.py 会在
    customer_service_test 里留下自己的会话行与消息行（每轮用例几条）。会话
    uuid 随机，不影响其他用例；要清干净得手工 DELETE 或重建库——seed.py 只
    重灌 faq，不会清 conversations/messages。
    """
    from app.config import get_settings

    get_settings.cache_clear()
    engine = create_async_engine(get_settings().db_url, echo=False)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async with maker() as session:
        trans = await session.begin()
        try:
            yield session
        finally:
            if trans.is_active:
                await trans.rollback()
    await engine.dispose()
