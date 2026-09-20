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
            await trans.rollback()
    await engine.dispose()
