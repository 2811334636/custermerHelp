from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    pass


engine = create_async_engine(get_settings().db_url, echo=False, pool_pre_ping=True)

# expire_on_commit=False：提交后仍可读对象属性，避免 async 下的隐式懒加载报错
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：每个请求一个 session。

    测试通过 app.dependency_overrides[get_db] 替换成测试 session。
    """
    async with AsyncSessionLocal() as session:
        yield session
