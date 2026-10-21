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


def _clear_test_tables() -> None:
    """清空测试库的 messages / tickets / conversations / low_confidence_questions。

    ★ 只对库名以 `_test` 结尾的库动手：哪怕配置被改错指向生产库，这里也绝不
    删真数据。宁可跳过清理（泄漏会以"行数又涨了"的形式暴露出来），也不能
    清错库（不可逆）。
    """
    import asyncio
    import sys

    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.config import get_settings

    db_url = get_settings().db_url
    db_name = make_url(db_url).database or ""
    if not db_name.endswith("_test"):
        print(f"[conftest] 跳过测试库清理：{db_name} 不是 *_test 库", file=sys.stderr)
        return

    async def _run() -> None:
        engine = create_async_engine(db_url)
        try:
            async with engine.begin() as conn:
                # tickets 的外键指向 conversations，得先删，否则删会话行撞外键。
                # low_confidence_questions 无外键，但拒答用例可能 commit，
                # 不清会跨 run 泄漏。
                for table in (
                    "messages",
                    "tickets",
                    "conversations",
                    "low_confidence_questions",
                ):
                    await conn.execute(text(f"DELETE FROM {table}"))
        finally:
            await engine.dispose()

    asyncio.run(_run())


@pytest.fixture(scope="session", autouse=True)
def isolated_checkpoint_db(tmp_path_factory):
    """把 LangGraph checkpointer 指到本次测试会话独占的临时目录。

    没有它，每个 ``with TestClient(app)`` 都会在仓库里留下
    ``./data/langgraph_checkpoints.sqlite``，而且不同用例会共用同一个
    checkpoint 文件（同一个 thread_id 可能读到别的用例留下的状态）。
    每个用例仍应使用独立的 session_id；这里只保证文件不落进仓库、不跨 run
    复用。
    """
    import os

    from app.config import get_settings

    path = tmp_path_factory.mktemp("langgraph-checkpoints") / "checkpoints.sqlite"
    previous = os.environ.get("APP_CHECKPOINT_DB")
    os.environ["APP_CHECKPOINT_DB"] = str(path)
    get_settings.cache_clear()
    try:
        yield path
    finally:
        if previous is None:
            os.environ.pop("APP_CHECKPOINT_DB", None)
        else:
            os.environ["APP_CHECKPOINT_DB"] = previous
        get_settings.cache_clear()


@pytest.fixture(scope="session", autouse=True)
def clean_test_db():
    """整个测试会话开始前清一次库：messages / tickets / conversations /
    low_confidence_questions。

    为什么需要它：`/api/chat` 路由会**真提交**本轮写入（commit 不可撤销），
    每个驱动真路由的用例都会往这几张表里留下自己的会话行/消息行。不清的话
    customer_service_test 会**无界增长**——实测跑一次 `uv run pytest -q`
    conversations +13、messages +6，而这个库同时还是 evals/harness.py 给工具
    评估用的那一个。`scripts/seed.py` 也不会清这两张表。

    为什么放在"会话开始"而不是"会话结束"：开始清是**自愈**的——上一次跑崩
    （Ctrl-C / 被 kill / teardown 没跑完），下一次照样从干净状态出发；放在
    结束则会把崩溃留下的垃圾永久留在库里。代价是并发跑第二个 pytest 会互相
    清表（本项目测试与 evals 都是串行跑，不这么用）。

    faq 表一个字都不动：工具评估（evals）要读它，它由 scripts/seed.py 灌。
    """
    from app.config import get_settings

    get_settings.cache_clear()
    _clear_test_tables()


@pytest.fixture
async def db_session():
    """每个用例一个 session，跑完回滚。

    外层的 engine 每次测试重建，避免事件循环切换导致的连接池串用。

    ★ 唯一例外：**被测代码自己 commit 过**时不再回滚。`/api/chat` 路由就是
    这样——它必须提交本轮写入（Task 5 裁决 + ch02 的会话行），于是 fixture
    手里那笔事务在请求结束时就已 CLOSED，回滚会抛 ResourceClosedError。
    这不是被测代码的 bug，是"commit 不可撤销"的必然结果：那些行已经真落库了，
    回滚也回不来。拦在这里只会让每个驱动真路由的用例在 teardown 报错。

    所以这些行是**真留在** customer_service_test 里的，跨 run 的累积由
    clean_test_db（session 级，跑之前清）兜住。而**同一个 run 之内**它们仍然
    在库里：提交过的固定 session_id 会与后跑的用例撞 UNIQUE key（这就是
    tests/test_api.py 那条用例把 "失败会话" 换成随机 uuid 的原因）。
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
