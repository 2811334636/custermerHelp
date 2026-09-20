import asyncio

import pytest
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.tools.context import ToolContext, current_context, tool_context
from app.tools.registry import ToolOutcome, ToolRegistry, ToolSpec


class EchoArgs(BaseModel):
    text: str = Field(description="要回显的文本")


@tool(args_schema=EchoArgs)
async def echo(text: str) -> str:
    """回显输入。"""
    return f"echo:{text}"


@tool(args_schema=EchoArgs)
async def slow(text: str) -> str:
    """故意超时。"""
    await asyncio.sleep(10)
    return "never"


_calls = {"n": 0}


@tool(args_schema=EchoArgs)
async def flaky(text: str) -> str:
    """前两次超时，第三次成功——用来验证重试确实发生了。"""
    _calls["n"] += 1
    if _calls["n"] < 3:
        await asyncio.sleep(10)
    return f"ok after {_calls['n']}"


def make_registry(*tools, timeout_s=0.1, max_retries=2):
    return ToolRegistry([ToolSpec(tool=t, timeout_s=timeout_s, max_retries=max_retries)
                         for t in tools])


async def test_registry_lists_tools_for_binding():
    reg = make_registry(echo)
    names = [t.name for t in reg.as_langchain_tools()]
    assert names == ["echo"]


async def test_execute_success():
    reg = make_registry(echo)
    out = await reg.execute("echo", {"text": "hi"}, "call_1")
    assert isinstance(out, ToolOutcome)
    assert out.ok is True
    assert "echo:hi" in out.content
    assert out.call_id == "call_1"
    assert out.elapsed_ms >= 0


async def test_execute_unknown_tool():
    reg = make_registry(echo)
    out = await reg.execute("不存在的工具", {}, "call_2")
    assert out.ok is False
    assert "没有" in out.content or "不存在" in out.content


async def test_invalid_args_rejected_without_retry():
    """参数不合法不重试——重试一次得到的还是同样的非法参数。"""
    calls_before = _calls["n"]
    reg = make_registry(flaky, timeout_s=5.0)
    out = await reg.execute("flaky", {"完全没有的参数": 1}, "call_3")
    assert out.ok is False
    assert _calls["n"] == calls_before, "参数校验失败时不该调用工具函数"


async def test_timeout_is_retried_then_reported():
    reg = make_registry(slow, timeout_s=0.05, max_retries=2)
    out = await reg.execute("slow", {"text": "x"}, "call_4")
    assert out.ok is False
    assert "超时" in out.content
    assert out.elapsed_ms >= 50


async def test_retry_can_succeed():
    _calls["n"] = 0
    reg = make_registry(flaky, timeout_s=0.05, max_retries=2)
    out = await reg.execute("flaky", {"text": "x"}, "call_5")
    assert out.ok is True
    assert _calls["n"] == 3, "应该试满 3 次才成功"


async def test_zero_retries_means_single_attempt():
    _calls["n"] = 0
    reg = make_registry(flaky, timeout_s=0.05, max_retries=0)
    out = await reg.execute("flaky", {"text": "x"}, "call_6")
    assert out.ok is False
    assert _calls["n"] == 1


async def test_context_is_visible_inside_tool():
    seen = {}

    @tool(args_schema=EchoArgs)
    async def spy(text: str) -> str:
        """窥探上下文。"""
        seen["cid"] = current_context().conversation_id
        return "ok"

    reg = make_registry(spy)
    with tool_context(42):
        await reg.execute("spy", {"text": "x"}, "call_7")
    assert seen["cid"] == 42


async def test_context_raises_outside_tool_context():
    with pytest.raises(LookupError):
        current_context()


# ---------------------------------------------------------------------------
# Ruling 3：DB session 随 ToolContext 走，不落模块级全局
# ---------------------------------------------------------------------------


async def test_session_reaches_tool_through_context():
    """工具靠 current_context().session 拿库会话——Task 7 的工具就建在这上面。"""
    seen = {}

    @tool(args_schema=EchoArgs)
    async def spy(text: str) -> str:
        """窥探上下文里的 session。"""
        seen["session"] = current_context().session
        return "ok"

    marker = object()
    reg = ToolRegistry([ToolSpec(tool=spy, timeout_s=1.0, max_retries=0)],
                       session=marker)
    out = await reg.execute("spy", {"text": "x"}, "call_8")
    assert out.ok is True
    assert seen["session"] is marker


async def test_two_registries_do_not_leak_sessions_into_each_other():
    """两个 registry 各带各的 session，**并发**执行时不能串号。

    这正是 Ruling 3 要挡的缺陷：模块级全局 `_session` 在单 worker 下会被
    后一个请求覆盖，重叠执行时 A 的工具读到 B 的库会话。用 ContextVar 承载
    后每个 asyncio 任务有自己的上下文副本，重叠也不串。

    用 barrier 把两个工具的执行卡在同一时刻，确保它们真的重叠；
    顺序执行(不重叠)是测不出串号的。
    """
    seen: dict[str, object] = {}
    rendezvous = asyncio.Barrier(2)

    @tool(args_schema=EchoArgs)
    async def spy(text: str) -> str:
        """截住，等对方也进来，再读上下文里的 session。"""
        await rendezvous.wait()
        seen[text] = current_context().session
        return "ok"

    session_a, session_b = object(), object()
    reg_a = ToolRegistry([ToolSpec(tool=spy, timeout_s=5.0, max_retries=0)],
                         session=session_a)
    reg_b = ToolRegistry([ToolSpec(tool=spy, timeout_s=5.0, max_retries=0)],
                         session=session_b)

    out_a, out_b = await asyncio.gather(
        reg_a.execute("spy", {"text": "a"}, "call_9a"),
        reg_b.execute("spy", {"text": "b"}, "call_9b"),
    )

    assert (out_a.ok, out_b.ok) == (True, True)
    assert seen["a"] is session_a
    assert seen["b"] is session_b


async def test_tool_context_carries_session():
    """tool_context() 收 session，current_context() 原样取回。"""
    marker = object()
    with tool_context(7, marker):
        ctx = current_context()
    assert ctx.conversation_id == 7
    assert ctx.session is marker
    assert isinstance(ctx, ToolContext)


async def test_code_bug_in_tool_becomes_result_not_exception():
    """工具里的代码 bug 不能掀掉对话——它要变成一条工具结果回灌给模型。

    这是 spec §11.1「查不到不是错误，坏了才是」里"坏了"那一支：
    异常被包成 ToolOutcome，且非瞬态故障不重试。
    """
    calls = {"n": 0}

    @tool(args_schema=EchoArgs)
    async def broken(text: str) -> str:
        """真坏了。"""
        calls["n"] += 1
        raise RuntimeError("boom")

    reg = make_registry(broken, timeout_s=1.0, max_retries=2)
    out = await reg.execute("broken", {"text": "x"}, "call_10")
    assert out.ok is False
    assert "执行失败" in out.content
    assert calls["n"] == 1, "非瞬态故障只试一次"
