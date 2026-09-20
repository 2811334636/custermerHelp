"""工具基础设施：注册管理、参数 Schema 校验、执行错误处理、超时重试。

设计原则（spec §11.1）：「查不到」不是错误，「坏了」才是。
业务空结果由工具自己正常返回；异常只留给真故障（超时、连接断开、代码 bug）。
"""

import asyncio
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.tools import BaseTool
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.tools.context import current_context, tool_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolSpec:
    tool: BaseTool
    timeout_s: float = 3.0
    max_retries: int = 2


@dataclass(frozen=True)
class ToolOutcome:
    call_id: str
    ok: bool
    content: str      # 回灌给模型的文本
    summary: str      # 给前端徽章 tooltip 的一句话
    elapsed_ms: int


class ToolRegistry:
    def __init__(
        self, specs: Iterable[ToolSpec], session: AsyncSession | None = None
    ) -> None:
        self._specs: dict[str, ToolSpec] = {s.tool.name: s for s in specs}
        # 本 registry 的执行要用哪个库会话。放实例上、由 execute() 塞进
        # ContextVar —— 不放模块级全局，否则重叠请求会串号（见 context.py）。
        self._session = session

    def as_langchain_tools(self) -> list[BaseTool]:
        """给 model.bind_tools() 用。"""
        return [s.tool for s in self._specs.values()]

    def get(self, name: str) -> ToolSpec:
        spec = self._specs.get(name)
        if spec is None:
            known = "、".join(sorted(self._specs))
            raise KeyError(f"没有名为 {name} 的工具。可用工具：{known}")
        return spec

    def _effective_conversation_id(self, conversation_id: int | None) -> int:
        """决定这次执行注入哪个会话 id。

        显式给了就用它；没给则沿用外层已有的上下文（调用方可能已经用
        `with tool_context(cid)` 包好了）；都没有才退到 0。
        """
        if conversation_id is not None:
            return conversation_id
        try:
            return current_context().conversation_id
        except LookupError:
            return 0

    def _effective_session(self, session: AsyncSession | None) -> AsyncSession | None:
        """决定这次执行注入哪个库会话 —— 与 _effective_conversation_id 对称。

        ★ 为什么要这一层（修复轮 · Minor 9）：会话原来只取 `self._session`，
        一个没带 session 的 registry 会把外层 `tool_context(cid, session=...)`
        里的 session **冲成 None**——同一个上下文里 conversation_id 会继承、
        session 却不会，两个字段行为不一致。显式给了就用它；没给则沿用外层
        已有的；都没有才退到 None。
        """
        if session is not None:
            return session
        try:
            return current_context().session
        except LookupError:
            return None

    async def execute(
        self, name: str, raw_args: dict, call_id: str, *, conversation_id: int | None = None
    ) -> ToolOutcome:
        started = time.monotonic()

        def elapsed() -> int:
            return int((time.monotonic() - started) * 1000)

        # ① 查注册表
        try:
            spec = self.get(name)
        except KeyError as exc:
            return ToolOutcome(call_id, False, str(exc), "工具不存在", elapsed())

        # ② Schema 校验 —— 失败不重试（重试一次得到的还是同样的非法参数）
        try:
            args = spec.tool.args_schema.model_validate(raw_args)
        except ValidationError as exc:
            msg = f"参数不合法：{exc.errors()}"
            logger.warning("工具 %s 参数校验失败 call_id=%s: %s", name, call_id, exc.errors())
            return ToolOutcome(call_id, False, msg, "参数不合法", elapsed())

        # ③④ 执行 + 超时重试（只重试超时/连接类故障）
        attempts = spec.max_retries + 1
        last_error: Exception | None = None
        # 实际跑了几次。回灌给模型的文案必须写**实际**次数：
        # 代码 bug 走 break 时只试了 1 次，写"已尝试 3 次"就是喂给模型一条假事实。
        tried = 0

        with tool_context(
            self._effective_conversation_id(conversation_id),
            self._effective_session(self._session),
        ):
            for attempt in range(1, attempts + 1):
                tried = attempt
                try:
                    result = await asyncio.wait_for(
                        spec.tool.ainvoke(args.model_dump()), timeout=spec.timeout_s
                    )
                    text = result if isinstance(result, str) else str(result)
                    return ToolOutcome(call_id, True, text, self._summarize(text), elapsed())
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    last_error = exc
                    logger.warning(
                        "工具 %s 第 %d/%d 次超时（%.1fs）call_id=%s",
                        name, attempt, attempts, spec.timeout_s, call_id,
                    )
                except (ConnectionError, OSError) as exc:
                    last_error = exc
                    logger.warning(
                        "工具 %s 第 %d/%d 次连接故障 call_id=%s: %s",
                        name, attempt, attempts, call_id, exc,
                    )
                except Exception as exc:  # noqa: BLE001 — 代码 bug 也归为工具失败，回灌而非中断对话
                    last_error = exc
                    logger.exception("工具 %s 执行异常 call_id=%s", name, call_id)
                    break  # 非瞬态故障不重试

        kind = "超时" if isinstance(last_error, (TimeoutError, asyncio.TimeoutError)) else "执行失败"
        content = f"工具 {name} {kind}（已尝试 {tried} 次）：{type(last_error).__name__}。暂时查不到相关信息。"
        return ToolOutcome(call_id, False, content, f"{kind}", elapsed())

    @staticmethod
    def _summarize(text: str, limit: int = 60) -> str:
        flat = " ".join(text.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"
