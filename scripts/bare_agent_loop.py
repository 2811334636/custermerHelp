"""裸 Agent 循环：手写 ReAct，给后面的 LangGraph 祛魅。

状态就是一个消息列表，控制流就是一个 ``for``：模型要工具就执行、把结果
回灌，直到模型不再要工具（或步数耗尽）。ch05 的图把同样的循环显式化成
节点和条件边，这里先证明“没有任何魔法”。

单元测试用脚本化的假模型/假 registry，不碰真实模型与数据库。CLI 才构造
真实 provider/registry，供手工演示：

    uv run python -m scripts.bare_agent_loop --query "查订单 1001"
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.workflow.texts import STEP_LIMIT_TEXT


async def run_bare_agent(
    model: Any,
    registry: Any,
    messages: list[BaseMessage],
    *,
    max_steps: int = 6,
) -> list[BaseMessage]:
    """跑一个手写 ReAct 循环，返回追加了 assistant/tool 消息的完整序列。

    返回新列表，不修改调用方传入的 ``messages``。模型每次返回的
    ``tool_calls`` 逐个交给 ``registry.execute``；工具失败也照常回灌
    ``outcome.content``（失败文案本身就是要给模型看的信息），不中断循环。
    步数耗尽时追加 ``STEP_LIMIT_TEXT`` 收敛，避免把“还需要工具”的
    assistant 消息当成最终回答。
    """
    convo = list(messages)
    for _ in range(max_steps):
        resp = await model.ainvoke(convo)
        convo.append(resp)
        if not resp.tool_calls:
            return convo
        for call in resp.tool_calls:
            outcome = await registry.execute(
                call["name"],
                call["args"],
                call["id"],
                conversation_id=1,
            )
            convo.append(ToolMessage(content=outcome.content, tool_call_id=call["id"]))

    convo.append(AIMessage(content=STEP_LIMIT_TEXT))
    return convo


class _TracedModel:
    """CLI 用的日志壳：每步打印工具调用与正文，不改变循环逻辑。"""

    def __init__(self, model: Any) -> None:
        self._model = model
        self.step = 0

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.step += 1
        resp = await self._model.ainvoke(messages)
        names = [call["name"] for call in resp.tool_calls] or ["（无）"]
        print(f"[step {self.step}] tool_calls={names}")
        if resp.content:
            print(f"          content={resp.content!r}")
        return resp


class _TracedRegistry:
    """CLI 用的日志壳：打印每次工具执行的结果。"""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    async def execute(
        self,
        name: str,
        args: dict,
        call_id: str,
        conversation_id: int | None = None,
    ) -> Any:
        outcome = await self._registry.execute(
            name, args, call_id, conversation_id=conversation_id
        )
        status = "ok" if outcome.ok else "失败"
        print(f"          tool {name}({args}) -> {status}: {outcome.content!r}")
        return outcome


async def main(query: str) -> None:
    from app.db.base import AsyncSessionLocal, engine
    from app.providers import get_chat_model
    from app.tools import build_registry

    try:
        async with AsyncSessionLocal() as session:
            registry = build_registry(session)
            model = get_chat_model().bind_tools(registry.as_langchain_tools())
            out = await run_bare_agent(
                _TracedModel(model),
                _TracedRegistry(registry),
                [HumanMessage(query)],
            )
    finally:
        await engine.dispose()

    print(f"\n[final] {out[-1].content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="裸 Agent 循环（祛魅演示）")
    parser.add_argument("--query", required=True, help="用户问题")
    args = parser.parse_args()
    asyncio.run(main(args.query))
