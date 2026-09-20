"""评估共用的「完整回合」执行器 —— 镜像生产路径（`app/chat.py` 的工具轮）。

**为什么两个评估集都必须用它**：只调一次模型拿到的不是用户看到的东西。
模型若决定调工具，那一轮的正文是**空的**（实测 fabricate-1/fabricate-3 就是
空字符串），judge 会对着空串判"否"、长度断言也变成 0 字——评估于是测了个
不存在的形态。生产是「先转发后决断 → 执行工具 → 第二次调用（不绑工具）」，
评估必须走同一条路。

时序对应 spec §7.1 的 ③⑤⑥：
  ③ 第一次调用**流式**并累积（与生产的流式探测一致，不用 ainvoke）
  ⑤ 执行工具，结果包成 ToolMessage
  ⑥ 第二次调用**不绑工具**——所以这里要传两个模型
"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from app.tools.registry import ToolRegistry


@dataclass(frozen=True)
class Turn:
    """一个完整回合的可断言产物。"""

    tool_calls: list[dict]
    tool_output: str   # 全部工具结果的拼接；没调工具时为空串
    final_text: str    # 用户实际看到的最终回复


async def run_turn(
    model,
    unbound_model,
    rendered: list[BaseMessage],
    registry: ToolRegistry,
    *,
    conversation_id: int = 0,
) -> Turn:
    """跑完整回合。`model` 须已绑定工具，`unbound_model` 须未绑定。"""
    gathered: AIMessage | None = None
    async for chunk in model.astream(rendered):
        gathered = chunk if gathered is None else gathered + chunk
    if gathered is None:  # 上游一个字都没吐：按"没调工具、没正文"处理，别让脚本崩
        return Turn([], "", "")

    calls = list(gathered.tool_calls or [])
    if not calls:
        return Turn([], "", gathered.text)

    tool_msgs: list[ToolMessage] = []
    for call in calls:
        outcome = await registry.execute(
            call["name"], call["args"], call["id"], conversation_id=conversation_id
        )
        tool_msgs.append(ToolMessage(content=outcome.content, tool_call_id=call["id"]))

    final = await unbound_model.ainvoke([*rendered, gathered, *tool_msgs])
    return Turn(calls, "\n\n".join(m.content for m in tool_msgs), final.text)
