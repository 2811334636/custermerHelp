"""LangChain 消息 ↔ messages 表行 的双向转换。

★ 本模块是全项目唯一同时知道 LangChain 消息类型和表结构的地方。
其他模块都不该做这个转换（spec §9.3）。
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from app.db.models import Message

_ROLE_TO_CLASS = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "tool": ToolMessage,
}


def to_rows(messages: list[BaseMessage]) -> list[dict]:
    """转成可直接喂给 repositories.message.append 的 kwargs 列表。"""
    out: list[dict] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content,
                        "tool_calls": None, "tool_call_id": None})
        elif isinstance(m, AIMessage):
            calls = m.tool_calls or None
            out.append({"role": "assistant", "content": m.content or None,
                        "tool_calls": calls, "tool_call_id": None})
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content,
                        "tool_calls": None, "tool_call_id": m.tool_call_id})
        else:
            raise TypeError(f"无法落库的消息类型: {type(m).__name__}")
    return out


def from_rows(rows: list[Message]) -> list[BaseMessage]:
    """表行还原成 LangChain 消息。"""
    out: list[BaseMessage] = []
    for r in rows:
        if r.role == "user":
            out.append(HumanMessage(r.content or ""))
        elif r.role == "assistant":
            out.append(AIMessage(content=r.content or "", tool_calls=r.tool_calls or []))
        elif r.role == "tool":
            out.append(ToolMessage(content=r.content or "", tool_call_id=r.tool_call_id or ""))
        else:
            raise ValueError(f"未知的 role: {r.role}")
    return out
