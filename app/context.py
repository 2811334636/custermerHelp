from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.messages.utils import trim_messages

from app.config import get_settings


def _count_tokens(messages: list[BaseMessage]) -> int:
    """按中文字符密度估算 token。

    不用 langchain_core 的 count_tokens_approximately：它按英文的 4 字符/token
    估算，对中文低估 58%（Task 2 实测 估算 25 / 实测 59）。中文实测密度
    1.47 字/token，取 1.5 后估算与实测比值 1.02。
    """
    chars = sum(len(str(m.content)) for m in messages)
    return int(chars / get_settings().token_chars_per_token)


def prepare_messages(
    history: list[BaseMessage],
    current: HumanMessage,
    *,
    token_budget: int,
) -> list[BaseMessage]:
    """剪裁历史到 token 预算内，并追加当前用户消息。

    history 不含当前消息 —— 当前消息由本函数无条件追加，
    确保它永远不会被剪裁掉。

    start_on="human" 保证剪裁后首条是 human 消息，
    避免出现"助手的回答还在、对应的提问已被剪掉"的错位。
    """
    if not history:
        return [current]

    try:
        trimmed = trim_messages(
            history,
            strategy="last",
            token_counter=_count_tokens,
            max_tokens=token_budget,
            start_on="human",
        )
    except ValueError:
        # 极端小的预算下 trim_messages 可能无解而抛错。
        # "当前消息永不被剪裁"这个不变量优先于复用库函数。
        return [current]

    return [*trimmed, current]
