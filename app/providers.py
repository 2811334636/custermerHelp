from langchain_openai import ChatOpenAI

from app.config import get_settings


def get_chat_model(temperature: float | None = None) -> ChatOpenAI:
    """构造上游模型。全项目唯一知道"上游是 DeepSeek"的地方。

    换 provider = 改 .env 的 base_url / api_key / model / extra_body 四行，
    正常情况下不需要动这个文件。
    """
    s = get_settings()

    # 输出上限必须走 extra_body，不能走 ChatOpenAI(max_tokens=...)：
    # langchain-openai 会无条件把它改名成 max_completion_tokens
    # （chat_models/base.py:3703、:3715-3718），DeepSeek 不认该字段，
    # 上限会静默失效（实测设 1024 却输出 1470 token）。
    extra_body = dict(s.extra_body)
    extra_body.setdefault("max_tokens", s.llm_max_output_tokens)

    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key or "MISSING",
        temperature=s.llm_temperature if temperature is None else temperature,
        extra_body=extra_body,
        max_retries=2,
    )
