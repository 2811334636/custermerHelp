from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings


def get_chat_model(temperature: float | None = None) -> ChatOpenAI:
    """构造上游模型。全项目唯一知道"上游是 DeepSeek"的地方。

    换 provider = 改 .env 的 base_url / api_key / model / extra_body 四行，
    正常情况下不需要动这个文件。

    `temperature` 参数仅供 evals/ 与测试按需覆盖；HTTP 依赖注入**必须**走下面
    的 get_default_chat_model()，绝不能在 Depends() 里直接用它（原因见该函数）。
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


@lru_cache(maxsize=1)
def get_default_chat_model() -> ChatOpenAI:
    """HTTP 端点的模型依赖：零参 + 进程内单例。

    这个包装函数存在有两个原因，都不能省：

    1. 安全：FastAPI 会按被依赖函数的签名做请求参数分析。直接把
       get_chat_model 交给 Depends() 会让它的 `temperature` 变成 /api/chat
       与 /api/extract 上的 **query 参数**（实测 /openapi.json 里两条路由都
       列出 ('temperature','query')），未鉴权调用者可以 `?temperature=0.9`
       改掉生产采样。
    2. 性能：每次请求都构造一个新的 ChatOpenAI，也就新建一个 httpx client、
       一次新的 TLS 握手。lru_cache 让全进程只建一次。

    保留签名带 temperature 的 get_chat_model 供 evals/ 与测试使用。
    """
    return get_chat_model()
