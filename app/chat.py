"""对话适配层仍需要的错误分类工具（spec §9）。

★ 为什么这里直接 import 了 openai SDK：``classify_exception`` 需要按上游异常
的具体类型（AuthenticationError / RateLimitError / ...）分错误码，而
``BaseChatModel.astream`` 的契约只保证抛 ``BaseException``，不暴露类型层级。
这是**有意为之的已知泄漏**：换 provider 时，这个文件是除 providers.py 外唯一
需要跟着改的地方。

ch05 起模型回合由 ``app/workflow`` 的 LangGraph 图驱动，
``stream_reply_with_tools``（ch01–ch04 的单轮循环）已删除；本文件只剩这条
被 API 适配层复用的分类函数。
"""

import openai

_AUTH_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError)
_UNAVAILABLE_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def classify_exception(exc: BaseException) -> str:
    """把上游异常映射成对外的错误码（spec §9）。"""
    if isinstance(exc, _AUTH_ERRORS):
        return "upstream_auth"
    if isinstance(exc, openai.RateLimitError):
        return "upstream_rate_limit"
    if isinstance(exc, _UNAVAILABLE_ERRORS):
        return "upstream_unavailable"
    if isinstance(exc, openai.APIStatusError):
        return "upstream_unavailable"
    return "internal_error"
