"""``app/chat.py`` 剩下的唯一职责：上游异常 → 对外错误码（spec §9）。

ch05 Task 9 删除了 ch01–ch04 的单轮循环 ``stream_reply_with_tools``：模型回合
现在由 ``app/workflow`` 的 LangGraph 图驱动。原文件里仍有价值的端到端断言
（工具事件顺序、先转发后决断、错误分类、收敛）已迁到
``tests/test_api_workflow.py``；这里只钉住被 API 适配层复用的分类函数。
"""

import openai
import pytest

from app.chat import classify_exception


def _resp(status):
    import httpx

    return httpx.Response(status, request=httpx.Request("POST", "https://x/"))


@pytest.mark.parametrize(
    "exc,expected",
    [
        (openai.AuthenticationError("x", response=_resp(401), body=None), "upstream_auth"),
        (openai.PermissionDeniedError("x", response=_resp(403), body=None), "upstream_auth"),
        (openai.RateLimitError("x", response=_resp(429), body=None), "upstream_rate_limit"),
        (openai.InternalServerError("x", response=_resp(500), body=None), "upstream_unavailable"),
        (openai.APIConnectionError(request=None), "upstream_unavailable"),
        (ValueError("随便什么"), "internal_error"),
    ],
)
def test_classify_exception(exc, expected):
    assert classify_exception(exc) == expected
