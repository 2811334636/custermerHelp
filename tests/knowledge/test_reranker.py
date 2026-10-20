"""Reranker（bge-reranker-v2-m3）封装测试。

HTTP 层通过替换模块内的 `_client` 注入假客户端，测试不依赖真实 API key
或外网。
"""

import httpx
import pytest

import app.knowledge.reranker as reranker_module
from app.config import get_settings
from app.knowledge.errors import RerankerUnavailable


class _FakeClient:
    """记录请求并返回预设响应的 httpx.AsyncClient 替身。"""

    def __init__(
        self,
        *,
        results: list[dict] | None = None,
        error: Exception | None = None,
    ):
        self.results = results if results is not None else []
        self.error = error
        self.requests: list[dict] = []

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def post(
        self,
        url: str,
        *,
        json: dict,
        headers: dict,
    ) -> httpx.Response:
        self.requests.append({"url": url, "json": json, "headers": headers})
        if self.error is not None:
            raise self.error
        return httpx.Response(
            200,
            json={"results": self.results},
            request=httpx.Request("POST", url),
        )


@pytest.mark.asyncio
async def test_rerank_returns_sorted_indices(monkeypatch):
    """rerank 返回按分数降序的 (索引, 分数) 列表，且只保留 top_k 条。"""
    passages = [
        "运费说明：满99元包邮，不满收8元配送费",
        "退货政策：7天无理由退货",
        "发货时效：下单后24小时内发货",
    ]
    # 上游故意打乱顺序：不能依赖服务端排序，封装层必须自己保证降序。
    fake = _FakeClient(
        results=[
            {"index": 2, "relevance_score": 0.12},
            {"index": 0, "relevance_score": 0.94},
            {"index": 1, "relevance_score": 0.37},
        ]
    )
    monkeypatch.setattr(reranker_module, "_client", lambda: fake)
    monkeypatch.setattr(
        reranker_module, "_RERANKER_URL", "https://reranker.test/v1/rerank"
    )

    results = await reranker_module.rerank("邮费是多少", passages, top_k=2)

    assert len(results) == 2, "应按 top_k 截断"
    assert results[0][0] == 0, "第一条passage与query最相关"
    assert results[0][1] > results[1][1], "分数降序"
    assert all(
        isinstance(idx, int) and isinstance(score, float) for idx, score in results
    )
    assert results == [(0, 0.94), (1, 0.37)]

    s = get_settings()
    request = fake.requests[0]
    assert request["url"] == "https://reranker.test/v1/rerank"
    assert request["json"] == {
        "model": s.reranker_model,
        "query": "邮费是多少",
        "documents": passages,
        "top_n": 2,
        "return_documents": False,
    }
    assert request["headers"]["Authorization"] == f"Bearer {s.reranker_api_key}"


@pytest.mark.asyncio
async def test_rerank_api_failure_raises(monkeypatch):
    """重排 API 失败时抛出 RerankerUnavailable。"""
    fake = _FakeClient(error=httpx.ConnectError("connection refused"))
    monkeypatch.setattr(reranker_module, "_client", lambda: fake)
    monkeypatch.setattr(
        reranker_module, "_RERANKER_URL", "https://reranker.test/v1/rerank"
    )

    with pytest.raises(RerankerUnavailable):
        await reranker_module.rerank("test", ["passage1"], top_k=1)
