"""重排模型封装：bge-reranker-v2-m3。

SiliconFlow 的 /rerank 接口与 OpenAI 兼容服务的调用方式一致：
POST {base_url}/rerank，返回 results 里每条的 index 与 relevance_score。
"""

import httpx

from app.config import get_settings
from app.knowledge.errors import RerankerUnavailable

_RERANKER_URL: str | None = None


def _get_reranker_url() -> str:
    """拼出 /rerank 端点；进程内只算一次。"""
    global _RERANKER_URL
    if _RERANKER_URL is None:
        s = get_settings()
        _RERANKER_URL = f"{s.reranker_base_url.rstrip('/')}/rerank"
    return _RERANKER_URL


def _client() -> httpx.AsyncClient:
    """创建 HTTP 客户端；独立成函数便于测试注入替身。"""
    return httpx.AsyncClient(timeout=float(get_settings().reranker_timeout))


async def rerank(
    query: str, passages: list[str], top_k: int = 10
) -> list[tuple[int, float]]:
    """重排：返回 (原索引, 分数) 列表，按分数降序，最多 top_k 条。

    Raises:
        RerankerUnavailable: API 连接、鉴权、HTTP 状态或响应格式异常。
    """
    if not passages or top_k <= 0:
        # 无候选或显然不需要结果时不必打上游。
        return []

    s = get_settings()
    url = _get_reranker_url()
    payload = {
        "model": s.reranker_model,
        "query": query,
        "documents": passages,
        "top_n": top_k,
        "return_documents": False,
    }

    try:
        async with _client() as client:
            resp = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {s.reranker_api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            # 响应格式：{"results": [{"index": 0, "relevance_score": 0.85}, ...]}
            ranked = [
                (int(item["index"]), float(item["relevance_score"]))
                for item in data["results"]
            ]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise RerankerUnavailable(f"Reranker API failed: {exc}") from exc

    # 不依赖上游一定按分数排序：封装层自己保证降序契约。
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:top_k]
