from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.knowledge.errors import KnowledgeBackendUnavailable


class EmbeddingUnavailable(KnowledgeBackendUnavailable):
    """Embedding 上游连接、超时或鉴权失败等暂时性故障。

    与 EmbeddingDimensionError 区分：维度不符是配置错误，重试也不会好；
    这里是外部依赖故障，在线检索应降级。
    """


class EmbeddingDimensionError(RuntimeError):
    """上游返回的向量维度与 APP_EMBEDDING_DIM 不一致。"""


@lru_cache(maxsize=1)
def _client() -> OpenAIEmbeddings:
    """进程内复用客户端；SiliconFlow 与 OpenAI 兼容，无需额外参数。"""
    s = get_settings()
    return OpenAIEmbeddings(
        base_url=s.embedding_base_url,
        api_key=s.embedding_api_key,
        model=s.embedding_model,
        # 默认 True 会用 tiktoken 把文本转成 token ID 数组再发给上游；
        # SiliconFlow 虽不报错，但会返回语义显著变差的向量（实测余弦 0.23 vs 0.635）。
        check_embedding_ctx_length=False,
    )


async def _embed_batch(texts: list[str]) -> list[list[float]]:
    """单批向量化。独立成模块级函数，便于测试按名字 monkeypatch。

    上游连接/超时/鉴权等异常在这里统一包装成 EmbeddingUnavailable：
    - 建库时它会让该批保持【待向量化】，下次重跑补齐；
    - 在线检索时它让 query_faq 返回降级文案，而不是变成工具故障。
    维度错误在 embed_texts 里单独抛出，属于配置问题，不在这里吞掉。
    """
    try:
        return await _client().aembed_documents(texts)
    except Exception as exc:  # noqa: BLE001 — 上游故障类型由 SDK 决定
        raise EmbeddingUnavailable(str(exc)) from exc


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """分批向量化，并校验每批返回的维度与配置一致。

    输出顺序即输入顺序：按切片顺序串行调用，逐批 extend，不做并发重排。
    """
    s = get_settings()
    out: list[list[float]] = []
    for i in range(0, len(texts), s.embed_batch_size):
        batch = texts[i : i + s.embed_batch_size]
        vectors = await _embed_batch(batch)
        if len(vectors) != len(batch):
            # 少返回或多返回都会让向量与文本错位，必须在进入下游前挡住。
            raise ValueError(
                f"embedding 返回数量不符：期望 {len(batch)}，实际 {len(vectors)}"
            )
        for v in vectors:
            if len(v) != s.embedding_dim:
                raise EmbeddingDimensionError(
                    f"embedding 维度不符：期望 {s.embedding_dim}，实际 {len(v)}"
                )
        out.extend(vectors)
    return out
