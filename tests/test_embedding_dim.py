from types import SimpleNamespace

import pytest

from app.knowledge.embedding import EmbeddingDimensionError, embed_texts


async def test_embed_texts_returns_configured_dim(monkeypatch):
    async def fake(*a, **kw):
        return [[0.0] * 1024]

    monkeypatch.setattr("app.knowledge.embedding._embed_batch", fake)
    out = await embed_texts(["测试"])
    assert len(out[0]) == 1024


async def test_embed_texts_rejects_wrong_dim(monkeypatch):
    async def fake(*a, **kw):
        return [[0.0] * 512]

    monkeypatch.setattr("app.knowledge.embedding._embed_batch", fake)
    with pytest.raises(EmbeddingDimensionError) as exc:
        await embed_texts(["测试"])
    assert "512" in str(exc.value) and "1024" in str(exc.value)


async def test_embed_texts_batches_and_preserves_input_order(monkeypatch):
    """按 embed_batch_size 分批，且输出顺序与输入严格一致。"""
    import app.knowledge.embedding as mod

    monkeypatch.setattr(
        mod,
        "get_settings",
        lambda: SimpleNamespace(embedding_dim=3, embed_batch_size=2),
    )
    seen: list[list[str]] = []

    async def fake(texts):
        seen.append(list(texts))
        return [[float(t), float(t), float(t)] for t in texts]

    monkeypatch.setattr(mod, "_embed_batch", fake)

    out = await embed_texts(["1", "2", "3", "4", "5"])

    assert seen == [["1", "2"], ["3", "4"], ["5"]]
    assert [int(v[0]) for v in out] == [1, 2, 3, 4, 5]


async def test_embed_texts_empty_input_returns_empty_without_calling_api(monkeypatch):
    async def fake(*a, **kw):  # pragma: no cover - 不应被调用
        raise AssertionError("空输入不应调用 embedding 接口")

    monkeypatch.setattr("app.knowledge.embedding._embed_batch", fake)
    assert await embed_texts([]) == []
