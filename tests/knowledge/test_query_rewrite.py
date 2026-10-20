"""Query 改写测试。

LLM 调用通过替换模块内的 `_call_llm` 注入替身，测试不依赖真实 API key
或外网，因此可在基础设施未启动时用 `--noconftest` 运行。
"""

import pytest
from langchain_core.messages import AIMessage, BaseMessage

import app.knowledge.query_rewrite as qr_module
from app.knowledge.query_rewrite import rewrite_query


@pytest.mark.asyncio
async def test_rewrite_query_normalizes(monkeypatch):
    """口语归一化：邮费咋算 → 邮费如何计算。"""

    async def _fake_call_llm(query: str) -> str:
        assert query == "邮费咋算"
        return "邮费如何计算"

    monkeypatch.setattr(qr_module, "_call_llm", _fake_call_llm)

    rewritten = await rewrite_query("邮费咋算")
    assert "邮费" in rewritten
    assert "咋" not in rewritten  # 口语词被归一化


@pytest.mark.asyncio
async def test_rewrite_query_failure_fallback(monkeypatch):
    """改写失败时返回原 query（不阻塞检索）。"""

    async def _failing_call_llm(query: str) -> str:
        raise RuntimeError("LLM error")

    monkeypatch.setattr(qr_module, "_call_llm", _failing_call_llm)

    result = await rewrite_query("原始query")
    assert result == "原始query"


@pytest.mark.asyncio
async def test_call_llm_uses_shared_chat_model(monkeypatch):
    """`_call_llm` 走 providers.get_chat_model（temperature=0），不另建客户端。"""
    constructor_kwargs: list[dict[str, object]] = []
    received_messages: list[list[BaseMessage]] = []

    class _FakeModel:
        async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
            received_messages.append(messages)
            return AIMessage(content="邮费如何计算")

    def _fake_get_chat_model(**kwargs: object) -> _FakeModel:
        constructor_kwargs.append(kwargs)
        return _FakeModel()

    monkeypatch.setattr(qr_module, "get_chat_model", _fake_get_chat_model)

    result = await qr_module._call_llm("邮费咋算")

    assert result == "邮费如何计算"
    assert constructor_kwargs == [{"temperature": 0}]
    assert "邮费咋算" in received_messages[0][0].content


@pytest.mark.asyncio
async def test_rewrite_query_overlong_output_fallback(monkeypatch):
    """模型返回过长解释（疑似没听指令）时回退到原 query。"""

    async def _verbose_call_llm(query: str) -> str:
        return "这是一段非常冗长的解释，" * 20

    monkeypatch.setattr(qr_module, "_call_llm", _verbose_call_llm)

    result = await rewrite_query("邮费咋算")
    assert result == "邮费咋算"
