import json

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings


def test_settings_defaults(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("APP_"):
            monkeypatch.delenv(k)
    s = Settings(_env_file=None)
    assert s.llm_base_url == "https://api.deepseek.com/"
    assert s.llm_model == "deepseek-flash"
    assert s.session_max_sessions == 200
    assert s.session_max_messages == 100
    assert s.extra_body == {"thinking": {"type": "disabled"}}


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("APP_LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("APP_SESSION_MAX_SESSIONS", "512")
    s = Settings(_env_file=None)
    assert s.llm_model == "deepseek-v4-pro"
    assert s.session_max_sessions == 512


def test_extra_body_parses_json_string(monkeypatch):
    monkeypatch.setenv("APP_LLM_EXTRA_BODY", "{}")
    assert Settings(_env_file=None).extra_body == {}


def test_get_settings_is_cached():
    assert get_settings() is get_settings()


@pytest.mark.parametrize(
    "env,value",
    [
        ("APP_TOKEN_CHARS_PER_TOKEN", "0"),   # _count_tokens 的除数，0 会 ZeroDivisionError
        ("APP_TOKEN_CHARS_PER_TOKEN", "-1"),
        ("APP_LLM_MAX_OUTPUT_TOKENS", "0"),
        ("APP_SESSION_MAX_SESSIONS", "0"),
        ("APP_SESSION_MAX_MESSAGES", "-5"),
        # 修复轮 · Minor 8：工具旋钮原先没有边界。负数 max_retries 会让
        # attempts = -1 + 1 = 0，重试循环一次都不跑（last_error 停在 None）。
        ("APP_TOOL_TIMEOUT_S", "0"),
        ("APP_TOOL_MAX_RETRIES", "-1"),
    ],
)
def test_non_positive_numeric_settings_fail_at_startup(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_zero_tool_retries_is_allowed(monkeypatch):
    """ge=0 的边界：0 次重试是合法配置（只试一次），别误伤成非法。"""
    monkeypatch.setenv("APP_TOOL_MAX_RETRIES", "0")
    assert Settings(_env_file=None).tool_max_retries == 0


def test_malformed_extra_body_fails_at_startup(monkeypatch):
    """坏 JSON 必须在启动时报错，而不是拖到第一个请求才 500。"""
    monkeypatch.setenv("APP_LLM_EXTRA_BODY", "{not json")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_extra_body_must_be_json_object(monkeypatch):
    monkeypatch.setenv("APP_LLM_EXTRA_BODY", "[1,2]")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_knowledge_defaults():
    from app.config import Settings

    s = Settings(_env_file=None)
    assert s.embedding_base_url == "https://api.siliconflow.cn/v1"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.embedding_dim == 1024
    assert s.milvus_uri == "http://127.0.0.1:19530"
    assert s.milvus_collection == "knowledge"
    assert s.retrieval_top_k == 3
    assert s.chunk_max_chars == 400
    assert s.chunk_overlap_chars == 80
    assert s.embed_batch_size == 16


def test_chunk_params_must_be_sane():
    import pytest
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_max_chars=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_overlap_chars=-1)
