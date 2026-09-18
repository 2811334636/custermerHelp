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
    assert s.history_token_budget == 4096
    assert s.session_max_sessions == 200
    assert s.session_max_messages == 100
    assert s.extra_body == {"thinking": {"type": "disabled"}}


def test_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("APP_LLM_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("APP_HISTORY_TOKEN_BUDGET", "512")
    s = Settings(_env_file=None)
    assert s.llm_model == "deepseek-v4-pro"
    assert s.history_token_budget == 512


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
        ("APP_HISTORY_TOKEN_BUDGET", "0"),
        ("APP_LLM_MAX_OUTPUT_TOKENS", "0"),
        ("APP_SESSION_MAX_SESSIONS", "0"),
        ("APP_SESSION_MAX_MESSAGES", "-5"),
    ],
)
def test_non_positive_numeric_settings_fail_at_startup(monkeypatch, env, value):
    monkeypatch.setenv(env, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_malformed_extra_body_fails_at_startup(monkeypatch):
    """坏 JSON 必须在启动时报错，而不是拖到第一个请求才 500。"""
    monkeypatch.setenv("APP_LLM_EXTRA_BODY", "{not json")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_extra_body_must_be_json_object(monkeypatch):
    monkeypatch.setenv("APP_LLM_EXTRA_BODY", "[1,2]")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
