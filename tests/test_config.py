import json

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
