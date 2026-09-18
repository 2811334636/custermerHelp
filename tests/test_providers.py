from langchain_openai import ChatOpenAI

from app.providers import get_chat_model


def test_model_uses_configured_endpoint(monkeypatch):
    monkeypatch.setenv("APP_LLM_BASE_URL", "https://api.deepseek.com/")
    monkeypatch.setenv("APP_LLM_MODEL", "deepseek-flash")
    monkeypatch.setenv("APP_LLM_API_KEY", "test-key")
    from app.config import get_settings

    get_settings.cache_clear()
    m = get_chat_model()
    assert isinstance(m, ChatOpenAI)
    assert m.model_name == "deepseek-flash"
    assert str(m.openai_api_base) == "https://api.deepseek.com/"
    get_settings.cache_clear()


def test_extra_body_carries_thinking_disabled(monkeypatch):
    """这是本项目的命门：thinking 没关掉，对话会一个字都吐不出来（spec §3.2）。"""
    monkeypatch.setenv("APP_LLM_EXTRA_BODY", '{"thinking":{"type":"disabled"}}')
    from app.config import get_settings

    get_settings.cache_clear()
    m = get_chat_model()
    # 只断言 thinking 这一项：extra_body 里还会带默认的 max_tokens（见 providers.py
    # 的 setdefault），整体相等断言会与 Step 4 的生产代码自相矛盾。
    assert m.extra_body["thinking"] == {"type": "disabled"}
    get_settings.cache_clear()


def test_temperature_override(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_chat_model(temperature=0.0).temperature == 0.0
    assert get_chat_model().temperature == get_settings().llm_temperature
    get_settings.cache_clear()


def test_max_tokens_goes_through_extra_body_not_constructor(monkeypatch):
    """Ruling 11：输出上限走构造参数会被 langchain-openai 改名成
    max_completion_tokens，DeepSeek 不认，上限静默失效（实测设 1024 输出 1470）。
    必须走 extra_body。"""
    monkeypatch.setenv("APP_LLM_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("APP_LLM_EXTRA_BODY", '{"thinking":{"type":"disabled"}}')
    from app.config import get_settings

    get_settings.cache_clear()
    m = get_chat_model()
    assert m.extra_body["max_tokens"] == 1024
    assert m.extra_body["thinking"] == {"type": "disabled"}
    assert m.max_tokens is None  # 绝不能同时走构造参数
    get_settings.cache_clear()


def test_config_extra_body_can_override_max_tokens(monkeypatch):
    """配置里显式给了 max_tokens 时，以配置为准，不被默认值覆盖。"""
    monkeypatch.setenv("APP_LLM_EXTRA_BODY", '{"max_tokens": 256}')
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_chat_model().extra_body["max_tokens"] == 256
    get_settings.cache_clear()
