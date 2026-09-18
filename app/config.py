import json
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        extra="ignore",
    )

    llm_base_url: str = "https://api.deepseek.com/"
    llm_api_key: str = ""
    llm_model: str = "deepseek-flash"
    llm_extra_body: str = '{"thinking":{"type":"disabled"}}'
    llm_max_output_tokens: int = Field(default=1024, gt=0)
    # 温度为 0 是合法且常用的（evals 就固定用 0），故这里是 ge=0 而非 gt=0
    llm_temperature: float = Field(default=0.3, ge=0)
    history_token_budget: int = Field(default=4096, gt=0)
    # token_chars_per_token 是 app/context.py::_count_tokens 的除数，
    # 0 会在每次请求时 ZeroDivisionError，必须在启动时就挡掉
    token_chars_per_token: float = Field(default=1.5, gt=0)
    session_max_sessions: int = Field(default=200, gt=0)
    session_max_messages: int = Field(default=100, gt=0)

    @field_validator("llm_extra_body")
    @classmethod
    def _validate_extra_body(cls, v: str) -> str:
        """启动时校验 JSON，而不是拖到第一次请求。

        APP_LLM_EXTRA_BODY 是手填的 JSON 字符串；写坏了属于配置错误，
        应该在进程启动时立刻报错，而不是运行中第一个请求才 500。
        """
        try:
            parsed = json.loads(v)
        except json.JSONDecodeError as exc:
            raise ValueError(f"APP_LLM_EXTRA_BODY 不是合法 JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("APP_LLM_EXTRA_BODY 必须是 JSON 对象")
        return v

    @property
    def extra_body(self) -> dict:
        """APP_LLM_EXTRA_BODY 是 JSON 字符串，原样透传给上游。

        换 provider 时改这一个值即可，无需改代码（见 spec §8）。
        """
        return json.loads(self.llm_extra_body)


@lru_cache
def get_settings() -> Settings:
    return Settings()
