import json
from functools import lru_cache

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
    llm_max_output_tokens: int = 1024
    llm_temperature: float = 0.3
    history_token_budget: int = 4096
    session_max_sessions: int = 200
    session_max_messages: int = 100

    @property
    def extra_body(self) -> dict:
        """APP_LLM_EXTRA_BODY 是 JSON 字符串，原样透传给上游。

        换 provider 时改这一个值即可，无需改代码（见 spec §8）。
        """
        return json.loads(self.llm_extra_body)


@lru_cache
def get_settings() -> Settings:
    return Settings()
