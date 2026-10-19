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
    # 3307 而非 3306：宿主 3306 被 MySQL80 服务 + VMware NAT 占用，
    # 容器只能映射到 3307（见 docker-compose.yml 注释）。
    db_url: str = "mysql+aiomysql://app:apppw@127.0.0.1:3307/customer_service?charset=utf8mb4"
    default_user_id: str = "anonymous"
    # 工具执行的两个数值旋钮，与文件里其他数值设置一样带边界（修复轮 · Minor 8）。
    # timeout_s=0 会让 asyncio.wait_for 立刻超时（工具永远"超时失败"）；
    # max_retries<0 更隐蔽：attempts = max_retries + 1 变成 0，重试循环一次都不跑，
    # last_error 保持 None，回灌给模型的是"已尝试 0 次：NoneType"——一句假事实。
    tool_timeout_s: float = Field(default=3.0, gt=0)
    tool_max_retries: int = Field(default=2, ge=0)

    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = Field(default=1024, gt=0)
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_collection: str = "knowledge"
    retrieval_top_k: int = Field(default=3, gt=0)
    chunk_max_chars: int = Field(default=400, gt=0)
    chunk_overlap_chars: int = Field(default=80, ge=0)
    embed_batch_size: int = Field(default=16, gt=0)

    reranker_base_url: str = "https://api.siliconflow.cn/v1"
    reranker_api_key: str = ""
    reranker_model: str = "bge-reranker-v2-m3"
    retrieval_dense_top_k: int = Field(default=50, gt=0)
    retrieval_bm25_top_k: int = Field(default=50, gt=0)
    retrieval_rerank_top_k: int = Field(default=10, gt=0)
    retrieval_rerank_threshold: float = Field(default=0.5, ge=0, le=1)
    retrieval_strategy: str = Field(default="hybrid_rerank", pattern="^(dense|bm25|hybrid|hybrid_rerank)$")
    rerank_arrangement_strategy: str = Field(default="first_last", pattern="^(score_desc|first_last)$")

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
