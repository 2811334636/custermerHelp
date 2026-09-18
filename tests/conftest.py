import os

import pytest

# 测试期间不读真实 .env，避免测试依赖本机密钥
os.environ.setdefault("APP_LLM_API_KEY", "test-key")


@pytest.fixture
def anyio_backend():
    return "asyncio"
