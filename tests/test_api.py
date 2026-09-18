import json
import uuid

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.api import get_store
from app.main import app
from app.providers import get_default_chat_model
from app.sessions import InMemorySessionStore


class FakeModel:
    def __init__(self, script=None, raises=None):
        self._script = script or ["您", "好"]
        self._raises = raises

    async def astream(self, messages):
        if self._raises:
            raise self._raises
        for t in self._script:
            yield AIMessageChunk(content=t)

    def with_structured_output(self, schema, method=None, **kwargs):
        raise AssertionError("extract 测试请覆盖 get_chat_model 为专用桩")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APP_LLM_API_KEY", "test-key")
    from app.config import get_settings

    get_settings.cache_clear()
    store = InMemorySessionStore(max_sessions=10, max_messages=20)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_default_chat_model] = lambda: FakeModel()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def parse_sse(raw: str) -> list[tuple[str, dict]]:
    """把 SSE 原始文本解析成 (event_name, data) 列表。"""
    out = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        out.append((name, data))
    return out


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_chat_emits_meta_then_deltas_then_done(client):
    r = client.post("/api/chat", json={"message": "你好"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert names.count("delta") == 2
    assert "".join(d["text"] for n, d in events if n == "delta") == "您好"
    assert uuid.UUID(events[0][1]["session_id"])  # 是合法 uuid


def test_chat_reuses_given_session_and_keeps_context(client):
    """会话复用（服务端侧）：带 session_id 的第二轮不再发 meta，且历史被累积。

    注意断言边界：打桩模型不看 messages，所以本用例**不**验证模型真的"看到"
    了第一轮内容（那是 evals/ 与 smoke.sh 验收 2 的职责）。它验证的是服务端行为：
    同一 session_id 下的历史确实被写入并保留。
    """
    r1 = client.post("/api/chat", json={"message": "我叫张三"})
    sid = parse_sse(r1.text)[0][1]["session_id"]

    r2 = client.post("/api/chat", json={"session_id": sid, "message": "我叫什么"})
    names = [n for n, _ in parse_sse(r2.text)]
    assert names[0] == "delta"  # 自带 session_id 时不再发 meta

    history = client.app.dependency_overrides[get_store]().get(sid)
    assert [m.content for m in history] == ["我叫张三", "您好", "我叫什么", "您好"]


def test_chat_rejects_empty_message(client):
    """流开始前的错误走 HTTP 状态码（spec §9）。"""
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_chat_unknown_session_id_is_not_an_error(client):
    r = client.post("/api/chat", json={"session_id": "从没见过的", "message": "你好"})
    assert r.status_code == 200


def test_chat_empty_session_id_is_rejected(client):
    """I1 回归（schema 层）：空串 session_id 现在被 422 挡下。

    修复前它是 200 —— 服务端生成了 id 却不发 meta，会话被静默孤立（spec §5.1）。
    """
    r = client.post("/api/chat", json={"session_id": "", "message": "你好"})
    assert r.status_code == 422


async def test_chat_emits_meta_when_session_id_is_empty_string():
    """I1 回归（推导层）：即便空串绕过 pydantic 校验，chat() 也必须回发 meta。

    直接驱动路由函数（model_construct 跳过校验），锁住 api.py 里的不变量：
    只要服务端替客户端生成了 id，就一定会通过 meta 告诉客户端。
    旧实现（is_new_session = session_id is None）在此会静默不发 meta。
    """
    from app.api import chat
    from app.schemas import ChatRequest

    req = ChatRequest.model_construct(session_id="", message="你好")
    store = InMemorySessionStore()
    resp = await chat(req, store=store, model=FakeModel())

    raw = "".join([chunk async for chunk in resp.body_iterator])
    events = parse_sse(raw)
    assert events[0][0] == "meta"
    assert events[0][1]["session_id"]  # 非空，且客户端学得到


def test_chat_upstream_error_becomes_error_event(client):
    import openai
    import httpx

    err = openai.RateLimitError(
        "slow down",
        response=httpx.Response(429, request=httpx.Request("POST", "https://x/")),
        body=None,
    )
    app.dependency_overrides[get_default_chat_model] = lambda: FakeModel(raises=err)
    r = client.post("/api/chat", json={"message": "你好"})
    assert r.status_code == 200  # 流已开始，状态码改不了
    events = parse_sse(r.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "upstream_rate_limit"


def test_failed_turn_leaves_session_store_untouched(client):
    """I4.3：失败的一轮不得在会话存储里留下任何痕迹（spec §7.2.1）。"""
    import openai
    import httpx

    store = client.app.dependency_overrides[get_store]()
    err = openai.AuthenticationError(
        "bad key",
        response=httpx.Response(401, request=httpx.Request("POST", "https://x/")),
        body=None,
    )
    app.dependency_overrides[get_default_chat_model] = lambda: FakeModel(raises=err)

    sid = "失败会话"
    r = client.post("/api/chat", json={"session_id": sid, "message": "你好"})
    assert parse_sse(r.text)[-1][0] == "error"
    assert not store.exists(sid)
    assert store.get(sid) == []


def test_temperature_is_not_an_endpoint_parameter():
    """I2 回归：get_chat_model 的 temperature 不得泄漏成 query 参数。

    一旦有人在 Depends() 里直接依赖带签名的工厂，FastAPI 会把 temperature
    变成 /api/chat 与 /api/extract 的 query 参数，未鉴权调用者即可改采样。
    """
    schema = app.openapi()
    for path in ("/api/chat", "/api/extract"):
        params = [
            p["name"] for p in schema["paths"][path]["post"].get("parameters", [])
        ]
        assert "temperature" not in params, f"{path} 泄漏了 temperature 参数"


def test_extract_returns_json(client):
    from app.schemas import AfterSalesTicket

    class S:
        async def ainvoke(self, messages):
            return AfterSalesTicket(order_id="A123", demand="退款", expected_solution="原路退回")

    class M:
        def with_structured_output(self, schema, method=None, **kwargs):
            return S()

    app.dependency_overrides[get_default_chat_model] = lambda: M()
    r = client.post("/api/extract", json={"text": "订单 A123 我要退款，希望原路退回"})
    assert r.status_code == 200
    assert r.json() == {"order_id": "A123", "demand": "退款", "expected_solution": "原路退回"}


def test_extract_failure_returns_502(client):
    class M:
        def with_structured_output(self, schema, method=None, **kwargs):
            raise ValueError("no tool call")

    app.dependency_overrides[get_default_chat_model] = lambda: M()
    r = client.post("/api/extract", json={"text": "随便"})
    assert r.status_code == 502
    assert r.json()["code"] == "extraction_failed"


def test_extract_rejects_blank_text(client):
    assert client.post("/api/extract", json={"text": ""}).status_code == 422
