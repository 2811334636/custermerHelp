import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api import get_store
from app.config import get_settings
from app.db.base import get_db
from app.main import app
from app.providers import get_default_chat_model
from app.workflow.texts import CHITCHAT_TEXT


class FakeModel:
    """API 层通用桩：意图分类走 `ainvoke`，Agent 走 `bind_tools().astream`。"""

    def __init__(self, intent: str = "闲聊", replies=None):
        self.intent = intent
        self.replies = list(replies or [])
        self.agent_calls = 0

    async def ainvoke(self, messages):
        return AIMessage(
            content=json.dumps(
                {"intent": self.intent, "confidence": 0.9}, ensure_ascii=False
            )
        )

    def bind_tools(self, tools):
        return self

    async def astream(self, messages):
        self.agent_calls += 1
        reply = self.replies[min(self.agent_calls - 1, len(self.replies) - 1)]
        for chunk in reply:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    def with_structured_output(self, schema, method=None, **kwargs):
        raise AssertionError("extract 测试请覆盖 get_chat_model 为专用桩")


class FakeStore:
    """测试用的内存桩，接口与 SessionStore 一致但全是 async。

    ch05 起它只承载**审计**写入；模型上下文由 checkpointer 提供。
    """

    def __init__(self):
        self._data: dict[str, list] = {}
        self.metadata: dict[str, dict] = {}
        self._next_id = 1

    async def get(self, session_id: str):
        return list(self._data.get(session_id, []))

    async def append(self, session_id: str, *messages, metadata=None):
        msgs = self._data.setdefault(session_id, [])
        ids = []
        for message in messages:
            msgs.append(message)
            ids.append(self._next_id)
            self._next_id += 1
        if metadata is not None and messages:
            # 与 MySQLSessionStore 语义一致：只属于最后一条（最终 assistant）。
            self.metadata[session_id] = metadata
        return ids

    async def exists(self, session_id: str) -> bool:
        return session_id in self._data


class SessionSpy:
    """包住真实 db session，只为了数 commit 次数；其余调用一律透传。"""

    def __init__(self, inner):
        self._inner = inner
        self.commits = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def commit(self):
        self.commits += 1
        await self._inner.commit()


@pytest.fixture
def client(monkeypatch, db_session):
    monkeypatch.setenv("APP_LLM_API_KEY", "test-key")
    get_settings.cache_clear()
    store = FakeStore()
    spy = SessionSpy(db_session)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_db] = lambda: spy
    app.dependency_overrides[get_default_chat_model] = lambda: FakeModel()
    with TestClient(app) as c:
        c.db = spy
        c.store = store
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
    """新会话：meta → delta → done；问答走固定话术出口（不调模型）。"""
    r = client.post("/api/chat", json={"message": "你好"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert "".join(d["text"] for name, d in events if name == "delta") == CHITCHAT_TEXT
    assert uuid.UUID(events[0][1]["session_id"])  # 是合法 uuid
    # 完成的一轮提交两次：落库后先 commit（保证客户端收到 done 时 message_id
    # 已持久化），finally 再提交一次作为无害的 no-op。
    assert client.db.commits == 2


async def test_chat_reuses_given_session_and_keeps_context(client):
    """带 session_id 的第二轮不再发 meta，且两轮都留在审计历史里。

    ★ 上下文本身由 checkpointer 提供（见 tests/test_api_workflow.py 的
    第二轮模型消息断言）；这里验证的是审计写入与 meta 契约。
    """
    r1 = client.post("/api/chat", json={"message": "我叫张三"})
    sid = parse_sse(r1.text)[0][1]["session_id"]

    r2 = client.post("/api/chat", json={"session_id": sid, "message": "我叫什么"})
    names = [name for name, _ in parse_sse(r2.text)]
    assert names[0] == "delta"  # 自带 session_id 时不再发 meta

    history = await client.app.dependency_overrides[get_store]().get(sid)
    assert [m.content for m in history] == [
        "我叫张三", CHITCHAT_TEXT, "我叫什么", CHITCHAT_TEXT,
    ]


def test_chat_rejects_empty_message(client):
    """流开始前的错误走 HTTP 状态码（spec §9）。"""
    assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_chat_rejects_whitespace_only_message(client):
    """Ruling 7：解析层 strip 后为空 → 422，而不是一轮空白对话。"""
    assert client.post("/api/chat", json={"message": "   "}).status_code == 422
    assert client.post("/api/chat", json={"message": "\n\t "}).status_code == 422


def test_chat_strips_surrounding_whitespace(client):
    """合法消息的首尾空白在进入工作流前被去掉（审计里是干净文本）。"""
    r = client.post("/api/chat", json={"message": "  你好  "})
    sid = parse_sse(r.text)[0][1]["session_id"]
    assert client.store._data[sid][0].content == "你好"


def test_chat_unknown_session_id_is_not_an_error(client):
    r = client.post("/api/chat", json={"session_id": "从没见过的", "message": "你好"})
    assert r.status_code == 200


def test_chat_empty_session_id_is_rejected(client):
    """I1 回归（schema 层）：空串 session_id 现在被 422 挡下。"""
    r = client.post("/api/chat", json={"session_id": "", "message": "你好"})
    assert r.status_code == 422


def test_chat_emits_meta_when_session_id_is_empty_string(client):
    """I1 回归（推导层）：即便空串绕过 pydantic 校验，chat() 也必须回发 meta。

    直接驱动路由函数（model_construct 跳过校验），锁住 api.py 里的不变量：
    只要服务端替客户端生成了 id，就一定会通过 meta 告诉客户端。
    """
    from app.api import chat
    from app.schemas import ChatRequest

    req = ChatRequest.model_construct(session_id="", message="你好")

    async def _drive():
        # 路由在 TestClient 的 portal 循环里跑，db 必须属于同一个循环。
        engine = create_async_engine(get_settings().db_url)
        maker = async_sessionmaker(engine, expire_on_commit=False)
        try:
            async with maker() as db:
                response = await chat(
                    req,
                    request=SimpleNamespace(app=app),
                    store=FakeStore(),
                    model=FakeModel(),
                    db=db,
                )
                return "".join([chunk async for chunk in response.body_iterator])
        finally:
            await engine.dispose()

    raw = client.portal.call(_drive)
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
    model = FakeModel(
        intent="物流",
        replies=[[err]],
    )
    app.dependency_overrides[get_default_chat_model] = lambda: model
    r = client.post(
        "/api/chat", json={"message": "订单 1001 的物流到哪了"}
    )
    assert r.status_code == 200  # 流已开始，状态码改不了
    events = parse_sse(r.text)
    assert events[-1][0] == "error"
    assert events[-1][1]["code"] == "upstream_rate_limit"


async def test_failed_turn_leaves_session_store_untouched(client):
    """I4.3：失败的一轮不得在会话存储里留下任何痕迹（spec §7.2.1）。

    会话行由 get_or_create 在请求一开始就建出来（工具的外键需要它），它
    **即使本轮失败也要提交**，否则下次请求又得重建。所以这里断言的是消息
    零痕迹 + 恰好一次提交（会话行）。
    """
    import openai
    import httpx

    from app.db.models import Conversation

    store = client.app.dependency_overrides[get_store]()
    err = openai.AuthenticationError(
        "bad key",
        response=httpx.Response(401, request=httpx.Request("POST", "https://x/")),
        body=None,
    )
    app.dependency_overrides[get_default_chat_model] = lambda: FakeModel(
        intent="物流", replies=[[err]]
    )

    sid = str(uuid.uuid4())
    r = client.post(
        "/api/chat", json={"session_id": sid, "message": "订单 1001 的物流到哪了"}
    )
    assert parse_sse(r.text)[-1][0] == "error"
    # 消息存储零痕迹：半截的一轮一句都不许写。
    assert not await store.exists(sid)
    assert await store.get(sid) == []
    # 会话行则必须留着（本轮唯一被提交的东西）。
    assert client.db.commits == 1

    async def _session_row_exists() -> bool:
        # ★ 另起一个引擎来查，而不是复用 client.db：那个 session 的连接建立在
        # TestClient 的 portal 循环里，在外层测试循环里 await 它会撞上
        # "got Future attached to a different loop"。
        engine = create_async_engine(get_settings().db_url)
        try:
            async with async_sessionmaker(engine)() as other:
                found = (
                    await other.execute(
                        select(Conversation).where(Conversation.session_uuid == sid)
                    )
                ).scalar_one_or_none()
                return found is not None
        finally:
            await engine.dispose()

    assert client.portal.call(_session_row_exists), "会话行即使本轮失败也要保留"


def test_temperature_is_not_an_endpoint_parameter():
    """I2 回归：get_chat_model 的 temperature 不得泄漏成 query 参数。"""
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
