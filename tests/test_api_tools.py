import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk

from app.main import app
from app.providers import get_default_chat_model
from app.sessions import SessionStore
from app.api import get_store
from app.db.base import get_db


class FakeStore:
    def __init__(self):
        self._data = {}

    async def get(self, session_id):
        return list(self._data.get(session_id, []))

    async def append(self, session_id, *messages):
        self._data.setdefault(session_id, []).extend(messages)

    async def exists(self, session_id):
        return session_id in self._data


class ToolCallingModel:
    def __init__(self):
        self._bound = []

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        if len(self._bound) > 0:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "query_faq", "args": '{"keyword": "退货"}',
                    "id": "call_x", "index": 0, "type": "tool_call_chunk",
                }],
            )
        else:
            yield AIMessageChunk(content="退货政策是七天无理由。")


@pytest.fixture
def client(db_session):
    """★ 必须注入真实的 db session：路由要用它做 get_or_create 与 build_registry。
    覆盖成 None 会让 get_or_create(None, ...) 直接崩。
    """
    store = FakeStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_default_chat_model] = lambda: ToolCallingModel()
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as c:
        c.store = store
        yield c
    app.dependency_overrides.clear()


def parse_sse(raw: str):
    out = []
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        out.append((name, data))
    return out


def test_tool_events_appear_before_deltas(client):
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "meta"
    assert "tool" in names
    assert names.index("tool") < names.index("delta")
    assert names[-1] == "done"


def test_tool_start_and_end_frames(client):
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    tools = [d for n, d in parse_sse(r.text) if n == "tool"]
    assert tools[0]["phase"] == "start"
    assert tools[0]["name"] == "query_faq"
    assert tools[0]["arguments"] == {"keyword": "退货"}
    assert tools[1]["phase"] == "end"
    assert tools[1]["call_id"] == tools[0]["call_id"]
    assert "ok" in tools[1]
    assert "summary" in tools[1]
    assert "content" not in tools[1], "SSE 帧只发 summary，不发回灌全文"


def test_full_message_sequence_is_stored(client):
    """★ 落库必须是**完整序列**：user → assistant(带 tool_calls) → tool → 最终 assistant。

    只存 user + 最终 assistant 的话，下一轮回放历史时 tool_call_id 会指向
    一条不存在的申请单，上游会直接报错。
    """
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    sid = parse_sse(r.text)[0][1]["session_id"]

    msgs = client.store._data[sid]
    assert [type(m).__name__ for m in msgs] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]
    assert msgs[0].content == "退货政策是什么"
    assert msgs[1].tool_calls and msgs[1].tool_calls[0]["name"] == "query_faq"
    assert msgs[2].tool_call_id == msgs[1].tool_calls[0]["id"], "tool_call_id 必须对得上"
    assert msgs[3].content, "最终回答不能为空"


def test_no_tool_turn_stores_two_messages(client):
    """不需要工具的轮次：只存 user + assistant，没有 tool 消息。"""

    class PlainModel:
        def bind_tools(self, tools):
            return self

        async def astream(self, messages):
            yield AIMessageChunk(content="您好，有什么可以帮您？")

    app.dependency_overrides[get_default_chat_model] = lambda: PlainModel()
    r = client.post("/api/chat", json={"message": "你好"})
    sid = parse_sse(r.text)[0][1]["session_id"]

    msgs = client.store._data[sid]
    assert [type(m).__name__ for m in msgs] == ["HumanMessage", "AIMessage"]
    assert msgs[1].content == "您好，有什么可以帮您？"
    assert msgs[1].tool_calls == []


# ---------------------------------------------------------------------------
# 以下是 Task 10 的三条控制项各自的钉子测试（brief 之外）——brief 的用例
# 只覆盖"一轮长什么样"，覆盖不到下面这三件事。
# ---------------------------------------------------------------------------


def test_stored_tool_message_holds_full_content_not_the_badge_summary(client):
    """控制项 (1)：两个文本字段各走各的路。

    `summary` 是截断到 60 字的一句话（前端徽章用），`content` 是回灌给模型的
    全文。库里必须存**全文**：存成摘要等于把截断过的工具结果喂给下一轮的模型。
    """
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    sid = parse_sse(r.text)[0][1]["session_id"]
    tools = [d for n, d in parse_sse(r.text) if n == "tool"]

    # 帧里只有 summary
    assert "content" not in tools[1]
    assert tools[1]["summary"].endswith("…"), "summary 应是截断过的一行摘要"

    stored = client.store._data[sid][2]
    assert type(stored).__name__ == "ToolMessage"
    assert stored.content != tools[1]["summary"], "库里存的不能是摘要"
    assert len(stored.content) > len(tools[1]["summary"])
    assert "退货" in stored.content, "存的应是 FAQ 工具的真实返回全文"


class MetadataModel:
    """第一次申请工具，第二次作答 —— 两次都带真实的 response_metadata / usage。

    第一次的 finish_reason 是真实上游会给的 "tool_calls"，第二次是 "length"
    （模拟被上游截断）。裸 chunk 的打桩模型看不到这两个值，所以这个桩
    专门用来锁住 DoneEvent 不再写死 "stop"。
    """

    USAGE = {"input_tokens": 31, "output_tokens": 7, "total_tokens": 38}

    def __init__(self):
        self._bound = []

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        if len(self._bound) > 0:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "query_faq", "args": '{"keyword": "退货"}',
                    "id": "call_m", "index": 0, "type": "tool_call_chunk",
                }],
                response_metadata={"finish_reason": "tool_calls"},
            )
        else:
            yield AIMessageChunk(
                content="退货政策是七天无理由。",
                response_metadata={"finish_reason": "length"},
                usage_metadata=MetadataModel.USAGE,
            )


def test_done_frame_carries_real_finish_reason_and_usage(client):
    """控制项 (3)：被上游截断的一轮不得被报成干净的 "stop"。

    旧实现两处都写死 `DoneEvent(finish_reason="stop", usage=None)`，于是
    finish_reason="length"（截断）与 usage 对客户端完全不可见。
    """
    app.dependency_overrides[get_default_chat_model] = lambda: MetadataModel()
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    done = [d for n, d in parse_sse(r.text) if n == "done"]

    assert done[-1]["finish_reason"] == "length", "截断必须如实上报，不能是 stop"
    assert done[-1]["usage"] == MetadataModel.USAGE


class FailsOnSecondCallModel:
    """第一次正常申请工具，第二次（最终回答那次）上游炸了。

    这是 ch01 错误边界在工具轮上的分支：已经开始流式（tool 帧都发了）之后
    才失败，仍然只许是 event: error，而且整轮不许落库。
    """

    def __init__(self):
        self._bound = []

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        if len(self._bound) > 0:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "query_faq", "args": '{"keyword": "退货"}',
                    "id": "call_f", "index": 0, "type": "tool_call_chunk",
                }],
            )
        else:
            raise RuntimeError("上游炸了")


def test_tool_round_failure_is_error_not_done_and_persists_nothing(client):
    app.dependency_overrides[get_default_chat_model] = lambda: FailsOnSecondCallModel()
    r = client.post("/api/chat", json={"message": "退货政策是什么"})
    names = [n for n, _ in parse_sse(r.text)]

    assert names[-1] == "error", "工具轮中途失败仍是 event: error"
    assert "done" not in names, "error 与 done 互斥"
    assert "tool" in names, "工具帧已经发过了"
    assert client.store._data == {}, "半截的一轮一句都不许落库"


class RecordingModel:
    """把每次 astream 收到的消息记进共享列表 rec（跨请求共享——override 的
    lambda 每个请求都会新建实例，实例内部的计数器跨不过请求）。

    只有**第一次**调用申请工具，之后再问都直接作答：这样第二轮的第一条
    消息就干净地反映"服务端回放了什么历史"。
    """

    def __init__(self, rec):
        self._rec = rec
        self._bound = []

    def bind_tools(self, tools):
        self._bound = list(tools)
        return self

    async def astream(self, messages):
        self._rec.append(list(messages))
        if len(self._rec) == 1:
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[{
                    "name": "query_faq", "args": '{"keyword": "退货"}',
                    "id": "call_r1", "index": 0, "type": "tool_call_chunk",
                }],
            )
        else:
            yield AIMessageChunk(content="退货政策是七天无理由。")


def test_second_turn_replays_the_first_turns_tool_round(client):
    """★ 本任务最有价值的用例：历史回放。

    "只存 user + 最终 assistant" 的实现，第一轮看着完全正常——帧对了、回复
    也对了。它只在**第二轮**才现形：回放出来的历史里没有 AIMessage(tool_calls)
    也没有 ToolMessage，模型看到一条凭空出现的 tool 结果（或 tool_call_id
    指向一张从没申请过的单子），上游直接报错。

    这里断言第二轮发给模型的 messages 逐条对得上第一轮的真实回合。
    """
    rec: list[list] = []
    app.dependency_overrides[get_default_chat_model] = lambda: RecordingModel(rec)

    r1 = client.post("/api/chat", json={"message": "退货政策是什么"})
    sid = parse_sse(r1.text)[0][1]["session_id"]
    client.post("/api/chat", json={"session_id": sid, "message": "那退了多久能退款"})

    assert len(rec) == 3, "第一轮两次调用 + 第二轮一次调用"
    seen = rec[2]  # 第二轮唯一的模型调用
    assert [type(m).__name__ for m in seen] == [
        "SystemMessage", "HumanMessage", "AIMessage", "ToolMessage",
        "AIMessage", "HumanMessage",
    ]
    assert seen[1].content == "退货政策是什么"
    assert seen[2].tool_calls and seen[2].tool_calls[0]["name"] == "query_faq"
    assert seen[3].tool_call_id == seen[2].tool_calls[0]["id"], "tool_call_id 必须对得上"
    assert seen[4].content == "退货政策是七天无理由。", "最终回答也要在历史里"
    assert seen[5].content == "那退了多久能退款"


def test_second_turn_replays_tool_round_from_mysql(client, db_session):
    """同上，但走**真实** MySQLSessionStore：锁住 tool_calls / tool_call_id
    在 messages 表里能原样往返。

    回放链路上还有一段 brief 的用例看不见的路：LangChain 消息 → to_rows →
    MySQL JSON 列 → from_rows。这段丢了 tool_calls 的话，FakeStore 版本
    依然全绿，线上第二轮却会炸。
    """
    from app.sessions import MySQLSessionStore

    rec: list[list] = []
    app.dependency_overrides[get_default_chat_model] = lambda: RecordingModel(rec)
    app.dependency_overrides[get_store] = lambda: MySQLSessionStore(session=db_session)

    r1 = client.post("/api/chat", json={"message": "退货政策是什么"})
    sid = parse_sse(r1.text)[0][1]["session_id"]
    client.post("/api/chat", json={"session_id": sid, "message": "那退了多久能退款"})

    seen = rec[2]
    assert [type(m).__name__ for m in seen] == [
        "SystemMessage", "HumanMessage", "AIMessage", "ToolMessage",
        "AIMessage", "HumanMessage",
    ]
    assert seen[2].tool_calls[0]["name"] == "query_faq"
    assert seen[3].tool_call_id == seen[2].tool_calls[0]["id"]
    assert seen[3].content.startswith("问："), "往返后仍是 query_faq 的全文"
    assert len(seen[3].content) > 60, "库里存的不是 60 字摘要，而是回灌给模型的全文"
