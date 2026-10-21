"""ReAct 子图：收敛、停止条件、全局引用编号、动作标记与流式过滤（Task 7）。"""

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from app.knowledge.errors import KnowledgeBackendUnavailable
from app.workflow import agent as agent_module
from app.workflow.agent import build_agent_subgraph, parse_action_markers
from app.workflow.texts import STEP_LIMIT_TEXT, TOKEN_LIMIT_TEXT


class ScriptedModel:
    """按脚本依次产出流式 chunk；记录每次调用看到的完整消息列表。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.seen: list[list] = []
        self.tools = None

    def bind_tools(self, tools):
        self.tools = list(tools)
        return self

    async def astream(self, messages):
        self.calls += 1
        self.seen.append(list(messages))
        chunks = self.replies.pop(0)
        if not isinstance(chunks, list):
            chunks = [chunks]
        for chunk in chunks:
            if isinstance(chunk, AIMessage):
                yield AIMessageChunk(
                    content=chunk.content,
                    tool_calls=chunk.tool_calls,
                    usage_metadata=chunk.usage_metadata,
                    id=chunk.id,
                )
            else:
                yield chunk


class FakeRegistry:
    def __init__(self, content="工具结果", ok=True, summary="工具结果"):
        self.content = content
        self.ok = ok
        self.summary = summary
        self.calls = []

    async def execute(self, name, args, call_id, *, conversation_id=None):
        self.calls.append((name, args, call_id, conversation_id))
        return SimpleNamespace(
            ok=self.ok,
            content=self.content,
            summary=self.summary,
            elapsed_ms=1,
            metadata=None,
        )


@dataclass(frozen=True)
class FakeChunk:
    chunk_id: str
    questions: str
    answer: str


@contextmanager
def patched_retrieval(hits=None, error=None):
    """把 agent_tools 内部的 search_with_metadata 换成固定结果。"""

    async def fake_search(
        session, query, top_k=None, *, strategy="dense", session_id="", **kwargs
    ):
        if error is not None:
            raise error
        return list(hits or []), {"strategy": strategy, "scores": None}

    with patch.object(agent_module.retrieval, "search_with_metadata", fake_search):
        yield


async def run_agent(
    model,
    registry,
    state=None,
    *,
    faq_hits=None,
    faq_error=None,
    max_steps=None,
    token_budget=None,
):
    base = {"messages": [], "steps": 0, "tokens_used": 0}
    if state:
        base.update(state)
    if max_steps is not None:
        base["max_steps"] = max_steps
    if token_budget is not None:
        base["token_budget"] = token_budget
    config = {
        "configurable": {
            "model": model,
            "registry": registry,
            "session": object(),
            "conversation_id": 1,
        }
    }
    with patched_retrieval(faq_hits, faq_error):
        return await build_agent_subgraph().ainvoke(base, config)


def test_parse_action_markers_strips_and_dedupes():
    text, actions = parse_action_markers(
        "帮您转人工 <SUGGEST_HANDOFF><SUGGEST_TICKET><SUGGEST_HANDOFF>"
    )
    assert text == "帮您转人工 "
    assert actions == ["handoff", "ticket"]
    assert parse_action_markers("普通回答") == ("普通回答", [])


async def test_agent_converges_without_tools():
    out = await run_agent(
        ScriptedModel([AIMessage(content="直接回答")]),
        FakeRegistry(),
        {"messages": [HumanMessage("你好")]},
    )
    assert out["messages"][-1].content == "直接回答"
    assert out["steps"] == 1
    assert out["actions"] == []


async def test_agent_calls_tool_then_converges():
    model = ScriptedModel([
        AIMessage(
            content="",
            tool_calls=[
                {"name": "query_order", "args": {"order_id": "1001"}, "id": "c1"}
            ],
        ),
        AIMessage(content="已发货"),
    ])
    registry = FakeRegistry("订单 1001 已发货")
    out = await run_agent(model, registry, {"messages": [HumanMessage("查订单")]})

    assert [m.type for m in out["messages"]] == ["human", "ai", "tool", "ai"]
    assert out["steps"] == 2
    assert registry.calls == [("query_order", {"order_id": "1001"}, "c1", 1)]
    assert out["messages"][-1].content == "已发货"


async def test_agent_stops_at_max_steps():
    model = ScriptedModel([
        AIMessage(
            content="",
            tool_calls=[
                {"name": "query_order", "args": {"order_id": "1"}, "id": f"c{i}"}
            ],
        )
        for i in range(5)
    ])
    registry = FakeRegistry()
    out = await run_agent(
        model, registry, {"messages": [HumanMessage("x")]}, max_steps=2
    )

    assert out["steps"] == 2
    assert out["messages"][-1].content == STEP_LIMIT_TEXT
    # 第 2 次模型调用已经触顶，不再执行第 2 批工具。
    assert len(registry.calls) == 1


async def test_agent_stops_at_token_budget():
    model = ScriptedModel([
        AIMessage(
            content="",
            tool_calls=[
                {"name": "query_order", "args": {"order_id": "1"}, "id": "c1"}
            ],
            usage_metadata={
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            },
        ),
        AIMessage(content="不应被调用"),
    ])
    registry = FakeRegistry()
    out = await run_agent(
        model,
        registry,
        {"messages": [HumanMessage("x")]},
        max_steps=10,
        token_budget=5,
    )

    assert out["steps"] == 1
    assert out["tokens_used"] == 5
    assert out["messages"][-1].content == TOKEN_LIMIT_TEXT
    assert registry.calls == []


async def test_query_faq_uses_global_citation_numbers():
    model = ScriptedModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "query_faq", "args": {"keyword": "运费"}, "id": "c1"}],
        ),
        AIMessage(content="见 [1] 和 [2]"),
    ])
    state = {
        "messages": [HumanMessage("运费")],
        "evidence": [],
        "citations": [],
        "citation_index": {},
        "citation_cursor": 1,
    }
    out = await run_agent(
        model,
        FakeRegistry(),
        state,
        faq_hits=[FakeChunk("faq:1", "运费", "满 99 包邮")],
    )

    tool_msg = next(m for m in out["messages"] if m.type == "tool")
    assert tool_msg.content.startswith("[1] 问：运费")
    assert out["citations"] == [{"index": 1, "chunk_id": "faq:1", "score": None}]
    assert out["citation_index"] == {"faq:1": 1}
    assert out["citation_cursor"] == 2


async def test_query_faq_falls_back_when_knowledge_backend_unavailable():
    model = ScriptedModel([
        AIMessage(
            content="",
            tool_calls=[{"name": "query_faq", "args": {"keyword": "运费"}, "id": "c1"}],
        ),
        AIMessage(content="稍后再试"),
    ])
    out = await run_agent(
        model,
        FakeRegistry(),
        {"messages": [HumanMessage("运费")]},
        faq_error=KnowledgeBackendUnavailable("milvus down"),
    )

    tool_msg = next(m for m in out["messages"] if m.type == "tool")
    assert tool_msg.content == "知识库暂时不可用，请稍后再试或转人工客服。"
    assert out["steps"] == 2
    assert out["citations"] == []


async def test_agent_done_maps_markers_to_actions_and_cleans_text():
    model = ScriptedModel([
        AIMessage(content="我可以帮您转人工 <SUGGEST_HANDOFF><SUGGEST_TICKET>")
    ])
    out = await run_agent(
        model,
        FakeRegistry(),
        {
            "messages": [HumanMessage("我要投诉")],
            "route": "knowledge",
            "user_input": "很长的原话" * 200,
        },
    )

    assert out["messages"][-1].content == "我可以帮您转人工 "
    assert out["actions"][0] == {"type": "handoff", "label": "转人工"}
    ticket = out["actions"][1]
    assert ticket["type"] == "ticket" and ticket["label"] == "建工单"
    assert ticket["ticket_type"] == "咨询"
    assert len(ticket["description"]) == 500


async def test_streamed_deltas_strip_action_markers_split_across_chunks():
    model = ScriptedModel([[
        AIMessageChunk(content="建议转"),
        AIMessageChunk(content="人工 <SUGGEST_"),
        AIMessageChunk(content="HANDOFF>"),
    ]])
    config = {
        "configurable": {
            "model": model,
            "registry": FakeRegistry(),
            "session": object(),
            "conversation_id": 1,
        }
    }
    state = {"messages": [HumanMessage("转人工")], "steps": 0, "tokens_used": 0}

    deltas: list[str] = []
    with patched_retrieval():
        async for event in build_agent_subgraph().astream(
            state, config, stream_mode="custom"
        ):
            if event.get("type") == "delta":
                deltas.append(event["text"])

    assert "".join(deltas) == "建议转人工 "
    assert all("<SUGGEST" not in text for text in deltas)


async def test_evidence_is_injected_as_system_message_before_model_call():
    model = ScriptedModel([AIMessage(content="按政策满 99 包邮。")])
    out = await run_agent(
        model,
        FakeRegistry(),
        {
            "messages": [HumanMessage("运费")],
            "route": "knowledge",
            "evidence": [
                {"index": 1, "chunk_id": "faq:1", "questions": "运费", "answer": "满 99 包邮"}
            ],
        },
    )

    first_seen = model.seen[0]
    assert isinstance(first_seen[0], SystemMessage)
    assert "[1] 问：运费" in first_seen[0].content
    assert "答：满 99 包邮" in first_seen[0].content
    assert isinstance(first_seen[1], HumanMessage)
    # 注入消息留在 State 里，供 Task 9 写审计时按 SystemMessage 过滤。
    assert any(isinstance(m, SystemMessage) for m in out["messages"])
