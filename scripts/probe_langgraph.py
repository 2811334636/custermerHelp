"""LangGraph 1.2.16 集成探测（ch05 Task 0，阻塞性）。

验证四件事，为后续 Workflow 实现定下真实 API 形态：
1. ``AsyncSqliteSaver.from_conn_string`` 是否开箱可用（``setup()`` 要不要手动调）；
2. ``astream(..., stream_mode="custom")`` 能收到节点内 ``get_stream_writer()`` 的事件；
3. 同一 ``thread_id`` 第二次 ``astream(None, ...)`` 能恢复上一次的 State；
4. 子图作为节点 + ``subgraphs=True`` 时自定义事件能冒泡（含事件元组形状）；
5. ``AIMessage.tool_calls`` 跨节点传递后 ``add_messages`` 能正确合并。

运行：``uv run python -m scripts.probe_langgraph``
任一检查 FAIL 时进程以非零码退出。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import traceback
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

RESULTS: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


class ProbeState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    counter: int


# ---------------------------------------------------------------------------
# 主图：step_a 写自定义事件 + AIMessage(tool_calls)；step_b 校验并写 ToolMessage
# ---------------------------------------------------------------------------


async def step_a(state: ProbeState) -> dict[str, Any]:
    writer = get_stream_writer()
    counter = state.get("counter", 0) + 1
    writer({"node": "step_a", "counter": counter})
    ai = AIMessage(
        id="ai-1",
        content="",
        tool_calls=[{"name": "query_faq", "args": {"query": "退款"}, "id": "call_1"}],
    )
    return {"counter": counter, "messages": [ai]}


async def step_b(state: ProbeState) -> dict[str, Any]:
    writer = get_stream_writer()
    messages = state.get("messages", [])
    ai = next((m for m in messages if isinstance(m, AIMessage)), None)
    tool_calls = list(ai.tool_calls) if ai is not None else []
    ok = (
        state.get("counter") == 1
        and ai is not None
        and ai.id == "ai-1"
        and len(tool_calls) == 1
        and tool_calls[0]["id"] == "call_1"
        and tool_calls[0]["name"] == "query_faq"
    )
    writer(
        {
            "node": "step_b",
            "counter": state.get("counter"),
            "resumed_messages": len(messages),
            "tool_calls_intact": ok,
        }
    )
    return {"messages": [ToolMessage(content="退款政策……", tool_call_id="call_1")]}


def build_main_graph(checkpointer: AsyncSqliteSaver):
    builder = StateGraph(ProbeState)
    builder.add_node("step_a", step_a)
    builder.add_node("step_b", step_b)
    builder.add_edge(START, "step_a")
    builder.add_edge("step_a", "step_b")
    builder.add_edge("step_b", END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 子图：sub_node 写自定义事件；外层图把编译后的子图直接当节点
# ---------------------------------------------------------------------------


async def sub_node(state: ProbeState) -> dict[str, Any]:
    writer = get_stream_writer()
    writer({"node": "sub_node", "from": "subgraph"})
    return {"counter": state.get("counter", 0) + 10}


def build_subgraph() -> Any:
    builder = StateGraph(ProbeState)
    builder.add_node("sub_node", sub_node)
    builder.add_edge(START, "sub_node")
    builder.add_edge("sub_node", END)
    return builder.compile()


def build_outer_graph(checkpointer: AsyncSqliteSaver):
    builder = StateGraph(ProbeState)
    builder.add_node("subgraph", build_subgraph())
    builder.add_edge(START, "subgraph")
    builder.add_edge("subgraph", END)
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 探测主体
# ---------------------------------------------------------------------------


async def probe(checkpoint_path: str) -> None:
    config = {"configurable": {"thread_id": "t1"}}

    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        # ---- 检查 1：from_conn_string 开箱可用，setup() 是否自动触发 ----
        record(
            "1a from_conn_string 可直接构造（未手动 setup）",
            saver is not None and saver.is_setup is False,
            f"is_setup(before)={saver.is_setup}",
        )
        record(
            "1b 文件在构造后即存在",
            os.path.exists(checkpoint_path),
            checkpoint_path,
        )

        graph = build_main_graph(saver)

        # ---- 检查 2：custom 事件 ----
        custom_events: list[Any] = []
        try:
            async for event in graph.astream(
                {"messages": [HumanMessage(content="怎么退款？")]},
                config,
                stream_mode="custom",
                interrupt_after=["step_a"],
            ):
                custom_events.append(event)
        except Exception:
            record("2 astream(stream_mode='custom') 收到节点事件", False)
            traceback.print_exc()
            raise

        first_events_ok = any(
            isinstance(e, dict) and e.get("node") == "step_a" for e in custom_events
        )
        record(
            "2 astream(stream_mode='custom') 收到节点事件",
            first_events_ok,
            f"events={custom_events!r}",
        )
        record(
            "1c setup() 在首次 checkpoint 读写时自动执行",
            saver.is_setup is True,
            f"is_setup(after)={saver.is_setup}",
        )

        # ---- 检查 3：checkpoint 恢复（第二次 astream(None) 继续到 step_b） ----
        snapshot = await graph.aget_state(config)
        pre_resume = snapshot.values
        pre_resume_ai = next(
            (m for m in pre_resume.get("messages", []) if isinstance(m, AIMessage)),
            None,
        )
        record(
            "3a 首次运行后 checkpoint 保留了 AIMessage.tool_calls",
            pre_resume.get("counter") == 1
            and pre_resume_ai is not None
            and pre_resume_ai.tool_calls[0]["id"] == "call_1",
            f"values={pre_resume!r}",
        )

        resume_events: list[Any] = []
        async for event in graph.astream(
            None, config, stream_mode="custom", interrupt_after=["step_a"]
        ):
            resume_events.append(event)

        step_b_event = next(
            (e for e in resume_events if isinstance(e, dict) and e.get("node") == "step_b"),
            None,
        )
        record(
            "3b 第二次 astream(None) 恢复 State 并继续执行",
            step_b_event is not None
            and step_b_event.get("counter") == 1
            and step_b_event.get("tool_calls_intact") is True,
            f"events={resume_events!r}",
        )

        # ---- 检查 5：add_messages 合并结果 ----
        final = (await graph.aget_state(config)).values
        msgs = final.get("messages", [])
        shapes = [type(m).__name__ for m in msgs]
        final_ai = next((m for m in msgs if isinstance(m, AIMessage)), None)
        tool_msg = next((m for m in msgs if isinstance(m, ToolMessage)), None)
        merge_ok = (
            shapes == ["HumanMessage", "AIMessage", "ToolMessage"]
            and final_ai is not None
            and final_ai.id == "ai-1"
            and final_ai.tool_calls[0]["id"] == "call_1"
            and tool_msg is not None
            and tool_msg.tool_call_id == "call_1"
            and final.get("counter") == 1
        )
        record(
            "5 add_messages 跨节点合并 AIMessage(tool_calls)+ToolMessage",
            merge_ok,
            f"messages={shapes}, ids={[getattr(m, 'id', None) for m in msgs]}",
        )

        # ---- 检查 4：子图 + subgraphs=True 自定义事件冒泡 ----
        outer = build_outer_graph(saver)
        sub_config = {"configurable": {"thread_id": "t2"}}
        raw_events: list[Any] = []
        async for event in outer.astream(
            {"messages": [HumanMessage(content="hi")]},
            sub_config,
            stream_mode="custom",
            subgraphs=True,
        ):
            raw_events.append(event)

        shape_desc = [type(e).__name__ for e in raw_events]
        bubbled = [
            e
            for e in raw_events
            if isinstance(e, tuple)
            and len(e) == 2
            and isinstance(e[0], tuple)
            and isinstance(e[1], dict)
            and e[1].get("node") == "sub_node"
        ]
        record(
            "4 子图自定义事件在 subgraphs=True 下冒泡",
            len(bubbled) > 0,
            f"raw_shapes={shape_desc}, bubbled={bubbled!r}",
        )
        record(
            "4b 事件形状固定为 (namespace_tuple, payload)",
            all(
                isinstance(e, tuple) and len(e) == 2 and isinstance(e[0], tuple)
                for e in raw_events
            ),
            f"raw={raw_events!r}",
        )


def main() -> int:
    tmp_dir = tempfile.mkdtemp(prefix="langgraph-probe-")
    checkpoint_path = os.path.join(tmp_dir, "checkpoints.sqlite")
    print(f"checkpoint: {checkpoint_path}\n")
    try:
        asyncio.run(probe(checkpoint_path))
    except Exception:
        traceback.print_exc()
        RESULTS.append(("probe 未崩溃", False, "见上方 traceback"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} PASS")
    if failed:
        print("FAILED:", ", ".join(r[0] for r in failed))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
