from langchain_core.messages import AIMessage, HumanMessage

from app.prompts import (
    EXTRACT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_chat_prompt,
    build_extract_prompt,
)


def test_chat_prompt_has_system_message_first():
    msgs = build_chat_prompt().format_messages(
        history=[HumanMessage("你好"), AIMessage("您好")]
    )
    assert msgs[0].type == "system"
    assert msgs[0].content == SYSTEM_PROMPT
    assert [m.content for m in msgs[1:]] == ["你好", "您好"]


def test_chat_prompt_accepts_empty_history():
    msgs = build_chat_prompt().format_messages(history=[])
    assert len(msgs) == 1
    assert msgs[0].type == "system"


def test_extract_prompt_renders_text_variable():
    msgs = build_extract_prompt().format_messages(text="订单 A123 要退款")
    assert msgs[0].type == "system"
    assert msgs[0].content == EXTRACT_SYSTEM_PROMPT
    assert msgs[1].type == "human"
    assert msgs[1].content == "订单 A123 要退款"


def test_system_prompt_states_no_order_system_access():
    """约束 1：无订单系统权限。这条要是丢了，模型会开始编物流。"""
    assert "没有" in SYSTEM_PROMPT and "订单系统" in SYSTEM_PROMPT


def test_system_prompt_has_no_placeholder_markers():
    for prompt in (SYSTEM_PROMPT, EXTRACT_SYSTEM_PROMPT):
        assert "TODO" not in prompt
        assert "TBD" not in prompt
