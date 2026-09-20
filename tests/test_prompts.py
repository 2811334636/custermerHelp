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


def test_system_prompt_bounds_facts_to_tool_output():
    """约束 1（ch02 改写后）：订单事实的唯一来源是工具，且只能转述。

    ch01 写的是「你**没有**订单系统的访问权限」，那条在 ch02 与能力直接冲突——
    实测（带旧文案、物流工具已绑）模型查到真实轨迹也**拒绝转述**，3/3 复现
    "我这边看不到订单的物流信息，没法帮您查"。所以断言跟着改动换了对象。

    约束的**意图没变**：不许凭空断言订单事实。变的是数据来源——从「没有」换成
    「有，但只能转述它返回的内容」。因此两条一起断言：工具这条路存在，且是唯一
    的事实来源、不许补充推测编造。少了后半句，模型就会拿工具的名义编细节。
    """
    for tool_name in ("query_order", "query_logistics", "query_product", "query_faq"):
        assert tool_name in SYSTEM_PROMPT
    assert "只能转述工具返回的内容" in SYSTEM_PROMPT
    assert "工具没返回就不要说" in SYSTEM_PROMPT


def test_system_prompt_has_no_placeholder_markers():
    for prompt in (SYSTEM_PROMPT, EXTRACT_SYSTEM_PROMPT):
        assert "TODO" not in prompt
        assert "TBD" not in prompt
