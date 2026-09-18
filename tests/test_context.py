from langchain_core.messages import AIMessage, HumanMessage

from app.context import prepare_messages


def test_empty_history_yields_only_current():
    cur = HumanMessage("你好")
    assert prepare_messages([], cur, token_budget=1000) == [cur]


def test_current_message_never_trimmed_even_with_zero_budget():
    """当前消息被剪掉 = 模型收到空提问。这是最严重的失效模式。"""
    cur = HumanMessage("这是一句很长很长的当前提问" * 20)
    got = prepare_messages(
        [HumanMessage("旧的"), AIMessage("旧的回复")], cur, token_budget=1
    )
    assert got[-1] is cur


def test_short_history_survives_intact():
    history = [HumanMessage("我叫张三"), AIMessage("好的张三")]
    got = prepare_messages(history, HumanMessage("我订单号多少"), token_budget=4096)
    assert [m.content for m in got] == ["我叫张三", "好的张三", "我订单号多少"]


def test_long_history_is_trimmed_to_recent():
    history = []
    for i in range(200):
        history.append(HumanMessage(f"问题{i}" + "啰嗦" * 50))
        history.append(AIMessage(f"回答{i}" + "啰嗦" * 50))
    got = prepare_messages(history, HumanMessage("最新提问"), token_budget=300)
    assert len(got) < len(history) + 1
    assert got[-1].content == "最新提问"


def test_trimmed_result_starts_with_human():
    """防止出现"助手的回答还在、对应的提问已被剪掉"的孤儿消息（spec §7.1）。"""
    history = []
    for i in range(50):
        history.append(HumanMessage(f"问题{i}" + "啰嗦" * 30))
        history.append(AIMessage(f"回答{i}" + "啰嗦" * 30))
    got = prepare_messages(history, HumanMessage("最新提问"), token_budget=200)
    assert got[0].type == "human"


def test_trimming_respects_chinese_budget():
    """用中文文本验证预算真的被遵守。

    旧的 count_tokens_approximately 对中文低估 58%，会放进约 2.4 倍的消息，
    这条测试在那时是过不了的。
    """
    from app.context import _count_tokens

    history = [
        HumanMessage("这是一句中文提问" * 20),
        AIMessage("这是一句中文回答" * 20),
    ] * 50
    got = prepare_messages(history, HumanMessage("最新"), token_budget=500)
    assert _count_tokens(got[:-1]) <= 600
    assert len(got) < len(history) + 1
