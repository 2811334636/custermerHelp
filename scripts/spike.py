"""Task 2 阻塞性探测。一次性脚本，用来回答 spec §12 的两项未决项。"""

import asyncio
import time

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings


def build() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key or "MISSING",
        temperature=s.llm_temperature,
        max_tokens=s.llm_max_output_tokens,
        extra_body=s.extra_body,
    )


async def probe_extra_body_passthrough() -> None:
    """未决项 1：extra_body 能否穿透 langchain-openai 把 thinking 送到上游。

    判据：若 thinking 被成功关闭，首个 chunk 应在 2 秒内到达且带非空 text；
    若 thinking 泄漏，会出现长时间静默或 text 始终为空。
    """
    print("=== 探测 1：extra_body 透传 thinking ===")
    model = build()
    t0 = time.time()
    first_text_at = None
    chunks = 0
    async for chunk in model.astream([HumanMessage("用一句话介绍退货政策")]):
        if chunk.text:
            chunks += 1
            if first_text_at is None:
                first_text_at = time.time() - t0

    print(f"  首个 text chunk: {first_text_at and round(first_text_at, 2)}s")
    print(f"  文本 chunk 数: {chunks}")
    if first_text_at is None or chunks == 0:
        print("  ❌ 结论：thinking 未被关闭（或 extra_body 未透传）—— 阻塞，需回报用户")
    elif first_text_at > 2.0:
        print("  ⚠️ 结论：文本有输出但首字延迟 > 2s，疑似 thinking 部分泄漏 —— 需回报用户")
    else:
        print("  ✅ 结论：extra_body 透传成功，thinking 已关闭")


async def probe_token_counting() -> None:
    """未决项 2：count_tokens_approximately 对中文的估算偏差（spec §7.3）。"""
    from langchain_core.messages.utils import count_tokens_approximately

    print("\n=== 探测 2：中文 token 估算偏差 ===")
    # 构造一段典型的中文客服文本
    zh_text = (
        "你好，我上周买的运动鞋尺码偏大，穿着不合脚，想换一双小一码的。"
        "订单号是 A123456，当时用了优惠券，换货之后优惠券还能用吗？"
        "另外希望你们承担退回的运费，谢谢。"
    )
    messages = [HumanMessage(zh_text)]
    estimate = count_tokens_approximately(messages)

    # 拿上游返回的真实 prompt_tokens 做基准
    model = build()
    reply = await model.ainvoke(messages)
    actual = reply.usage_metadata.get("input_tokens") if reply.usage_metadata else None

    print(f"  估算: {estimate}")
    print(f"  实测: {actual}")
    if not actual:
        print("  ⚠️ 上游未返回 usage，无法标定 —— 需回报用户")
        return
    # 估算与实测的偏差比例，仅比较正文部分（实测含 chat 模板开销，通常略高）
    ratio = estimate / actual
    print(f"  估算/实测 = {ratio:.2f}")
    if ratio < 0.8:
        print("  ❌ 结论：低估超过 20% —— 阻塞，需换成显式 chars_per_token 的可配计数器")
    elif ratio < 0.9:
        print("  ⚠️ 结论：低估 10%~20%，建议把 HISTORY_TOKEN_BUDGET 调低 20% 作为补偿即可")
    else:
        print("  ✅ 结论：偏差可接受，维持 trim_messages + count_tokens_approximately")


async def main() -> None:
    await probe_extra_body_passthrough()
    await probe_token_counting()


if __name__ == "__main__":
    asyncio.run(main())
