"""Query 改写：LLM 驱动的口语归一化。

Task 5 的检索策略在召回前调用 `rewrite_query`，把「邮费咋算」这类口语问法
归一成知识库更容易匹配的标准问法。

改写是**可选增强**，不是检索的前置条件：LLM 不可用、超时、或返回内容不可用
时必须回退到原 query，绝不阻塞检索。

模型客户端统一走 `app.providers.get_chat_model`：本模块不直接构造 OpenAI
客户端，这样 base_url / api_key / extra_body（尤其是 DeepSeek 的
`thinking: disabled`）只由 providers.py 一处决定。
"""

import asyncio
import logging

from langchain_core.messages import HumanMessage

from app.providers import get_chat_model

logger = logging.getLogger(__name__)

# 改写发生在召回之前，耗时直接占用整条检索链路的预算。宁可放弃改写，也不能
# 让一次卡住的 LLM 请求拖垮检索。
_REWRITE_TIMEOUT_S = 3.0

REWRITE_PROMPT = """将用户的口语化问题改写为标准问法，保留核心意图。

规则：
- 去除口语词（咋、啥、咋办）
- 统一疑问词（多少钱 → 价格、咋算 → 如何计算）
- 保留核心名词

示例：
用户：邮费咋算
改写：邮费如何计算

用户：能退吗
改写：退货政策

现在改写：{query}
只返回改写后的问题，不要解释。"""


async def _call_llm(query: str) -> str:
    """调用 LLM 改写 query。返回去掉首尾空白的结果。"""
    model = get_chat_model(temperature=0)
    resp = await model.ainvoke(
        [HumanMessage(content=REWRITE_PROMPT.format(query=query))]
    )
    return resp.text.strip()


async def rewrite_query(query: str) -> str:
    """改写 query，失败返回原 query（不阻塞检索）。"""
    try:
        rewritten = await asyncio.wait_for(
            _call_llm(query), timeout=_REWRITE_TIMEOUT_S
        )
        # 空结果或明显超长的"解释"（模型没听指令）都不可信，按失败处理。
        if rewritten and len(rewritten) <= len(query) * 3:
            logger.info("Query rewrite: %s → %s", query, rewritten)
            return rewritten
        logger.warning(
            "Query rewrite returned unusable result (len=%d), fallback to original",
            len(rewritten or ""),
        )
    except Exception as exc:  # noqa: BLE001 — 改写失败绝不能阻塞检索
        logger.warning("Query rewrite failed: %s, fallback to original", exc)

    return query
