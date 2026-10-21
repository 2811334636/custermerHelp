"""FAQ 检索工具。ch03 起改为 Milvus dense 向量语义检索。

工具签名保持不变：query_faq(keyword: str) -> str。
"""

from langchain_core.tools import tool

from app.config import get_settings
from app.knowledge import retrieval
from app.knowledge.errors import KnowledgeBackendUnavailable
from app.tools.context import current_context

# 知识库降级与空结果的固定文案：Agent 子图内直接调检索（绕过 ToolRegistry）
# 时也要产出与工具完全一致的文本，集中在这里避免两处措辞漂移。
KNOWLEDGE_UNAVAILABLE_TEXT = "知识库暂时不可用，请稍后再试或转人工客服。"
NO_FAQ_RESULT_TEMPLATE = (
    "没有找到与「{keyword}」相关的常见问题。可以换个说法，或帮你转人工客服。"
)


def format_hits(hits) -> list[str]:
    """把召回块格式化成 ``问：…\n答：…`` 列表（问答块格式的唯一真相）。

    Agent 子图给结果加 ``[i]`` 前缀时复用本函数，保证工具路径与图内路径
    交给模型的正文逐字一致。
    """
    return [f"问：{hit.questions}\n答：{hit.answer}" for hit in hits]


def _publish_retrieval_meta(
    ctx, hits, meta: dict | None
) -> None:
    """把本次检索结果写进请求级 metadata 容器（registry 随后挂到 ToolOutcome）。

    citations 的 index 与工具回灌给模型的问答顺序一致（1-based），前端
    用它与回答里的 [n] 交叉验证。scores 与 chunks 已由策略层对齐；非重排
    策略没有可比较分数，score 填 null。
    """
    container = ctx.retrieval_meta
    if container is None:
        return

    meta = meta or {}
    scores = meta.get("scores")
    citations = [
        {
            "index": i + 1,
            "chunk_id": hit.chunk_id,
            "score": (
                scores[i] if scores is not None and i < len(scores) else None
            ),
        }
        for i, hit in enumerate(hits)
    ]
    container.clear()
    container.update({
        "citations": citations,
        "retrieval_strategy": meta.get("strategy"),
        "top_score": meta.get("top_score"),
    })
    # 检索层已经拒答并入池时才记录原因；api.py 据此避免模型再吐
    # <INSUFFICIENT_KNOWLEDGE> 时对同一问题重复入池。成功检索不写 null，
    # 避免污染 messages.metadata。
    if meta.get("refusal_reason"):
        container["refusal_reason"] = meta["refusal_reason"]


@tool
async def query_faq(keyword: str) -> str:
    """按语义检索常见问题与政策手册（退货、运费、发票、尺码、价保等）。

    keyword 请传**用户的完整问题或其中的核心说法**——
    例如「邮费是多少」「退货运费谁承担」「怎么开发票」。

    回答时请标注引用编号 [1] [2]，编号对应召回的知识块顺序。

    禁止承诺的内容（负面知识）：
    - 不承诺到账时间
    - 不承诺具体库存
    - 不承诺物流异常赔偿金额

    如果召回的知识不足以回答问题，输出：<INSUFFICIENT_KNOWLEDGE>
    """
    # 库会话从上下文取，不落模块级全局：单 worker 下重叠请求会互相覆盖
    # （见 context.py 顶部注释），A 请求的查询就会打到 B 的会话上。
    ctx = current_context()
    session = ctx.session
    if session is None:
        return "FAQ 查询暂时不可用（数据库未绑定）。"

    try:
        hits, meta = await retrieval.search_with_metadata(
            session,
            keyword,
            strategy=get_settings().retrieval_strategy,
            session_id=str(ctx.conversation_id),
        )
    except KnowledgeBackendUnavailable:
        # Milvus 与 embedding 上游都在这个基类下：基础设施故障一律降级，
        # 不让 ToolRegistry 把它当成工具故障重试/回灌错误码。
        # 本次没有可信召回，清掉可能残留的引用，避免前端张冠李戴。
        _publish_retrieval_meta(ctx, [], None)
        return KNOWLEDGE_UNAVAILABLE_TEXT

    _publish_retrieval_meta(ctx, hits, meta)

    if not hits:
        # ★ 「查不到」是正常业务结果，不是异常：走成功返回，不触发 registry 的
        # 故障重试。
        return NO_FAQ_RESULT_TEMPLATE.format(keyword=keyword)

    return "\n\n".join(format_hits(hits))
