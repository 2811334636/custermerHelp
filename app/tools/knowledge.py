"""FAQ 检索工具。ch03 起改为 Milvus dense 向量语义检索。

工具签名保持不变：query_faq(keyword: str) -> str。
"""

from langchain_core.tools import tool

from app.config import get_settings
from app.knowledge import retrieval
from app.knowledge.errors import KnowledgeBackendUnavailable
from app.tools.context import current_context


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
        hits = await retrieval.search(
            session,
            keyword,
            strategy=get_settings().retrieval_strategy,
            session_id=str(ctx.conversation_id),
        )
    except KnowledgeBackendUnavailable:
        # Milvus 与 embedding 上游都在这个基类下：基础设施故障一律降级，
        # 不让 ToolRegistry 把它当成工具故障重试/回灌错误码。
        return "知识库暂时不可用，请稍后再试或转人工客服。"

    if not hits:
        # ★ 「查不到」是正常业务结果，不是异常：走成功返回，不触发 registry 的
        # 故障重试。
        return f"没有找到与「{keyword}」相关的常见问题。可以换个说法，或帮你转人工客服。"

    return "\n\n".join(f"问：{h.questions}\n答：{h.answer}" for h in hits)
