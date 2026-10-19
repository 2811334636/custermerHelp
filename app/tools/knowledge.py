"""FAQ 检索工具。ch03 起改为 Milvus dense 向量语义检索。

工具签名保持不变：query_faq(keyword: str) -> str。
"""

from langchain_core.tools import tool

from app.knowledge import retrieval
from app.knowledge.vector_store import VectorStoreUnavailable
from app.tools.context import current_context


@tool
async def query_faq(keyword: str) -> str:
    """按语义检索常见问题与政策手册（退货、运费、发票、尺码、价保等）。

    keyword 请传**用户的完整问题或其中的核心说法**——
    例如「邮费是多少」「退货运费谁承担」「怎么开发票」。
    """
    # 库会话从上下文取，不落模块级全局：单 worker 下重叠请求会互相覆盖
    # （见 context.py 顶部注释），A 请求的查询就会打到 B 的会话上。
    session = current_context().session
    if session is None:
        return "FAQ 查询暂时不可用（数据库未绑定）。"

    try:
        hits = await retrieval.search(session, keyword)
    except VectorStoreUnavailable:
        return "知识库暂时不可用，请稍后再试或转人工客服。"

    if not hits:
        # ★ 「查不到」是正常业务结果，不是异常：走成功返回，不触发 registry 的
        # 故障重试。
        return f"没有找到与「{keyword}」相关的常见问题。可以换个说法，或帮你转人工客服。"

    return "\n\n".join(f"问：{h.questions}\n答：{h.answer}" for h in hits)
