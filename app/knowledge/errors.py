"""知识库后端故障的公共异常类型。

Milvus 与 embedding 上游都属于「外部基础设施」：它们暂时不可用时，在线
检索应返回降级文案（spec §9.3/§11），而不是把异常抛给 ToolRegistry 变成
一次工具故障。两者共用一个基类，工具层只需捕获它。
"""


class KnowledgeBackendUnavailable(RuntimeError):
    """知识库后端（Milvus / embedding 上游）暂时不可用。"""
