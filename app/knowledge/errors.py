"""知识库后端故障的公共异常类型。

Milvus 与 embedding 上游都属于「外部基础设施」：它们暂时不可用时，在线
检索应返回降级文案（spec §9.3/§11），而不是把异常抛给 ToolRegistry 变成
一次工具故障。两者共用一个基类，工具层只需捕获它。
"""


class KnowledgeBackendUnavailable(RuntimeError):
    """知识库后端（Milvus / embedding 上游）暂时不可用。"""


class RerankerUnavailable(Exception):
    """重排服务不可用（连接、超时、鉴权或响应格式异常）。

    刻意不继承 KnowledgeBackendUnavailable：重排失败只应让检索退回
    「不重排」的混合结果，而不是把整个知识库工具降级成不可用。
    """
