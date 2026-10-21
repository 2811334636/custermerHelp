"""全局引用编号分配：预检索与工具检索共用同一套编号（spec §5.7）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.db.models import KnowledgeChunk


def assign_citations(
    hits: list[KnowledgeChunk],
    citations: list[dict],
    index_map: dict[str, int],
    cursor: int,
) -> tuple[str, list[dict], dict[str, int], int]:
    """给每块知识分配全局唯一编号，返回（带编号文本、citations、index_map、游标）。

    ``index_map`` 里已有的 ``chunk_id`` 复用原编号——同一块知识在预检索和
    Agent 工具检索里被重复召回时，模型看到的始终是同一个 ``[i]``。返回值是
    新的列表/字典，不原地修改调用方传入的 state 结构。
    """
    merged_citations = [dict(citation) for citation in citations]
    merged_index = dict(index_map)
    next_index = cursor
    blocks: list[str] = []

    for hit in hits:
        index = merged_index.get(hit.chunk_id)
        if index is None:
            index = next_index
            next_index += 1
            merged_index[hit.chunk_id] = index
            merged_citations.append(
                {"index": index, "chunk_id": hit.chunk_id, "score": None}
            )
        blocks.append(f"[{index}] 问：{hit.questions}\n答：{hit.answer}")

    return "\n\n".join(blocks), merged_citations, merged_index, next_index
