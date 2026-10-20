"""引用溯源：GET /api/chunks/{chunk_id}。"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.repositories import knowledge_chunk as repo

router = APIRouter()


@router.get("/chunks/{chunk_id}")
async def get_chunk(
    chunk_id: str, session: AsyncSession = Depends(get_db)
) -> dict:
    """按 chunk_id 返回原文与元数据（前端点击引用编号时调用）。

    `source` 不是表列，由确定性主键 `source:ordinal` 的前缀还原。
    """
    chunks = await repo.get_by_ids(session, [chunk_id])
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "chunk_not_found",
                "message": f"没有找到知识块：{chunk_id}",
            },
        )

    chunk = chunks[0]
    return {
        "chunk_id": chunk.chunk_id,
        "content": chunk.answer,
        "questions": chunk.questions,
        "category": chunk.category,
        "section_path": chunk.section_path,
        "content_type": chunk.content_type,
        "is_key_clause": chunk.is_key_clause,
        "source": chunk.chunk_id.split(":", 1)[0],
    }
