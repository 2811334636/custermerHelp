"""满意度反馈：POST /api/feedback（预留接口，本章不落库）。"""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class FeedbackRequest(BaseModel):
    message_id: int
    feedback: Literal["thumbs_up", "thumbs_down"]


@router.post("/feedback")
async def post_feedback(req: FeedbackRequest) -> dict:
    """接收满意度反馈（本章只校验并返回成功，落库留待后续章节）。"""
    # TODO(ch05+): 把 (message_id, feedback) 写入反馈表。
    return {"status": "ok"}
