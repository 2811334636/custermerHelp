import hashlib
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    category: str
    questions: str
    answer: str
    section_path: str
    content_type: Literal["policy", "faq", "mined"]
    is_key_clause: bool = False
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None

    @property
    def embedding_text(self) -> str:
        """送去向量化的文本：三字段用换行拼接。"""
        return f"{self.category}\n{self.questions}\n{self.answer}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.embedding_text.encode("utf-8")).hexdigest()
