import hashlib
from dataclasses import dataclass
from typing import Literal

# 与 scripts/schema.sql / app/db/models.py 的列长度保持一致。
# 超限必须给出明确错误，不能静默截断：截断会让 chunk_id 不再确定性，
# 也会让 questions/category 与源文档悄悄不一致。
_MAX_CHUNK_ID = 64
_MAX_QUESTIONS = 512
_MAX_CATEGORY = 255


def _check_length(field: str, value: str, limit: int) -> None:
    if len(value) > limit:
        preview = value if len(value) <= 40 else value[:40] + "…"
        raise ValueError(
            f"{field} 超过 MySQL VARCHAR({limit}) 上限："
            f"{len(value)} 字符（{preview}）"
        )


def format_embedding_text(category: str, questions: str, answer: str) -> str:
    """送去向量化的文本：三字段用换行拼接。唯一格式定义处。"""
    return f"{category}\n{questions}\n{answer}"


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

    def __post_init__(self) -> None:
        # 所有来源（Markdown/FAQ/挖掘）最终都构造 Chunk，因此在这里统一
        # 挡住超长字段，错误信息能直接指向是哪一列超了（final review M6）。
        _check_length("chunk_id", self.chunk_id, _MAX_CHUNK_ID)
        _check_length("questions", self.questions, _MAX_QUESTIONS)
        _check_length("category", self.category, _MAX_CATEGORY)

    @property
    def embedding_text(self) -> str:
        """送去向量化的文本。"""
        return format_embedding_text(self.category, self.questions, self.answer)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.embedding_text.encode("utf-8")).hexdigest()
