import logging
from pathlib import Path

from app.knowledge.chunk import Chunk
from app.knowledge.splitter import split_markdown

logger = logging.getLogger(__name__)


def load_markdown_dir(
    path: Path,
    *,
    content_type: str = "policy",
    max_chars: int = 400,
    overlap_chars: int = 80,
) -> list[Chunk]:
    out: list[Chunk] = []
    for md in sorted(path.glob("*.md")):
        try:
            text = md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # spec §11：单个坏文件不能中断整次构建，只记日志并跳过。
            logger.warning("跳过无法读取的知识文件 %s: %s", md.name, exc)
            continue
        out.extend(
            split_markdown(
                text,
                source_key=md.name,
                content_type=content_type,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )
        )
    return out
