from pathlib import Path

from app.knowledge.chunk import Chunk
from app.knowledge.splitter import split_markdown


def load_markdown_dir(
    path: Path,
    *,
    content_type: str = "policy",
    max_chars: int = 400,
    overlap_chars: int = 80,
) -> list[Chunk]:
    out: list[Chunk] = []
    for md in sorted(path.glob("*.md")):
        text = md.read_text(encoding="utf-8")
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
