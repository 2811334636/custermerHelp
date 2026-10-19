"""Markdown 结构感知切分器。

硬规则（spec §5）：
- 超长递归切：段落 → 句子 → 才硬切
- 切点必须落在句末标点（。！？；）之后
- 表格按行切，每块复述表头
- 代码块整体保留，绝不切
- chunk_id 确定性：f"{source_key}:{ordinal}"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.knowledge.chunk import Chunk

_SENTENCE_ENDS = "。！？；"
_ASCII_ENDS = "!?"
_CLOSERS = "”’」』）】》\"')]"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")


@dataclass
class _Block:
    kind: str                     # paragraph | list | table | code | quote
    lines: list[str]
    section_path: list[str]
    is_key_clause: bool = False


def _split_sentences(text: str) -> list[str]:
    """按句末标点/换行切，保留标点与紧随的右引号、右括号。无边界时整体返回。"""
    parts: list[str] = []
    buf = ""
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        buf += ch
        i += 1
        boundary = ch in _SENTENCE_ENDS or ch in _ASCII_ENDS or ch == "\n"
        if not boundary and ch == ".":
            boundary = i >= n or text[i].isspace()
        if boundary:
            while i < n and text[i] in _CLOSERS:
                buf += text[i]
                i += 1
            carry = ""
            if buf.endswith("\n"):
                buf = buf.rstrip("\n")
                carry = "\n"
            parts.append(buf)
            buf = carry
    if buf:
        parts.append(buf)
    return parts


def _overlap_tail(chunk: str, overlap_chars: int, room: int) -> str:
    """取 chunk 尾部若干完整句子，总长不超过 min(overlap_chars, room)。"""
    limit = min(overlap_chars, room)
    if limit <= 0:
        return ""
    tail: list[str] = []
    acc = 0
    for s in reversed(_split_sentences(chunk)):
        if acc + len(s) > limit:
            break
        tail.insert(0, s)
        acc += len(s)
    return "".join(tail)


def _pack_sentences(sentences: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    """把句子组装成不超过 max_chars 的块，相邻块重叠整句。"""
    out: list[str] = []
    cur = ""
    for sent in sentences:
        # 单句超长：先冲掉当前块，再把这句话硬切（spec 规则 ③）
        while len(sent) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            out.append(sent[:max_chars])
            sent = sent[max_chars:]
        if not sent:
            continue
        if len(cur) + len(sent) <= max_chars:
            cur += sent
            continue
        if cur:
            out.append(cur)
            # 重叠只能占 sent 之外的剩余额度，绝不能把块挤出上限
            tail = _overlap_tail(cur, overlap_chars, max_chars - len(sent))
            cur = tail + sent
        else:
            cur = sent
    if cur.strip():
        out.append(cur)
    return out


def _parse_blocks(text: str) -> list[_Block]:
    lines = text.splitlines()
    blocks: list[_Block] = []
    stack: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            i += 1
            continue

        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # 代码块
        if stripped.startswith("```"):
            fence_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                fence_lines.append(lines[i])
                i += 1
            if i < len(lines):
                fence_lines.append(lines[i])
                i += 1
            blocks.append(_Block("code", fence_lines, [s for s in stack if s]))
            continue

        # 表格
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            blocks.append(_Block("table", table_lines, [s for s in stack if s]))
            continue

        # 引用块
        if stripped.startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i])
                i += 1
            blocks.append(
                _Block(
                    "quote",
                    quote_lines,
                    [s for s in stack if s],
                    is_key_clause="[!IMPORTANT]" in quote_lines[0],
                )
            )
            continue

        # 列表
        if re.match(r"^(\s*)([-*+]|\d+\.)\s+", line):
            list_lines = []
            while i < len(lines) and re.match(r"^(\s*)([-*+]|\d+\.)\s+", lines[i]):
                list_lines.append(lines[i])
                i += 1
            blocks.append(_Block("list", list_lines, [s for s in stack if s]))
            continue

        # 段落：直到空行或块起始
        para = []
        while i < len(lines):
            nxt = lines[i]
            if not nxt.strip():
                break
            if _HEADING_RE.match(nxt) or nxt.strip().startswith(("```", "|", ">")):
                break
            if re.match(r"^(\s*)([-*+]|\d+\.)\s+", nxt):
                break
            para.append(nxt)
            i += 1
        blocks.append(
            _Block(
                "paragraph",
                para,
                [s for s in stack if s],
                is_key_clause=para[0].strip().startswith("**关键条款**"),
            )
        )

    return blocks


def _table_row_chunks(lines: list[str], max_chars: int) -> list[str]:
    """表格按行切，每块复述表头（仅当存在分隔行时）。行本身不硬切。"""
    lines = [ln for ln in lines if ln.strip()]
    if not lines:
        return []
    has_sep = len(lines) >= 2 and bool(_SEPARATOR_RE.match(lines[1])) and "-" in lines[1]
    header = lines[:2] if has_sep else []
    rows = lines[2:] if has_sep else lines
    if not rows:
        return ["\n".join(header)] if header else []
    return ["\n".join(header + [row]) for row in rows]


def split_markdown(
    text: str,
    *,
    source_key: str,
    content_type: str = "policy",
    max_chars: int = 400,
    overlap_chars: int = 80,
) -> list[Chunk]:
    if max_chars <= 0:
        raise ValueError(f"max_chars 必须大于 0，当前为 {max_chars}")
    if overlap_chars < 0:
        raise ValueError(f"overlap_chars 不能为负，当前为 {overlap_chars}")
    if overlap_chars >= max_chars:
        raise ValueError(
            f"overlap_chars（{overlap_chars}）必须小于 max_chars（{max_chars}）"
        )

    chunks: list[Chunk] = []
    ordinal = 0

    for block in _parse_blocks(text):
        path = " > ".join(block.section_path)
        category = " > ".join(block.section_path[:-1]) or path
        questions = block.section_path[-1] if block.section_path else source_key

        if block.kind == "table":
            pieces = _table_row_chunks(block.lines, max_chars)
        elif block.kind == "code":
            pieces = ["\n".join(block.lines)]
        else:
            raw = "\n".join(block.lines)
            if len(raw) <= max_chars:
                pieces = [raw]
            else:
                pieces = _pack_sentences(_split_sentences(raw), max_chars, overlap_chars)

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{source_key}:{ordinal}",
                    category=category,
                    questions=questions,
                    answer=piece,
                    section_path=path,
                    content_type=content_type,  # type: ignore[arg-type]
                    is_key_clause=block.is_key_clause,
                )
            )
            ordinal += 1

    # 回填前后指针
    linked: list[Chunk] = []
    for idx, c in enumerate(chunks):
        linked.append(
            Chunk(
                chunk_id=c.chunk_id,
                category=c.category,
                questions=c.questions,
                answer=c.answer,
                section_path=c.section_path,
                content_type=c.content_type,
                is_key_clause=c.is_key_clause,
                prev_chunk_id=chunks[idx - 1].chunk_id if idx > 0 else None,
                next_chunk_id=chunks[idx + 1].chunk_id if idx < len(chunks) - 1 else None,
            )
        )
    return linked
