import pytest

from app.knowledge.splitter import split_markdown


BASIC = """# 售后政策

## 退换货

### 退货

签收后 7 天内可无理由退货。定制类商品不支持。

### 运费

因商品质量问题导致的退换货，往返运费由平台承担。因个人原因发起的退换货，寄回运费需您自行承担。
"""


def test_heading_stack_builds_section_path():
    chunks = split_markdown(BASIC, source_key="policy.md")
    paths = [c.section_path for c in chunks]
    assert "售后政策 > 退换货 > 退货" in paths
    assert "售后政策 > 退换货 > 运费" in paths


def test_questions_is_leaf_heading_and_category_is_parent_path():
    chunks = split_markdown(BASIC, source_key="policy.md")
    c = next(c for c in chunks if c.section_path.endswith("运费"))
    assert c.questions == "运费"
    assert c.category == "售后政策 > 退换货"


def test_chunk_ids_are_deterministic_and_unique():
    a = [c.chunk_id for c in split_markdown(BASIC, source_key="policy.md")]
    b = [c.chunk_id for c in split_markdown(BASIC, source_key="policy.md")]
    assert a == b
    assert len(set(a)) == len(a)
    assert all(x.startswith("policy.md:") for x in a)


def test_prev_next_pointers_are_linked():
    chunks = split_markdown(BASIC, source_key="policy.md")
    assert chunks[0].prev_chunk_id is None
    assert chunks[-1].next_chunk_id is None
    for i in range(len(chunks) - 1):
        assert chunks[i].next_chunk_id == chunks[i + 1].chunk_id
        assert chunks[i + 1].prev_chunk_id == chunks[i].chunk_id


def test_long_paragraph_split_ends_on_sentence_boundary():
    text = "# 政策\n\n" + "这是一句话。" * 200
    chunks = split_markdown(text, source_key="long.md", max_chars=100, overlap_chars=20)
    assert len(chunks) > 1
    for c in chunks:
        assert c.answer.endswith("。"), f"chunk 未以句末标点结尾: ...{c.answer[-20:]!r}"
        assert len(c.answer) <= 100


def test_overlap_is_whole_sentences_not_char_cut():
    # 每句 7 字：30 不是句长的整数倍，char-cut 会落在句子中间
    text = "# 政策\n\n" + "".join(f"第{i:03d}号句。" for i in range(1, 101))
    chunks = split_markdown(text, source_key="over2.md", max_chars=60, overlap_chars=30)
    assert len(chunks) >= 2
    for i in range(1, len(chunks)):
        prev = chunks[i - 1].answer
        cur = chunks[i].answer
        prev_sents = [s + "。" for s in prev.split("。") if s]
        assert prev_sents, "上一块应至少包含一个完整句子"
        matched = any(
            cur.startswith("".join(prev_sents[-k:]))
            for k in range(1, len(prev_sents) + 1)
        )
        assert matched, f"重叠不是上一块尾部的完整句子: {cur[:40]!r}"


def test_table_split_by_row_and_repeats_header():
    text = """# 商品

## 库存

| 商品 | 尺码 | 库存 |
|---|---|---|
| 跑步鞋 | 42 | 12 |
| 运动袜 | L | 88 |
"""
    chunks = split_markdown(text, source_key="goods.md")
    table_chunks = [c for c in chunks if "跑步鞋" in c.answer or "运动袜" in c.answer]
    assert len(table_chunks) == 2
    for c in table_chunks:
        assert "| 商品 | 尺码 | 库存 |" in c.answer
        assert "|---|" in c.answer


def test_code_fence_is_never_split():
    body = "```python\n" + "# 这不是标题\n" * 100 + "```"
    text = f"# 规则\n\n{body}\n"
    # overlap 必须显式小于 max_chars（Task 1 review 新增的参数校验）
    chunks = split_markdown(text, source_key="code.md", max_chars=80, overlap_chars=20)
    code_chunks = [c for c in chunks if "这不是标题" in c.answer]
    assert len(code_chunks) == 1
    assert code_chunks[0].answer.startswith("```python")
    assert code_chunks[0].answer.rstrip().endswith("```")


def test_important_blockquote_sets_key_clause():
    text = "# 规则\n\n> [!IMPORTANT]\n> 每人限购 2 件。\n"
    chunks = split_markdown(text, source_key="rule.md")
    assert any(c.is_key_clause for c in chunks)


def test_text_without_punctuation_terminates():
    text = "# 规则\n\n" + "啊" * 1000
    chunks = split_markdown(text, source_key="nopunct.md", max_chars=100)
    assert sum(len(c.answer) for c in chunks) >= 1000
    assert all(len(c.answer) <= 100 for c in chunks)


def test_invalid_chunk_params_are_rejected():
    with pytest.raises(ValueError):
        split_markdown("# 标题\n\n内容。", source_key="bad.md", max_chars=0)
    with pytest.raises(ValueError):
        split_markdown("# 标题\n\n内容。", source_key="bad.md", max_chars=100, overlap_chars=100)


def test_single_overlong_sentence_is_hard_cut_within_max():
    text = "# 规则\n\n" + "超" * 250
    chunks = split_markdown(text, source_key="huge.md", max_chars=100, overlap_chars=20)
    assert all(len(c.answer) <= 100 for c in chunks)
    assert sum(len(c.answer) for c in chunks) >= 250


def test_overlap_never_pushes_chunk_over_max():
    text = "# 政策\n\n" + "短句。" * 30 + "超长句" + "啊" * 160 + "。" + "短句。" * 30
    chunks = split_markdown(text, source_key="mix.md", max_chars=100, overlap_chars=30)
    assert all(len(c.answer) <= 100 for c in chunks)


def test_table_without_separator_does_not_duplicate_row():
    text = "# 表\n\n| a | b |\n| c | d |\n"
    chunks = split_markdown(text, source_key="noseg.md")
    assert sum(c.answer.count("| a | b |") for c in chunks) == 1


def test_paragraph_starting_with_key_clause_marker_is_flagged():
    text = "# 规则\n\n**关键条款**每人限购 2 件。\n"
    chunks = split_markdown(text, source_key="key.md")
    assert any(c.is_key_clause for c in chunks)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError):
        split_markdown("# 标题\n\n内容。", source_key="bad.md", overlap_chars=-1)


def test_newline_does_not_absorb_next_line_opening_quote():
    text = '# 政策\n\n第一行\n"第二行"\n'
    # 必须强制按句分块：max_chars=200 时两行会原样拼回同一块，
    # '第一行\n"' 是源文本本身的子串，无法区分吸收 bug。
    chunks = split_markdown(text, source_key="nl.md", max_chars=6, overlap_chars=0)
    assert any(c.answer.startswith("第一行") for c in chunks)
    assert all('第一行\n"' not in c.answer for c in chunks)


def test_top_level_section_uses_heading_as_category_fallback():
    text = "# 退货政策\n\n七天无理由退货。\n"
    chunks = split_markdown(text, source_key="top.md")
    assert chunks[0].category == "退货政策"
    assert chunks[0].questions == "退货政策"


def test_document_without_headings_falls_back_to_source_key():
    """final review M5：没有标题的文档不能让 category 变成空串。"""
    chunks = split_markdown("没有标题的一段内容。", source_key="plain.md")
    assert chunks[0].section_path == ""
    assert chunks[0].category == "plain.md"
    assert chunks[0].questions == "plain.md"


def test_chunk_id_over_mysql_limit_is_rejected():
    """final review M6：超长 source_key 不能静默截断，也不能拖到写库才报错。"""
    with pytest.raises(ValueError, match=r"chunk_id.*VARCHAR\(64\)"):
        split_markdown("# 标题\n\n内容。", source_key="a" * 65)


def test_questions_over_mysql_limit_is_rejected():
    text = "# " + "标" * 513 + "\n\n内容。\n"
    with pytest.raises(ValueError, match=r"questions.*VARCHAR\(512\)"):
        split_markdown(text, source_key="ok.md")


def test_category_over_mysql_limit_is_rejected():
    text = (
        "# " + "甲" * 200 + "\n\n"
        "## " + "乙" * 200 + "\n\n"
        "### 叶子\n\n内容。\n"
    )
    with pytest.raises(ValueError, match=r"category.*VARCHAR\(255\)"):
        split_markdown(text, source_key="ok.md")
