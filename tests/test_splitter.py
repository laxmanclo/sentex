from sentex.ingestion.splitter import split_sentences


def test_basic_prose():
    text = "The sky is blue. The grass is green. Water is wet."
    result = split_sentences(text)
    assert len(result) == 3
    assert result[0] == "The sky is blue."


def test_code_block_is_atomic():
    text = "Here is some code.\n```python\nx = 1\ny = 2\nz = x + y\n```\nAnd that is it."
    result = split_sentences(text)
    code_blocks = [s for s in result if "```" in s]
    assert len(code_blocks) == 1
    assert "x = 1" in code_blocks[0]


def test_list_items_atomic():
    text = "Items:\n- First item here\n- Second item here\n- Third item here"
    result = split_sentences(text)
    list_items = [s for s in result if s.startswith("-")]
    assert len(list_items) == 3


def test_abbreviations_not_split():
    text = "Dr. Smith works at St. Mary's Hospital. He is a great doctor."
    result = split_sentences(text)
    # NLTK's Punkt handles common abbreviations; result is 1 or 2 sentences
    assert len(result) in (1, 2)
    # The full text content must be preserved
    combined = " ".join(result)
    assert "Dr." in combined
    assert "great doctor" in combined


def test_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_single_sentence():
    result = split_sentences("Just one sentence here.")
    assert len(result) == 1
