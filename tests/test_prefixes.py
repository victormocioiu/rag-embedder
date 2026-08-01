"""The prefix rule is a correctness issue, so it gets a real test."""

import pytest

from rag_embedder.tokenizer import apply_prefix


def test_query_prefix():
    assert apply_prefix(["hello"], "query") == ["query: hello"]


def test_passage_prefix():
    assert apply_prefix(["hello"], "passage") == ["passage: hello"]


def test_unknown_input_type_raises():
    # Loud failure beats a silently wrong prefix.
    with pytest.raises(ValueError):
        apply_prefix(["hello"], "document")
