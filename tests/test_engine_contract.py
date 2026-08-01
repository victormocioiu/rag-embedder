"""Contract tests. These define what the engine must do; they are the spec.

Marked skip until a model is exported locally -- run
`make export` first, then `pytest -v`.
"""

import math
import os
from pathlib import Path

import pytest

MODEL = Path(os.environ.get("MODEL_PATH", "onnx-int8"))
needs_model = pytest.mark.skipif(
    not (MODEL / "tokenizer.json").exists(),
    reason="no exported model -- run `make export`",
)


@needs_model
def test_returns_384_dims():
    from rag_embedder.engine import EmbeddingEngine

    e = EmbeddingEngine(str(MODEL))
    v = e.embed(["hello"], "query")
    assert v.shape == (1, 384)


@needs_model
def test_vectors_are_l2_normalised():
    """halfvec_ip_ops + <#> only equals cosine for unit vectors."""
    from rag_embedder.engine import EmbeddingEngine

    e = EmbeddingEngine(str(MODEL))
    v = e.embed(["hello", "a much longer sentence with more tokens in it"], "passage")
    for row in v:
        assert math.isclose(float((row**2).sum()) ** 0.5, 1.0, abs_tol=1e-4)


@needs_model
def test_query_and_passage_differ():
    """If the prefix were being dropped these would be identical."""
    from rag_embedder.engine import EmbeddingEngine

    e = EmbeddingEngine(str(MODEL))
    q = e.embed(["refund policy"], "query")[0]
    p = e.embed(["refund policy"], "passage")[0]
    assert float((q * p).sum()) < 0.999


@needs_model
def test_unknown_input_type_raises():
    from rag_embedder.engine import EmbeddingEngine

    e = EmbeddingEngine(str(MODEL))
    with pytest.raises(ValueError):
        e.embed(["hello"], "document")
