"""Batcher contract tests.

The one that matters most is the un-permute round-trip on shuffled input:
getting it wrong returns correct embeddings attached to the wrong texts --
every vector is unit-norm, every response well-formed, and retrieval is
quietly garbage.

These use a fake embed_fn so they run without a model. The fake encodes the
text's identity into the vector, which is exactly what makes permutation
bugs visible.
"""

import asyncio
import hashlib

import numpy as np
import pytest

from rag_embedder.batching import DynamicBatcher, bucket_by_length, padding_waste


def identity_vector(text: str, input_type: str) -> list[float]:
    h = hashlib.sha256(f"{input_type}:{text}".encode()).digest()
    return [b / 255 for b in h[:4]]


class FakeEmbed:
    """Records every call; returns vectors derived from each text's identity."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def __call__(self, texts: list[str], input_type: str) -> np.ndarray:
        self.calls.append((list(texts), input_type))
        return np.array([identity_vector(t, input_type) for t in texts])


# --- bucket_by_length ---------------------------------------------------------


def test_bucket_sorts_by_length():
    texts = ["aaaa", "a", "aaa", "aa"]
    sorted_texts, _ = bucket_by_length(texts)
    assert sorted_texts == ["a", "aa", "aaa", "aaaa"]


def test_bucket_roundtrip_unpermutes():
    texts = [f"{'x' * (i * 37 % 19)}-{i}" for i in range(50)]
    sorted_texts, original_indices = bucket_by_length(texts)
    restored: list[str | None] = [None] * len(texts)
    for pos, orig in enumerate(original_indices):
        restored[orig] = sorted_texts[pos]
    assert restored == texts


# --- padding_waste ------------------------------------------------------------


def test_padding_waste_empty():
    assert padding_waste([]) == 0.0


def test_padding_waste_uniform():
    assert padding_waste([7, 7, 7]) == 0.0


def test_padding_waste_mixed():
    # padded = 2 * 9 = 18, real = 10 -> waste 8/18
    assert padding_waste([1, 9]) == pytest.approx(8 / 18)


# --- DynamicBatcher -----------------------------------------------------------


async def test_roundtrip_on_shuffled_concurrent_input():
    """THE test: 64 concurrent callers, every caller gets ITS vector back."""
    fake = FakeEmbed()
    batcher = DynamicBatcher(max_size=16, max_wait_ms=5, embed_fn=fake)
    runner = asyncio.create_task(batcher.run())
    try:
        texts = [f"{'x' * (i * 31 % 41)} text {i}" for i in range(64)]
        results = await asyncio.gather(
            *(batcher.submit([t], "passage") for t in texts))
        for text, res in zip(texts, results):
            expected = identity_vector(text, "passage")
            assert np.allclose(res.vectors[0], expected), f"wrong vector for {text!r}"
    finally:
        runner.cancel()


async def test_flushes_at_max_size_without_waiting():
    fake = FakeEmbed()
    batcher = DynamicBatcher(max_size=8, max_wait_ms=10_000, embed_fn=fake)
    runner = asyncio.create_task(batcher.run())
    try:
        submits = [batcher.submit([f"t{i}"], "query") for i in range(8)]
        results = await asyncio.wait_for(asyncio.gather(*submits), timeout=2.0)
        assert all(r.batch_size == 8 for r in results)
        assert sum(len(c[0]) for c in fake.calls) == 8
    finally:
        runner.cancel()


async def test_flushes_at_max_wait_when_batch_not_full():
    fake = FakeEmbed()
    batcher = DynamicBatcher(max_size=1000, max_wait_ms=30, embed_fn=fake)
    runner = asyncio.create_task(batcher.run())
    try:
        res = await asyncio.wait_for(batcher.submit(["solo"], "query"), timeout=2.0)
        assert res.batch_size == 1
    finally:
        runner.cancel()


async def test_input_types_are_never_mixed_in_one_engine_call():
    """query: and passage: prefixes differ; a merged call would apply one
    prefix to the other's texts."""
    fake = FakeEmbed()
    batcher = DynamicBatcher(max_size=8, max_wait_ms=5, embed_fn=fake)
    runner = asyncio.create_task(batcher.run())
    try:
        await asyncio.gather(
            batcher.submit(["q1"], "query"),
            batcher.submit(["p1"], "passage"),
            batcher.submit(["q2"], "query"),
            batcher.submit(["p2"], "passage"),
        )
        for texts, input_type in fake.calls:
            assert len({input_type}) == 1
            for t in texts:
                assert t.startswith("q") == (input_type == "query")
    finally:
        runner.cancel()


async def test_engine_error_propagates_to_caller():
    def broken(texts: list[str], input_type: str) -> np.ndarray:
        raise ValueError("input_type must be one of ['passage', 'query']")

    batcher = DynamicBatcher(max_size=4, max_wait_ms=5, embed_fn=broken)
    runner = asyncio.create_task(batcher.run())
    try:
        with pytest.raises(ValueError):
            await asyncio.wait_for(batcher.submit(["x"], "query"), timeout=2.0)
    finally:
        runner.cancel()


async def test_multi_text_request_stays_intact():
    fake = FakeEmbed()
    batcher = DynamicBatcher(max_size=32, max_wait_ms=5, embed_fn=fake)
    runner = asyncio.create_task(batcher.run())
    try:
        texts = [f"{'y' * (7 - i)} multi {i}" for i in range(7)]
        res = await asyncio.wait_for(batcher.submit(texts, "passage"), timeout=2.0)
        assert len(res.vectors) == 7
        for text, vec in zip(texts, res.vectors):
            assert np.allclose(vec, identity_vector(text, "passage"))
    finally:
        runner.cancel()


async def test_token_budget_caps_flush_of_long_texts():
    """The memory-cap contract: many long texts queued at once must NOT be
    reassembled into one giant flush. 12 requests of 8 x ~1920-char texts
    (~480 estimated tokens each) against a 3840-token budget -> every
    engine call stays at or under ~8 long texts."""
    fake = FakeEmbed()
    batcher = DynamicBatcher(max_size=32, max_wait_ms=5, embed_fn=fake,
                             token_budget=3840, max_length=512)
    runner = asyncio.create_task(batcher.run())
    try:
        long_text = "x" * 1920
        submits = [
            batcher.submit([f"{long_text}{i}-{j}" for j in range(8)], "passage")
            for i in range(12)
        ]
        await asyncio.wait_for(asyncio.gather(*submits), timeout=5.0)
        assert len(fake.calls) >= 12, "long-text requests were merged"
        for texts, _ in fake.calls:
            est = sum(min(len(t) // 4 + 1, 512) for t in texts)
            # a flush may exceed the budget only when it is a single
            # request (which the engine bounds via max_length truncation)
            if len(texts) > 8:
                assert est <= 3840, f"multi-request flush over budget: {est}"
    finally:
        runner.cancel()


async def test_carryover_request_is_not_lost():
    """A request stashed as carryover must still be answered."""
    fake = FakeEmbed()
    batcher = DynamicBatcher(max_size=32, max_wait_ms=20, embed_fn=fake,
                             token_budget=100, max_length=512)
    runner = asyncio.create_task(batcher.run())
    try:
        texts = ["y" * 300, "z" * 300, "w" * 300]  # ~76 est tokens each
        results = await asyncio.wait_for(
            asyncio.gather(*(batcher.submit([t], "passage") for t in texts)),
            timeout=5.0)
        for text, res in zip(texts, results):
            assert np.allclose(res.vectors[0],
                               identity_vector(text, "passage"))
    finally:
        runner.cancel()


async def test_short_texts_still_batch_up_under_budget():
    """Queries must keep batching: the budget only bites on long texts."""
    fake = FakeEmbed()
    batcher = DynamicBatcher(max_size=8, max_wait_ms=10_000, embed_fn=fake,
                             token_budget=3840, max_length=512)
    runner = asyncio.create_task(batcher.run())
    try:
        submits = [batcher.submit([f"short {i}"], "query") for i in range(8)]
        results = await asyncio.wait_for(asyncio.gather(*submits), timeout=2.0)
        assert all(r.batch_size == 8 for r in results)
    finally:
        runner.cancel()
