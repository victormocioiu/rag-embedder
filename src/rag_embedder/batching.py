"""Dynamic batching with length bucketing.

Two mechanisms, and the second is the one people skip:

**Dynamic batching.** Requests accumulate in a queue; a collector flushes when
the batch reaches ``batch_max_size`` or ``batch_max_wait_ms`` elapses, whichever
comes first. Turns bursty single-item traffic into efficient batched inference
without adding meaningful latency.

**Length bucketing.** Sort by length BEFORE padding. A batch mixing a 12-token
query with a 512-token passage pads everything to 512 and wastes ~97% of the
compute on the short one. Sorting first cuts that dramatically. It is close to
free -- ``embed_padding_waste`` in /metrics quantifies the win.

The collector runs the (GIL-releasing) engine call in a worker thread via
``asyncio.to_thread``, so the event loop keeps accepting requests while ORT
computes. One flush at a time: ORT's internal threading (intra_op) is better
at using the cores than concurrent sessions fighting over them.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

EmbedFn = Callable[[list[str], str], np.ndarray]


@dataclass
class PendingRequest:
    texts: list[str]
    input_type: str
    future: asyncio.Future


@dataclass
class BatchResult:
    """What one caller gets back: its rows, plus how it was served."""

    vectors: np.ndarray  # rows in the caller's original text order
    batch_size: int      # total texts in the flush this request rode in
    inference_ms: float  # engine time for that flush


class DynamicBatcher:
    def __init__(self, max_size: int, max_wait_ms: int, embed_fn: EmbedFn) -> None:
        self.max_size = max_size
        self.max_wait_ms = max_wait_ms
        self.embed_fn = embed_fn
        self.queue: asyncio.Queue[PendingRequest] = asyncio.Queue()

    async def submit(self, texts: list[str], input_type: str) -> BatchResult:
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.queue.put(PendingRequest(texts, input_type, future))
        return await future

    async def run(self) -> None:
        """Collector loop: drain up to max_size texts, or flush after
        max_wait_ms -- whichever comes first."""
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            batch = [first]
            total = len(first.texts)
            deadline = loop.time() + self.max_wait_ms / 1000
            while total < self.max_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    request = await asyncio.wait_for(self.queue.get(), remaining)
                except TimeoutError:
                    break
                batch.append(request)
                total += len(request.texts)
            await self._flush(batch, total)

    async def _flush(self, batch: list[PendingRequest], total: int) -> None:
        # query: and passage: prefixes differ -- never merge input types into
        # one engine call.
        groups: dict[str, list[PendingRequest]] = {}
        for request in batch:
            groups.setdefault(request.input_type, []).append(request)

        started = time.perf_counter()
        sliced: list[tuple[PendingRequest, np.ndarray]] = []
        try:
            for input_type, requests in groups.items():
                flat = [t for r in requests for t in r.texts]
                sorted_texts, original_indices = bucket_by_length(flat)
                embedded = await asyncio.to_thread(
                    self.embed_fn, sorted_texts, input_type)
                # un-permute: row for sorted_texts[pos] belongs at
                # flat[original_indices[pos]]
                restored = np.empty_like(embedded)
                for pos, orig in enumerate(original_indices):
                    restored[orig] = embedded[pos]
                offset = 0
                for r in requests:
                    sliced.append((r, restored[offset:offset + len(r.texts)]))
                    offset += len(r.texts)
        except Exception as exc:  # noqa: BLE001 -- forward ANY engine failure to
            # the waiting callers; the collector loop itself must never die
            for r in batch:
                if not r.future.done():
                    r.future.set_exception(exc)
            return

        inference_ms = (time.perf_counter() - started) * 1000
        for r, rows in sliced:
            if not r.future.done():
                r.future.set_result(BatchResult(rows, total, inference_ms))


def bucket_by_length(texts: list[str]) -> tuple[list[str], list[int]]:
    """Sort by length, return (sorted_texts, original_indices) with
    sorted_texts[i] == texts[original_indices[i]].

    Caller must un-permute the results. Getting this wrong returns the right
    embeddings attached to the wrong texts -- silent and extremely annoying;
    tests/test_batching.py covers the round-trip explicitly.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    return [texts[i] for i in order], order


def padding_waste(lengths: list[int]) -> float:
    """Fraction of the padded batch that is padding. Reported in /metrics to
    quantify the bucketing win."""
    if not lengths:
        return 0.0
    padded = max(lengths) * len(lengths)
    return 1.0 - (sum(lengths) / padded)
