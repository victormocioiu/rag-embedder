"""FastAPI app.

The model loads in the lifespan, before readiness flips. `/readyz` returning 503
until then is what stops Kubernetes routing traffic into a pod that will time out --
without it every rollout drops requests.

No dynamic batching yet: session 1.3 adds it, informed by the batch knee that
session 1.2 measures. Until then `/embed` runs the request batch directly.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from rag_embedder.batching import DynamicBatcher, padding_waste
from rag_embedder.config import get_settings
from rag_embedder.engine import EmbeddingEngine
from rag_embedder.schemas import EmbedRequest, EmbedResponse

state: dict[str, Any] = {}

REQUESTS = Counter("embed_requests_total", "Embed requests", ["input_type"])
TEXTS = Counter("embed_texts_total", "Texts embedded", ["input_type"])
LATENCY = Histogram("embed_inference_seconds", "Inference latency", ["input_type"])
BATCH = Histogram(
    "embed_batch_size", "Texts per request",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
)
FLUSH = Histogram(
    "embed_flush_size", "Texts per engine call after dynamic batching",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
)
PADDING = Histogram(
    "embed_padding_waste", "Fraction of the padded batch that is padding",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    engine = EmbeddingEngine(
        model_path=s.model_path,
        intra_op_num_threads=s.intra_op_num_threads,
        inter_op_num_threads=s.inter_op_num_threads,
        max_length=s.max_sequence_length,
    )

    def embed_and_record(texts: list[str], input_type: str) -> np.ndarray:
        vectors = engine.embed(texts, input_type)
        FLUSH.observe(len(texts))
        PADDING.observe(padding_waste(engine.last_token_lengths))
        return vectors

    batcher = DynamicBatcher(
        max_size=s.batch_max_size,
        max_wait_ms=s.batch_max_wait_ms,
        embed_fn=embed_and_record,
        token_budget=s.batch_token_budget,
        max_length=s.max_sequence_length,
    )
    collector = asyncio.create_task(batcher.run())
    state["engine"] = engine
    state["batcher"] = batcher
    yield
    collector.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await collector
    state.clear()


app = FastAPI(title="rag-embedder", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, str]:
    engine = state.get("engine")
    if engine is None or not engine.is_ready:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    engine = state.get("engine")
    if engine is None or not engine.is_ready:
        raise HTTPException(status_code=503, detail="model not loaded")

    s = get_settings()
    REQUESTS.labels(req.input_type).inc()
    TEXTS.labels(req.input_type).inc(len(req.texts))
    BATCH.observe(len(req.texts))

    started = time.perf_counter()
    try:
        result = await state["batcher"].submit(req.texts, req.input_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    elapsed = time.perf_counter() - started
    LATENCY.labels(req.input_type).observe(elapsed)

    return EmbedResponse(
        embeddings=result.vectors.tolist(),
        dimensions=int(result.vectors.shape[1]),
        model=s.model_id,
        # the flush this request rode in -- shows batcher effectiveness
        batch_size=result.batch_size,
        inference_ms=result.inference_ms,
    )
