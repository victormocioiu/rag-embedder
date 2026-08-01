# The quantized ONNX artifact is baked into the image -- no model download at
# pod start, so readiness is fast and has no network dependency. It comes
# prebuilt from the model image (Dockerfile.model, built on GitHub runners:
# the export needs ~3GB RAM, more than in-cluster builders have), which keeps
# this build light enough to run anywhere.
#
# The quant target lives in the model image tag: avx512_vnni, for the AMD
# EPYC Genoa embed node (AVX-512 WITH VNNI).

ARG MODEL_IMAGE=ghcr.io/victormocioiu/rag-embedder-model:e5-small-avx512vnni-v1
FROM ${MODEL_IMAGE} AS model

FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock* ./
RUN uv venv /venv && VIRTUAL_ENV=/venv uv pip install --no-cache .

FROM python:3.12-slim
RUN useradd -r -u 10001 app
COPY --from=builder /venv /venv
COPY --from=model /models /models
COPY src/ /app/src/
ENV PATH="/venv/bin:$PATH" PYTHONPATH=/app/src MODEL_PATH=/models/onnx-int8
# numeric UID so Kubernetes runAsNonRoot can verify it
USER 10001
EXPOSE 8001
CMD ["uvicorn", "rag_embedder.main:app", "--host", "0.0.0.0", "--port", "8001"]
