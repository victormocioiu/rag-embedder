# TASKS — rag-embedder

Ordered. Each is independently verifiable.

## Session 1.1 — a callable endpoint

- [x] Run `scripts/export_model.py --target avx512` locally; confirm
      `onnx-int8/tokenizer.json` and a `.onnx` file both exist
- [x] `uv run uvicorn rag_embedder.main:app --port 8001`, then
      `POST /embed {"texts":["hello"],"input_type":"query"}` → 384 floats, **norm 1.0**
- [x] `GET /readyz` 503s before the model loads, 200 after
- [x] `POST /embed` with `input_type: "document"` → 422, not a silent wrong prefix
- [x] Add `.github/workflows/build.yml`, push to GHCR, make the package public
- [x] Deploy to the `embed` pool; verify `kubectl -n rag get pod -o wide` shows
      `edka-pool-embed-worker1`
- [x] Tailscale `Ingress`, `ingressClassName: tailscale`, host `embed`
- [x] Finish `scripts/parity_check.py`; expand `SENTENCES` to ~200; assert
      int8-vs-PyTorch cosine ≥ 0.99

## Session 1.2 — measured, before optimising

- [x] Batch-size sweep {1,8,16,32,64,128} → find the knee
- [x] `intra_op_num_threads` sweep — {1,2} on this node
- [x] fp32 vs int8: throughput **and** retrieval quality
- [x] Record RSS per config to size the pod memory limit

## Session 1.3 — dynamic batching

- [x] `DynamicBatcher.submit` / `.run` — flush at `max_size` **or** `max_wait_ms`
- [x] `bucket_by_length` — sort, batch, un-permute
- [x] **Test the un-permute round-trip on shuffled input.** Getting it wrong returns
      correct embeddings attached to the wrong texts: silent, and horrible to find
- [x] `padding_waste()` reported in `/metrics`
- [x] Re-run 1.2's sweeps through HTTP; the diff is serving overhead
- [x] Concurrency sweep {1,4,16,64} single-text clients — proves batching works
