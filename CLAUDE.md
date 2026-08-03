# CLAUDE.md — rag-embedder

Self-hosted ONNX embedding service. `multilingual-e5-small`, int8, CPU.

**No internal dependencies.** Not on `rag-core`, not on a database. The embedder is a
pure function of its input — small image, fast start, failure modes independent of
everything else. Keep it that way.

## Invariants

1. **`input_type` is a required field**, never a defaulted kwarg. e5 needs
   `query:` / `passage:` prefixes and a default is a value someone forgets.
2. **Mean pooling over the attention mask**, not CLS.
3. **L2-normalise before returning.** Downstream uses `halfvec_ip_ops` + `<#>`,
   which only equals cosine for unit vectors. A returned norm != 1.0 is a bug.
4. **Build the ORT feed dict from `session.get_inputs()`.** XLM-R has no
   `token_type_ids`; BERT does.
5. **`/readyz` 503s until the model is loaded.** Without it, rollouts drop requests.
6. **Quantization target is `--avx512_vnni`** — AMD EPYC Genoa, AVX-512 with
   VNNI. Read `/proc/cpuinfo` flags before changing it. Nuance, verified: in
   optimum 1.26 the `avx512` and `avx512_vnni` presets emit byte-identical
   models (both default `reduce_range=False`); the target records intent —
   the parity gate is what actually verifies accuracy on the serving CPU.
7. **The batcher's flush cap is a TOKEN budget, not a text count**
   (`batch_token_budget`, chars/4 estimate, carryover for overflow). Learned
   in production: concurrent ingest re-assembled 8-text requests into
   32-long-passage flushes → arena past the pod limit → all replicas
   OOMKilled (2026-08-02, EnterpriseRAG-Bench ingest). `max_size` alone
   does NOT bound memory.

## State

| | |
|---|---|
| implemented | everything in `src/` and `scripts/` — engine, batching, parity check, engine + HTTP benchmarks |
| stubbed | nothing; part 1 complete |

## Commands

```bash
uv sync
uv sync --group export
uv run python scripts/export_model.py --target avx512 --out onnx-int8
uv run uvicorn rag_embedder.main:app --port 8001
uv run pytest -v
docker build -t rag-embedder:dev .   # model comes from the rag-embedder-model image
```

## Deployment

Runs on the tainted `embed` pool. Deployed through edka's deployment UI
(GitHub-repo mode: in-cluster build → zot registry → auto-deploy); its
PLACEMENT tab sets `nodeSelector`/`tolerations`.

Node is 2 vCPU (AMD EPYC Genoa). ORT threads must not exceed the pod's cpu
**limit**; measured settings and the batch knee are in
`docs/benchmarks-1.2.md` (cpu limit 2 + `INTRA_OP_NUM_THREADS=2` ≈ 1.5×
over 1/1; knee at batch 32, ~150 texts/s).

## Next

Part 1 complete (all TASKS.md boxes ticked). Serving numbers:
`docs/benchmarks-1.2.md` (engine) and `docs/benchmarks-1.3.md` (HTTP,
batching). Known cheap win if throughput ever matters: the response path
spends ~7ms/text on JSON + pydantic validation of the vectors.
