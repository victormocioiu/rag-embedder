# rag-embedder

Self-hosted embedding service. `multilingual-e5-small` → ONNX → int8, served on CPU.

**No internal dependencies** — not on `rag-core`, not on a database. Pure function of
its input.

## Check your CPU before building

```bash
grep -o 'avx512[a-z_]*' /proc/cpuinfo | sort -u
```

| You see | Target | Why |
|---|---|---|
| `avx512f` etc **with** `avx512_vnni` | `avx512_vnni` | VNNI's `VPDPBUSD` does int8 dot-product in one instruction |
| `avx512f` etc **without** `avx512_vnni` | `avx512` | no VNNI — if the parity check fails here, re-quantize with `reduce_range` (`VPMADDUBSW` saturation) |
| nothing | `avx2` | no AVX-512 at all |
| ARM | `arm64` | `SDOT`/`UDOT` |

**Current node: AMD EPYC Genoa, AVX-512 with VNNI → `avx512_vnni`.** On hardware
*without* VNNI, that preset is an accuracy bug (it skips the `reduce_range`
saturation mitigation) — check the flags before you build.

Note the CPU model string on Hetzner is a generic QEMU model presented for live
migration — trust the flags, not the name.

## The API detail that is a correctness issue

E5 is trained with asymmetric prefixes: `passage: ` for documents, `query: ` for
queries. Right at index time and wrong at query time gives you a system that works
well enough to ship and badly enough to frustrate. So `input_type` is **required**
on `/embed`.

## Endpoints

| | |
|---|---|
| `POST /embed` | `{texts: [...], input_type: "query"\|"passage"}` |
| `GET /healthz` | liveness |
| `GET /readyz` | **503 until the model is loaded** |
| `GET /metrics` | Prometheus |

## Local

```bash
uv sync --group export
uv run python scripts/export_model.py --target avx512 --out onnx-int8
MODEL_PATH=./onnx-int8 uv run uvicorn rag_embedder.main:app --port 8001
```

## Docs, in reading order

1. [docs/benchmarks.md](docs/benchmarks.md) — every figure, one dataset, how to reproduce
2. [docs/benchmarks-1.2.md](docs/benchmarks-1.2.md) — engine numbers: knee, threads, int8 vs fp32, RSS
3. [docs/benchmarks-1.3.md](docs/benchmarks-1.3.md) — HTTP numbers: serving overhead, batching under load
4. `results/*.json` — the raw datasets the figures are drawn from
