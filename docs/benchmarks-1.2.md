# Session 1.2 — measurements

2026-08-01, on `hrag-pool-embed-worker1` (Hetzner CPX-class, AMD EPYC Genoa,
2 vCPU, AVX-512 **with** VNNI). Bench pod ran the serving image (`fa440dd`,
model `e5-small-avx512vnni-v1`, per-channel int8) with **no cpu limit**; the
serving pod was idle. Engine-level (`scripts/benchmark.py`), no HTTP: 1.3
re-runs these through HTTP and the diff is serving overhead.

## Batch × threads × model

| model | intra | b=1 p50 | b=8 | b=32 | b=128 | peak texts/s | peak RSS |
|---|---|---|---|---|---|---|---|
| int8 | 1 | 12.1ms | 80.2ms | 346.7ms | 1578.4ms | 99.8 @ b8 | 830MB |
| int8 | 2 | 10.5ms | 59.3ms | 210.4ms | 921.8ms | **152.1 @ b32** | 832MB |
| fp32 | 1 | 28.2ms | 187.1ms | 810.8ms | 3703.0ms | 42.8 @ b8 | 1182MB |
| fp32 | 2 | 19.3ms | 113.4ms | 452.9ms | 2008.5ms | 71.4 @ b16 | 1180MB |

## Findings

1. **Batch knee: b=32 with intra=2** (152 texts/s; decay past b=64). With
   intra=1 the knee is early (b=8) and shallow. Consequence for 1.3: the
   `batch_max_size=32` default sits exactly on the knee.
2. **intra=2 is ~1.5× intra=1** on both models. Genoa's single core is strong
   enough that the second thread buys less than a weaker CPU would show, but
   it is still the single best setting change available. The rule stands:
   threads must be backed by the cpu **limit** — thread counts above the
   cgroup quota measure *worse* than one thread.
3. **int8 is ~2.1× fp32** at equal threads, ~350MB less RSS. Quality cost is
   the parity number: mean cosine 0.9961 / p5 0.9941 vs fp32's 1.000
   (`scripts/parity_check.py`, measured against the deployed endpoint).
4. **RSS: int8 peaks at 832MB** after b=128 runs (ORT's arena grows with the
   largest batch seen and does not shrink). The 2Gi pod limit has ample
   headroom; 1536Mi would also be safe.

## Deployment settings (edka form)

- cpu request `500m`–`1`, cpu **limit `2`** — the limit backs the threads
- env `INTRA_OP_NUM_THREADS=2`, `INTER_OP_NUM_THREADS=1`
- memory request `512Mi`, limit `2Gi` (832MB measured peak)
- expected: single-text engine time ~10ms, ~150 texts/s at the knee
