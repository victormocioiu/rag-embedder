# Session 1.3 — dynamic batching, measured over HTTP

2026-08-01, image `fa440dd` (batching: max_size 32, max_wait 10ms), serving
pod on the Genoa embed node with cpu limit 2 / `INTRA_OP_NUM_THREADS=2`.
Client: `scripts/benchmark_http.py` in a pod on the default worker, calling
the in-cluster Service (port 80 → container 8001).

## Batch sweep (single client) vs engine-level (1.2)

| batch | HTTP p50 | engine p50 | serving overhead |
|---|---|---|---|
| 1 | 22.6ms | 10.5ms | +12.1ms |
| 8 | 74.9ms | 59.3ms | +15.6ms |
| 32 | 221.0ms | 210.4ms | +10.6ms |
| 128 | 991.9ms | 921.8ms | +70.1ms |

Overhead is a fixed ~12ms at batch 1 — mostly the batcher's 10ms max_wait
window, paid in full by solo traffic — and settles to **~0.5ms per text** at
large batches (JSON serialization + response validation). At the knee (b=32)
the serving stack costs 5% of the engine time.

## Concurrency sweep (single-text clients, 10s per level)

| clients | p50 | p95 | req/s | mean flush size |
|---|---|---|---|---|
| 1 | 26.0ms | 28.9ms | 38.8 | 1.0 |
| 4 | 49.8ms | 59.1ms | 81.1 | 4.0 |
| 16 | 140.5ms | 174.2ms | 111.7 | 15.7 |
| 64 | 488.2ms | 550.9ms | **130.1** | 31.8 |

- **Batching works**: mean flush climbs 1 → 4 → 16 → 31.8 (saturating at
  `batch_max_size`), and throughput scales 38.8 → 130 req/s (~3.4×) while
  p95 stays a constant factor above p50 — queueing, not collapse.
- Sustained ceiling ~130 req/s vs 152 texts/s engine-side; the gap is the
  serving overhead above.

## Verdict

max_size 32 / max_wait 10ms sits exactly on this node's knee; keep it.
Solo-traffic latency pays the 10ms window — lower max_wait to ~5ms if solo
p50 matters more than peak coalescing.
