"""Session 1.3 sweeps, through HTTP. stdlib only -- runs anywhere, including
inside the serving image (no httpx there).

Two sweeps against a live endpoint:

1. Batch sweep (single client, growing request batch) -- compare with the
   engine-level numbers in docs/benchmarks-1.2.md; the diff is serving
   overhead (HTTP, JSON, queueing).
2. Concurrency sweep {1,4,16,64} single-text clients -- the dynamic batcher's
   whole reason to exist. Without it, throughput is flat at ~1/latency; with
   it, reported batch_size climbs and throughput approaches the engine's
   batched rate.

    python scripts/benchmark_http.py --url http://rag-embedder.rag.svc:8001
"""

import argparse
import json
import statistics
import threading
import time
import urllib.request

WORDS = ["retrieval", "augmented", "generation", "embeds", "document", "chunks",
         "into", "dense", "vectors", "stored", "beside", "their", "metadata",
         "the", "service", "exposes", "an", "endpoint", "accepting", "batches"]


def make_text(i: int) -> str:
    return " ".join(WORDS[(i * 3 + j) % len(WORDS)] for j in range(40)) + f" ({i})"


def post(url: str, texts: list[str]) -> dict:
    body = json.dumps({"texts": texts, "input_type": "passage"}).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/embed", data=body,
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def batch_sweep(url: str, batches: list[int]) -> list[dict]:
    print(f"\nbatch sweep (single client)\n  {'batch':>5} {'p50_ms':>8} {'texts/s':>8}")
    rows = []
    for b in batches:
        texts = [make_text(i) for i in range(b)]
        post(url, texts)  # warmup
        times = []
        deadline = time.perf_counter() + 3.0
        while len(times) < 3 or (time.perf_counter() < deadline and len(times) < 30):
            t0 = time.perf_counter()
            post(url, texts)
            times.append(time.perf_counter() - t0)
        p50 = statistics.median(times)
        rows.append({"batch": b, "p50_ms": round(p50 * 1000, 1),
                     "texts_per_s": round(b / p50, 1)})
        print(f"  {b:>5} {p50 * 1000:>8.1f} {b / p50:>8.1f}")
    return rows


def concurrency_sweep(url: str, levels: list[int], seconds: float) -> list[dict]:
    print(f"\nconcurrency sweep (single-text clients, {seconds:.0f}s each)")
    print(f"  {'clients':>7} {'reqs':>6} {'p50_ms':>8} {'p95_ms':>8} "
          f"{'req/s':>7} {'mean_flush':>10}")
    rows = []
    for n in levels:
        latencies: list[float] = []
        flushes: list[int] = []
        lock = threading.Lock()
        stop = time.perf_counter() + seconds

        def client(worker: int, stop: float = stop, lock: threading.Lock = lock,
                   latencies: list[float] = latencies,
                   flushes: list[int] = flushes) -> None:
            i = 0
            while time.perf_counter() < stop:
                t0 = time.perf_counter()
                r = post(url, [make_text(worker * 1000 + i)])
                dt = time.perf_counter() - t0
                with lock:
                    latencies.append(dt)
                    flushes.append(r["batch_size"])
                i += 1

        threads = [threading.Thread(target=client, args=(w,)) for w in range(n)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        latencies.sort()
        p50 = latencies[len(latencies) // 2] * 1000
        p95 = latencies[int(len(latencies) * 0.95)] * 1000
        rows.append({"clients": n, "requests": len(latencies),
                     "p50_ms": round(p50, 1), "p95_ms": round(p95, 1),
                     "req_per_s": round(len(latencies) / wall, 1),
                     "mean_flush": round(statistics.mean(flushes), 1)})
        print(f"  {n:>7} {len(latencies):>6} {p50:>8.1f} {p95:>8.1f} "
              f"{len(latencies) / wall:>7.1f} {statistics.mean(flushes):>10.1f}")
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--batches", default="1,8,16,32,64,128")
    p.add_argument("--concurrency", default="1,4,16,64")
    p.add_argument("--seconds", type=float, default=10.0,
                   help="duration per concurrency level")
    p.add_argument("--json", default=None, help="also write results to this file")
    a = p.parse_args()
    batches = batch_sweep(a.url, [int(x) for x in a.batches.split(",")])
    conc = concurrency_sweep(
        a.url, [int(x) for x in a.concurrency.split(",")], a.seconds)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"http_batch": batches, "http_concurrency": conc}, f, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
