"""Session 1.2 sweeps: batch knee, intra-op threads, fp32 vs int8, RSS.

Engine-level, no HTTP: this measures the model, not the serving stack.
Session 1.3 re-runs the sweeps through HTTP; the diff is serving overhead.

Run on the TARGET hardware. Quantized kernels are ISA-specific and thread
behaviour is cgroup-specific -- laptop numbers do not transfer:

    kubectl -n rag exec <bench-pod> -- python /tmp/benchmark.py \
        --model-path /models/onnx-int8,/tmp/onnx-fp32

Each (model, intra) config runs in a subprocess so ru_maxrss isolates the
peak RSS per config -- that number sizes the pod memory limit.
"""

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

WORDS = ["retrieval", "augmented", "generation", "embeds", "document", "chunks", "into", "dense", "vectors", "stored", "beside", "their", "metadata", "the", "service", "exposes", "an", "endpoint", "accepting", "batches", "applies", "the", "passage", "prefix", "mean", "pools", "over", "the", "attention", "mask", "and", "normalises", "every", "vector", "before", "returning", "it", "to", "the", "caller"]


def make_texts(n: int = 128, words: int = 40) -> list[str]:
    """~`words`-word synthetic passages, deterministic, all distinct.
    40 words ~ a short sentence; 330 words ~ a 480-token chunk."""
    out = []
    for i in range(n):
        picks = [WORDS[(i * 7 + j * 3) % len(WORDS)] for j in range(words)]
        out.append(" ".join(picks) + f" ({i})")
    return out


def bench_worker(model_path: str, intra: int, batches: list[int],
                 words: int = 40) -> None:
    from rag_embedder.engine import EmbeddingEngine

    engine = EmbeddingEngine(model_path, intra_op_num_threads=intra)
    texts = make_texts(words=words)
    results = []
    for b in batches:
        batch = (texts * (b // len(texts) + 1))[:b]
        for _ in range(2):
            engine.embed(batch, "passage")
        times: list[float] = []
        deadline = time.perf_counter() + 3.0
        while len(times) < 3 or (time.perf_counter() < deadline and len(times) < 50):
            t0 = time.perf_counter()
            engine.embed(batch, "passage")
            times.append(time.perf_counter() - t0)
        times.sort()
        p50 = times[len(times) // 2]
        results.append({"batch": b, "iters": len(times),
                        "p50_ms": round(p50 * 1000, 1),
                        "texts_per_s": round(b / p50, 1)})
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = round(maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    print(json.dumps({"model": model_path, "intra": intra,
                      "rss_mb": rss_mb, "results": results}))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="comma-separated model dirs")
    p.add_argument("--intra", default="1,2", help="comma-separated thread counts")
    p.add_argument("--batches", default="1,8,16,32,64,128")
    p.add_argument("--words", type=int, default=40,
                   help="words per synthetic text (330 ~ a 480-token chunk)")
    p.add_argument("--json", default=None, help="also write results to this file")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()

    batches = [int(x) for x in a.batches.split(",")]
    if a.worker:
        bench_worker(a.model_path, int(a.intra), batches, a.words)
        return 0

    configs = []
    for model in a.model_path.split(","):
        for intra in (int(x) for x in a.intra.split(",")):
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", "--model-path", model,
                 "--intra", str(intra), "--batches", a.batches,
                 "--words", str(a.words)],
                capture_output=True, text=True, check=True)
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            r["model"] = Path(model).name
            configs.append(r)
            print(f"\n{Path(model).name}  intra={intra}  peak_rss={r['rss_mb']}MB")
            print(f"  {'batch':>5} {'iters':>5} {'p50_ms':>8} {'texts/s':>8}")
            for row in r["results"]:
                print(f"  {row['batch']:>5} {row['iters']:>5} "
                      f"{row['p50_ms']:>8} {row['texts_per_s']:>8}")
    if a.json:
        Path(a.json).write_text(json.dumps({"engine": configs}, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
