"""Run every benchmark in sequence and merge results into one JSON.

Stages and where they must run:

- engine sweep    on the serving hardware (quantized kernels are ISA-specific)
- HTTP sweeps     from inside the cluster (WAN RTT would drown the signal)
- parity check    anywhere with the parity dep group + reach to the endpoint

Each stage is optional, so the suite runs wherever it is and JSONs from
stages run elsewhere are merged via --ingest:

    # on the embed-node bench pod
    python benchmark.py --model-path /models/onnx-int8,/tmp/onnx-fp32 --json engine.json
    # on an in-cluster client pod
    python benchmark_http.py --url http://rag-embedder.rag.svc --json http.json
    # locally: parity + merge everything
    uv run python scripts/benchmark_suite.py --url https://<endpoint> --parity \
        --ingest engine.json http.json --label tuned-v2 --out results/tuned-v2.json

Plot with scripts/plot_benchmarks.py.
"""

import argparse
import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).parent


def run_stage(cmd: list[str]) -> dict:
    print("+", " ".join(cmd), flush=True)
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        subprocess.run([*cmd, "--json", tmp.name], check=True)
        return json.loads(Path(tmp.name).read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=None, help="live endpoint")
    p.add_argument("--engine", default=None,
                   help="comma-separated model dirs: run the engine sweep HERE")
    p.add_argument("--http", action="store_true", help="run HTTP sweeps HERE")
    p.add_argument("--parity", action="store_true", help="run parity check HERE")
    p.add_argument("--ingest", nargs="*", default=[],
                   help="JSON files from stages run elsewhere")
    p.add_argument("--label", default="run")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    merged: dict = {"meta": {
        "label": a.label,
        "url": a.url,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }}

    for path in a.ingest:
        merged.update(json.loads(Path(path).read_text()))
    if a.engine:
        merged.update(run_stage(
            [sys.executable, str(SCRIPTS / "benchmark.py"), "--model-path", a.engine]))
    if a.http:
        if not a.url:
            p.error("--http needs --url")
        merged.update(run_stage(
            [sys.executable, str(SCRIPTS / "benchmark_http.py"), "--url", a.url]))
    if a.parity:
        if not a.url:
            p.error("--parity needs --url")
        merged.update(run_stage(
            [sys.executable, str(SCRIPTS / "parity_check.py"), "--url", a.url]))

    out = Path(a.out or f"results/{a.label}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, indent=1))
    stages = [k for k in merged if k != "meta"]
    print(f"\nwrote {out} with stages: {', '.join(stages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
