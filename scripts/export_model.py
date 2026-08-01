"""Export + quantize multilingual-e5-small to ONNX. Runs at IMAGE BUILD time.

Baking the artifact into the layer keeps pod start fast and removes a network
dependency from readiness.

TARGET SELECTION -- this is not cosmetic:

    avx512        AVX-512 WITHOUT VNNI (Skylake-SP and similar). Applies
                  reduce_range to mitigate VPMADDUBSW saturation, which
                  otherwise degrades accuracy.
    avx512_vnni   AVX-512 WITH VNNI (Cascade Lake+, Zen 4). Skips reduce_range
                  because VNNI has no saturation problem. Using this on
                  non-VNNI hardware is a correctness bug. THIS IS OUR NODE
                  (AMD EPYC Genoa).
    avx2          no AVX-512 at all.
    arm64         Ampere/Graviton.

Check with:  grep -o 'avx512[a-z_]*' /proc/cpuinfo | sort -u
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

TARGET_FLAG = {
    "avx2": "--avx2",
    "avx512": "--avx512",
    "avx512_vnni": "--avx512_vnni",
    "arm64": "--arm64",
}

# `optimum-cli onnxruntime quantize` writes only the model. Without these the
# tokenizer cannot load and the container dies on startup.
SIDECAR_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "sentencepiece.bpe.model",
    "spiece.model",
    "vocab.txt",
]


# The CLI must be invoked via its entry point, not `python -m`: running the
# module as __main__ creates a second copy of the subcommand registry, and the
# `onnxruntime` command group (registered by the subpackage loader into the
# first copy) is missing from it.
OPTIMUM_CLI = str(Path(sys.executable).with_name("optimum-cli"))


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--target", choices=sorted(TARGET_FLAG), default="avx512_vnni")
    p.add_argument("--out", default="onnx-int8")
    p.add_argument("--fp32-dir", default=None,
                   help="where the unquantized export lands (default: <out>/../onnx-fp32)")
    p.add_argument("--skip-quantize", action="store_true",
                   help="fp32 only -- part 1 needs both for the comparison")
    a = p.parse_args()

    out = Path(a.out)
    fp32 = Path(a.fp32_dir) if a.fp32_dir else out.parent / "onnx-fp32"

    # --library-name pins the load path: if sentence-transformers happens to be
    # installed (parity group), optimum would infer it and crash on ST >= 5
    run([OPTIMUM_CLI, "export", "onnx", "--model", a.model,
         "--task", "feature-extraction", "--library-name", "transformers",
         str(fp32)])

    if a.skip_quantize:
        shutil.copytree(fp32, out, dirs_exist_ok=True)
        print(f"fp32 export at {out}")
        return 0

    # --per_channel: per-tensor weight scales lose ~0.013 mean cosine vs the
    # fp32 reference on this model; per-channel recovers it (0.987 -> 0.996,
    # measured on the 200-sentence parity set)
    run([OPTIMUM_CLI, "onnxruntime", "quantize", TARGET_FLAG[a.target],
         "--per_channel", "--onnx_model", str(fp32), "-o", str(out)])

    out.mkdir(parents=True, exist_ok=True)
    copied = []
    for f in SIDECAR_FILES:
        src = fp32 / f
        if src.exists():
            shutil.copy(src, out / f)
            copied.append(f)
    print(f"copied tokenizer files: {copied}")

    if not (out / "tokenizer.json").exists():
        print("ERROR: tokenizer.json missing from output -- the service will not start",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
