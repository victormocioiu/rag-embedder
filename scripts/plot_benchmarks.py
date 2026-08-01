"""Render every figure from a merged benchmark-suite JSON.

    uv sync --group plots
    uv run python scripts/plot_benchmarks.py results/tuned-v2.json --out docs/figures

Encoding rules (kept deliberately boring):
- color follows the entity: int8 blue, fp32 orange, HTTP aqua -- never the rank
- thread count is a line style (intra=2 solid, intra=1 dashed), not a hue
- one y-axis per plot; where two measures matter, two stacked panels
- batch/client axes are log2 (the sweeps are geometric)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots
CRITICAL = "#d03b3b"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "figure.dpi": 150,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6.5,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
}

ENGINE_STYLE = {
    ("onnx-int8", 2): {"color": BLUE, "ls": "-", "label": "int8, 2 threads"},
    ("onnx-int8", 1): {"color": BLUE, "ls": "--", "label": "int8, 1 thread"},
    ("onnx-fp32", 2): {"color": ORANGE, "ls": "-", "label": "fp32, 2 threads"},
    ("onnx-fp32", 1): {"color": ORANGE, "ls": "--", "label": "fp32, 1 thread"},
}


def log2_axis(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()


def save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def engine_series(data):
    for cfg in data.get("engine", []):
        key = (Path(cfg["model"]).name, cfg["intra"])
        style = ENGINE_STYLE.get(key)
        if style:
            rows = cfg["results"]
            yield style, [r["batch"] for r in rows], rows, cfg


def fig_throughput(data, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    batches = None
    for style, batches, rows, _ in engine_series(data):
        ax.plot(batches, [r["texts_per_s"] for r in rows], marker="o", **style)
    http = data.get("http_batch")
    if http:
        ax.plot([r["batch"] for r in http], [r["texts_per_s"] for r in http],
                marker="o", color=AQUA, ls="-", label="int8 via HTTP")
    log2_axis(ax, batches)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("batch size (texts per engine call)")
    ax.set_ylabel("throughput (texts / s)")
    ax.set_title("Embedding throughput vs batch size — engine and HTTP")
    ax.legend(loc="lower center", ncols=3)
    best = next((rows for _, _, rows, c in engine_series(data)
                 if c["intra"] == 2 and "int8" in c["model"]), None)
    if best:
        knee = max(best, key=lambda r: r["texts_per_s"])
        ax.annotate("knee: batching stops paying",
                    (knee["batch"], knee["texts_per_s"]),
                    xytext=(-118, 8), textcoords="offset points",
                    fontsize=8.5, color=INK2,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    save(fig, out, "throughput_vs_batch.png")


def fig_latency(data, out):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    batches = None
    for style, batches, rows, _ in engine_series(data):
        ax.plot(batches, [r["p50_ms"] for r in rows], marker="o", **style)
    http = data.get("http_batch")
    if http:
        ax.plot([r["batch"] for r in http], [r["p50_ms"] for r in http],
                marker="o", color=AQUA, label="int8 via HTTP")
    log2_axis(ax, batches)
    ax.set_yscale("log")
    ax.set_xlabel("batch size (texts per engine call)")
    ax.set_ylabel("p50 latency (ms, log)")
    ax.set_title("Latency vs batch size — linear cost past the knee")
    ax.legend(loc="upper left")
    save(fig, out, "latency_vs_batch.png")


def fig_overhead(data, out):
    engine = {r["batch"]: r["p50_ms"]
              for _, _, rows, cfg in engine_series(data)
              if "int8" in cfg["model"] and cfg["intra"] == 2 for r in rows}
    http = data.get("http_batch", [])
    common = [r for r in http if r["batch"] in engine]
    if not common:
        return
    b = [r["batch"] for r in common]
    diff = [r["p50_ms"] - engine[r["batch"]] for r in common]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    ax1.plot(b, diff, marker="o", color=BLUE)
    ax1.set_ylabel("HTTP p50 − engine p50 (ms)")
    ax1.set_title("Serving overhead, absolute")
    ax2.plot(b, [d / bb for d, bb in zip(diff, b)], marker="o", color=BLUE)
    ax2.set_ylabel("overhead per text (ms)")
    ax2.set_title("…and per text (JSON + validation)")
    for ax in (ax1, ax2):
        log2_axis(ax, b)
        ax.set_xlabel("batch size")
        ax.set_ylim(bottom=0)
    save(fig, out, "serving_overhead.png")


def fig_concurrency(data, out):
    rows = data.get("http_concurrency", [])
    if not rows:
        return
    clients = [r["clients"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.6), sharex=True)
    ax1.plot(clients, [r["req_per_s"] for r in rows], marker="o", color=BLUE)
    ax1.set_ylabel("throughput (req / s)")
    ax1.set_ylim(bottom=0)
    ax1.set_title("Dynamic batching under concurrent single-text load")
    ax2.plot(clients, [r["mean_flush"] for r in rows], marker="o", color=BLUE)
    ax2.plot(clients, clients, ls=":", color=MUTED, lw=1.2)
    ax2.annotate("perfect coalescing (flush = clients)",
                 (clients[-2], clients[-2]), xytext=(-4, 10),
                 textcoords="offset points", ha="right",
                 fontsize=8, color=MUTED)
    ax2.axhline(32, color=BASELINE, lw=1)
    ax2.annotate("batch_max_size = 32", (clients[0], 32), xytext=(0, 4),
                 textcoords="offset points", fontsize=8, color=INK2)
    ax2.set_ylabel("mean flush size (texts)")
    ax2.set_xlabel("concurrent single-text clients")
    log2_axis(ax2, clients)
    save(fig, out, "concurrency.png")


def fig_concurrency_latency(data, out):
    rows = data.get("http_concurrency", [])
    if not rows:
        return
    clients = [r["clients"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(clients, [r["p50_ms"] for r in rows], marker="o",
            color=BLUE, label="p50")
    ax.plot(clients, [r["p95_ms"] for r in rows], marker="o",
            color=BLUE, ls="--", label="p95")
    log2_axis(ax, clients)
    ax.set_yscale("log")
    ax.set_xlabel("concurrent single-text clients")
    ax.set_ylabel("latency (ms, log)")
    ax.set_title("Latency under load — queueing, not collapse (p95 tracks p50)")
    ax.legend(loc="upper left")
    save(fig, out, "concurrency_latency.png")


def fig_parity(data, out):
    par = data.get("parity")
    if not par:
        return
    cos = par["cos"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.histplot(cos, bins=40, ax=ax, color=BLUE, edgecolor=SURFACE,
                 linewidth=0.8, alpha=1.0)
    ax.axvline(par["threshold"], color=CRITICAL, lw=1.2, ls="--")
    ax.annotate(f"threshold {par['threshold']}", (par["threshold"], ax.get_ylim()[1]),
                xytext=(-6, -12), textcoords="offset points", ha="right",
                fontsize=8.5, color=CRITICAL)
    mean = sum(cos) / len(cos)
    ax.annotate(f"mean {mean:.4f}\nmin {min(cos):.4f}\nn = {len(cos)}",
                xy=(0.03, 0.95), xycoords="axes fraction", va="top",
                fontsize=8.5, color=INK2)
    ax.set_xlabel("cosine similarity, int8 endpoint vs fp32 PyTorch reference")
    ax.set_ylabel("sentence pairs")
    ax.set_title("Quantization parity across 200 sentences")
    save(fig, out, "parity_distribution.png")


def fig_parity_vs_length(data, out):
    par = data.get("parity")
    if not par or "char_len" not in par:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(par["char_len"], par["cos"], s=22, color=BLUE, alpha=0.75,
               edgecolors=SURFACE, linewidths=0.6)
    ax.axvline(2000, color=MUTED, lw=1, ls=":")
    ax.annotate("≈512-token\ntruncation boundary", (2000, ax.get_ylim()[1]),
                xytext=(-6, -6), textcoords="offset points",
                ha="right", va="top", fontsize=8, color=INK2)
    ax.axhline(par["threshold"], color=CRITICAL, lw=1.2, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("text length (characters, log)")
    ax.set_ylabel("cosine vs fp32 reference")
    ax.set_title("Quantization parity vs text length")
    save(fig, out, "parity_vs_length.png")


def fig_rss(data, out):
    cfgs = list(engine_series(data))
    if not cfgs:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    labels, values, colors = [], [], []
    for style, _, _, cfg in cfgs:
        labels.append(style["label"].replace(", ", "\n"))
        values.append(cfg["rss_mb"])
        colors.append(style["color"])
    bars = ax.bar(labels, values, color=colors, width=0.55,
                  edgecolor=SURFACE, linewidth=2)
    for bar, v in zip(bars, values):
        ax.annotate(f"{v:.0f}", (bar.get_x() + bar.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK2)
    ax.axhline(2048, color=CRITICAL, lw=1.2, ls="--")
    ax.annotate("pod memory limit (2 GiB)", (0, 2048), xytext=(2, 5),
                textcoords="offset points", fontsize=8.5, color=CRITICAL)
    ax.set_ylabel("peak RSS after batch-128 sweep (MB)")
    ax.set_title("Memory per configuration")
    ax.grid(axis="x", visible=False)
    save(fig, out, "rss_per_config.png")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("results", help="merged JSON from benchmark_suite.py")
    p.add_argument("--out", default="docs/figures")
    a = p.parse_args()
    data = json.loads(Path(a.results).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", rc=RC)
    print(f"figures -> {out}/")
    for fn in (fig_throughput, fig_latency, fig_overhead, fig_concurrency,
               fig_concurrency_latency, fig_parity, fig_parity_vs_length,
               fig_rss):
        fn(data, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
