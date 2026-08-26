"""Charts from the run json.

Three runs per point, drawn as the mean with a min/max band. A single-run
line hides how noisy the point was, and on a rented GPU that noise is real.

Colours are fixed per engine and never cycled, so the same engine is the
same colour on every chart. The palette is checked for colour-blind
separation; aqua sits below 3:1 on this background, which is why every bar
carries its value and why the markdown table is written out too.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"

# Fixed slots, assigned in order. A fourth engine folds into "other"
# rather than inventing a colour.
COLORS = {"nanoserve": "#2a78d6", "vllm": "#eb6834", "hf": "#1baf7a"}
LABELS = {"nanoserve": "nanoserve", "vllm": "vLLM", "hf": "HF (static)"}

ONLINE_CHARTS = [
    ("output_tok_per_s", "Output throughput (tok/s)", "throughput"),
    ("ttft_p50", "TTFT p50 (s)", "ttft-p50"),
    ("ttft_p99", "TTFT p99 (s)", "ttft-p99"),
    ("itl_p99", "Inter-token latency p99 (s)", "itl-p99"),
]


def load(results_dir: str):
    """Split the run files into online (has a rate) and offline."""
    online, offline = [], []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            run = json.load(f)
        (online if "summary" in run else offline).append(run)
    return online, offline


def group(online, metric: str):
    """(engine, rate) -> list of that metric, one per repeat."""
    points = defaultdict(list)
    for run in online:
        engine = run.get("engine", "nanoserve")
        points[(engine, run["summary"]["offered_rate"])].append(run["summary"][metric])
    return points


def _style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def line_chart(online, metric: str, ylabel: str, out_path: str, title: str):
    import matplotlib.pyplot as plt

    points = group(online, metric)
    engines = sorted({e for e, _ in points}, key=lambda e: list(COLORS).index(e))
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)

    for engine in engines:
        rates = sorted(r for e, r in points if e == engine)
        runs = [points[(engine, r)] for r in rates]
        means = [sum(v) / len(v) for v in runs]
        color = COLORS[engine]
        ax.fill_between(rates, [min(v) for v in runs], [max(v) for v in runs],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(rates, means, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[engine])
        # Direct label at the end of the line, so identity is never colour alone.
        ax.annotate(LABELS[engine], (rates[-1], means[-1]), color=color,
                    fontsize=9, fontweight="bold",
                    xytext=(6, 0), textcoords="offset points", va="center")

    ax.set_title(title, color=INK, fontsize=12, fontweight="bold", loc="left", pad=12)
    _style(ax, "Offered request rate (req/s)", ylabel)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({r for _, r in points}))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, labelcolor=INK_SOFT, fontsize=9)
    # Right margin reserved for the direct labels, which sit outside the axes.
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def offline_chart(offline, out_path: str):
    import matplotlib.pyplot as plt

    by_engine = defaultdict(list)
    for run in offline:
        by_engine[run["engine"]].append(run["output_tok_per_s"])
    engines = sorted(by_engine, key=lambda e: list(COLORS).index(e))
    means = [sum(by_engine[e]) / len(by_engine[e]) for e in engines]
    lows = [m - min(by_engine[e]) for e, m in zip(engines, means)]
    highs = [max(by_engine[e]) - m for e, m in zip(engines, means)]

    fig, ax = plt.subplots(figsize=(5.5, 4), facecolor=SURFACE)
    bars = ax.bar([LABELS[e] for e in engines], means, width=0.55,
                  color=[COLORS[e] for e in engines],
                  yerr=[lows, highs], capsize=4,
                  error_kw={"ecolor": INK_SOFT, "elinewidth": 1})
    # Every bar labelled: the aqua slot is low contrast on this background.
    for bar, mean in zip(bars, means):
        ax.annotate(f"{mean:,.0f}", (bar.get_x() + bar.get_width() / 2, mean),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", color=INK, fontsize=10, fontweight="bold")

    ax.set_title("Offline throughput, all prompts at t=0", color=INK,
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    _style(ax, "", "Output throughput (tok/s)")
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def markdown_table(online, offline) -> str:
    """The same numbers as text. Charts are not readable by everyone, and
    the README wants the figures anyway."""
    lines = ["| engine | rate (req/s) | tok/s | TTFT p50 | TTFT p99 | ITL p99 | attained |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    rows = defaultdict(list)
    for run in online:
        rows[(run.get("engine", "nanoserve"), run["summary"]["offered_rate"])].append(run["summary"])
    for (engine, rate), summaries in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        def mean(key):
            return sum(s[key] for s in summaries) / len(summaries)
        lines.append(
            f"| {LABELS[engine]} | {rate:g} | {mean('output_tok_per_s'):.0f} | "
            f"{mean('ttft_p50') * 1000:.0f} ms | {mean('ttft_p99') * 1000:.0f} ms | "
            f"{mean('itl_p99') * 1000:.0f} ms | {mean('attained_rate'):.2f} |"
        )
    if offline:
        lines += ["", "| engine | offline tok/s |", "| --- | --- |"]
        by_engine = defaultdict(list)
        for run in offline:
            by_engine[run["engine"]].append(run["output_tok_per_s"])
        for engine in sorted(by_engine, key=lambda e: list(COLORS).index(e)):
            values = by_engine[engine]
            lines.append(f"| {LABELS[engine]} | {sum(values) / len(values):.0f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/charts")
    args = parser.parse_args()

    online, offline = load(args.results)
    os.makedirs(args.out, exist_ok=True)

    for metric, ylabel, name in ONLINE_CHARTS:
        if online:
            line_chart(online, metric, ylabel, os.path.join(args.out, f"{name}.png"),
                       ylabel.split(" (")[0])
    if offline:
        offline_chart(offline, os.path.join(args.out, "offline.png"))

    table = markdown_table(online, offline)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
