"""Render docs/before_after.png from results/benchmark.json.

Three panels compare the unstructured baseline against tokenbeat on the real
loopback streams. The fourth panel is the sensitivity study on the calibrated
simulator and includes the point where the advantage inverts.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BEFORE = "#eb6834"
AFTER = "#2a78d6"

LABELS = ("Threshold baseline", "tokenbeat")


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=1.0)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def bar_panel(ax, title, note, values, fmt, higher_is_better=False):
    bars = ax.bar([0, 1], values, width=0.46, color=[BEFORE, AFTER], linewidth=0, zorder=3)
    style(ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(LABELS, color=INK_2, fontsize=9.5)
    ax.set_xlim(-0.55, 1.55)
    top = max(values) if max(values) > 0 else 1.0
    ax.set_ylim(0, top * 1.28)
    for rect, value in zip(bars, values):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            value + top * 0.04,
            fmt % value,
            ha="center",
            va="bottom",
            color=INK,
            fontsize=11,
            fontweight="bold",
        )
    direction = "higher is better" if higher_is_better else "lower is better"
    ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", pad=14, loc="left")
    ax.text(
        0.0, 1.015, "%s, %s" % (note, direction), transform=ax.transAxes,
        color=MUTED, fontsize=8.5, ha="left", va="bottom",
    )


def line_panel(ax, rates, base, beat):
    style(ax)
    x = np.arange(len(rates))
    ax.plot(x, base, color=BEFORE, linewidth=2.0, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=3, label=LABELS[0])
    ax.plot(x, beat, color=AFTER, linewidth=2.0, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2.0, zorder=4, label=LABELS[1])
    ax.set_xticks(x)
    ax.set_xticklabels(["%.0f%%" % (100 * r) for r in rates], color=INK_2, fontsize=9.5)
    ax.set_xlim(-0.3, len(rates) - 0.5)
    ax.set_ylim(0, max(max(base), max(beat)) * 1.32)
    ax.set_xlabel("frames hit by a queueing delay", color=INK_2, fontsize=9.5, labelpad=6)
    for series in (base, beat):
        ax.annotate(
            "%.3f" % series[-1], xy=(x[-1], series[-1]), xytext=(8, 0),
            textcoords="offset points", color=INK, fontsize=10, va="center",
            fontweight="bold",
        )
    ax.set_title(
        "Sensitivity: where the advantage holds", color=INK, fontsize=11.5,
        fontweight="bold", pad=14, loc="left",
    )
    ax.text(
        0.0, 1.015,
        "acceptance-rate error on the calibrated simulator, lower is better",
        transform=ax.transAxes, color=MUTED, fontsize=8.5, ha="left", va="bottom",
    )
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9.5, handlelength=1.6)
    for text in legend.get_texts():
        text.set_color(INK_2)


def main() -> None:
    with open(os.path.join(ROOT, "results", "benchmark.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    loop = data["loopback"]
    base, beat = loop["methods"]["baseline"], loop["methods"]["tokenbeat"]
    rates = sorted(float(k) for k in data["sweep"])
    sweep_base = [data["sweep"]["%.2f" % r]["methods"]["baseline"]["alpha_mae"] for r in rates]
    sweep_beat = [data["sweep"]["%.2f" % r]["methods"]["tokenbeat"]["alpha_mae"] for r in rates]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    fig.patch.set_facecolor(SURFACE)

    bar_panel(
        axes[0][0], "Acceptance rate alpha",
        "mean absolute error over %d streams" % loop["n_spec"],
        [base["alpha_mae"], beat["alpha_mae"]], "%.3f",
    )
    bar_panel(
        axes[0][1], "Draft length gamma",
        "exact recovery over %d streams" % loop["n_spec"],
        [base["gamma_exact_pct"], beat["gamma_exact_pct"]], "%.0f%%",
        higher_is_better=True,
    )
    bar_panel(
        axes[1][0], "Recovered burst law",
        "total variation from the true law",
        [base["tv_mean"], beat["tv_mean"]], "%.3f",
    )
    line_panel(axes[1][1], rates, sweep_base, sweep_beat)

    fig.suptitle(
        "Reading a serving stack's decode parameters off token arrival times",
        color=INK, fontsize=15, fontweight="bold", x=0.055, ha="left", y=0.972,
    )
    lines = [
        "%d real loopback SSE streams. Both methods fit the same censored gap mixture."
        % (loop["n_spec"] + loop["n_nonspec"]),
        "The baseline then cuts bursts at a hard threshold; tokenbeat imposes the"
        " speculative run-length law and picks the draft length by BIC.",
        "On streams with no speculation the baseline reports speculation %.0f%% of the"
        " time, tokenbeat %.0f%%." % (base["detect_fpr_pct"], beat["detect_fpr_pct"]),
    ]
    for i, line in enumerate(lines):
        fig.text(0.055, 0.932 - 0.022 * i, line, color=INK_2, fontsize=9.5, ha="left")

    fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.862))
    out = os.path.join(ROOT, "docs", "before_after.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
