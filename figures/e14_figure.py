"""Plot E14's outcome-meaning shift regret curves from the summary CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Mapping, Tuple


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "analysis" / "e14_summary.csv"
FIGURE_DIR = ROOT / "figures"
WEEKS = (4, 13, 52, 104)
DIRECTION_TITLES = {
    "gain": "B's doctors also test on a hidden risk factor (tests mean more)",
    "lose": "A's doctors tested on it, B's do not (tests mean less)",
}
METHODS = (
    "keep",
    "dial_only",
    "head_U",
    "head_L",
    "finetune_lr05",
    "finetune_lr005",
)
COLORS = {
    "keep": "#666666",
    "dial_only": "#0072B2",
    "head_U": "#D55E00",
    "head_L": "#CC79A7",
    "finetune_lr05": "#009E73",
    "finetune_lr005": "#56B4E9",
}
MARKERS = {
    "keep": "o",
    "dial_only": "s",
    "head_U": "^",
    "head_L": "D",
    "finetune_lr05": "P",
    "finetune_lr005": "X",
}


def _read_summary(path: Path) -> Dict[Tuple[str, str, int], Mapping[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (row["direction"], row["method"], int(row["week"])): row
            for row in csv.DictReader(handle)
        }


def make_figure(summary_path: Path = SUMMARY_PATH) -> Tuple[Path, Path]:
    import matplotlib.pyplot as plt

    data = _read_summary(summary_path)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.7), sharex=True, sharey=True)
    label_handles = {}

    for ax, direction in zip(axes, DIRECTION_TITLES):
        case_counts = []
        for week in WEEKS:
            row = data.get((direction, "dial_only", week))
            case_counts.append(float(row["n_cases_mean"]) if row else float("nan"))

        for method in METHODS:
            method_rows = [data.get((direction, method, week)) for week in WEEKS]
            if not all(method_rows):
                continue
            means = [float(row["regret_mean"]) for row in method_rows]
            lows = [float(row["regret_min"]) for row in method_rows]
            highs = [float(row["regret_max"]) for row in method_rows]
            # Regret is theoretically nonnegative against the exact Bayes rule;
            # clip only at the plotting floor for the requested log y axis.
            floor = 1e-8
            means = [max(value, floor) for value in means]
            lows = [max(value, floor) for value in lows]
            highs = [max(value, floor) for value in highs]
            linestyle = "--" if method == "keep" else "-"
            line, = ax.plot(
                WEEKS,
                means,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=5,
                linewidth=1.8,
                linestyle=linestyle,
                label=method,
                zorder=3,
            )
            ax.fill_between(
                WEEKS,
                lows,
                highs,
                color=COLORS[method],
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )
            label_handles[method] = line

        tick_labels = [
            f"{week}\n{count:.1f}" if count == count else str(week)
            for week, count in zip(WEEKS, case_counts)
        ]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(WEEKS)
        ax.set_xticklabels(tick_labels)
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_title(DIRECTION_TITLES[direction], fontsize=10, pad=9)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=1)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Regret (Brier minus oracle Brier)")
    for ax in axes:
        ax.set_xlabel("Weeks since deployment\nMean labelled outcome cases")

    handles = [label_handles[method] for method in METHODS if method in label_handles]
    fig.legend(
        handles,
        [method for method in METHODS if method in label_handles],
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=1.5)
    png_path = FIGURE_DIR / "e14_meaning_shift.png"
    pdf_path = FIGURE_DIR / "e14_meaning_shift.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    png_path, pdf_path = make_figure()
    print(f"WROTE {png_path}")
    print(f"WROTE {pdf_path}")


if __name__ == "__main__":
    main()
