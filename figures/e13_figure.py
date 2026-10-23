"""Plot the E13 regret curves from ``analysis/e13_summary.csv``."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "analysis" / "e13_summary.csv"
FIGURE_DIR = ROOT / "figures"
WEEKS = (1, 2, 4, 8, 13, 26, 52, 104)
POLICY_TITLES = {
    "B_less": "B less selective",
    "B_more": "B more selective",
    "B_match": "B tests at random (same rate)",
    "B_dense": "B tests at random (twice as often)",
}
METHODS = ("keep", "dial_only", "head_U", "head_U_dial", "finetune")
COLORS = {
    "keep": "#666666",
    "dial_only": "#0072B2",
    "head_U": "#D55E00",
    "head_U_dial": "#CC79A7",
    "finetune": "#009E73",
}
LABELS = {
    "keep": "keep (A model)",
    "dial_only": "dial only",
    "head_U": "head_U",
    "head_U_dial": "head_U + dial",
    "finetune": "fine-tune all parameters",
}
MARKERS = {"keep": "o", "dial_only": "s", "head_U": "^", "head_U_dial": "D", "finetune": "P"}


def _read_summary(path: Path) -> Dict[Tuple[str, str, int], Mapping[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        return {
            (row["policy"], row["method"], int(row["week"])): row
            for row in rows
        }


def make_figure(summary_path: Path = SUMMARY_PATH) -> Tuple[Path, Path]:
    import matplotlib.pyplot as plt

    data = _read_summary(summary_path)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.7), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    label_handles = {}

    for ax, policy_name in zip(axes_flat, POLICY_TITLES):
        policy_rows = {
            (method, week): data[(policy_name, method, week)]
            for method in (*METHODS, "oracle")
            for week in WEEKS
            if (policy_name, method, week) in data
        }
        case_counts = []
        for week in WEEKS:
            row = policy_rows.get(("dial_only", week))
            case_counts.append(float(row["n_cases_mean"]) if row else float("nan"))

        x = list(WEEKS)
        log_midpoints = [
            (WEEKS[index] * WEEKS[index + 1]) ** 0.5
            for index in range(len(WEEKS) - 1)
        ]
        left_edge = WEEKS[0] / (WEEKS[1] / WEEKS[0]) ** 0.5
        right_edge = WEEKS[-1] * (WEEKS[-1] / WEEKS[-2]) ** 0.5
        bounds = [left_edge, *log_midpoints, right_edge]
        for index, week in enumerate(WEEKS):
            available = [
                method for method in METHODS
                if (method, week) in policy_rows
                and policy_rows[(method, week)]["regret_mean"] not in ("", "nan")
            ]
            if available:
                best = min(
                    available,
                    key=lambda method: float(policy_rows[(method, week)]["regret_mean"]),
                )
                ax.axvspan(
                    bounds[index],
                    bounds[index + 1],
                    color=COLORS[best],
                    alpha=0.085,
                    linewidth=0,
                    zorder=0,
                )

        for method in METHODS:
            method_rows = [policy_rows.get((method, week)) for week in WEEKS]
            if not all(method_rows):
                continue
            means = [float(row["regret_mean"]) for row in method_rows]
            lows = [float(row["regret_min"]) for row in method_rows]
            highs = [float(row["regret_max"]) for row in method_rows]
            linestyle = "--" if method == "keep" else "-"
            line, = ax.plot(
                x,
                means,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4.5,
                linewidth=1.8,
                linestyle=linestyle,
                label=LABELS[method],
                zorder=3,
            )
            ax.fill_between(
                x,
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
        ax.set_xticks(WEEKS)
        ax.set_xticklabels(tick_labels)
        ax.set_title(POLICY_TITLES[policy_name], fontsize=10, pad=8)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, zorder=1)
        ax.set_axisbelow(True)

    axes_flat[0].set_ylabel("Regret (Brier minus B optimum)")
    axes_flat[2].set_ylabel("Regret (Brier minus B optimum)")
    for ax in (axes_flat[2], axes_flat[3]):
        ax.set_xlabel("Weeks since deployment\nMean labelled outcome cases")
    for ax in (axes_flat[0], axes_flat[1]):
        ax.set_xlabel("Weeks since deployment\nMean labelled outcome cases")

    handles = [label_handles[method] for method in METHODS if method in label_handles]
    fig.legend(
        handles,
        [LABELS[method] for method in METHODS if method in label_handles],
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1), h_pad=1.5, w_pad=1.4)
    png_path = FIGURE_DIR / "e13_rungs_over_time.png"
    pdf_path = FIGURE_DIR / "e13_rungs_over_time.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main(argv: Iterable[str] | None = None) -> None:
    del argv
    png_path, pdf_path = make_figure()
    print(f"WROTE {png_path}")
    print(f"WROTE {pdf_path}")


if __name__ == "__main__":
    main()
