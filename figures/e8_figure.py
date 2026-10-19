"""Plot E8 primary-pair AUROC gains over keep by run-in duration."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "analysis" / "e8_runin_summary.csv"
OUTPUT = ROOT / "figures" / "e8_runin.png"
PRIMARY_PAIRS = (
    ("A-MICU", "B-MICU"),
    ("B-MICU", "A-MICU"),
    ("A-SICU", "B-SICU"),
    ("B-SICU", "A-SICU"),
)
MONTHS = {54: 1, 108: 2, 162: 3, 325: 6, 650: 12, 1300: 24}
LINES = (
    ("end", "end", "#4c78a8"),
    ("dial", "dial", "#f58518"),
    ("end_recal", "end_recal", "#54a24b"),
    ("dial_recal", "dial_recal", "#e45756"),
)


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (source, target) in zip(axes.flat, PRIMARY_PAIRS):
        pair = summary[(summary["source"] == source) & (summary["target"] == target)]
        for method, label, color in LINES:
            curve = pair[pair["method"] == method].sort_values("n_run")
            x = np.asarray([MONTHS[int(n)] for n in curve["n_run"]], dtype=float)
            mean = curve["delta_to_keep_auroc_mean"].to_numpy(dtype=float)
            low = curve["delta_to_keep_auroc_p10"].to_numpy(dtype=float)
            high = curve["delta_to_keep_auroc_p90"].to_numpy(dtype=float)
            ax.plot(x, mean, marker="o", linewidth=1.5, label=label, color=color)
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 6, 12, 24], labels=["1", "2", "3", "6", "12", "24"])
        ax.set_title(f"{source} -> {target}")
        ax.grid(axis="y", alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel("Run-in size (months)")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC difference to keep")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("E8: unit-level run-in adaptation")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
