"""Render E7 target log-loss trust-dial curves (run by the figure renderer)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "analysis" / "e7_pairs.csv"
CURVES = ROOT / "analysis" / "e7_curves.csv"
OUTPUT = ROOT / "figures" / "e7_pairs.png"


def main() -> None:
    pairs = pd.read_csv(PAIRS)
    curves = pd.read_csv(CURVES)
    fig, axes = plt.subplots(6, 4, figsize=(17, 19), sharex=True)
    for dataset_ix, dataset in enumerate(("P19", "P12")):
        subset = pairs[pairs["dataset"] == dataset]
        for pair_ix, (_, pair) in enumerate(subset.iterrows()):
            row = dataset_ix * 3 + pair_ix // 4
            col = pair_ix % 4
            ax = axes[row, col]
            c = curves[(curves["dataset"] == dataset)
                       & (curves["source"] == pair["source"])
                       & (curves["target"] == pair["target"])].sort_values("w")
            w1_ll = float(c.loc[np.isclose(c["w"], 1.0), "log_loss"].iloc[0])
            y = c["log_loss"].to_numpy(dtype=float) - w1_ll
            ax.plot(c["w"], y, color="#2463a6", linewidth=1.5)
            best_ix = int(np.argmin(c["log_loss"].to_numpy(dtype=float)))
            ax.scatter([c["w"].iloc[best_ix]], [y[best_ix]], color="#c43b3b", s=22, zorder=3)
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.set_xlim(0, 1.5)
            ax.set_title(f"{pair['source']} -> {pair['target']}", fontsize=9)
            if row in (2, 5):
                ax.set_xlabel("Trust weight w")
            if col == 0:
                ax.set_ylabel(f"{dataset}\nTarget log loss minus w=1")
    fig.suptitle("E7 trust dial across directed site pairs", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
