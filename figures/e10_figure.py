"""E10 scatter of physiology and pattern AUROC transfer gaps."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GAPS = ROOT / "analysis" / "e10_gaps.csv"
OUTPUT = ROOT / "figures" / "e10_gaps.png"
COLORS = {
    "P19 same unit, other hospital": "#0072B2",
    "P19 other unit, other hospital": "#D55E00",
    "P19 other unit, same hospital": "#009E73",
    "P19 whole hospital": "#CC79A7",
    "P12": "#E69F00",
}


def main() -> None:
    gaps = pd.read_csv(GAPS)
    physiology = gaps[gaps["feature_set"] == "physiology"][[
        "dataset", "source", "target", "pair", "pair_group", "auroc_gap",
    ]].rename(columns={"auroc_gap": "physiology_gap"})
    pattern = gaps[gaps["feature_set"] == "pattern"][[
        "dataset", "source", "target", "auroc_gap",
    ]].rename(columns={"auroc_gap": "pattern_gap"})
    data = physiology.merge(pattern, on=["dataset", "source", "target"], validate="one_to_one")

    fig, ax = plt.subplots(figsize=(9.0, 7.2), constrained_layout=True)
    for group, part in data.groupby("pair_group", sort=False):
        ax.scatter(
            part["physiology_gap"], part["pattern_gap"],
            label=group, color=COLORS[group], s=52, alpha=0.88,
            edgecolor="white", linewidth=0.6, zorder=3,
        )
    bound = max(0.05, float(data[["physiology_gap", "pattern_gap"]].abs().to_numpy().max()) * 1.08)
    ax.plot([-bound, bound], [-bound, bound], color="#555555", linestyle="--", linewidth=1.0,
            label="equal gaps", zorder=1)
    ax.axhline(0, color="#888888", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#888888", linewidth=0.8, zorder=0)
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Physiology AUROC gap (local − transferred)")
    ax.set_ylabel("Pattern AUROC gap (local − transferred)")
    ax.set_title("E10: which part shifts between sites?")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.6)
    ax.legend(frameon=False, loc="best", fontsize=8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
