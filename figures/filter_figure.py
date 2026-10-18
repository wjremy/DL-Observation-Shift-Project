"""Plot E5's exact Bayes and learned latent-filter trade-offs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PANELS = {
    "B_match": "B tests everyone at A's average rate",
    "B_dense": "B tests everyone twice as often",
}


def main() -> None:
    data = pd.read_csv(ROOT / "analysis" / "e5_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (policy, title) in zip(axes, PANELS.items()):
        subset = data[data["test_policy"] == policy]
        exact = subset[subset["family"] == "exact_blend"].sort_values("w")
        learned = subset[subset["family"] == "filter"].sort_values("w")
        plain = subset[subset["family"] == "plain_gru"].iloc[0]

        ax.plot(exact["brier_A"], exact["brier_B"], color="0.75", lw=2,
                label="Exact Bayes blend", zorder=1)
        exact_zero = exact.loc[exact["w"] == 0.0].iloc[0]
        exact_one = exact.loc[exact["w"] == 1.0].iloc[0]
        ax.scatter(exact_one["brier_A"], exact_one["brier_B"], marker="*", s=260,
                   color="black", zorder=3, label="Exact A rule (w=1)")
        ax.scatter(exact_zero["brier_A"], exact_zero["brier_B"], marker="s", s=65,
                   color="0.45", zorder=3, label="Values-only rule (w=0)")

        ax.plot(learned["brier_A"], learned["brier_B"], color="#d62728", lw=1.8,
                marker="o", markersize=4.5, zorder=2, label="Latent filter (w varies)")
        for weight in (0.0, 0.5, 1.0):
            point = learned.loc[learned["w"] == weight].iloc[0]
            ax.annotate(f"w={weight:g}", (point["brier_A"], point["brier_B"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8,
                        color="#9f1d20")
        ax.scatter(plain["brier_A"], plain["brier_B"], marker="D", s=75,
                   color="#1f77b4", edgecolor="black", zorder=4, label="Plain E1 GRU")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("error at hospital A  (Brier, lower = better)")
        ax.set_ylabel("error at hospital B  (Brier, lower = better)")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Discounting the testing-pattern signal", fontsize=12)
    fig.tight_layout()
    output = ROOT / "figures" / "filter_sigma1_gamma2.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
