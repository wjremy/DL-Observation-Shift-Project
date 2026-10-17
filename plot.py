"""Create the three study figures from analyze.py CSV outputs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARCH_COLORS = {
    "gru": "#1f77b4", "grud": "#ff7f0e", "cde": "#2ca02c",
    "transformer": "#9467bd",
}
POLICY_MARKERS = {"B_match": "o", "B_dense": "s"}


def make_plots(analysis_dir="analysis", out_dir=None):
    source = Path(analysis_dir)
    output = Path(out_dir) if out_dir else source
    output.mkdir(parents=True, exist_ok=True)
    e1 = pd.read_csv(source / "e1_summary.csv")
    e2 = pd.read_csv(source / "e2_summary.csv")

    targets = e1[e1["test_policy"].isin(["B_match", "B_dense"])].copy()
    sigmas = sorted(targets["sigma"].dropna().unique().tolist())
    if sigmas:
        rows = int(np.ceil(len(sigmas) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(13, 5.3 * rows), squeeze=False)
        for axis, sigma in zip(axes.flat, sigmas):
            panel = targets[np.isclose(targets["sigma"], sigma)]
            for _, item in panel.iterrows():
                arch = str(item["arch"])
                policy = str(item["test_policy"])
                marker = POLICY_MARKERS.get(policy, "o")
                color = ARCH_COLORS.get(arch, None)
                axis.scatter(item["X_B"], item["S"], marker=marker, color=color,
                             s=58, label=f"{arch} / {policy}")
                axis.annotate(
                    f"J={item['J_B']:.3g}; γ={item['gamma']:g}",
                    (item["X_B"], item["S"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7,
                )
            axis.scatter([0], [0], marker="*", color="black", s=140,
                         label="A_rule (0, 0)")
            axis.axhline(0, color="#999999", linewidth=0.7)
            axis.axvline(0, color="#999999", linewidth=0.7)
            axis.set_title(f"σ = {sigma:g}")
            axis.set_xlabel("X_B: model minus transferred A_rule Brier")
            axis.set_ylabel("S: model minus A_rule Brier on A")
            axis.grid(alpha=0.2)
            handles, labels = axis.get_legend_handles_labels()
            dedup = dict(zip(labels, handles))
            axis.legend(dedup.values(), dedup.keys(), fontsize=7, loc="best")
        for axis in axes.flat[len(sigmas):]:
            axis.set_visible(False)
        fig.tight_layout()
        fig.savefig(output / "e1_transfer_cost.png", dpi=180)
        plt.close(fig)

    source_rows = e1[e1["test_policy"] == "A"].copy()
    if not source_rows.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for arch in sorted(source_rows["arch"].dropna().unique()):
            subset = source_rows[source_rows["arch"] == arch]
            grouped = subset.groupby("sigma", as_index=False)[["b_mask", "b_values"]].mean()
            color = ARCH_COLORS.get(str(arch), None)
            axes[0].plot(grouped["sigma"], grouped["b_mask"], marker="o",
                         color=color, label=str(arch))
            axes[1].plot(grouped["sigma"], grouped["b_values"], marker="o",
                         color=color, label=str(arch))
        axes[0].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        axes[0].set_title("Mask coefficient")
        axes[1].set_title("Values coefficient")
        for axis in axes:
            axis.set_xlabel("Observation noise σ")
            axis.grid(alpha=0.2)
            axis.legend()
        fig.tight_layout()
        fig.savefig(output / "e1_reliance_probe.png", dpi=180)
        plt.close(fig)

    if not e2.empty:
        grouped = e2.groupby(["arch", "test_policy"], as_index=False)["excess_brier"].mean()
        archs = sorted(grouped["arch"].unique())
        policies = sorted(grouped["test_policy"].unique())
        x = np.arange(len(archs), dtype=np.float64)
        width = 0.8 / max(1, len(policies))
        fig, axis = plt.subplots(figsize=(9, 5))
        for index, policy in enumerate(policies):
            values = [
                float(grouped[(grouped["arch"] == arch) &
                              (grouped["test_policy"] == policy)]["excess_brier"].mean())
                for arch in archs
            ]
            axis.bar(x + (index - (len(policies) - 1) / 2) * width,
                     values, width, label=policy)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, archs)
        axis.set_ylabel("Brier excess over values-only Bayes")
        axis.set_title("E2: sampling coverage shift")
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / "e2_excess.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", default="analysis", help="directory with summary CSVs")
    parser.add_argument("--out", default=None, help="figure output directory")
    args = parser.parse_args()
    make_plots(args.analysis, args.out)


if __name__ == "__main__":
    main()

