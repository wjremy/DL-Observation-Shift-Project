"""Render the E6 two-hospital dial and silent run-in figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
OUT = ROOT / "figures" / "e6_two_hospitals.png"


def main() -> None:
    curve = pd.read_csv(ANALYSIS / "e6_wcurve.csv")
    curve = curve[curve["model"] == "combined_w"].sort_values("w")
    runin = pd.read_csv(ANALYSIS / "e6_runin.csv")

    fig = plt.figure(figsize=(15, 6.4), facecolor="#fbfbf8")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.18, 1.0), height_ratios=(1, 1),
                            left=0.07, right=0.98, bottom=0.13, top=0.84,
                            wspace=0.30, hspace=0.43)
    ax_auc = fig.add_subplot(grid[0, 0])
    ax_loss = fig.add_subplot(grid[1, 0])
    ax_run = fig.add_subplot(grid[:, 1])
    axes = (ax_auc, ax_loss, ax_run)
    for ax in axes:
        ax.set_facecolor("#fbfbf8")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#767676")
        ax.spines["bottom"].set_color("#767676")
        ax.tick_params(colors="#30343b", labelsize=9)
        ax.grid(axis="y", color="#d9dcdf", linewidth=0.75, alpha=0.8)

    # Panel 1: the same source model's testing-pattern coefficient is dialed at deployment.
    color_a, color_b = "#275d8c", "#b04c36"
    xw = curve["w"].to_numpy()
    ax_auc.plot(xw, curve["auroc_A_test"], color=color_a, linestyle="--", linewidth=2,
                label="A test")
    ax_auc.plot(xw, curve["auroc_B"], color=color_b, linewidth=2.2, label="B test")
    ax_auc.fill_between(xw, curve["auroc_B_low"], curve["auroc_B_high"], color=color_b, alpha=0.16,
                        linewidth=0)
    ax_auc.set_title("AUROC", loc="left", fontsize=11, color="#20242a", weight="bold")
    ax_auc.set_ylabel("Area under ROC curve", fontsize=9, color="#30343b")
    ax_auc.set_xlim(0, 1.5)
    ax_auc.legend(frameon=False, ncol=2, fontsize=8, loc="lower left")

    ax_loss.plot(xw, curve["log_loss_A_test"], color=color_a, linestyle="--", linewidth=2,
                 label="A test")
    ax_loss.plot(xw, curve["log_loss_B"], color=color_b, linewidth=2.2, label="B test")
    ax_loss.fill_between(xw, curve["log_loss_B_low"], curve["log_loss_B_high"], color=color_b, alpha=0.16,
                         linewidth=0)
    ax_loss.set_title("Log loss", loc="left", fontsize=11, color="#20242a", weight="bold")
    ax_loss.set_ylabel("Lower is better", fontsize=9, color="#30343b")
    ax_loss.set_xlabel("Trust in the testing pattern (w)", fontsize=9, color="#30343b")
    ax_loss.set_xlim(0, 1.5)

    # Panel 2: each point summarizes 50 random run-in samples; bands show the 10th–90th percentiles.
    palette = {
        "w_tuned": "#386f91",
        "intercept_slope": "#9a6a2f",
        "w_plus_intercept": "#3c8069",
        "full_refit": "#795a8d",
        "w0_fixed": "#777777",
        "w1_fixed": "#b04c36",
    }
    labels = {
        "w_tuned": "Choose w",
        "intercept_slope": "Intercept + slope",
        "w_plus_intercept": "Choose w + intercept",
        "full_refit": "Refit all coefficients",
        "w0_fixed": "Fixed w = 0",
        "w1_fixed": "Fixed w = 1",
    }
    for method, group in runin.groupby("method", sort=False):
        group = group.sort_values("n_run")
        x = group["n_run"].to_numpy(dtype=float)
        mean = group["auroc_mean"].to_numpy()
        low = group["auroc_p10"].to_numpy()
        high = group["auroc_p90"].to_numpy()
        ax_run.plot(x, mean, marker="o", markersize=3.5, linewidth=1.7,
                    color=palette.get(method, "#333333"), label=labels.get(method, method))
        ax_run.fill_between(x, low, high, color=palette.get(method, "#333333"), alpha=0.11,
                            linewidth=0)
    ax_run.set_xscale("log")
    ax_run.set_ylim(0.55, 0.73)
    ax_run.set_title("2. B-test AUROC after a silent run-in of n patients", loc="left", fontsize=12,
                     color="#20242a", weight="bold")
    ax_run.set_xlabel("Labeled B patients used for adaptation (log scale)", fontsize=9, color="#30343b")
    ax_run.set_ylabel("B-test AUROC (higher is better)", fontsize=9, color="#30343b")
    ax_run.legend(frameon=False, fontsize=8, loc="best", ncol=1)

    fig.suptitle("Can a small B run-in tune trust in the testing pattern?", x=0.07, ha="left",
                 fontsize=15, weight="bold", color="#20242a")
    fig.text(0.07, 0.885,
             "Lines show means; shaded areas show patient-bootstrap 95% intervals for the B dial "
             "and 10–90% ranges across 50 run-in samples.",
             fontsize=9, color="#50565e")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
