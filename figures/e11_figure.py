"""Plot E11 AUROC gains over keep for all P12 source-target pairs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "analysis" / "e11_summary.csv"
OUTPUT = ROOT / "figures" / "e11_runin.png"
SITES = ("CCU", "CSRU", "MICU", "SICU")
SUBGROUP_PAIRS = {
    ("CCU", "SICU"),
    ("CSRU", "CCU"),
    ("CSRU", "SICU"),
    ("SICU", "CSRU"),
}
LINES = (
    ("end_recal", "End recal", "#6b7280", "--"),
    ("dial_recal", "Dial recal", "#2563eb", "-"),
    ("dial_recal_shrink", "Dial recal shrink", "#dc2626", "-"),
    ("dial_auc_shrink", "Dial AUROC shrink", "#059669", "-"),
)


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    for ax, (source, target) in zip(
            axes_flat, ((s, t) for s in SITES for t in SITES if s != t)):
        pair = summary[(summary.source == source) & (summary.target == target)]
        subgroup_mark = " *" if (source, target) in SUBGROUP_PAIRS else ""
        for method, label, color, linestyle in LINES:
            rows = pair[pair.method == method].sort_values("runin_months")
            x = rows["runin_months"].to_numpy()
            mean = rows["delta_to_keep_auroc_mean"].to_numpy()
            ax.plot(x, mean, label=label, color=color, linestyle=linestyle, linewidth=1.8)
            if method == "dial_recal_shrink":
                ax.fill_between(
                    x,
                    rows["delta_to_keep_auroc_p10"].to_numpy(),
                    rows["delta_to_keep_auroc_p90"].to_numpy(),
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                    label="Dial recal shrink 10–90%",
                )
        ax.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.7)
        ax.set_xscale("log")
        ax.set_title(f"{source} → {target}{subgroup_mark}", fontsize=10)
        ax.grid(True, which="both", alpha=0.2)
    for ax in axes_flat[8:]:
        ax.set_xlabel("Run-in duration (months)")
    for ax in axes_flat[::4]:
        ax.set_ylabel("AUROC difference to keep")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("E11: safe tuning on 2012 mortality pairs", y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
