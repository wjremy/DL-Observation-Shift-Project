"""Plot E15 target AUROC changes by source-target pair and run-in size."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("keep", "drop", "dial_only", "head_U", "head_U_pool", "head_L", "finetune")
METHOD_LABELS = {
    "keep": "keep", "drop": "drop", "dial_only": "dial only", "head_U": "head U",
    "head_U_pool": "head U pool", "head_L": "head L", "finetune": "fine-tune",
}


def make_figure(*, smoke: bool = False, input_path: str | Path | None = None,
                output_stem: str | Path | None = None) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt

    csv_path = Path(input_path) if input_path is not None else (
        ROOT / "analysis" / ("e15_smoke_summary.csv" if smoke else "e15_summary.csv")
    )
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"E15 summary CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        frame = list(reader)
    required = {
        "dataset", "source", "target", "pair_type", "pair_label", "n_run", "method",
        "auroc_diff_to_keep_mean", "auroc_diff_to_keep_min", "auroc_diff_to_keep_max",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"E15 summary is missing columns: {missing}")
    pair_order = []
    seen_pairs = set()
    for row in frame:
        key = tuple(row[name] for name in ("dataset", "source", "target", "pair_type", "pair_label"))
        if key not in seen_pairs:
            pair_order.append(key)
            seen_pairs.add(key)
    # Stable sort keeps the pre-specified order within each pair group.
    pair_order.sort(key=lambda row: 0 if row[3] == "shift" else 1)
    if not pair_order:
        raise ValueError("E15 summary has no pairs to plot")

    run_sizes = sorted({int(row["n_run"]) for row in frame if row.get("n_run")})
    color_map = plt.get_cmap("tab10")
    colors = {size: color_map(index) for index, size in enumerate(run_sizes)}
    height = max(2.3 * len(pair_order), 3.8)
    fig, axes = plt.subplots(len(pair_order), 1, figsize=(11.5, height), sharex=True, sharey=True)
    if len(pair_order) == 1:
        axes = [axes]
    x_positions = list(range(len(METHODS)))

    for ax, (dataset, source, target, pair_type, pair_label) in zip(axes, pair_order):
        pair_frame = [row for row in frame if (
            row["dataset"] == dataset and row["source"] == source and row["target"] == target
        )]
        ax.axhline(0.0, color="#333333", linewidth=0.9, linestyle="--", zorder=0)
        # The keep method itself is the x-axis anchor at zero change.
        ax.axvline(0.0, color="#999999", linewidth=0.6, alpha=0.55, zorder=0)
        for size in run_sizes:
            size_frame = [row for row in pair_frame if int(row["n_run"]) == size]
            for position, method in enumerate(METHODS):
                selected = [row for row in size_frame if row["method"] == method]
                if selected.empty:
                    continue
                row = selected.iloc[0]
                mean = float(row["auroc_diff_to_keep_mean"])
                low = float(row["auroc_diff_to_keep_min"])
                high = float(row["auroc_diff_to_keep_max"])
                if not all(math.isfinite(value) for value in (mean, low, high)):
                    continue
                ax.errorbar(
                    position, mean,
                    yerr=[[max(0.0, mean - low)], [max(0.0, high - mean)]],
                    fmt="o", markersize=5.5, capsize=3, elinewidth=1.2,
                    color=colors[size], label=f"n = {size}" if position == 0 else None,
                    zorder=3,
                )
        kind = "shift" if pair_type == "shift" else "control"
        ax.set_ylabel("AUROC change\nvs keep")
        ax.set_title(f"{pair_label}  ({kind})", loc="left", fontsize=10, pad=5)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[-1].set_xticks(x_positions, [METHOD_LABELS[name] for name in METHODS], rotation=20, ha="right")
    axes[-1].set_xlabel("Method")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Run-in size", loc="upper right", frameon=False,
                   bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("E15: AUROC change on target evaluation patients", x=0.03, ha="left", y=0.999,
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.01, 0.98, 0.985))

    if output_stem is None:
        stem = ROOT / "figures" / ("e15_real_pairs_smoke" if smoke else "e15_real_pairs")
    else:
        stem = Path(output_stem)
        if not stem.is_absolute():
            stem = ROOT / stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="plot e15_smoke_summary.csv")
    parser.add_argument("--input", type=Path, default=None, help="summary CSV override")
    parser.add_argument("--output-stem", type=Path, default=None, help="output path without extension")
    args = parser.parse_args()
    png, pdf = make_figure(smoke=args.smoke, input_path=args.input, output_stem=args.output_stem)
    print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
