"""E7 follow-up: does the dial's gain grow with the size of the observation shift?

Shift size per pair = curvature c of a quadratic fitted to target log loss over w in [0, 1.5]
(how steeply the target error depends on the trust weight). For a quadratic bowl with minimum
at w*, the best dial beats the better end by c * d^2, where d is the distance from w* to the
nearest of {0, 1}. Writes analysis/e7_shift_size.csv and figures/e7_shift_size.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
curves = pd.read_csv(ROOT / "analysis" / "e7_curves.csv")
pairs = pd.read_csv(ROOT / "analysis" / "e7_pairs.csv")
e6 = pd.read_csv(ROOT / "analysis" / "e6_wcurve.csv")
e6 = e6[e6["model"] == "combined_w"].rename(columns={"log_loss_B": "log_loss"})
e6 = e6.assign(dataset="P19", source="A-all", target="B-all")[["dataset", "source", "target", "w", "log_loss"]]
curves = pd.concat([curves[["dataset", "source", "target", "w", "log_loss"]], e6], ignore_index=True)


def kind(row):
    if row.dataset == "P12":
        return "P12: other ICU type, same hospital (mortality)"
    if row.source.split("-")[0] == row.target.split("-")[0]:
        return "P19: other ICU type, same hospital"
    if row.source.split("-")[1] == row.target.split("-")[1] or row.source.endswith("all"):
        return "P19: same ICU type, other hospital"
    return "P19: other ICU type, other hospital"


rows = []
for (ds, s, t), g in curves.groupby(["dataset", "source", "target"], sort=False):
    g = g.sort_values("w")
    w, ll = g["w"].to_numpy(), g["log_loss"].to_numpy()
    c, b, a = np.polyfit(w, ll, 2)
    l0 = ll[np.isclose(w, 0)][0]
    l1 = ll[np.isclose(w, 1)][0]
    w_best = w[np.argmin(ll)]
    row = {"dataset": ds, "source": s, "target": t, "curvature": c,
           "w_star_quadratic": -b / (2 * c), "best_w": w_best,
           "end_gap": abs(l0 - l1), "gain_full": min(l0, l1) - ll.min(),
           "gain_full_pct": 100 * (min(l0, l1) - ll.min()) / min(l0, l1)}
    match = pairs[(pairs.dataset == ds) & (pairs.source == s) & (pairs.target == t)]
    row["gain_tuned"] = -float(match["dial_minus_end_log_loss_mean"].iloc[0]) if len(match) else np.nan
    rows.append(row)
out = pd.DataFrame(rows)
out["kind"] = out.apply(kind, axis=1)
d = np.minimum(np.abs(out["w_star_quadratic"]), np.abs(out["w_star_quadratic"] - 1))
out["gain_predicted"] = out["curvature"] * np.where((out["w_star_quadratic"] > 0) & (out["w_star_quadratic"] < 1), d, np.maximum(out["w_star_quadratic"] - 1, 0)) ** 2
out.to_csv(ROOT / "analysis" / "e7_shift_size.csv", index=False)

colors = {"P19: same ICU type, other hospital": "#C0392B", "P19: other ICU type, other hospital": "#E67E22",
          "P19: other ICU type, same hospital": "#2E86C1", "P12: other ICU type, same hospital (mortality)": "#7F8C8D"}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for k, g in out.groupby("kind"):
    axes[0].scatter(g["curvature"], g["gain_full"], s=55, color=colors[k], label=k, edgecolor="white")
    axes[1].scatter(g["curvature"], g["gain_tuned"], s=55, color=colors[k], label=k, edgecolor="white")
axes[0].set_title("Best possible dial vs better end (whole target)", loc="left", fontsize=11)
axes[1].set_title("Dial tuned on half the target vs better end (other half)", loc="left", fontsize=11)
for ax in axes:
    ax.set_xscale("log")
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("Size of observation shift: curvature of target error in w (log scale)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Log-loss gain of the dial (higher = dial better)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "e7_shift_size.png", dpi=150)
print(out[["dataset", "source", "target", "kind", "curvature", "best_w", "end_gap", "gain_full", "gain_full_pct", "gain_tuned", "gain_predicted"]].round(5).to_string())
print("Spearman curvature vs gain_full:", round(out["curvature"].rank().corr(out["gain_full"].rank()), 2))
print("Spearman curvature vs gain_tuned:", round(out["curvature"].rank().corr(out["gain_tuned"].rank()), 2))
