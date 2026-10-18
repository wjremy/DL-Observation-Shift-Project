"""Explainer figure: error at hospital A against error at hospital B (E1 results).

Default: the perfect model of A (star, exact Bayes rule under A) with a dashed line
at its error at B. The gap down to the model that ignores who is tested (square,
values-only Bayes rule, optimal at B) is the price of learning A: the same for
every model. A model above the dashed line loses more at B than a perfect model
of A would; below it, less.
With --curve, also the grey curve that blends the two exact rules in logit space,
logit p_w = logit p_V + w (logit p_A - logit p_V), w from 0 to 1: the trade-off
from discounting the testing pattern.
Usage: python figures/tradeoff_figure.py [config] [--curve]
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
CONFIG = ARGS[0] if ARGS else "sigma1_gamma2"
SHOW_CURVE = "--curve" in sys.argv
SEEDS = (0, 1, 2)
ARCHS = {"gru": "GRU", "grud": "GRU-D", "cde": "Neural CDE", "transformer": "Transformer"}
COLORS = {"gru": "#1f77b4", "grud": "#2ca02c", "cde": "#d62728", "transformer": "#9467bd"}
PANELS = {"B_match": "B tests everyone at A's average rate",
          "B_dense": "B tests everyone twice as often"}
W = np.linspace(0.0, 1.0, 41)


def brier(y, p):
    return float(np.mean((y - p) ** 2))


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def curve(b, pol):
    la, lv = logit(b["A_rule_" + pol]), logit(b["V_rule_" + pol])
    return np.array([brier(b["y"], 1 / (1 + np.exp(-(lv + w * (la - lv))))) for w in W])


fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, (pol, title) in zip(axes, PANELS.items()):
    ca = np.zeros_like(W)
    cb = np.zeros_like(W)
    pts = {a: np.zeros(2) for a in ARCHS}
    for s in SEEDS:
        b = dict(np.load(ROOT / f"results/E1/{CONFIG}/seed{s}/bayes.npz"))
        ca += curve(b, "A") / len(SEEDS)
        cb += curve(b, pol) / len(SEEDS)
        for a in ARCHS:
            p = np.load(ROOT / f"results/E1/{CONFIG}/{a}/seed{s}/preds.npz")
            pts[a] += np.array([brier(p["y"], p["A"]), brier(p["y"], p[pol])]) / len(SEEDS)
    star, square = (ca[-1], cb[-1]), (ca[0], cb[0])
    if SHOW_CURVE:
        ax.plot(ca, cb, color="0.75", lw=2, zorder=1,
                label="perfect A model that discounts\nthe testing pattern a bit")
    ax.axhline(star[1], color="black", ls="--", lw=1, zorder=1)
    ax.annotate("", xy=(star[0], star[1]), xytext=(star[0], square[1]),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.5))
    ax.text(star[0] + 0.0003, (star[1] + square[1]) / 2,
            "price of learning A\n(same for every model)", fontsize=9, color="0.3", va="center")
    ax.text(square[0], star[1] + 0.0003,
            "above dashed line: loses more at B\nthan a perfect model of A would",
            fontsize=8.5, color="0.3", ha="right", va="bottom")
    ax.scatter(*star, marker="*", s=280, color="black", zorder=3, label="perfect model of A")
    ax.scatter(*square, marker="s", s=70, color="0.45", zorder=3,
               label="model that ignores who is tested")
    for a, name in ARCHS.items():
        ax.scatter(*pts[a], s=90, color=COLORS[a], edgecolor="black", zorder=4, label=name)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("error at hospital A  (Brier, lower = better)")
    ax.set_ylabel("error at hospital B  (Brier, lower = better)")
    ax.margins(y=0.12)
    ax.grid(alpha=0.25)
axes[1].legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.suptitle("Trained at hospital A (sick patients tested more), "
             "evaluated at hospital B (testing unrelated to sickness)", fontsize=11)
fig.tight_layout()
out = ROOT / f"figures/tradeoff_{CONFIG}{'_curve' if SHOW_CURVE else ''}.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(out)
