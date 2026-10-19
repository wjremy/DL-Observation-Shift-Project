Created and ran [e9_fixed_tuning.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/e9_fixed_tuning.py) with seed **91020**. The E8 definition checks passed: `keep` stayed at w=1, and `dial_recal` reused E8’s fitting and selection helpers.

**Primary pairs, 3-month run-in** — AUROC difference to `keep` [p10, p90]; share of draws with a gain; mean chosen w:

| Pair | `dial_recal` | `dial_recal_shrink` | `dial_auc` | `dial_auc_shrink` |
|---|---|---|---|---|
| A-MICU → B-MICU | −0.0261 [−0.1100, +0.0227]; 44%; 1.06 | −0.0013 [−0.0170, +0.0112]; 39%; 1.02 | −0.0128 [−0.1079, +0.0237]; 57%; 1.58 | −0.0129 [−0.1076, +0.0219]; 55%; 1.16 |
| B-MICU → A-MICU | +0.0019 [−0.0578, +0.0243]; 72%; 2.01 | +0.0011 [−0.0014, +0.0040]; 63%; 1.06 | −0.0012 [−0.0560, +0.0216]; 64%; 1.79 | −0.0040 [−0.0544, +0.0158]; 62%; 1.28 |
| A-SICU → B-SICU | −0.0061 [−0.0148, +0.0021]; 28%; 0.61 | −0.0007 [−0.0022, +0.0005]; 23%; 0.93 | −0.0045 [−0.0148, +0.0027]; 46%; 1.29 | −0.0041 [−0.0146, +0.0022]; 45%; 0.92 |
| B-SICU → A-SICU | −0.0150 [−0.0431, +0.0084]; 30%; 0.74 | −0.0005 [−0.0028, +0.0017]; 23%; 0.96 | −0.0047 [−0.0434, +0.0149]; 55%; 1.68 | −0.0055 [−0.0410, +0.0071]; 50%; 1.15 |

The all-size primary table and secondary-pair results are in [e9_summary.csv](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/analysis/e9_summary.csv) and [e9_summary.json](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/analysis/e9_summary.json); three pair-size combinations were skipped by E8’s evaluation-size rule. The run took **191 seconds (3.2 minutes)**, and the figure script [e9_figure.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/figures/e9_figure.py) passed syntax checking; it is configured to save `figures/e9_runin.png` when run, but I did not render it.