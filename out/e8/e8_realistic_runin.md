Created the analysis and summary files:

- [e8_runin.py](<C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/e8_runin.py>)
- [e8_runin_draws.csv](<C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/analysis/e8_runin_draws.csv>) — 69,000 rows
- [e8_runin_summary.csv](<C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/analysis/e8_runin_summary.csv>) — 345 rows
- [e8_summary.json](<C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/analysis/e8_summary.json>)
- [e8_figure.py](<C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/figures/e8_figure.py>) — syntax-check passed with `py_compile`

The fixed run-in seed is **81019**. All four selected source penalties were **0.1**, matching E7. The PNG was not generated: the specified Python runtime lacks `matplotlib`; the figure script is set to save it as `figures/e8_runin.png`.

**Primary pairs, 3-month run-in (n=162)**

| Pair | Mean cases | `dial_recal − end_recal` AUROC [p10, p90]; gain share | `dial_recal − keep` AUROC [p10, p90]; gain share |
|---|---:|---:|---:|
| A-MICU → B-MICU | 2.89 | +0.0137 [−0.0033, +0.0338]; 64% | −0.0252 [−0.1114, +0.0236]; 50% |
| B-MICU → A-MICU | 5.65 | +0.0137 [+0.0000, +0.0252]; 72% | −0.0017 [−0.0574, +0.0248]; 68% |
| A-SICU → B-SICU | 4.31 | +0.0010 [−0.0009, +0.0041]; 36% | −0.0070 [−0.0154, +0.0023]; 22% |
| B-SICU → A-SICU | 2.77 | +0.0033 [−0.0019, +0.0120]; 40% | −0.0158 [−0.0448, +0.0094]; 30% |

**Primary pairs across sizes:** AUROC difference to `keep`; values in each method column are ordered 1, 2, 3, 6, 12, 24 months.

| Pair | `end` | `dial` | `end_recal` | `dial_recal` |
|---|---|---|---|---|
| A-MICU → B-MICU | −.0302, −.0594, −.0618, −.0757, −.0741, −.0900 | −.0234, −.0552, −.0564, −.0694, −.0667, −.0761 | −.0255, −.0424, −.0389, −.0404, −.0223, −.0090 | −.0174, −.0338, −.0252, −.0285, −.0134, +.0039 |
| B-MICU → A-MICU | −.0331, −.0392, −.0432, −.0477, −.0523, −.0530 | −.0300, −.0361, −.0407, −.0464, −.0519, −.0530 | −.0162, −.0146, −.0154, −.0068, −.0032, +.0000 | −.0063, −.0014, −.0017, +.0088, +.0150, +.0201 |
| A-SICU → B-SICU | −.0038, −.0054, −.0064, −.0057, −.0052, −.0046 | −.0027, −.0044, −.0055, −.0043, −.0040, −.0035 | −.0045, −.0064, −.0080, −.0075, −.0086, −.0105 | −.0036, −.0056, −.0070, −.0068, −.0072, −.0084 |
| B-SICU → A-SICU | −.0146, −.0261, −.0307, −.0370, −.0363, skipped | −.0129, −.0250, −.0295, −.0365, −.0363, skipped | −.0117, −.0184, −.0191, −.0210, −.0221, skipped | −.0082, −.0151, −.0158, −.0178, −.0176, skipped |

**Secondary pairs at 3 months:** `dial_recal − keep` AUROC.

| Pair | Mean cases | Difference [p10, p90] | Gain share | Zero-case draws |
|---|---:|---:|---:|---:|
| A-MICU → A-SICU | 2.70 | −.0319 [−.1062, +.0059] | 38% | 12/200 |
| A-MICU → B-SICU | 4.17 | −.0253 [−.1031, +.0139] | 52% | 4/200 |
| A-SICU → A-MICU | 5.59 | −.0133 [−.0456, +.0002] | 14% | 1/200 |
| A-SICU → B-MICU | 2.92 | +.0032 [−.0014, +.0081] | 74% | 11/200 |
| B-MICU → A-SICU | 2.77 | −.0155 [−.1160, +.0117] | 52% | 14/200 |
| B-MICU → B-SICU | 4.66 | −.0159 [−.0650, +.0031] | 44% | 2/200 |
| B-SICU → A-MICU | 5.86 | −.0130 [−.0502, +.0037] | 46% | 2/200 |
| B-SICU → B-MICU | 2.74 | −.0098 [−.0813, +.0156] | 62% | 11/200 |

Across the four primary pairs, mean `dial_recal` AUROC change versus `keep` was −0.0124, while its change versus `end_recal` was +0.0079.  
At three months, 2.9% of primary run-in draws had no cases; three pair-size combinations were skipped because they would leave fewer than 500 evaluation patients.  
All source penalties matched E7, and the analysis produced 69,000 draw rows and 345 summary rows.

**Runtime: 3.0 min (178.6 s).**