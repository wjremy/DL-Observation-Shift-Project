Implemented and ran E7. Source-heldout metrics and selected penalties are in `e7_sites.csv`; the 24 pair results and 744 curve rows are in the other CSVs. The figure script was syntax-checked only, as requested; I did not render the PNG. E6 source files and outputs are unchanged.

### Sites

| Dataset | Site | n | Cases |
|---|---|---:|---:|
| P19 | A-MICU | 3,065 | 105 |
| P19 | A-SICU | 1,470 | 25 |
| P19 | B-MICU | 5,018 | 88 |
| P19 | B-SICU | 4,844 | 130 |
| P12 | CCU | 1,769 | 224 |
| P12 | CSRU | 2,530 | 123 |
| P12 | MICU | 4,293 | 852 |
| P12 | SICU | 3,408 | 508 |

### Directed pairs

Here, Δ = dial minus the better endpoint on the 100-split evaluation halves; ranges are the 10th–90th percentiles.

| Dataset | Source → target | Best w | LL at 0 | LL at 1 | LL best | Mean tuned w | Interior splits | Δ log loss | Δ AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P19 | A-MICU → A-SICU | 0.10 | 0.0904 | 0.1092 | 0.0903 | 0.14 | 61% | +0.00074 [−0.00009, +0.00243] | +0.0308 [+0.0000, +0.0716] |
| P19 | A-MICU → B-MICU | 0.25 | 0.0991 | 0.1008 | 0.0990 | 0.28 | 82% | −0.00011 [−0.00021, +0.00054] | +0.0260 [+0.0000, +0.0525] |
| P19 | A-MICU → B-SICU | 0.40 | 0.1260 | 0.1275 | 0.1252 | 0.43 | 99% | −0.00126 [−0.00305, −0.00040] | +0.0383 [−0.0143, +0.0704] |
| P19 | A-SICU → A-MICU | 0.00 | 0.1637 | 0.1653 | 0.1637 | 0.23 | 41% | −0.00005 [−0.00165, +0.00060] | +0.0046 [+0.0000, +0.0162] |
| P19 | A-SICU → B-MICU | 0.35 | 0.0879 | 0.0887 | 0.0876 | 0.36 | 90% | −0.00035 [−0.00133, +0.00006] | +0.0004 [−0.0127, +0.0085] |
| P19 | A-SICU → B-SICU | 0.85 | 0.1292 | 0.1278 | 0.1277 | 0.86 | 59% | +0.00007 [−0.00006, +0.00047] | −0.0008 [−0.0059, +0.0010] |
| P19 | B-MICU → A-MICU | 0.00 | 0.1642 | 0.1691 | 0.1642 | 0.00 | 0% | +0.00000 [+0.00000, +0.00000] | +0.0000 [+0.0000, +0.0000] |
| P19 | B-MICU → A-SICU | 1.50 | 0.0852 | 0.0834 | 0.0830 | 1.38 | 7% | −0.00014 [−0.00050, +0.00024] | +0.0060 [−0.0045, +0.0152] |
| P19 | B-MICU → B-SICU | 1.00 | 0.1243 | 0.1198 | 0.1198 | 1.00 | 28% | +0.00027 [+0.00000, +0.00081] | −0.0003 [−0.0022, +0.0018] |
| P19 | B-SICU → A-MICU | 1.00 | 0.1499 | 0.1473 | 0.1473 | 0.98 | 44% | +0.00038 [+0.00000, +0.00138] | −0.0008 [−0.0067, +0.0015] |
| P19 | B-SICU → A-SICU | 0.00 | 0.0922 | 0.0985 | 0.0922 | 0.00 | 0% | +0.00000 [+0.00000, +0.00000] | +0.0000 [+0.0000, +0.0000] |
| P19 | B-SICU → B-MICU | 1.50 | 0.0900 | 0.0869 | 0.0863 | 1.45 | 0% | −0.00049 [−0.00083, −0.00012] | +0.0064 [−0.0021, +0.0153] |
| P12 | CCU → CSRU | 0.00 | 0.1665 | 0.1758 | 0.1665 | 0.05 | 30% | +0.00024 [+0.00000, +0.00106] | +0.0002 [−0.0001, +0.0014] |
| P12 | CCU → MICU | 0.90 | 0.4759 | 0.4651 | 0.4649 | 0.90 | 58% | +0.00013 [−0.00027, +0.00084] | +0.0012 [−0.0006, +0.0025] |
| P12 | CCU → SICU | 0.85 | 0.3718 | 0.3618 | 0.3615 | 0.83 | 86% | +0.00023 [−0.00047, +0.00144] | +0.0060 [+0.0020, +0.0103] |
| P12 | CSRU → CCU | 1.40 | 0.3758 | 0.3274 | 0.3221 | 1.39 | 0% | −0.00475 [−0.00702, −0.00146] | −0.0130 [−0.0172, −0.0077] |
| P12 | CSRU → MICU | 1.40 | 0.5278 | 0.4670 | 0.4608 | 1.39 | 0% | −0.00580 [−0.00773, −0.00330] | −0.0110 [−0.0143, −0.0081] |
| P12 | CSRU → SICU | 1.50 | 0.4370 | 0.3905 | 0.3804 | 1.50 | 0% | −0.01035 [−0.01310, −0.00798] | −0.0210 [−0.0237, −0.0184] |
| P12 | MICU → CCU | 1.00 | 0.3103 | 0.3031 | 0.3031 | 1.02 | 30% | +0.00057 [+0.00004, +0.00131] | −0.0008 [−0.0024, +0.0000] |
| P12 | MICU → CSRU | 0.50 | 0.2101 | 0.2101 | 0.2092 | 0.49 | 99% | −0.00111 [−0.00122, −0.00097] | +0.0030 [−0.0040, +0.0091] |
| P12 | MICU → SICU | 1.40 | 0.3372 | 0.3276 | 0.3267 | 1.40 | 0% | −0.00063 [−0.00120, +0.00025] | −0.0015 [−0.0037, +0.0003] |
| P12 | SICU → CCU | 0.50 | 0.3758 | 0.3764 | 0.3738 | 0.52 | 99% | −0.00270 [−0.00324, −0.00224] | +0.0028 [−0.0025, +0.0090] |
| P12 | SICU → CSRU | 0.00 | 0.1787 | 0.1844 | 0.1787 | 0.05 | 21% | +0.00021 [+0.00000, +0.00078] | −0.0003 [−0.0004, +0.0002] |
| P12 | SICU → MICU | 0.75 | 0.4417 | 0.4352 | 0.4340 | 0.73 | 100% | −0.00091 [−0.00165, +0.00067] | −0.0002 [−0.0019, +0.0008] |

The full-target best weight was strictly interior for 10 of 24 pairs (41.7%); the split-tuned weight was interior in 43.1% of pair-splits. Relative to the better fixed end, the full-target best dial gained 0.00123 log-loss units per patient on average; the split-tuned dial’s mean difference was −0.00108. Data checks found 9,484 P19 patients with neither unit, 16,241 without complete hours 1–24, and 214 excluded for early sepsis; P12 had 5,726 missing Heights, 1,001 missing Weights, and 12 missing Genders, which were handled through source-specific imputation.

The analysis checked all 24 source-train/target ID pairs with zero overlaps and validated the P19/P12 feature horizons and static Weight handling. Runtime was about 5.1 minutes total: 4.1 minutes to build the cache and 1.0 minute for analysis.

Files: [e7_sites.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/e7_sites.py), [e7_dial.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/e7_dial.py), [e7_figure.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/figures/e7_figure.py), [e7_sites.csv](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/analysis/e7_sites.csv), [e7_pairs.csv](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/analysis/e7_pairs.csv), [e7_curves.csv](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/analysis/e7_curves.csv), [e7_summary.json](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/analysis/e7_summary.json). The cache [e7_cache.npz](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/data/e7_cache.npz) is gitignored.