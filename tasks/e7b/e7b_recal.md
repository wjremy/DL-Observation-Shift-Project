# Task E7b: is the dial's gain above w = 1 real, or recalibration?

You write and run Python analysis code in this repository. Read first: `SPEC.md` sections "E7" and "E7b" (E7b is binding), `e7_dial.py` (reuse its source fits, features and helpers by importing or copying functions; do not modify `e7_dial.py`, `e7_sites.py` or any E7 output), and `out/e7/e7_many_pairs.md` for context.

Python: run with the Codex runtime only (numpy, pandas):
`%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
Data: the cache `data/e7_cache.npz` already exists (built by `e7_sites.py`).

## Write and run `e7b_recal.py`

For each of the 24 E7 pairs (same source fits as E7: same split seed, penalties and preprocessing, so the w = 0..1.5 curve reproduces E7 exactly; check this and report the maximum absolute difference in target log loss):

1. Full-target curve on the extended grid w = 0 to 3.0, step 0.05: log loss and AUROC. Report the best w and whether it is at the grid edge.
2. 100 random 50/50 tune/evaluation splits of the target (new seed, stated). On the tune half choose each option below by tune-half log loss; evaluate on the evaluation half (log loss, AUROC):
   - `end`: w in {0, 1};
   - `dial01`: w on the grid 0 to 1;
   - `dial03`: w on the grid 0 to 3;
   - `end_recal`, `dial01_recal`, `dial03_recal`: the same choice sets, but every candidate w is followed by logistic recalibration (intercept and slope on the candidate logit, fitted by unpenalised Newton on the tune half); choose the w whose recalibrated tune-half log loss is lowest, and apply that w with its tune-half recalibration to the evaluation half.
   For each option report the mean evaluation log loss and AUROC, and the mean and 10-90% range of the paired difference to `end` and to `end_recal`. Also the mean chosen w per option, and the share of splits with chosen w > 1.

Outputs: `analysis/e7b_pairs.csv` (one row per pair and option), `analysis/e7b_curves.csv` (extended grid), `analysis/e7b_summary.json`.

## Answer (stdout, compact)

- One table per pair group is not needed; give one table of 24 pairs with: best w on 0-3 (edge flag), and for `dial03` minus `end` and `dial03_recal` minus `end_recal`: mean log-loss difference [10-90%] and mean AUROC difference.
- Totals: in how many pairs `dial03_recal` beats `end_recal` (mean log-loss difference below 0 and 90th percentile below 0), the mean difference across pairs, and the same for `dial01_recal`.
- Three sentences on whether the w > 1 gains survive recalibration.
- Runtime.
