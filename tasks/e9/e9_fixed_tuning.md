# Task E9: the realistic run-in with corrected tuning

You write and run Python analysis code in this repository. Read first: `SPEC.md` sections "E8" and "E9" (E9 is binding), and `e8_runin.py`. Reuse E8's source fits, run-in sampling, penalised recalibration, AUROC and helpers (import or copy); do not modify any E7/E7b/E8 file or output.

Python: Codex runtime only (numpy, pandas):
`%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
Data: `data/e7_cache.npz`. P19 sites only.

## Write and run `e9_fixed_tuning.py`

- All 12 P19 pairs, E8's run-in sizes and skip rule, 200 draws, a new fixed seed (state it), evaluation on all remaining target patients.
- Methods exactly as SPEC E9: `keep`, `dial_recal`, `dial_recal_shrink` (tau = 2), `dial_auc`, `dial_auc_shrink` (0.02 penalty). Grid 0 to 3 step 0.05.
- Run-in AUROC for `dial_auc`: rank-based, as E8's helper; precompute per draw efficiently (the run-in is small).
- Check that `keep` and `dial_recal` reproduce E8's method definitions (same code path); different draws are expected because the seed differs.

Outputs: `analysis/e9_summary.csv` (per pair, size, method: mean AUROC and log loss; differences to `keep`: mean, p10, p90, share of draws with a gain; mean and p10/p90 of chosen w; mean run-in cases), `analysis/e9_summary.json`. Do not write a per-draw CSV larger than 10 MB (aggregate instead). Also `figures/e9_figure.py` (matplotlib; syntax-check only): same layout as `figures/e8_figure.py`, lines for the five methods' AUROC difference to `keep`; save `figures/e9_runin.png`.

## Answer (stdout, compact)

- Primary: the 4 primary pairs at 3 months, each method: AUROC difference to `keep` mean [p10, p90], share of gains, mean chosen w.
- The same for all sizes in one compact table (means only).
- Secondary pairs at 3 months, `dial_recal_shrink` and `dial_auc_shrink` only.
- Three sentences; runtime.
