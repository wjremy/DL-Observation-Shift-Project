# Task E11: safe dial tuning with realistic run-ins on the 2012 mortality pairs

You write and run Python analysis code in this repository. Read first: `SPEC.md` section "E11" (binding, pre-specified; do not change the design), sections E8 and E9, and the code `e8_runin.py`, `e9_fixed_tuning.py`, `e7_dial.py`. Reuse their source fits, preprocessing, run-in sampling, penalised recalibration, tuning rules and AUROC (import or copy); do not modify any existing file or output.

Python: Codex runtime only (numpy, pandas):
`%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
Data: `data/e7_cache.npz`, P12 sites only (CCU, CSRU, MICU, SICU).

## Write and run `e11_p12_runin.py`

- E7's P12 source fits (check selected penalties match `analysis/e7_sites.csv`).
- All 12 ordered P12 pairs, SPEC run-in sizes and skip rule, 200 draws, a new fixed seed (state it), evaluation on all remaining target patients.
- Methods exactly as SPEC E11: `keep`, `end_recal`, `dial_recal`, `dial_recal_shrink` (tau = 2), `dial_auc_shrink` (0.02).

Outputs: `analysis/e11_summary.csv` (per pair, size, method: mean AUROC and log loss; differences to `keep` and to `end_recal`: mean, p10, p90, share of draws with a gain; chosen w mean, p10, p90; mean run-in cases), `analysis/e11_summary.json` (including the primary result). No per-draw file over 10 MB. `figures/e11_figure.py` (matplotlib; syntax-check only): 12 panels (one per pair, the four subgroup pairs marked in the title), x = run-in months (log axis), y = AUROC difference to `keep`, lines for `end_recal`, `dial_recal`, `dial_recal_shrink`, `dial_auc_shrink` with 10-90% bands for `dial_recal_shrink`; save `figures/e11_runin.png`.

## Answer (stdout, compact)

- Primary: `dial_recal_shrink` minus `keep` AUROC at 3 months, mean over the 12 pairs, and per pair mean [p10, p90], share of gains, mean run-in deaths, mean chosen w.
- Same for `dial_recal_shrink` minus `end_recal` and for `dial_auc_shrink` minus `keep` (means per pair only).
- Subgroup (four named pairs) means at all sizes.
- Three sentences; runtime.
