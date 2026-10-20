# Task E10: which part shifted between sites?

You write and run Python analysis code in this repository. Read first: `SPEC.md` section "E10" (binding) and sections E6/E7; code `e6_cohort.py`, `e6_dial.py`, `e7_sites.py`, `e7_dial.py` (reuse their caches, raw feature arrays, logistic fit with CV, AUROC; import or copy functions; do not modify existing files or outputs).

Python: Codex runtime only (numpy, pandas):
`%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
Data: `data/e7_cache.npz` (P19 unit sites and P12 sites, raw unimputed features) and `data/e6_cache.npz` (whole hospitals; if it lacks raw unimputed features, rebuild the whole-hospital raw arrays from `data/raw_physionet2019` with E6's cohort rules, or derive A-all and B-all from the raw arrays used by `e7_sites.py`, and say which).

## Write and run `e10_diagnostic.py`

As SPEC E10. Outputs:
- `analysis/e10_gaps.csv`: one row per pair and feature set: AUROC local, AUROC transferred, gap with 95% interval; log loss local, transferred, gap with interval.
- `analysis/e10_sites.csv`: per site and feature set, local AUROC; per site, mean imputed share of physiology variables.
- `analysis/e10_coverage.csv`: per site and variable, measurement rate and imputed share.
- `analysis/e10_summary.json`.
- `figures/e10_figure.py` (matplotlib; syntax-check only): scatter of physiology AUROC gap (x) against pattern AUROC gap (y), one point per pair, colours by pair group (P19 same unit other hospital; P19 other unit other hospital; P19 other unit same hospital; P19 whole hospital; P12), diagonal line, zero lines; save `figures/e10_gaps.png`.

## Answer (stdout, compact)

- Table: pair, group, AUROC gap for pattern, physiology, combined (with intervals), mean measurement-rate difference, mean imputed-share difference.
- Per group: mean pattern gap and mean physiology gap.
- Three sentences: in which pairs the testing pattern's meaning shifted, in which physiology shifted, and how coverage relates to the physiology gap. Runtime.
