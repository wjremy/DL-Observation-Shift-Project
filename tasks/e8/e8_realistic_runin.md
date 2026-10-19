# Task E8: the trust dial with a realistic unit-level run-in

You write and run Python analysis code in this repository. Read first: `SPEC.md` section "E8" (binding; pre-specified, do not change the design), sections "E7" and "E7b", and the code `e7_dial.py` and `e7b_recal.py`. Reuse their source fits, preprocessing and helpers (import or copy functions); do not modify any E7/E7b file or output.

Python: Codex runtime only (numpy, pandas):
`%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`
Data: `data/e7_cache.npz` exists. Only the P19 sites are used.

## Write and run `e8_runin.py`

- Same source fits as E7 for the four P19 sites (check the selected penalties match E7 and report).
- For each of the 12 P19 pairs (4 primary, 8 secondary, as in SPEC), each run-in size and 200 draws (fixed seed, stated): draw the run-in from the target unit without replacement, evaluate every method on all remaining target patients.
- Methods exactly as SPEC E8. The penalised recalibration: minimise sum over run-in patients of log loss of sigmoid(a + s * z) + 1.0 * (a^2 + (s - 1)^2) by Newton, where z is the candidate logit; start at a = 0, s = 1. Report how often a run-in has 0 cases.
- If a run-in size exceeds what the target unit can supply while leaving at least 500 evaluation patients, skip that size for that pair and say so.

Outputs:
- `analysis/e8_runin_draws.csv`: one row per pair, size, draw, method (AUROC, log loss, chosen w, run-in cases).
- `analysis/e8_runin_summary.csv`: per pair, size, method: mean AUROC and log loss; and paired differences of every method to `keep` and to `end_recal`: mean, 10th and 90th percentile, share of draws with a gain; mean run-in cases.
- `analysis/e8_summary.json`.
- `figures/e8_figure.py` (matplotlib; syntax-check only with py_compile): for the 4 primary pairs, one panel each, x = run-in size in months (log axis), y = AUROC difference to `keep`, lines for `end`, `dial`, `end_recal`, `dial_recal` with 10-90% bands; dashed line at 0; title "source -> target". Save `figures/e8_runin.png`.

## Answer (stdout, compact)

- Primary result: for the 4 primary pairs at n_run = 162 (3 months): mean run-in cases; `dial_recal` minus `end_recal` and `dial_recal` minus `keep`, AUROC mean [10-90%] and share of draws with a gain.
- One compact table for the primary pairs across all sizes (AUROC difference to `keep` for `end`, `dial`, `end_recal`, `dial_recal`).
- Secondary pairs at 3 months in one short table.
- Three sentences, then runtime.
