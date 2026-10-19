# Task E7: the trust dial across many real site pairs

You write and run Python analysis code in this repository. Read first: `SPEC.md` section "E7" (binding) and section "E6"; the existing code `e6_cohort.py` and `e6_dial.py` (reuse their logic: logistic fit by Newton/IRLS with L2 and 5-fold CV over the same penalty grid, AUROC by ranks, metrics). Do not modify E6 files or outputs.

## Python

Run everything with the Codex runtime (numpy and pandas only; no scikit-learn, no matplotlib):
`%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

## Data (already downloaded, read-only)

- `data/raw_physionet2019/training_setA/*.psv`, `training_setB/*.psv` (pipe-separated hourly rows; columns include Unit1, Unit2, ICULOS, SepsisLabel).
- `data/raw_physionet2012/set-a`, `set-b`, `set-c` (one `RecordID.txt` per stay; long format `Time,Parameter,Value`, time as hh:mm since ICU admission; static descriptors at 00:00: RecordID, Age, Gender, Height, ICUType, Weight; -1 means missing) and `Outcomes-a.txt`, `Outcomes-b.txt`, `Outcomes-c.txt` (RecordID, ..., In-hospital_death). Note Weight can also appear later as a time-varying value; treat Weight as static using the 00:00 value.

## Write

1. `e7_sites.py`: builds both site cohorts and caches feature matrices to `data/e7_cache.npz` (add it to `.gitignore`). P19: E6 cohort and features plus a site label (hospital x unit; exclude patients with neither unit). Keep raw (unimputed, unscaled) physiology and pattern features so each source site can impute and standardise with its own 80% training statistics. P12: as SPEC.
2. `e7_dial.py`: per source site, fit; per target, w curve and the 100-split tuned comparison (SPEC). Writes `analysis/e7_pairs.csv`, `analysis/e7_curves.csv`, `analysis/e7_sites.csv`, `analysis/e7_summary.json`. Print progress.
3. `figures/e7_figure.py` (matplotlib; you cannot run it, syntax-check with `python -m py_compile`): two figures or one with two blocks (P19, P12); each panel one directed pair, x = w (0 to 1.5), y = target log loss minus log loss at w = 1, dot at the best w, horizontal line at 0, title "source -> target". Save `figures/e7_pairs.png`.

Run both scripts end to end. Check: no feature uses P19 hours after 24 or P12 hours after 48; no target patient is used in fitting the source model; site sizes and outcome counts are printed.

## Answer (to stdout, compact)

- Table of sites: n, outcome cases.
- Table of pairs: dataset, source, target, best w on full target, log loss at w = 0, 1, best; mean chosen w (tuned); share of splits with interior w; mean (dial minus end) difference in log loss and AUROC with 10-90% range.
- Three sentences: how often the best w is interior, how large the gain over the better end is, and any data problems.
- Runtime.
