# Task: write `e15_sequences.py` (hourly sequence cache for experiment E15)

You write one new Python script in the project folder (your working directory). Read first:

- `SPEC.md`, section "E15: re-learning the testing head on real hospital pairs", paragraph "Data" (binding).
- `e7_sites.py` (cohort rules, raw file layout, patient ids, outcome definitions for PhysioNet 2019 and 2012; reuse its constants and helpers by import, e.g. `P19_VITAL_LABS`, `_parse_time`, paths). `data/raw_physionet2019/` and `data/raw_physionet2012/` hold the raw files.
- `data/e7_cache.npz` holds E7's patient ids per site (`P19_<site>_patient_ids`, `P12_<site>_patient_ids`) and outcomes (`..._y`).

Do not modify any existing file. Another agent writes the model and experiment against the cache format below, so keep it exactly.

## What the script does

1. Same patients as E7, in the same order: for each E7 site, read the raw records of exactly the ids in `data/e7_cache.npz` and check that the outcome you derive equals E7's `y` (assert; report mismatches if any).
2. PhysioNet 2019: hourly rows with ICULOS 1-24 (t = ICULOS - 1, T = 24); variables `P19_VITAL_LABS`. Sites in the cache: `P19_A-all` = A-MICU then A-SICU ids concatenated, `P19_B-all` = B-MICU then B-SICU (also keep the four unit sites).
3. PhysioNet 2012: hours 0-47 from the `Time` column (hour h = floor(time in hours), times exactly at 48:00 go to hour 47); per variable the last value within the hour; measured = any value within the hour. Variables: all time-varying variables present in the raw files except the static ones in `P12_STATIC` (list them in a fixed order, sorted by name). Sites CCU, CSRU, MICU, SICU. T = 48.
4. Cleaning (per SPEC): pH outside 6.5-8.0 -> missing; negative values of variables that are nonnegative by nature (everything except base excess) -> missing; for P12 also values of -1 that the raw files use as "unknown", if any appear in time-varying variables -> missing. The mask m is 1 only where a valid value remains. Count and report removed values per variable.
5. Log list: per dataset, variables whose valid values are all positive and whose pooled skewness (all sites of the dataset) is above 2 -> stored as log1p(value). Record the list.
6. Write `data/e15_cache.npz` with keys, per site `<dataset>_<site>`: `<key>_x` float32 [n, T, D] with NaN where missing, `<key>_m` int8 [n, T, D], `<key>_y` int8 [n], `<key>_ids` (as in E7); per dataset `<dataset>_variables` (array of str), `<dataset>_log_variables`, `<dataset>_T`. No standardisation in the cache.
7. Print a summary: per site n, outcome rate, mean measurement rate overall and for the 10 most- and least-measured variables; per dataset D, T and the log list. Also write that summary to `out/e15/e15_sequences_summary.json`.

Speed: the raw files are about 40,000 small text files; read each once, vectorise per file, and print progress every 2,000 files.

## Checks you can do

Your Python: `%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` (numpy, pandas). Run the script fully if it finishes in under about 15 minutes with that interpreter; if the sandbox denies reading the raw files, say so and only `py_compile` it. If it runs, check: ids and y equal E7's; a few hand-checked patients (pick 2 per dataset, compare 3 raw values with the cache); x is NaN exactly where m == 0.

## Your answer

Short report: what you built, whether it ran (runtime), the summary numbers, any departure from the SPEC and why, the exact command.
