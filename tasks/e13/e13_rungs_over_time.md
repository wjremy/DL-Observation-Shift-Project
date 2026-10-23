# Task: write `e13_rungs_over_time.py` and `figures/e13_figure.py` (experiment E13)

You write two Python scripts in the project folder (your working directory). Read first:

- `SPEC.md`, section "E13: which rung when? Performance over weeks of deployment" (binding design; follow it exactly). Also section E12 for context.
- `e12_unlabelled_head.py` (reuse its helpers by import where they fit: `auroc_rank`, `_metrics`, `_refit_testing_head`, `_policies`, `_load_parameters`; do not copy them).
- `latent_filter.py` (`LatentFilter`, `numpy_forward`, `train_filter`, `PARAMETER_NAMES`), `sim.py`, `bayes.py` (`bayes_predict`), `run_grid.py`.

## `e13_rungs_over_time.py`

1. At import time: `sim.THRESHOLD = 2.05` and `bayes.THRESHOLD = 2.05` (bayes imported the constant by value). Put this at module top level so spawned workers that import the module get it too. Assert in `main` that the A training cohort's outcome rate is between 1% and 3%.
2. Train the A model once and cache it: `results/E13/a_model/params.npz` plus `meta.json` (skip training if both exist). A policy informative gamma 2, rate 0.3, sigma 1; n_train 16,000, n_val 4,000 from new fixed seeds (e.g. 1_500_000 base); `latent_filter.train_filter` with E5's settings.
3. Test cohort: 10,000 patients, fixed seeds; per B policy a fixed observation seed; oracle with `bayes.bayes_predict`.
4. Per policy and draw (5 draws), one deployment stream of 2,600 patients (fresh latent and observation seeds, deterministic from policy and draw, disjoint from all other seeds). Week t uses the first 25 t patients (t = 1, 2, 4, 8, 13, 26, 52, 104).
5. Methods per (policy, draw, week), as the SPEC defines: `keep`, `head_U`, `head_U_dial`, `finetune`, `dial_only`, `oracle` (`keep` and `oracle` once per policy).
   - `head_U`: `_refit_testing_head(stream_records, None, A_params, ...)` with lr and steps from the command line (defaults lr 0.05, steps 160).
   - `head_U_dial` and `dial_only`: w on the grid 0, 0.05, ..., 1 minimising `sum of log losses on the labelled patients + 2 * (w - 1)**2`; ties to the larger w; zero outcome cases -> w = 1. Use `numpy_forward`.
   - `finetune`: new function: load A params into `LatentFilter`, all parameters trainable, Adam (same lr and steps), minibatch 256 with replacement, loss BCE + `-log_marginal.mean() / T` (as `train_filter`), w = 1. Zero outcome cases -> return A params unchanged.
   - Evaluate every method on the policy's test cohort with `numpy_forward` (w as chosen).
6. Rows: policy, method, week, n_patients, n_cases (outcome cases among the labelled patients), draw, brier, log_loss, auroc, regret (Brier minus oracle Brier), w, wall_time_seconds. Write `analysis/e13_runs.csv` and `analysis/e13_summary.csv` (mean, min, max over draws per policy x method x week).
7. Jobs = (policy, draw, week); run them in a spawn-safe `multiprocessing` pool, `--workers` (default 4), `torch.set_num_threads(1)` per worker, one progress line per finished job with elapsed seconds. Load the test cohort and oracle once per worker via an initializer (as E12 does).
8. Flags: `--smoke` (A model n_train 1,000, n_val 300, 3 epochs, cached separately under `results_smoke/E13/`; test cohort 500; weeks 1 and 4; 1 draw; 2 Adam steps; outputs `analysis/e13_smoke_*.csv`), `--workers`, `--lr`, `--steps`, `--train-only` (train and cache the A model, then stop).

## `figures/e13_figure.py`

Reads `analysis/e13_summary.csv`, writes `figures/e13_rungs_over_time.png` (and `.pdf`). One panel per B policy (2 x 2). x: weeks, log scale, ticks at the weeks used, with a second label row giving approximate outcome cases (mean n_cases). y: regret (Brier minus optimum). Lines: `keep` (flat, grey dashed), `dial_only`, `head_U`, `head_U_dial`, `finetune`; mean with a light min-max band. Shade the background of each week's interval by the method with the lowest mean regret (light colour of that method). Readable in print: distinct colours plus markers, legend once for the figure, panel titles with the policy in words (B less selective, B more selective, B tests at random (same rate), B tests at random (twice as often)). Use matplotlib only.

## Checks you can do

You cannot import torch or matplotlib. Your Python: `%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` (numpy, pandas). Run `py_compile` on both scripts; check with numpy only that the threshold patch gives an outcome rate near 2% on 20,000 simulated patients, that `bayes.bayes_predict` then uses the new threshold (its mean prediction near 2%), and that the stream nesting gives 25 t patients per week. Keep torch and matplotlib imports inside functions or guarded so these checks run. Do not modify existing files. Do not run anything heavy.

## Your answer

Short report: what you built, any departure from the SPEC and why, check results, exact commands (smoke, train-only, full, figure). Do not paste the scripts.
