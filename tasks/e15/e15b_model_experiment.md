# Task: write `latent_filter_mv.py`, `e15_real_pairs.py` and `figures/e15_figure.py` (experiment E15)

You write three new Python files in the project folder (your working directory). Read first:

- `SPEC.md`, section "E15: re-learning the testing head on real hospital pairs" (binding design; follow it exactly), and sections E12-E14 for the methods.
- `latent_filter.py` (the one-variable model you generalise: `LatentFilter`, `numpy_forward`, `train_filter`, initialisation, loss), `e12_unlabelled_head.py` (`_refit_testing_head`, `auroc_rank`), `e13_rungs_over_time.py` (`_finetune`, `_choose_penalized_weight`, the worker/pool pattern, model caching). Do not import `e13_rungs_over_time` (it patches simulator constants); write the multivariate versions in `latent_filter_mv.py`.
- `analysis/e7_*` files for E7's linear-model AUROC per site (for the sanity comparison; find the right file and column and say which you used).

Another agent is writing `e15_sequences.py`, which produces `data/e15_cache.npz` with this format (it may not exist yet): per site key `<dataset>_<site>` (e.g. `P12_CSRU`, `P19_A-all`): `<key>_x` float32 [n, T, D] with NaN where missing, `<key>_m` int8 [n, T, D], `<key>_y` int8 [n], `<key>_ids`; per dataset `<dataset>_variables`, `<dataset>_log_variables`, `<dataset>_T`. Values are cleaned and log-transformed but not standardised. Do not modify any existing file.

## `latent_filter_mv.py`

- Parameters: `transition_logits` [K, K], `initial_logits` [K], `mu` [K, D], `log_sd` [D], `c` [K, D], `u` [K]. Initialisation as E5 (banded transition, initial from a grid), `mu[k, d]` from a linspace like E5 for every variable, `c` = logit of the variable's source measurement rate, `u` = logit of the source outcome rate.
- `numpy_forward(x, m, params, w=1.0)` and a torch `LatentFilterMV(nn.Module)` with `forward(x, m, w)` returning (probabilities from the final belief, log marginal), exactly as the one-variable model but summing value and testing log-likelihoods over variables: value term only where m == 1, testing term `w * sum_d [m log sigmoid(c) + (1 - m) log sigmoid(-c)]`. NumPy and torch must agree (test to 1e-5 on random data, you can only check the NumPy side; write the torch check as a test function for Claude to run).
- `train_filter_mv(...)`: as `train_filter` (Adam lr 1e-2, batch 256, max 100 epochs, patience 10, best-validation state), loss BCE + lambda * (-log_marginal.mean()) / (T * D), `lambda_nll` argument default 1.0. Return the model and a history including validation AUROC per epoch.
- `refit_testing_head_mv(x, m, labels_or_None, params, *, batch_seed, steps, lr)` (c only; NLL / (T * D), plus BCE when labels are given) and `finetune_mv(x, m, labels, params, *, batch_seed, steps, lr)` (all parameters), minibatch 256 with replacement, `torch.set_num_threads(1)`; zero outcome cases in `finetune_mv` -> return the input parameters.
- `standardise(x, mean, sd)` with mean and sd per variable from the source training split (NaN-aware), result clipped to [-5, 5], NaN kept.

## `e15_real_pairs.py`

1. Sources and pairs exactly as the SPEC. A model per source: 80/20 split (fixed seed), standardisation from the 80%, `train_filter_mv`, cached at `results/E15/a_model_<dataset>_<site>/params.npz` plus `meta.json` (standardisation means/sds, split ids, best epoch, wall time, validation AUROC, the E7 linear AUROC for comparison). Skip training if cached.
2. Per pair: target standardised with the source's statistics; fixed 50/50 split into pool and evaluation half; run-ins n_run 162 and 650 drawn without replacement from the pool, 3 draws (`--draws`). Skip a size if the pool is smaller.
3. Methods per (pair, size, draw) as the SPEC: `keep`, `drop`, `dial_only`, `head_U`, `head_L`, `finetune`; `head_U_pool` once per pair (it does not depend on the run-in), copied to each size and draw like `keep`. Evaluate on the evaluation half: AUROC, log loss, Brier, n cases in the run-in, chosen w.
4. Outputs: `analysis/e15_runs.csv`, `analysis/e15_summary.csv` (mean, min, max over draws; also AUROC difference to `keep`), `analysis/e15_sources.csv` (per source: n, cases, best epoch, validation AUROC, E7 linear AUROC, training time).
5. Jobs in a spawn-safe `multiprocessing` pool (`--workers`, default 4), one torch thread per worker, data loaded once per worker; train the A models first, sequentially, with 4 torch threads. Flags: `--smoke` (2 pairs, 1 draw, size 162 only, 2 epochs, 2 steps, cached under `results_smoke/E15/`, outputs `analysis/e15_smoke_*`), `--train-only`, `--workers`, `--draws`, `--lambda-nll`.
6. If a source's validation AUROC is more than 0.05 below its E7 linear AUROC, print a clear warning line (SPEC: Claude then stops and reports).

## `figures/e15_figure.py`

`figures/e15_real_pairs.png` and `.pdf`: one row per pair (shift pairs first, then controls, labelled), x = method, y = AUROC difference to `keep` (mean with min-max over draws), one marker colour per run-in size; a vertical zero line. matplotlib only.

## Checks you can do

You cannot import torch or matplotlib. Your Python: `%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` (numpy, pandas). `py_compile` all files. With numpy only: `numpy_forward` with D = 1 equals `latent_filter.numpy_forward` on the same parameters and data (to 1e-10); with w = 0 the predictions do not depend on the mask pattern beyond which values are seen; standardisation keeps NaN and clips. Test the experiment's data handling on a small synthetic cache if the real one is missing. Keep torch and matplotlib imports inside functions. Do not run anything heavy.

## Your answer

Short report: what you built, departures from the SPEC and why, check results, which E7 file/column gives the linear AUROC, exact commands (smoke, train-only, full, figure, torch test).
