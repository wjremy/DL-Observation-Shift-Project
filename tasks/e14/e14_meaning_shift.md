# Task: write `sim_u.py`, `e14_meaning_shift.py` and `figures/e14_figure.py` (experiment E14)

You write three new Python files in the project folder (your working directory). Read first:

- `SPEC.md`, section "E14: a shift only outcomes can reveal" (binding design; follow it exactly). Sections E12 and E13 for context.
- `sim.py` (`simulate_latent`, `observe`, `informative`, `_alpha_for_rate`, `_sigmoid`, constants T, RHO, HORIZON), `bayes.py` (`_grid_and_transition`, `_bayes_predict_grid`, `_normal_cdf`), `latent_filter.py` (`numpy_forward`, `train_filter`, `LatentFilter`, `PARAMETER_NAMES`).
- `e12_unlabelled_head.py` (`_refit_testing_head`, `_metrics`, `_load_parameters`) and `e13_rungs_over_time.py` (`_finetune`, `_choose_penalized_weight`, `_stream_seeds`, the worker/pool pattern, A-model caching). Reuse these by import; do not copy them. Note: importing `e13_rungs_over_time` sets `sim.THRESHOLD` and `bayes.THRESHOLD` to 2.05 as a side effect. E14 must not depend on either constant: compute y and the oracle with E14's own threshold, passed explicitly.
- `figures/e13_figure.py` (style to follow: log x and log y, no background shading, week labels with a second row of mean outcome cases, legend once).

Do not modify any existing file.

## `sim_u.py` (NumPy only)

- `U_RATE = 0.3`, `DELTA = 1.0`. `outcome_threshold()`: solve 0.7 * (1 - Phi(t)) + 0.3 * (1 - Phi(t - DELTA)) = 0.02 for t (z_future is N(0, 1) under the stationary AR(1)); cache it.
- `simulate_latent_u(n, seed)`: z and z_future from `sim.simulate_latent(n, seed)`; u ~ Bernoulli(U_RATE) from a separate generator seeded deterministically from `seed` (e.g. `np.random.default_rng([seed, 7])`); y = 1[z_future + DELTA * u > threshold]. Return z, z_future, u, y. Do not use `sim.THRESHOLD`.
- A frozen dataclass policy with name, rate, gamma, beta, alpha; constructor `u_informative(gamma, beta, rate=0.3)` solving E[sigmoid(alpha + gamma Z + beta U)] = rate with Z ~ N(0, 1) (Gauss-Hermite, as `sim._alpha_for_rate`) and U ~ Bernoulli(U_RATE).
- `observe_u(latent, policy, sigma, seed)`: mask with P = sigmoid(alpha + gamma z_t + beta u), values z + sigma * noise; same output format as `sim.observe` (`x` float32 with NaN, `m` int8).
- `bayes_predict_u(x, m, sigma, policy, threshold=None, delta=DELTA)`: exact mixture predictor from SPEC (241-point grid, `bayes._grid_and_transition`; per u a forward filter with intercept alpha + beta * u that also accumulates the log evidence; P(u | h) from the evidences and the prior; P(y | h, u) = posterior over z times P(z_future > threshold - delta * u | z) with the AR(1) horizon formula of `bayes.py`). Optionally return P(u = 1 | h) too.

## `e14_meaning_shift.py`

1. Directions: `gain` (A: gamma 2, beta 0; B: gamma 2, beta 2) and `lose` (A: gamma 2, beta 2; B: gamma 2, beta 0). sigma 1.
2. A model per direction, trained once and cached at `results/E14/a_model_<direction>/params.npz` plus `meta.json` (skip if both exist): n_train 16,000, n_val 4,000 from fixed new seeds (base e.g. 1_700_000, disjoint from E12/E13 seeds), `latent_filter.train_filter` with E5/E13 settings (K 32). Assert the training outcome rate is 1-3%. Record in meta: outcome rate, cases, best epoch, wall time.
3. Test cohort: 10,000 patients, fixed seeds, the same latent cohort for both directions, observed under each direction's B policy; oracle from `bayes_predict_u` under B's policy. Also record the oracle's AUROC and keep's AUROC per direction (the shift's size).
4. Streams as E13: per direction and draw, one stream of 2,600 patients (latent via `simulate_latent_u`, observed under B's policy), week t uses the first 25 t; weeks 4, 13, 52, 104; draws 2 (`--draws` to change). Seeds deterministic and disjoint from everything else.
5. Methods per (direction, draw, week): `head_U`, `dial_only`, `head_L`, `finetune_lr05`, `finetune_lr005`; plus `keep` and `oracle` once per direction (copied to each week, as E13). Definitions in the SPEC. Steps 160, minibatch 256 with replacement; `head_U` and `head_L` lr 0.05. Zero outcome cases: labelled methods return A's parameters (and `dial_only` w = 1).
6. Rows: direction, method, week, n_patients, n_cases, draw, brier, log_loss, auroc, regret (Brier minus oracle Brier), w, wall_time_seconds. Write `analysis/e14_runs.csv` and `analysis/e14_summary.csv` (mean, min, max over draws per direction x method x week), and `analysis/e14_shift_size.csv` (direction, keep AUROC, oracle AUROC, keep regret, mean P(u=1|h) for u=1 and u=0 test patients under the oracle).
7. Jobs = (direction, draw, week) in a spawn-safe `multiprocessing` pool (`--workers`, default 4), `torch.set_num_threads(1)` per worker, test data loaded once per worker via an initializer, one progress line per finished job.
8. Flags: `--smoke` (n_train 1,000, n_val 300, 3 epochs, cached under `results_smoke/E14/`; test cohort 500; weeks 4 and 13; 1 draw; 2 Adam steps; outputs `analysis/e14_smoke_*.csv`), `--workers`, `--draws`, `--train-only`.

## `figures/e14_figure.py`

Reads `analysis/e14_summary.csv`, writes `figures/e14_meaning_shift.png` and `.pdf`: two panels (titles: "B's doctors also test on a hidden risk factor (tests mean more)", "A's doctors tested on it, B's do not (tests mean less)"); lines `keep` (grey dashed), `dial_only`, `head_U`, `head_L`, `finetune_lr05`, `finetune_lr005`; mean with min-max band; log x (weeks with a second row of mean cases) and log y (regret). matplotlib only.

## Checks you can do

You cannot import torch or matplotlib. Your Python: `%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` (numpy, pandas). Run `py_compile` on all three files. With numpy only, check: (a) outcome rate near 2% on 20,000 patients; (b) the marginal testing rate of each policy near 0.3 on 20,000 patients; (c) with beta 0, P(u = 1 | h) from `bayes_predict_u` is 0.3 for every patient; (d) with beta 0 and delta 0, `bayes_predict_u` equals `bayes._bayes_predict_grid` at the same threshold to 1e-10 (patch `bayes.THRESHOLD` temporarily for that comparison, or compute the reference inline); (e) under `gain`'s B policy, P(u = 1 | h) separates u = 1 from u = 0 patients (report the two means) and the oracle's mean prediction is near the outcome rate. Keep torch and matplotlib imports inside functions so these checks run. Do not run anything heavy (no training).

## Your answer

Short report: what you built, any departure from the SPEC and why, check results with numbers, exact commands (smoke, train-only, full, figure). Do not paste the scripts.
