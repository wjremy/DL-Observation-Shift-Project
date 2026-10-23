# Task: write `e12_unlabelled_head.py` (experiment E12)

You write one Python script in the project folder (your working directory). Read first:

- `SPEC.md`, section "E12: re-learning the testing head at B without outcomes" (the binding design; follow it exactly).
- `latent_filter.py` (the E5 filter: `LatentFilter` torch module, `numpy_forward`, parameter names `transition_logits, initial_logits, mu, log_sd, c, u`).
- `sim.py` (`simulate_latent`, `observe`, `informative`, `uniform`), `run_grid.py` (`_seeds_for`, `_policy_setup`), `bayes.py` (`bayes_predict(x, m, sigma, assumed_policy)`).
- The trained A parameters: `results/E5/sigma1_gamma2/seed0/filter/params.npz`.

## What the script does

1. Load the A parameters. Build the four B policies of the SPEC.
2. Test cohort: `sim.simulate_latent(8000, seed=<E5 test latent seed from run_grid._seeds_for(0)>)`; per policy observe with a new fixed observation seed (e.g. 1_200_000 + policy index). Oracle predictions with `bayes.bayes_predict` under that policy.
3. For each policy, size and draw (3 draws), draw fresh pools from new latent and observation seeds (deterministic from policy, size, draw; disjoint from the test seed and from E1/E5 seeds, e.g. base 1_300_000), and run the methods `keep`, `drop`, `dial_L`, `head_L`, `head_U` as the SPEC defines. `keep`, `drop`, `dial_L` need no torch: use `numpy_forward`. `head_L` and `head_U`: load the A parameters into `LatentFilter`, set `requires_grad=False` on every parameter except `c`, Adam on `c` only (lr 1e-2, minibatch 256 drawn with a seeded generator, 80 steps). `head_U` loss: `-log_marginal.mean() / T`. `head_L` loss: BCE + `-log_marginal.mean() / T`, as `train_filter` does. Evaluate the refitted model on the test cohort with `numpy_forward` and the refitted `c`.
   - `keep`, `drop` and `oracle` do not depend on size or draw: compute once per policy.
   - `dial_L` and `head_L` use the labelled sizes n_L; `head_U` uses the unlabelled sizes n_U. Record which size type each row has.
4. Metrics on the test cohort: Brier, log loss (clip 1e-7), AUROC (write a small rank-based AUROC with ties averaged; no sklearn), regret = Brier minus oracle Brier. Record the chosen w for `dial_L` and wall time per fit.
5. For `head_U` at n_U = 2000, draw 0: save the refitted `c` (sigmoid of it, i.e. test probability per state, with the state means `mu`) next to the true B testing probability `sigmoid(alpha + gamma * mu)` (uniform policies: the constant rate).
6. Write `analysis/e12_runs.csv` (one row per policy x method x size x draw), `analysis/e12_summary.csv` (mean and min/max over draws of each metric per policy x method x size), `analysis/e12_c_curves.csv`.

## Engineering requirements

- `--smoke` flag: test cohort 500, sizes n_L = 30, n_U = 100, 1 draw, 2 Adam steps, output to `analysis/e12_smoke_*.csv`.
- `--workers N` (default 4): run the (policy, size, draw) refit jobs in a `multiprocessing` pool (spawn-safe: `if __name__ == "__main__":`), each worker with `torch.set_num_threads(1)`. Print one progress line per finished job with elapsed seconds.
- Deterministic given seeds. No new dependencies beyond numpy, pandas, torch (already used in the project).
- Do not modify any existing file. Do not run anything heavy.

## Checks you can do

You cannot import torch in your sandbox. Python available to you: `%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` (numpy and pandas only). Use it to: `py_compile` the script; and run a numpy-only check that `numpy_forward` with the A parameters on a small B sample gives probabilities in (0, 1) and that your AUROC function matches a brute-force pairwise AUROC on random data. Structure the code so the torch import is inside the refit functions or guarded, so these checks can run.

## Your answer

Short report: what the script does, any place you departed from the SPEC and why, the check results, and the exact command to run it (smoke and full). Do not paste the script.
