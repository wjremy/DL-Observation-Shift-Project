"""E12: re-learn the E5 testing head at B with and without outcomes.

Run from the project folder with ``python e12_unlabelled_head.py``.  The
``--smoke`` mode uses small cohorts and short refits; normal mode follows the
sizes and optimisation steps in SPEC.md.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
import time
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import bayes
import latent_filter
import run_grid
import sim


ROOT = Path(__file__).resolve().parent
PARAMS_PATH = ROOT / "results" / "E5" / "sigma1_gamma2" / "seed0" / "filter" / "params.npz"
SIGMA = 1.0
POLICY_NAMES = ("B_less", "B_more", "B_match", "B_dense")
POOL_SEED_BASE = 1_300_000
TEST_OBS_SEED_BASE = 1_200_000
RUN_COLUMNS = (
    "policy", "method", "size_type", "size", "draw", "brier", "log_loss",
    "auroc", "regret", "w", "wall_time_seconds",
)
CURVE_COLUMNS = (
    "policy", "n_U", "draw", "state_index", "mu", "refitted_probability",
    "true_testing_probability",
)


def _policies() -> Dict[str, sim.Policy]:
    """Build the four target policies specified for E12."""
    return {
        "B_less": sim.informative(gamma=1.0, rate=0.3),
        "B_more": sim.informative(gamma=3.5, rate=0.3),
        "B_match": sim.uniform(rate=0.3),
        "B_dense": sim.uniform(rate=0.6),
    }


def _load_parameters(path: Path | str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]).copy() for name in latent_filter.PARAMETER_NAMES}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def auroc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC from average ranks, assigning average ranks to ties."""
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.size != score.size:
        raise ValueError("y_true and scores must have the same length")
    positives = int(y.sum())
    negatives = y.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    ordered = score[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        stop = start + 1
        while stop < y.size and ordered[stop] == ordered[start]:
            stop += 1
        # Ranks are one-based; all tied items receive their average rank.
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive_rank_sum = float(ranks[y].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _metrics(y: np.ndarray, probability: np.ndarray, oracle_brier: float) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if y.shape != probability.shape:
        raise ValueError("prediction and outcome arrays must have matching shapes")
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((probability - y) ** 2))
    log_loss = float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped)))
    return {
        "brier": brier,
        "log_loss": log_loss,
        "auroc": auroc_rank(y, probability),
        "regret": brier - float(oracle_brier),
    }


def _pool_seeds(policy_index: int, size_type: str, size_index: int, draw: int) -> Tuple[int, int, int]:
    """Return disjoint latent, observation, and minibatch seeds for one pool."""
    type_offset = 0 if size_type == "n_L" else 25_000
    latent_seed = POOL_SEED_BASE + policy_index * 50_000 + type_offset + size_index * 1_000 + draw * 3
    return latent_seed, latent_seed + 1, latent_seed + 2


def _observe_pool(n: int, policy: sim.Policy, latent_seed: int, obs_seed: int):
    latent = sim.simulate_latent(n, seed=latent_seed)
    observed = sim.observe(latent, policy, sigma=SIGMA, seed=obs_seed)
    return observed, np.asarray(latent["y"], dtype=np.float32)


def _choose_dial_weight(x: np.ndarray, m: np.ndarray, y: np.ndarray,
                        parameters: Mapping[str, np.ndarray]) -> float:
    """Choose the labelled run-in log-loss minimiser, breaking ties toward 1."""
    weights = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    losses = []
    for weight in weights:
        probability, _ = latent_filter.numpy_forward(x, m, parameters, w=float(weight))
        clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
        loss = -np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
        losses.append(float(loss))
    best_loss = min(losses)
    return float(max(weight for weight, loss in zip(weights, losses) if loss == best_loss))


def _refit_testing_head(
    observed: Mapping[str, np.ndarray],
    labels: np.ndarray | None,
    parameters: Mapping[str, np.ndarray],
    *,
    batch_seed: int,
    steps: int,
    lr: float = 1e-2,
) -> Dict[str, np.ndarray]:
    """Refit c only, using the E5 likelihood and optional outcome BCE."""
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:  # Keep NumPy checks usable without PyTorch.
        raise ImportError("PyTorch is required for the E12 head refits") from exc

    torch.set_num_threads(1)
    torch.manual_seed(int(batch_seed))
    model = latent_filter.LatentFilter(n_states=int(np.asarray(parameters["mu"]).size))
    with torch.no_grad():
        for name in latent_filter.PARAMETER_NAMES:
            tensor = torch.as_tensor(np.asarray(parameters[name]), dtype=getattr(model, name).dtype)
            getattr(model, name).copy_(tensor)

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "c")
    optimizer = torch.optim.Adam([model.c], lr=float(lr))

    x = torch.from_numpy(np.ascontiguousarray(observed["x"], dtype=np.float32))
    m = torch.from_numpy(np.ascontiguousarray(observed["m"], dtype=np.float32))
    y_tensor = None if labels is None else torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))
    n = int(x.shape[0])
    if n == 0:
        raise ValueError("cannot refit the testing head on an empty pool")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(batch_seed))
    model.train()

    for _ in range(int(steps)):
        # Sampling with replacement keeps every optimizer step at 256 records,
        # including the small labelled run-ins in E12.
        indices = torch.randint(0, n, (256,), generator=generator)
        xb = x.index_select(0, indices)
        mb = m.index_select(0, indices)
        probabilities, log_marginal = model(xb, mb, w=1.0)
        nll_per_step = -log_marginal.mean() / x.shape[1]
        loss = nll_per_step
        if y_tensor is not None:
            yb = y_tensor.index_select(0, indices)
            bce = F.binary_cross_entropy(probabilities.clamp(1e-7, 1.0 - 1e-7), yb)
            loss = bce + nll_per_step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model.numpy_parameters()


_WORKER_TEST_OBS: Dict[str, Mapping[str, np.ndarray]] = {}
_WORKER_TEST_Y: np.ndarray | None = None
_WORKER_PARAMS_PATH: str | None = None
_WORKER_POLICIES: Dict[str, sim.Policy] = {}
_WORKER_ORACLE_BRIER: Dict[str, float] = {}


def _init_worker(
    test_observations: Dict[str, Mapping[str, np.ndarray]],
    test_y: np.ndarray,
    params_path: str,
    oracle_brier: Dict[str, float],
) -> None:
    global _WORKER_TEST_OBS, _WORKER_TEST_Y, _WORKER_PARAMS_PATH, _WORKER_POLICIES
    global _WORKER_ORACLE_BRIER
    _WORKER_TEST_OBS = test_observations
    _WORKER_TEST_Y = test_y
    _WORKER_PARAMS_PATH = params_path
    _WORKER_POLICIES = _policies()
    _WORKER_ORACLE_BRIER = oracle_brier


def _run_refit_job(task: Mapping[str, object]) -> Dict[str, object]:
    """Run one policy/size/draw job and return metric rows plus any curve."""
    if _WORKER_TEST_Y is None or _WORKER_PARAMS_PATH is None:
        raise RuntimeError("worker was not initialised")
    started = time.perf_counter()
    policy_name = str(task["policy"])
    policy_index = int(task["policy_index"])
    size_type = str(task["size_type"])
    size_index = int(task["size_index"])
    size = int(task["size"])
    draw = int(task["draw"])
    policy = _WORKER_POLICIES[policy_name]
    latent_seed, obs_seed, batch_seed = _pool_seeds(policy_index, size_type, size_index, draw)
    pool, pool_y = _observe_pool(size, policy, latent_seed, obs_seed)
    test_observed = _WORKER_TEST_OBS[policy_name]
    test_y = _WORKER_TEST_Y
    parameters = _load_parameters(_WORKER_PARAMS_PATH)
    oracle_brier = _WORKER_ORACLE_BRIER[policy_name]
    rows: List[Dict[str, object]] = []
    curves: List[Dict[str, object]] = []

    if size_type == "n_L":
        method_started = time.perf_counter()
        dial_w = _choose_dial_weight(pool["x"], pool["m"], pool_y, parameters)
        dial_probability, _ = latent_filter.numpy_forward(
            test_observed["x"], test_observed["m"], parameters, w=dial_w
        )
        dial_elapsed = time.perf_counter() - method_started
        rows.append({
            "policy": policy_name,
            "method": "dial_L",
            "size_type": size_type,
            "size": size,
            "draw": draw,
            **_metrics(test_y, dial_probability, oracle_brier),
            "w": dial_w,
            "wall_time_seconds": dial_elapsed,
        })

        method_started = time.perf_counter()
        refitted = _refit_testing_head(
            pool, pool_y, parameters, batch_seed=batch_seed, steps=int(task["steps"]),
            lr=float(task["lr"])
        )
        head_probability, _ = latent_filter.numpy_forward(
            test_observed["x"], test_observed["m"], refitted, w=1.0
        )
        head_elapsed = time.perf_counter() - method_started
        rows.append({
            "policy": policy_name,
            "method": "head_L",
            "size_type": size_type,
            "size": size,
            "draw": draw,
            **_metrics(test_y, head_probability, oracle_brier),
            "w": 1.0,
            "wall_time_seconds": head_elapsed,
        })
    elif size_type == "n_U":
        method_started = time.perf_counter()
        refitted = _refit_testing_head(
            pool, None, parameters, batch_seed=batch_seed, steps=int(task["steps"]),
            lr=float(task["lr"])
        )
        head_probability, _ = latent_filter.numpy_forward(
            test_observed["x"], test_observed["m"], refitted, w=1.0
        )
        head_elapsed = time.perf_counter() - method_started
        rows.append({
            "policy": policy_name,
            "method": "head_U",
            "size_type": size_type,
            "size": size,
            "draw": draw,
            **_metrics(test_y, head_probability, oracle_brier),
            "w": 1.0,
            "wall_time_seconds": head_elapsed,
        })
        if size == int(task["curve_size"]) and draw == 0:
            mu = np.asarray(refitted["mu"], dtype=np.float64)
            refitted_probability = _sigmoid(np.asarray(refitted["c"], dtype=np.float64))
            if policy.name.startswith("informative"):
                true_probability = _sigmoid(policy.alpha + policy.gamma * mu)
            else:
                true_probability = np.full(mu.shape, policy.rate, dtype=np.float64)
            curves = [
                {
                    "policy": policy_name,
                    "n_U": size,
                    "draw": draw,
                    "state_index": index,
                    "mu": float(mu[index]),
                    "refitted_probability": float(refitted_probability[index]),
                    "true_testing_probability": float(true_probability[index]),
                }
                for index in range(mu.size)
            ]
    else:
        raise ValueError(f"unknown size_type: {size_type}")

    return {
        "policy": policy_name,
        "size_type": size_type,
        "size": size,
        "draw": draw,
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "curves": curves,
    }


def _baseline_rows(
    test_observations: Mapping[str, Mapping[str, np.ndarray]],
    test_y: np.ndarray,
    parameters: Mapping[str, np.ndarray],
    policies: Mapping[str, sim.Policy],
    size_specs: Sequence[Tuple[str, Sequence[int]]],
    draws: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Compute keep/drop/oracle once per policy, then attach rows to each size."""
    rows: List[Dict[str, object]] = []
    oracle_briers: Dict[str, float] = {}
    for policy_name in POLICY_NAMES:
        observed = test_observations[policy_name]
        policy = policies[policy_name]
        baseline_predictions = {}
        timings = {}
        for method, weight in (("keep", 1.0), ("drop", 0.0)):
            started = time.perf_counter()
            baseline_predictions[method], _ = latent_filter.numpy_forward(
                observed["x"], observed["m"], parameters, w=weight
            )
            timings[method] = time.perf_counter() - started
        started = time.perf_counter()
        baseline_predictions["oracle"] = bayes.bayes_predict(
            observed["x"], observed["m"], SIGMA, policy
        )
        timings["oracle"] = time.perf_counter() - started
        baseline_metrics = {
            method: _metrics(test_y, probability, 0.0)
            for method, probability in baseline_predictions.items()
        }
        oracle_brier = baseline_metrics["oracle"]["brier"]
        oracle_briers[policy_name] = oracle_brier
        for size_type, sizes in size_specs:
            for size in sizes:
                for draw in range(draws):
                    for method in ("keep", "drop", "oracle"):
                        weight = {"keep": 1.0, "drop": 0.0, "oracle": float("nan")}[method]
                        rows.append({
                            "policy": policy_name,
                            "method": method,
                            "size_type": size_type,
                            "size": int(size),
                            "draw": draw,
                            **{
                                **baseline_metrics[method],
                                "regret": baseline_metrics[method]["brier"] - oracle_brier,
                            },
                            "w": weight,
                            "wall_time_seconds": timings[method],
                        })
    return rows, oracle_briers


def _write_outputs(rows: Sequence[Mapping[str, object]], curves: Sequence[Mapping[str, object]],
                   *, smoke: bool, tag: str = "") -> Tuple[Path, Path, Path]:
    analysis_dir = ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    prefix = ("e12_smoke_" if smoke else "e12_") + (f"{tag}_" if tag else "")
    runs_path = analysis_dir / f"{prefix}runs.csv"
    summary_path = analysis_dir / f"{prefix}summary.csv"
    curves_path = analysis_dir / f"{prefix}c_curves.csv"

    runs = pd.DataFrame(rows, columns=RUN_COLUMNS)
    runs = runs.sort_values(
        ["policy", "size_type", "size", "draw", "method"], kind="mergesort"
    ).reset_index(drop=True)
    runs.to_csv(runs_path, index=False)

    group_columns = ["policy", "method", "size_type", "size"]
    summary_metrics = ["brier", "log_loss", "auroc", "regret", "w", "wall_time_seconds"]
    summary = runs.groupby(group_columns, dropna=False)[summary_metrics].agg(["mean", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(group_columns, kind="mergesort")
    summary.to_csv(summary_path, index=False)

    pd.DataFrame(curves, columns=CURVE_COLUMNS).sort_values(
        ["policy", "state_index"], kind="mergesort"
    ).to_csv(curves_path, index=False)
    return runs_path, summary_path, curves_path


def run(*, workers: int = 4, smoke: bool = False, steps: int | None = None,
        lr: float = 1e-2, tag: str = "") -> Tuple[Path, Path, Path]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"E5 parameter file not found: {PARAMS_PATH}")
    parameters = _load_parameters(PARAMS_PATH)
    policies = _policies()
    n_test = 500 if smoke else 8_000
    labelled_sizes = (30,) if smoke else (30, 125, 500)
    unlabelled_sizes = (100,) if smoke else (200, 2_000)
    draws = 1 if smoke else 3
    if steps is None:
        steps = 2 if smoke else 80
    size_specs = (("n_L", labelled_sizes), ("n_U", unlabelled_sizes))

    # E5's test latent cohort is reused, while every B policy gets a fresh,
    # fixed observation draw as specified for E12.
    test_latent_seed = run_grid._seeds_for(0)[2]
    test_latent = sim.simulate_latent(n_test, seed=test_latent_seed)
    test_y = np.asarray(test_latent["y"], dtype=np.float32)
    test_observations = {
        policy_name: sim.observe(
            test_latent,
            policies[policy_name],
            sigma=SIGMA,
            seed=TEST_OBS_SEED_BASE + policy_index,
        )
        for policy_index, policy_name in enumerate(POLICY_NAMES)
    }

    rows, oracle_briers = _baseline_rows(
        test_observations, test_y, parameters, policies, size_specs, draws
    )
    tasks = []
    max_unlabelled_size = max(unlabelled_sizes)
    for policy_index, policy_name in enumerate(POLICY_NAMES):
        for size_type, sizes in size_specs:
            for size_index, size in enumerate(sizes):
                for draw in range(draws):
                    tasks.append({
                        "policy": policy_name,
                        "policy_index": policy_index,
                        "size_type": size_type,
                        "size_index": size_index,
                        "size": int(size),
                        "draw": draw,
                        "steps": steps,
                        "lr": lr,
                        "curve_size": max_unlabelled_size,
                    })

    context = mp.get_context("spawn")
    curves: List[Dict[str, object]] = []
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(test_observations, test_y, str(PARAMS_PATH), oracle_briers),
    ) as pool:
        for result in pool.imap_unordered(_run_refit_job, tasks, chunksize=1):
            rows.extend(result["rows"])
            curves.extend(result["curves"])
            print(
                f"DONE {result['policy']} {result['size_type']}={result['size']} "
                f"draw={result['draw']} wall={result['elapsed_seconds']:.1f}s",
                flush=True,
            )

    outputs = _write_outputs(rows, curves, smoke=smoke, tag=tag)
    for path in outputs:
        print(f"WROTE {path.relative_to(ROOT)}", flush=True)
    return outputs


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=None, help="Adam steps per refit (default 80; smoke 2)")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tag", default="", help="suffix for output file names")
    args = parser.parse_args(argv)
    run(workers=args.workers, smoke=args.smoke, steps=args.steps, lr=args.lr, tag=args.tag)


if __name__ == "__main__":
    mp.freeze_support()
    main()
