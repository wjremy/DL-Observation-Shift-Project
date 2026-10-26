"""E14b: a dial allowed above 1, alone and on top of head_U, in the E14 directions.

Same A models, streams, seeds and test cohort as E14. The dial's weight is
chosen on the deployment stream's labelled patients by the E13 rule (summed log
loss + 2 * (w - 1)^2), now on the grid 0, 0.05, ..., 2. ``*_best`` rows choose
w on the test cohort instead: the most any single dial could gain.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

import e12_unlabelled_head as e12
import e14_meaning_shift as e14
import latent_filter
import sim_u

WEIGHTS = np.round(np.arange(0.0, 2.0001, 0.05), 2)


def _choose_weight(x, m, y, parameters) -> float:
    """E13's penalised rule on the wider grid; no case -> w = 1."""
    if int(np.asarray(y).sum()) == 0:
        return 1.0
    labels = np.asarray(y, dtype=np.float64)
    losses = []
    for weight in WEIGHTS:
        probability, _ = latent_filter.numpy_forward(x, m, parameters, w=float(weight))
        clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
        log_loss_sum = -np.sum(labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped))
        losses.append(float(log_loss_sum + 2.0 * (float(weight) - 1.0) ** 2))
    best = min(losses)
    # Ties go to the weight closest to 1.
    return float(min((w for w, loss in zip(WEIGHTS, losses) if loss == best), key=lambda w: abs(w - 1.0)))


def _best_weight_on_test(observed, y, parameters, oracle_brier):
    """Weight with the lowest test Brier, and its metrics (upper bound for a dial)."""
    best = None
    for weight in WEIGHTS:
        probability, _ = latent_filter.numpy_forward(observed["x"], observed["m"], parameters, w=float(weight))
        metrics = e12._metrics(y, probability, oracle_brier)
        if best is None or metrics["brier"] < best[1]["brier"]:
            best = (float(weight), metrics)
    return best


def _row(direction, method, week, n_patients, n_cases, draw, metrics, w, started):
    return {
        "direction": direction, "method": method, "week": week, "n_patients": n_patients,
        "n_cases": n_cases, "draw": draw, **metrics, "w": w,
        "wall_time_seconds": time.perf_counter() - started,
    }


def _run_job(task: Mapping[str, object]) -> List[Dict[str, object]]:
    direction = str(task["direction"])
    draw = int(task["draw"])
    week = int(task["week"])
    n_patients = e14.N_PER_WEEK * week
    latent_seed, observation_seed, batch_seed = e14._stream_seeds(int(task["direction_index"]), draw)
    stream_latent = sim_u.simulate_latent_u(e14.STREAM_N, seed=latent_seed)
    stream_all = sim_u.observe_u(
        stream_latent, e14._WORKER_B_POLICIES[direction], sigma=e14.SIGMA, seed=observation_seed
    )
    stream = {
        "x": np.asarray(stream_all["x"][:n_patients], dtype=np.float32),
        "m": np.asarray(stream_all["m"][:n_patients], dtype=np.int8),
    }
    stream_y = np.asarray(stream_latent["y"][:n_patients], dtype=np.float32)
    n_cases = int(stream_y.sum())
    observed = e14._WORKER_TEST_OBS[direction]
    test_y = e14._WORKER_TEST_Y
    oracle_brier = e14._WORKER_ORACLE_BRIER[direction]
    parameters = e14._WORKER_PARAMS[direction]
    rows = []

    # Same call and batch seed as E14's head_U, so these rows must match E14.
    started = time.perf_counter()
    head_u = e12._refit_testing_head(stream, None, parameters, batch_seed=batch_seed, steps=e14._WORKER_STEPS, lr=0.05)
    prediction, _ = latent_filter.numpy_forward(observed["x"], observed["m"], head_u, w=1.0)
    rows.append(_row(direction, "head_U", week, n_patients, 0, draw, e12._metrics(test_y, prediction, oracle_brier), 1.0, started))

    for method, base in (("head_U_dial2", head_u), ("dial2_only", parameters)):
        started = time.perf_counter()
        weight = _choose_weight(stream["x"], stream["m"], stream_y, base)
        prediction, _ = latent_filter.numpy_forward(observed["x"], observed["m"], base, w=weight)
        rows.append(_row(direction, method, week, n_patients, n_cases, draw, e12._metrics(test_y, prediction, oracle_brier), weight, started))

    started = time.perf_counter()
    weight, metrics = _best_weight_on_test(observed, test_y, head_u, oracle_brier)
    rows.append(_row(direction, "head_U_dial_best", week, n_patients, 0, draw, metrics, weight, started))
    return rows


def run(*, workers: int, draws: int) -> None:
    policies = e14._direction_policies()
    params_paths = {d: e14._train_or_load_a_model(d, False, policies[d]["A"])[0] for d in e14.DIRECTIONS}
    test_y, _, test_observations, oracle_predictions, _ = e14._make_test_cohort(10_000, policies, params_paths)
    rows: List[Dict[str, object]] = []
    for direction in e14.DIRECTIONS:
        params = e14._load_a_parameters(params_paths[direction])
        oracle_brier = float(np.mean((oracle_predictions[direction] - test_y) ** 2))
        started = time.perf_counter()
        weight, metrics = _best_weight_on_test(test_observations[direction], test_y, params, oracle_brier)
        rows.append(_row(direction, "dial_best", 0, 0, 0, 0, metrics, weight, started))
    tasks = [
        {"direction": d, "direction_index": i, "draw": draw, "week": week}
        for i, d in enumerate(e14.DIRECTIONS) for draw in range(draws) for week in e14.WEEKS
    ]
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers, initializer=e14._init_worker,
        initargs=(test_observations, test_y, oracle_predictions,
                  {d: str(p) for d, p in params_paths.items()},
                  {d: policies[d]["B"] for d in e14.DIRECTIONS}, 160),
    ) as pool:
        for job_rows in pool.imap_unordered(_run_job, tasks, chunksize=1):
            rows.extend(job_rows)
            first = job_rows[0]
            print(f"DONE {first['direction']} draw={first['draw']} week={first['week']}", flush=True)
    path = e14.ROOT / "analysis" / "e14b_runs.csv"
    pd.DataFrame(rows).sort_values(["direction", "week", "method", "draw"]).to_csv(path, index=False)
    print(f"WROTE {path}", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--draws", type=int, default=2)
    args = parser.parse_args(argv)
    run(workers=args.workers, draws=args.draws)


if __name__ == "__main__":
    mp.freeze_support()
    main()
