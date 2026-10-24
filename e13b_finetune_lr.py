"""E13b: rerun only the E13 full fine-tune baseline at smaller learning rates.

Same A model, streams, seeds, test cohort and objective as E13; only the
fine-tune learning rate changes. Quick version by default (weeks 4, 13, 52,
104; draws 0-1); ``--full`` uses all E13 weeks and 5 draws.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

# e13 first: importing it patches the outcome threshold to 2.05 (2% outcome).
import e13_rungs_over_time as e13
import e12_unlabelled_head as e12
import latent_filter
import sim

QUICK_WEEKS = (4, 13, 52, 104)
QUICK_DRAWS = 2


def _run_job(task: Mapping[str, object]) -> Dict[str, object]:
    started = time.perf_counter()
    policy_name = str(task["policy"])
    draw = int(task["draw"])
    week = int(task["week"])
    lr = float(task["lr"])
    n_patients = e13.N_PER_WEEK * week
    latent_seed, observation_seed, batch_seed = e13._stream_seeds(int(task["policy_index"]), draw)
    stream_latent = sim.simulate_latent(e13.STREAM_N, seed=latent_seed)
    stream_all = sim.observe(
        stream_latent, e13._WORKER_POLICIES[policy_name], sigma=e13.SIGMA, seed=observation_seed
    )
    stream = {
        "x": np.asarray(stream_all["x"][:n_patients], dtype=np.float32),
        "m": np.asarray(stream_all["m"][:n_patients], dtype=np.int8),
    }
    stream_y = np.asarray(stream_latent["y"][:n_patients], dtype=np.float32)
    # Same batch seed as the E13 fine-tune, so lr is the only difference.
    params = e13._finetune(
        stream, stream_y, e13._WORKER_PARAMS,
        batch_seed=batch_seed + 17, steps=e13._WORKER_STEPS, lr=lr,
    )
    observed = e13._WORKER_TEST_OBS[policy_name]
    prediction, _ = latent_filter.numpy_forward(observed["x"], observed["m"], params, w=1.0)
    return {
        "policy": policy_name, "method": "finetune", "lr": lr, "week": week,
        "n_patients": n_patients, "n_cases": int(stream_y.sum()), "draw": draw,
        **e12._metrics(e13._WORKER_TEST_Y, prediction, e13._WORKER_ORACLE_BRIER[policy_name]),
        "wall_time_seconds": time.perf_counter() - started,
    }


def run(*, lrs: Iterable[float], full: bool, workers: int, steps: int) -> None:
    params_path, _ = e13._train_or_load_a_model(False)
    _, test_observations, test_y, oracle_predictions = e13._make_test_cohort(10_000)
    weeks = e13.WEEKS if full else QUICK_WEEKS
    draws = 5 if full else QUICK_DRAWS
    tasks = [
        {"policy": name, "policy_index": index, "draw": draw, "week": week, "lr": lr}
        for lr in lrs
        for index, name in enumerate(e13.POLICY_NAMES)
        for draw in range(draws)
        for week in weeks
    ]
    rows: List[Dict[str, object]] = []
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=e13._init_worker,
        initargs=(test_observations, test_y, oracle_predictions, str(params_path), 0.05, steps),
    ) as pool:
        for row in pool.imap_unordered(_run_job, tasks, chunksize=1):
            rows.append(row)
            print(
                f"DONE {row['policy']} lr={row['lr']} draw={row['draw']} "
                f"week={row['week']} wall={row['wall_time_seconds']:.1f}s",
                flush=True,
            )
    prefix = "e13b_full" if full else "e13b_quick"
    frame = pd.DataFrame(rows).sort_values(["policy", "lr", "week", "draw"])
    runs_path = e13.ROOT / "analysis" / f"{prefix}_runs.csv"
    frame.to_csv(runs_path, index=False)
    print(f"WROTE {runs_path}", flush=True)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.005, 0.001])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=160)
    args = parser.parse_args(argv)
    run(lrs=args.lrs, full=args.full, workers=args.workers, steps=args.steps)


if __name__ == "__main__":
    mp.freeze_support()
    main()
