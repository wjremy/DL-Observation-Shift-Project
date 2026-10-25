"""E14: outcome-only meaning shifts and adaptation over deployment weeks."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import sim
import sim_u
import e12_unlabelled_head as e12
import e13_rungs_over_time as e13
import latent_filter


ROOT = Path(__file__).resolve().parent
SIGMA = 1.0
DIRECTIONS = ("gain", "lose")
WEEKS = (4, 13, 52, 104)
SMOKE_WEEKS = (4, 13)
N_PER_WEEK = 25
STREAM_N = 2_600

# E14's seed ranges are disjoint from E12 and E13. The latent test cohort is
# shared by both directions; each direction receives its own B observation draw.
A_TRAIN_LATENT_SEED = 1_700_000
A_VAL_LATENT_SEED = 1_701_000
A_TRAIN_OBS_SEED = 1_702_000
A_VAL_OBS_SEED = 1_703_000
A_MODEL_SEED = 1_704_337
TEST_LATENT_SEED = 1_750_000
TEST_OBS_SEED_BASE = 1_760_000
STREAM_SEED_BASE = 1_800_000

RUN_COLUMNS = (
    "direction", "method", "week", "n_patients", "n_cases", "draw",
    "brier", "log_loss", "auroc", "regret", "w", "wall_time_seconds",
)
SUMMARY_METRICS = (
    "n_patients", "n_cases", "brier", "log_loss", "auroc", "regret",
    "w", "wall_time_seconds",
)
METHOD_ORDER = {
    "keep": 0,
    "head_U": 1,
    "dial_only": 2,
    "head_L": 3,
    "finetune_lr05": 4,
    "finetune_lr005": 5,
    "oracle": 6,
}


def _direction_policies() -> Dict[str, Dict[str, sim_u.UPolicy]]:
    return {
        "gain": {
            "A": sim_u.u_informative(gamma=2.0, beta=0.0),
            "B": sim_u.u_informative(gamma=2.0, beta=2.0),
        },
        "lose": {
            "A": sim_u.u_informative(gamma=2.0, beta=2.0),
            "B": sim_u.u_informative(gamma=2.0, beta=0.0),
        },
    }


def _a_model_paths(direction: str, smoke: bool) -> Tuple[Path, Path]:
    root = ROOT / ("results_smoke" if smoke else "results") / "E14"
    folder = root / f"a_model_{direction}"
    return folder / "params.npz", folder / "meta.json"


def _load_a_parameters(path: Path | str) -> Dict[str, np.ndarray]:
    return e12._load_parameters(path)


def _train_or_load_a_model(
    direction: str, smoke: bool, policy: sim_u.UPolicy
) -> Tuple[Path, Dict[str, object]]:
    n_train, n_val, max_epochs = (1_000, 300, 3) if smoke else (16_000, 4_000, 200)
    params_path, meta_path = _a_model_paths(direction, smoke)
    params_path.parent.mkdir(parents=True, exist_ok=True)

    train_latent = sim_u.simulate_latent_u(n_train, seed=A_TRAIN_LATENT_SEED)
    train_y = np.asarray(train_latent["y"], dtype=np.float32)
    training_rate = float(train_y.mean())
    assert 0.01 <= training_rate <= 0.03, (
        f"E14 {direction} A training outcome rate must be 1%-3%; "
        f"got {training_rate:.4%}"
    )

    if params_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        print(f"SKIP A model training (cached at {params_path.parent})", flush=True)
        return params_path, metadata

    if latent_filter.torch is None:
        raise ImportError("PyTorch is required to train the E14 A filter")
    started = time.perf_counter()
    train_observed = sim_u.observe_u(
        train_latent, policy, sigma=SIGMA, seed=A_TRAIN_OBS_SEED
    )
    val_latent = sim_u.simulate_latent_u(n_val, seed=A_VAL_LATENT_SEED)
    val_observed = sim_u.observe_u(
        val_latent, policy, sigma=SIGMA, seed=A_VAL_OBS_SEED
    )
    val_y = np.asarray(val_latent["y"], dtype=np.float32)
    model, fit_meta = latent_filter.train_filter(
        train_observed["x"], train_observed["m"], train_y,
        val_observed["x"], val_observed["m"], val_y,
        n_states=32,
        max_epochs=max_epochs,
        batch_size=512,
        patience=15,
        seed=A_MODEL_SEED,
    )
    wall_time = time.perf_counter() - started
    metadata: Dict[str, object] = {
        "experiment": "E14",
        "direction": direction,
        "model": "latent_filter",
        "sigma": SIGMA,
        "training_policy": policy.name,
        "threshold": sim_u.outcome_threshold(),
        "outcome_rate": training_rate,
        "cases": int(train_y.sum()),
        "n_train": n_train,
        "n_val": n_val,
        "n_train_cases": int(train_y.sum()),
        "train_outcome_rate": training_rate,
        "n_val_cases": int(val_y.sum()),
        "train_latent_seed": A_TRAIN_LATENT_SEED,
        "val_latent_seed": A_VAL_LATENT_SEED,
        "train_observation_seed": A_TRAIN_OBS_SEED,
        "val_observation_seed": A_VAL_OBS_SEED,
        "model_seed": A_MODEL_SEED,
        "best_epoch": fit_meta["best_epoch"],
        "epochs": fit_meta["epochs"],
        "train_loss": fit_meta["train_loss"],
        "val_loss": fit_meta["val_loss"],
        "train_bce": fit_meta["train_bce"],
        "val_bce": fit_meta["val_bce"],
        "train_nll_per_step": fit_meta["train_nll_per_step"],
        "val_nll_per_step": fit_meta["val_nll_per_step"],
        "wall_time_seconds": wall_time,
    }
    latent_filter._atomic_npz(params_path, model.numpy_parameters())
    latent_filter._atomic_json(meta_path, metadata)
    print(
        f"DONE A model {direction} at {params_path.parent} "
        f"rate={training_rate:.3%} best_epoch={fit_meta['best_epoch']} "
        f"wall={wall_time:.1f}s",
        flush=True,
    )
    return params_path, metadata


def _stream_seeds(direction_index: int, draw: int) -> Tuple[int, int, int]:
    base = STREAM_SEED_BASE + direction_index * 100_000 + draw * 1_000
    return base, base + 1, base + 2


def _copy_parameters(parameters: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        name: np.asarray(parameters[name]).copy()
        for name in latent_filter.PARAMETER_NAMES
    }


_WORKER_TEST_OBS: Dict[str, Mapping[str, np.ndarray]] = {}
_WORKER_TEST_Y: np.ndarray | None = None
_WORKER_ORACLE: Dict[str, np.ndarray] = {}
_WORKER_ORACLE_BRIER: Dict[str, float] = {}
_WORKER_PARAMS: Dict[str, Dict[str, np.ndarray]] = {}
_WORKER_B_POLICIES: Dict[str, sim_u.UPolicy] = {}
_WORKER_STEPS = 160


def _init_worker(
    test_observations: Dict[str, Mapping[str, np.ndarray]],
    test_y: np.ndarray,
    oracle_predictions: Dict[str, np.ndarray],
    params_paths: Dict[str, str],
    b_policies: Dict[str, sim_u.UPolicy],
    steps: int,
) -> None:
    global _WORKER_TEST_OBS, _WORKER_TEST_Y, _WORKER_ORACLE
    global _WORKER_ORACLE_BRIER, _WORKER_PARAMS, _WORKER_B_POLICIES, _WORKER_STEPS
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to run E14 adaptation jobs") from exc
    torch.set_num_threads(1)
    _WORKER_TEST_OBS = test_observations
    _WORKER_TEST_Y = np.asarray(test_y)
    _WORKER_ORACLE = oracle_predictions
    _WORKER_ORACLE_BRIER = {
        direction: float(np.mean((prediction - _WORKER_TEST_Y) ** 2))
        for direction, prediction in oracle_predictions.items()
    }
    _WORKER_PARAMS = {
        direction: _load_a_parameters(path)
        for direction, path in params_paths.items()
    }
    _WORKER_B_POLICIES = b_policies
    _WORKER_STEPS = int(steps)


def _run_job(task: Mapping[str, object]) -> Dict[str, object]:
    if _WORKER_TEST_Y is None:
        raise RuntimeError("E14 worker was not initialized")
    started = time.perf_counter()
    direction = str(task["direction"])
    direction_index = int(task["direction_index"])
    draw = int(task["draw"])
    week = int(task["week"])
    n_patients = N_PER_WEEK * week
    latent_seed, observation_seed, batch_seed = _stream_seeds(direction_index, draw)

    stream_latent = sim_u.simulate_latent_u(STREAM_N, seed=latent_seed)
    stream_all = sim_u.observe_u(
        stream_latent,
        _WORKER_B_POLICIES[direction],
        sigma=SIGMA,
        seed=observation_seed,
    )
    stream = {
        "x": np.asarray(stream_all["x"][:n_patients], dtype=np.float32),
        "m": np.asarray(stream_all["m"][:n_patients], dtype=np.int8),
    }
    stream_y = np.asarray(stream_latent["y"][:n_patients], dtype=np.float32)
    n_cases = int(stream_y.sum())
    test_observed = _WORKER_TEST_OBS[direction]
    test_y = _WORKER_TEST_Y
    oracle_brier = _WORKER_ORACLE_BRIER[direction]
    parameters = _WORKER_PARAMS[direction]
    rows: List[Dict[str, object]] = []

    method_started = time.perf_counter()
    head_u_params = e12._refit_testing_head(
        stream,
        None,
        parameters,
        batch_seed=batch_seed,
        steps=_WORKER_STEPS,
        lr=0.05,
    )
    head_u_prediction, _ = latent_filter.numpy_forward(
        test_observed["x"], test_observed["m"], head_u_params, w=1.0
    )
    rows.append({
        "direction": direction,
        "method": "head_U",
        "week": week,
        "n_patients": n_patients,
        "n_cases": 0,
        "draw": draw,
        **e12._metrics(test_y, head_u_prediction, oracle_brier),
        "w": 1.0,
        "wall_time_seconds": time.perf_counter() - method_started,
    })

    method_started = time.perf_counter()
    dial_weight = e13._choose_penalized_weight(
        stream["x"], stream["m"], stream_y, parameters
    )
    dial_prediction, _ = latent_filter.numpy_forward(
        test_observed["x"], test_observed["m"], parameters, w=dial_weight
    )
    rows.append({
        "direction": direction,
        "method": "dial_only",
        "week": week,
        "n_patients": n_patients,
        "n_cases": n_cases,
        "draw": draw,
        **e12._metrics(test_y, dial_prediction, oracle_brier),
        "w": dial_weight,
        "wall_time_seconds": time.perf_counter() - method_started,
    })

    method_started = time.perf_counter()
    if n_cases == 0:
        head_l_params = _copy_parameters(parameters)
    else:
        head_l_params = e12._refit_testing_head(
            stream,
            stream_y,
            parameters,
            batch_seed=batch_seed + 7,
            steps=_WORKER_STEPS,
            lr=0.05,
        )
    head_l_prediction, _ = latent_filter.numpy_forward(
        test_observed["x"], test_observed["m"], head_l_params, w=1.0
    )
    rows.append({
        "direction": direction,
        "method": "head_L",
        "week": week,
        "n_patients": n_patients,
        "n_cases": n_cases,
        "draw": draw,
        **e12._metrics(test_y, head_l_prediction, oracle_brier),
        "w": 1.0,
        "wall_time_seconds": time.perf_counter() - method_started,
    })

    for method, lr in (("finetune_lr05", 0.05), ("finetune_lr005", 0.005)):
        method_started = time.perf_counter()
        finetuned = e13._finetune(
            stream,
            stream_y,
            parameters,
            batch_seed=batch_seed + 17,
            steps=_WORKER_STEPS,
            lr=lr,
        )
        prediction, _ = latent_filter.numpy_forward(
            test_observed["x"], test_observed["m"], finetuned, w=1.0
        )
        rows.append({
            "direction": direction,
            "method": method,
            "week": week,
            "n_patients": n_patients,
            "n_cases": n_cases,
            "draw": draw,
            **e12._metrics(test_y, prediction, oracle_brier),
            "w": 1.0,
            "wall_time_seconds": time.perf_counter() - method_started,
        })

    return {
        "direction": direction,
        "week": week,
        "draw": draw,
        "rows": rows,
        "elapsed": time.perf_counter() - started,
    }


def _make_test_cohort(
    n_test: int,
    policies: Dict[str, Dict[str, sim_u.UPolicy]],
    params_paths: Dict[str, Path],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[str, Mapping[str, np.ndarray]],
    Dict[str, np.ndarray],
    Dict[str, Dict[str, float]],
]:
    latent = sim_u.simulate_latent_u(n_test, seed=TEST_LATENT_SEED)
    test_y = np.asarray(latent["y"], dtype=np.float32)
    test_u = np.asarray(latent["u"], dtype=np.int8)
    threshold = sim_u.outcome_threshold()
    observations: Dict[str, Mapping[str, np.ndarray]] = {}
    oracle_predictions: Dict[str, np.ndarray] = {}
    shift_size: Dict[str, Dict[str, float]] = {}
    for direction_index, direction in enumerate(DIRECTIONS):
        b_policy = policies[direction]["B"]
        observed = sim_u.observe_u(
            latent,
            b_policy,
            sigma=SIGMA,
            seed=TEST_OBS_SEED_BASE + direction_index,
        )
        prediction, p_u1 = sim_u.bayes_predict_u(
            observed["x"],
            observed["m"],
            SIGMA,
            b_policy,
            threshold=threshold,
            return_u_probability=True,
        )
        keep, _ = latent_filter.numpy_forward(
            observed["x"], observed["m"],
            _load_a_parameters(params_paths[direction]),
            w=1.0,
        )
        oracle_brier = float(np.mean((prediction - test_y) ** 2))
        shift_size[direction] = {
            "direction": direction,
            "keep_auroc": e12.auroc_rank(test_y, keep),
            "oracle_auroc": e12.auroc_rank(test_y, prediction),
            "keep_regret": float(np.mean((keep - test_y) ** 2) - oracle_brier),
            "mean_p_u1_given_h_u1": float(p_u1[test_u == 1].mean()),
            "mean_p_u1_given_h_u0": float(p_u1[test_u == 0].mean()),
        }
        observations[direction] = observed
        oracle_predictions[direction] = prediction
    return test_y, test_u, observations, oracle_predictions, shift_size


def _summary_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=RUN_COLUMNS)
    grouped = frame.groupby(["direction", "method", "week"], sort=False)[
        list(SUMMARY_METRICS)
    ].agg(["mean", "min", "max"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    return grouped.reset_index()


def run(
    *, smoke: bool = False, workers: int = 4, draws: int = 2, train_only: bool = False
) -> Tuple[Path | None, Path | None, Path | None]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if draws < 1:
        raise ValueError("--draws must be at least 1")

    policies = _direction_policies()
    params_paths: Dict[str, Path] = {}
    for direction in DIRECTIONS:
        path, _ = _train_or_load_a_model(
            direction, smoke, policies[direction]["A"]
        )
        params_paths[direction] = path
    if train_only:
        return None, None, None

    n_test = 500 if smoke else 10_000
    weeks = SMOKE_WEEKS if smoke else WEEKS
    n_draws = 1 if smoke else draws
    test_y, _test_u, test_observations, oracle_predictions, shift_size = (
        _make_test_cohort(n_test, policies, params_paths)
    )
    params = {
        direction: _load_a_parameters(params_paths[direction])
        for direction in DIRECTIONS
    }
    oracle_brier = {
        direction: float(np.mean((oracle_predictions[direction] - test_y) ** 2))
        for direction in DIRECTIONS
    }

    static_rows: List[Dict[str, object]] = []
    for direction in DIRECTIONS:
        observed = test_observations[direction]
        keep_prediction, _ = latent_filter.numpy_forward(
            observed["x"], observed["m"], params[direction], w=1.0
        )
        for week in weeks:
            n_patients = N_PER_WEEK * week
            for method, prediction in (
                ("keep", keep_prediction),
                ("oracle", oracle_predictions[direction]),
            ):
                static_rows.append({
                    "direction": direction,
                    "method": method,
                    "week": week,
                    "n_patients": n_patients,
                    "n_cases": 0,
                    "draw": 0,
                    **e12._metrics(test_y, prediction, oracle_brier[direction]),
                    "w": 1.0 if method == "keep" else float("nan"),
                    "wall_time_seconds": 0.0,
                })

    tasks = [
        {
            "direction": direction,
            "direction_index": direction_index,
            "draw": draw,
            "week": week,
        }
        for direction_index, direction in enumerate(DIRECTIONS)
        for draw in range(n_draws)
        for week in weeks
    ]
    job_rows: List[Dict[str, object]] = []
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(
            test_observations,
            test_y,
            oracle_predictions,
            {direction: str(params_paths[direction]) for direction in DIRECTIONS},
            {direction: policies[direction]["B"] for direction in DIRECTIONS},
            2 if smoke else 160,
        ),
    ) as pool:
        for result in pool.imap_unordered(_run_job, tasks, chunksize=1):
            job_rows.extend(result["rows"])
            print(
                f"DONE {result['direction']} draw={result['draw']} "
                f"week={result['week']} wall={result['elapsed']:.1f}s",
                flush=True,
            )

    rows = static_rows + job_rows
    analysis_dir = ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    prefix = "e14_smoke" if smoke else "e14"
    runs_path = analysis_dir / f"{prefix}_runs.csv"
    summary_path = analysis_dir / f"{prefix}_summary.csv"
    shift_path = analysis_dir / f"{prefix}_shift_size.csv"

    runs_frame = pd.DataFrame(rows, columns=RUN_COLUMNS)
    direction_order = {name: index for index, name in enumerate(DIRECTIONS)}
    runs_frame["_direction_order"] = runs_frame["direction"].map(direction_order)
    runs_frame["_method_order"] = runs_frame["method"].map(METHOD_ORDER)
    runs_frame.sort_values(
        ["_direction_order", "week", "_method_order", "draw"], inplace=True
    )
    runs_frame.drop(columns=["_direction_order", "_method_order"]).to_csv(
        runs_path, index=False
    )
    _summary_frame(rows).to_csv(summary_path, index=False)
    pd.DataFrame([shift_size[name] for name in DIRECTIONS]).to_csv(
        shift_path, index=False
    )
    print(f"WROTE {runs_path}", flush=True)
    print(f"WROTE {summary_path}", flush=True)
    print(f"WROTE {shift_path}", flush=True)
    return runs_path, summary_path, shift_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args(argv)
    run(
        smoke=args.smoke,
        workers=args.workers,
        draws=args.draws,
        train_only=args.train_only,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
