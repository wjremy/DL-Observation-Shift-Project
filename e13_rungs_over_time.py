"""E13: compare adaptation rungs as a deployment run-in grows week by week.

Run from the project folder with ``python e13_rungs_over_time.py``.  The
``--smoke`` option uses short training/refits and writes separate smoke files.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import bayes
import sim

# bayes imports THRESHOLD by value, so patch both modules at import time.  This
# module is imported afresh by spawned workers, which keeps their outcome rule
# aligned with the A model and the main process.
sim.THRESHOLD = 2.05
bayes.THRESHOLD = 2.05

import e12_unlabelled_head as e12
import latent_filter


ROOT = Path(__file__).resolve().parent
SIGMA = 1.0
WEEKS = (1, 2, 4, 8, 13, 26, 52, 104)
SMOKE_WEEKS = (1, 4)
N_PER_WEEK = 25
STREAM_N = 2_600
POLICY_NAMES = ("B_less", "B_more", "B_match", "B_dense")

# All E13 seed ranges are separate from the E5/E12 seeds and from one another.
A_TRAIN_LATENT_SEED = 1_500_000
A_VAL_LATENT_SEED = 1_501_000
A_TRAIN_OBS_SEED = 1_502_000
A_VAL_OBS_SEED = 1_503_000
A_MODEL_SEED = 1_504_337
TEST_LATENT_SEED = 1_550_000
TEST_OBS_SEED_BASE = 1_560_000
STREAM_SEED_BASE = 1_600_000

RUN_COLUMNS = (
    "policy", "method", "week", "n_patients", "n_cases", "draw",
    "brier", "log_loss", "auroc", "regret", "w", "wall_time_seconds",
)
SUMMARY_METRICS = (
    "n_patients", "n_cases", "brier", "log_loss", "auroc", "regret",
    "w", "wall_time_seconds",
)


def _a_model_paths(smoke: bool) -> Tuple[Path, Path]:
    folder = ROOT / ("results_smoke" if smoke else "results") / "E13" / "a_model"
    return folder / "params.npz", folder / "meta.json"


def _observe_cohort(n: int, latent_seed: int, obs_seed: int, policy):
    latent = sim.simulate_latent(n, seed=latent_seed)
    observed = sim.observe(latent, policy, sigma=SIGMA, seed=obs_seed)
    return observed, np.asarray(latent["y"], dtype=np.float32)


def _train_or_load_a_model(smoke: bool) -> Tuple[Path, Dict[str, object]]:
    n_train, n_val, max_epochs = (1_000, 300, 3) if smoke else (16_000, 4_000, 200)
    params_path, meta_path = _a_model_paths(smoke)
    train_policy = sim.informative(gamma=2.0, rate=0.3)

    train_latent = sim.simulate_latent(n_train, seed=A_TRAIN_LATENT_SEED)
    train_y = np.asarray(train_latent["y"], dtype=np.float32)
    training_rate = float(train_y.mean())
    assert 0.01 <= training_rate <= 0.03, (
        f"E13 A training outcome rate must be 1%-3%; got {training_rate:.4%}"
    )

    if params_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        print(f"SKIP A model training (cached at {params_path.parent})", flush=True)
        return params_path, metadata

    if latent_filter.torch is None:
        raise ImportError("PyTorch is required to train the E13 A filter")
    started = time.perf_counter()
    train_observed = sim.observe(
        train_latent, train_policy, sigma=SIGMA, seed=A_TRAIN_OBS_SEED
    )
    val_observed, val_y = _observe_cohort(
        n_val, A_VAL_LATENT_SEED, A_VAL_OBS_SEED, train_policy
    )
    model, fit_meta = latent_filter.train_filter(
        train_observed["x"], train_observed["m"], train_y,
        val_observed["x"], val_observed["m"], val_y,
        n_states=32,
        max_epochs=max_epochs,
        batch_size=512,
        patience=15,
        seed=A_MODEL_SEED,
    )
    metadata: Dict[str, object] = {
        "experiment": "E13",
        "model": "latent_filter",
        "sigma": SIGMA,
        "training_policy": "informative_gamma2_rate0.3",
        "threshold": 2.05,
        "n_train": n_train,
        "n_val": n_val,
        "n_train_cases": int(train_y.sum()),
        "train_outcome_rate": training_rate,
        "n_val_cases": int(np.asarray(val_y).sum()),
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
        "wall_time_seconds": time.perf_counter() - started,
    }
    latent_filter._atomic_npz(params_path, model.numpy_parameters())
    latent_filter._atomic_json(meta_path, metadata)
    print(
        f"DONE A model training at {params_path.parent} "
        f"rate={training_rate:.3%} best_epoch={fit_meta['best_epoch']} "
        f"wall={metadata['wall_time_seconds']:.1f}s",
        flush=True,
    )
    return params_path, metadata


def _stream_seeds(policy_index: int, draw: int) -> Tuple[int, int, int]:
    base = STREAM_SEED_BASE + policy_index * 10_000 + draw * 100
    return base, base + 1, base + 2


def _choose_penalized_weight(
    x: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    parameters: Mapping[str, np.ndarray],
) -> float:
    """Minimize summed labelled log loss plus the E9 quadratic penalty."""
    if int(np.asarray(y).sum()) == 0:
        return 1.0
    weights = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    losses: List[float] = []
    labels = np.asarray(y, dtype=np.float64)
    for weight in weights:
        probability, _ = latent_filter.numpy_forward(x, m, parameters, w=float(weight))
        clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
        log_loss_sum = -np.sum(
            labels * np.log(clipped) + (1.0 - labels) * np.log1p(-clipped)
        )
        losses.append(float(log_loss_sum + 2.0 * (float(weight) - 1.0) ** 2))
    best_loss = min(losses)
    return float(max(weight for weight, loss in zip(weights, losses) if loss == best_loss))


def _finetune(
    observed: Mapping[str, np.ndarray],
    labels: np.ndarray,
    parameters: Mapping[str, np.ndarray],
    *,
    batch_seed: int,
    steps: int,
    lr: float,
) -> Dict[str, np.ndarray]:
    """Fine-tune every A parameter with the E5 objective and replacement batches."""
    if int(np.asarray(labels).sum()) == 0:
        return {name: np.asarray(parameters[name]).copy() for name in latent_filter.PARAMETER_NAMES}
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:
        raise ImportError("PyTorch is required for E13 fine-tuning") from exc

    torch.set_num_threads(1)
    torch.manual_seed(int(batch_seed))
    model = latent_filter.LatentFilter(n_states=int(np.asarray(parameters["mu"]).size))
    with torch.no_grad():
        for name in latent_filter.PARAMETER_NAMES:
            tensor = torch.as_tensor(
                np.asarray(parameters[name]), dtype=getattr(model, name).dtype
            )
            getattr(model, name).copy_(tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    x = torch.from_numpy(np.ascontiguousarray(observed["x"], dtype=np.float32))
    m = torch.from_numpy(np.ascontiguousarray(observed["m"], dtype=np.float32))
    y = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))
    n = int(x.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(batch_seed))
    model.train()

    for _ in range(int(steps)):
        indices = torch.randint(0, n, (256,), generator=generator)
        xb = x.index_select(0, indices)
        mb = m.index_select(0, indices)
        yb = y.index_select(0, indices)
        probability, log_marginal = model(xb, mb, w=1.0)
        bce = F.binary_cross_entropy(
            probability.clamp(1e-7, 1.0 - 1e-7), yb
        )
        nll_per_step = -log_marginal.mean() / x.shape[1]
        loss = bce + nll_per_step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.numpy_parameters()


_WORKER_TEST_OBS: Dict[str, Mapping[str, np.ndarray]] = {}
_WORKER_TEST_Y: np.ndarray | None = None
_WORKER_ORACLE: Dict[str, np.ndarray] = {}
_WORKER_ORACLE_BRIER: Dict[str, float] = {}
_WORKER_PARAMS: Dict[str, np.ndarray] = {}
_WORKER_POLICIES: Dict[str, sim.Policy] = {}
_WORKER_LR = 0.05
_WORKER_STEPS = 160


def _init_worker(
    test_observations: Dict[str, Mapping[str, np.ndarray]],
    test_y: np.ndarray,
    oracle_predictions: Dict[str, np.ndarray],
    params_path: str,
    lr: float,
    steps: int,
) -> None:
    global _WORKER_TEST_OBS, _WORKER_TEST_Y, _WORKER_ORACLE, _WORKER_ORACLE_BRIER
    global _WORKER_PARAMS, _WORKER_POLICIES, _WORKER_LR, _WORKER_STEPS
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to run E13 adaptation jobs") from exc
    torch.set_num_threads(1)
    _WORKER_TEST_OBS = test_observations
    _WORKER_TEST_Y = np.asarray(test_y)
    _WORKER_ORACLE = oracle_predictions
    _WORKER_ORACLE_BRIER = {
        policy: float(np.mean((prediction - _WORKER_TEST_Y) ** 2))
        for policy, prediction in oracle_predictions.items()
    }
    _WORKER_PARAMS = e12._load_parameters(params_path)
    _WORKER_POLICIES = e12._policies()
    _WORKER_LR = float(lr)
    _WORKER_STEPS = int(steps)


def _run_job(task: Mapping[str, object]) -> Dict[str, object]:
    if _WORKER_TEST_Y is None:
        raise RuntimeError("E13 worker was not initialized")
    started = time.perf_counter()
    policy_name = str(task["policy"])
    policy_index = int(task["policy_index"])
    draw = int(task["draw"])
    week = int(task["week"])
    n_patients = N_PER_WEEK * week
    latent_seed, observation_seed, batch_seed = _stream_seeds(policy_index, draw)
    stream_latent = sim.simulate_latent(STREAM_N, seed=latent_seed)
    stream_all = sim.observe(
        stream_latent,
        _WORKER_POLICIES[policy_name],
        sigma=SIGMA,
        seed=observation_seed,
    )
    stream = {
        "x": np.asarray(stream_all["x"][:n_patients], dtype=np.float32),
        "m": np.asarray(stream_all["m"][:n_patients], dtype=np.int8),
    }
    stream_y = np.asarray(stream_latent["y"][:n_patients], dtype=np.float32)
    n_cases = int(stream_y.sum())
    test_observed = _WORKER_TEST_OBS[policy_name]
    test_y = _WORKER_TEST_Y
    oracle_brier = _WORKER_ORACLE_BRIER[policy_name]
    params = _WORKER_PARAMS
    rows: List[Dict[str, object]] = []

    # R1 and R2 share the same refitted testing head. Its fit is unlabelled.
    method_started = time.perf_counter()
    head_u_params = e12._refit_testing_head(
        stream,
        None,
        params,
        batch_seed=batch_seed,
        steps=_WORKER_STEPS,
        lr=_WORKER_LR,
    )
    head_u_prediction, _ = latent_filter.numpy_forward(
        test_observed["x"], test_observed["m"], head_u_params, w=1.0
    )
    rows.append({
        "policy": policy_name,
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
    head_u_dial_w = _choose_penalized_weight(
        stream["x"], stream["m"], stream_y, head_u_params
    )
    head_u_dial_prediction, _ = latent_filter.numpy_forward(
        test_observed["x"], test_observed["m"], head_u_params, w=head_u_dial_w
    )
    rows.append({
        "policy": policy_name,
        "method": "head_U_dial",
        "week": week,
        "n_patients": n_patients,
        "n_cases": n_cases,
        "draw": draw,
        **e12._metrics(test_y, head_u_dial_prediction, oracle_brier),
        "w": head_u_dial_w,
        "wall_time_seconds": time.perf_counter() - method_started,
    })

    method_started = time.perf_counter()
    dial_only_w = _choose_penalized_weight(
        stream["x"], stream["m"], stream_y, params
    )
    dial_only_prediction, _ = latent_filter.numpy_forward(
        test_observed["x"], test_observed["m"], params, w=dial_only_w
    )
    rows.append({
        "policy": policy_name,
        "method": "dial_only",
        "week": week,
        "n_patients": n_patients,
        "n_cases": n_cases,
        "draw": draw,
        **e12._metrics(test_y, dial_only_prediction, oracle_brier),
        "w": dial_only_w,
        "wall_time_seconds": time.perf_counter() - method_started,
    })

    method_started = time.perf_counter()
    finetuned_params = _finetune(
        stream,
        stream_y,
        params,
        batch_seed=batch_seed + 17,
        steps=_WORKER_STEPS,
        lr=_WORKER_LR,
    )
    finetune_prediction, _ = latent_filter.numpy_forward(
        test_observed["x"], test_observed["m"], finetuned_params, w=1.0
    )
    rows.append({
        "policy": policy_name,
        "method": "finetune",
        "week": week,
        "n_patients": n_patients,
        "n_cases": n_cases,
        "draw": draw,
        **e12._metrics(test_y, finetune_prediction, oracle_brier),
        "w": 1.0,
        "wall_time_seconds": time.perf_counter() - method_started,
    })

    return {
        "policy": policy_name,
        "week": week,
        "draw": draw,
        "rows": rows,
        "elapsed": time.perf_counter() - started,
    }


def _make_test_cohort(n_test: int):
    policies = e12._policies()
    test_latent = sim.simulate_latent(n_test, seed=TEST_LATENT_SEED)
    test_y = np.asarray(test_latent["y"], dtype=np.float32)
    test_observations: Dict[str, Mapping[str, np.ndarray]] = {}
    oracle_predictions: Dict[str, np.ndarray] = {}
    for policy_index, policy_name in enumerate(POLICY_NAMES):
        policy = policies[policy_name]
        observed = sim.observe(
            test_latent,
            policy,
            sigma=SIGMA,
            seed=TEST_OBS_SEED_BASE + policy_index,
        )
        test_observations[policy_name] = observed
        oracle_predictions[policy_name] = bayes.bayes_predict(
            observed["x"], observed["m"], SIGMA, policy
        )
    return policies, test_observations, test_y, oracle_predictions


def _summary_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=RUN_COLUMNS)
    grouped = frame.groupby(["policy", "method", "week"], sort=False)[
        list(SUMMARY_METRICS)
    ].agg(["mean", "min", "max"])
    grouped.columns = [f"{column}_{stat}" for column, stat in grouped.columns]
    return grouped.reset_index()


def run(
    *,
    smoke: bool = False,
    workers: int = 4,
    lr: float = 0.05,
    steps: int = 160,
    train_only: bool = False,
) -> Tuple[Path | None, Path | None]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if steps < 1:
        raise ValueError("--steps must be at least 1")
    if lr <= 0:
        raise ValueError("--lr must be positive")

    params_path, _ = _train_or_load_a_model(smoke)
    if train_only:
        return None, None

    n_test = 500 if smoke else 10_000
    weeks = SMOKE_WEEKS if smoke else WEEKS
    draws = 1 if smoke else 5
    policies, test_observations, test_y, oracle_predictions = _make_test_cohort(n_test)
    params = e12._load_parameters(params_path)
    oracle_brier = {
        name: float(np.mean((oracle_predictions[name] - test_y) ** 2))
        for name in POLICY_NAMES
    }

    # Compute static R0 and oracle predictions once per policy. A copy of each
    # metric row is placed at every week so the summary and figure have flat
    # reference curves on the same x axis as the adapting methods.
    static_rows: List[Dict[str, object]] = []
    for policy_name in POLICY_NAMES:
        observed = test_observations[policy_name]
        keep_prediction, _ = latent_filter.numpy_forward(
            observed["x"], observed["m"], params, w=1.0
        )
        for week in weeks:
            n_patients = N_PER_WEEK * week
            static_rows.append({
                "policy": policy_name,
                "method": "keep",
                "week": week,
                "n_patients": n_patients,
                "n_cases": 0,
                "draw": 0,
                **e12._metrics(test_y, keep_prediction, oracle_brier[policy_name]),
                "w": 1.0,
                "wall_time_seconds": 0.0,
            })
            static_rows.append({
                "policy": policy_name,
                "method": "oracle",
                "week": week,
                "n_patients": n_patients,
                "n_cases": 0,
                "draw": 0,
                **e12._metrics(
                    test_y, oracle_predictions[policy_name], oracle_brier[policy_name]
                ),
                "w": float("nan"),
                "wall_time_seconds": 0.0,
            })

    tasks = [
        {
            "policy": policy_name,
            "policy_index": policy_index,
            "draw": draw,
            "week": week,
        }
        for policy_index, policy_name in enumerate(POLICY_NAMES)
        for draw in range(draws)
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
            str(params_path),
            lr,
            2 if smoke else steps,
        ),
    ) as pool:
        for result in pool.imap_unordered(_run_job, tasks, chunksize=1):
            job_rows.extend(result["rows"])
            print(
                f"DONE {result['policy']} draw={result['draw']} "
                f"week={result['week']} wall={result['elapsed']:.1f}s",
                flush=True,
            )

    rows = static_rows + job_rows
    analysis_dir = ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    prefix = "e13_smoke" if smoke else "e13"
    runs_path = analysis_dir / f"{prefix}_runs.csv"
    summary_path = analysis_dir / f"{prefix}_summary.csv"
    runs_frame = pd.DataFrame(rows, columns=RUN_COLUMNS)
    policy_order = {name: index for index, name in enumerate(POLICY_NAMES)}
    method_order = {
        "keep": 0, "head_U": 1, "head_U_dial": 2, "finetune": 3,
        "dial_only": 4, "oracle": 5,
    }
    runs_frame["_policy_order"] = runs_frame["policy"].map(policy_order)
    runs_frame["_method_order"] = runs_frame["method"].map(method_order)
    runs_frame.sort_values(
        ["_policy_order", "week", "_method_order", "draw"], inplace=True
    )
    runs_frame.drop(columns=["_policy_order", "_method_order"]).to_csv(
        runs_path, index=False
    )
    _summary_frame(rows).to_csv(summary_path, index=False)
    print(f"WROTE {runs_path}", flush=True)
    print(f"WROTE {summary_path}", flush=True)
    return runs_path, summary_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args(argv)
    run(
        smoke=args.smoke,
        workers=args.workers,
        lr=args.lr,
        steps=args.steps,
        train_only=args.train_only,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
