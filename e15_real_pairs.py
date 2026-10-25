"""E15: re-learning the testing head on real hospital pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

import latent_filter_mv


ROOT = Path(__file__).resolve().parent
CACHE_PATH = ROOT / "data" / "e15_cache.npz"
SOURCES = (
    ("P12", "CCU"), ("P12", "CSRU"), ("P12", "MICU"), ("P12", "SICU"),
    ("P19", "A-all"), ("P19", "B-all"),
)
PAIRS = (
    ("P12", "SICU", "CSRU", "shift"),
    ("P12", "CSRU", "CCU", "shift"),
    ("P12", "CSRU", "SICU", "shift"),
    ("P12", "CCU", "SICU", "shift"),
    ("P19", "B-all", "A-all", "shift"),
    ("P12", "MICU", "SICU", "control"),
    ("P12", "SICU", "MICU", "control"),
    ("P19", "A-all", "B-all", "control"),
)
SPLIT_SEED = 1_510_025
TARGET_SPLIT_SEED = 1_510_125
RUNIN_SEED = 1_510_225
FULL_RUN_SIZES = (162, 650)
SMOKE_RUN_SIZES = (162,)
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
METRIC_COLUMNS = ("auroc", "log_loss", "brier", "n_cases", "w", "auroc_diff_to_keep")
RUN_COLUMNS = (
    "dataset", "source", "target", "pair_type", "pair_label", "n_run", "draw",
    "method", "n_pool", "n_evaluation", "n_cases", "auroc", "log_loss", "brier",
    "w", "auroc_diff_to_keep",
)
SOURCE_COLUMNS = (
    "dataset", "site", "n", "cases", "n_train", "n_validation", "validation_cases",
    "best_epoch", "validation_auroc", "e7_linear_auroc", "linear_comparison_source",
    "training_time_seconds",
)


def _stable_offset(text: str) -> int:
    return sum((i + 1) * ord(char) for i, char in enumerate(str(text)))


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_cache(path: Path = CACHE_PATH) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing E15 sequence cache: {path}. Run e15_sequences.py first.")
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name].copy() for name in stored.files}


def _source_reference_aurocs() -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], str]]:
    """Read E7 source holdout AUROC; use E10 combined AUROC for whole hospitals."""
    import pandas as pd

    scores: dict[tuple[str, str], float] = {}
    provenance: dict[tuple[str, str], str] = {}
    e7_path = ROOT / "analysis" / "e7_sites.csv"
    if e7_path.is_file():
        frame = pd.read_csv(e7_path)
        for row in frame.itertuples(index=False):
            key = (str(row.dataset), str(row.site))
            scores[key] = float(row.source_holdout_auroc_w1)
            provenance[key] = "analysis/e7_sites.csv:source_holdout_auroc_w1"
    # E7's source-site table did not include whole-hospital P19 fits.  E10
    # reports the same E7-style combined linear model on its held-out half.
    e10_path = ROOT / "analysis" / "e10_sites.csv"
    if e10_path.is_file():
        frame = pd.read_csv(e10_path)
        rows = frame[(frame["dataset"] == "P19")
                     & (frame["site"].isin(["A-all", "B-all"]))
                     & (frame["feature_set"] == "combined")]
        for row in rows.itertuples(index=False):
            key = (str(row.dataset), str(row.site))
            scores[key] = float(row.local_auroc)
            provenance[key] = "analysis/e10_sites.csv:local_auroc (combined; E7 whole-hospital rows absent)"
    return scores, provenance


def _source_folder(dataset: str, site: str, smoke: bool) -> Path:
    base = ROOT / ("results_smoke" if smoke else "results") / "E15"
    return base / f"a_model_{dataset}_{site}"


def _load_parameters(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]).copy() for name in latent_filter_mv.PARAMETER_NAMES}


def _source_split(n: int, dataset: str, site: str) -> tuple[np.ndarray, np.ndarray, int]:
    seed = SPLIT_SEED + _stable_offset(f"{dataset}_{site}")
    order = np.random.default_rng(seed).permutation(n)
    n_train = int(math.floor(0.8 * n))
    return order[:n_train], order[n_train:], seed


def _target_split_indices(n: int, dataset: str, site: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the reproducible fixed half-pool, half-evaluation split."""
    order = np.random.default_rng(
        TARGET_SPLIT_SEED + _stable_offset(f"{dataset}_{site}")
    ).permutation(int(n))
    n_pool = len(order) // 2
    return order[:n_pool], order[n_pool:]


def _draw_runin_indices(n_pool: int, n_run: int, seed: int) -> np.ndarray:
    if n_pool < n_run:
        raise ValueError(f"pool n={n_pool} is smaller than run-in n={n_run}")
    return np.random.default_rng(int(seed)).choice(int(n_pool), size=int(n_run), replace=False)


def _standardise_target(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return latent_filter_mv.standardise(x, mean, sd).astype(np.float32)


def _train_or_load_source(
    cache: Mapping[str, np.ndarray],
    dataset: str,
    site: str,
    *,
    smoke: bool,
    lambda_nll: float,
    references: Mapping[tuple[str, str], float],
    reference_sources: Mapping[tuple[str, str], str],
) -> dict[str, object]:
    folder = _source_folder(dataset, site, smoke)
    params_path = folder / "params.npz"
    meta_path = folder / "meta.json"
    if params_path.is_file() and meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as stream:
            meta = json.load(stream)
        _load_parameters(params_path)
        if "standardisation_mean" not in meta or "standardisation_sd" not in meta:
            raise ValueError(f"Cached E15 model lacks standardisation metadata: {folder}")
        print(f"E15 source {dataset}/{site}: using cached model", flush=True)
        return {
            "dataset": dataset, "site": site, "folder": str(folder), "meta": meta,
            "params_path": str(params_path), "source_row": _source_row_from_meta(meta),
        }

    prefix = f"{dataset}_{site}"
    x = np.asarray(cache[f"{prefix}_x"], dtype=np.float32)
    m = np.asarray(cache[f"{prefix}_m"], dtype=np.int8)
    y = np.asarray(cache[f"{prefix}_y"], dtype=np.int8).reshape(-1)
    ids = np.asarray(cache[f"{prefix}_ids"]).astype(str)
    if x.ndim != 3 or m.shape != x.shape or len(y) != len(x) or len(ids) != len(x):
        raise ValueError(f"Malformed E15 cache arrays for {dataset}/{site}")
    train_idx, val_idx, split_seed = _source_split(len(y), dataset, site)
    if len(np.unique(y[train_idx])) < 2:
        raise ValueError(f"Source training split has one outcome class: {dataset}/{site}")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(np.where(m.astype(bool), x, np.nan), axis=(0, 1))
        sd = np.nanstd(np.where(m.astype(bool), x, np.nan), axis=(0, 1))
    mean = np.where(np.isfinite(mean), mean, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0.0), sd, 1.0)
    x_standard = latent_filter_mv.standardise(x, mean, sd).astype(np.float32)
    started = time.perf_counter()
    model, history = latent_filter_mv.train_filter_mv(
        x_standard[train_idx], m[train_idx], y[train_idx],
        x_standard[val_idx], m[val_idx], y[val_idx],
        n_states=32, max_epochs=(2 if smoke else 100), batch_size=256,
        patience=10, seed=split_seed + 17, lambda_nll=lambda_nll,
    )
    training_time = time.perf_counter() - started
    params = model.numpy_parameters()
    reference = float(references.get((dataset, site), float("nan")))
    reference_source = reference_sources.get((dataset, site), "unavailable")
    best_epoch = int(history["best_epoch"])
    val_auc = float(history["best_val_auroc"])
    meta = {
        "spec": "E15",
        "dataset": dataset,
        "site": site,
        "n_states": 32,
        "n_variables": int(x.shape[2]),
        "time_steps": int(x.shape[1]),
        "lambda_nll": float(lambda_nll),
        "split_seed": int(split_seed),
        "train_ids": ids[train_idx].tolist(),
        "validation_ids": ids[val_idx].tolist(),
        "source_cases": int(y.sum()),
        "validation_cases": int(y[val_idx].sum()),
        "standardisation_mean": mean.astype(float).tolist(),
        "standardisation_sd": sd.astype(float).tolist(),
        "best_epoch": best_epoch,
        "epochs": int(history["epochs"]),
        "history": history,
        "validation_auroc": val_auc,
        "e7_linear_auroc": reference,
        "linear_comparison_source": reference_source,
        "training_time_seconds": float(training_time),
    }
    _atomic_npz(params_path, params)
    _atomic_json(meta_path, meta)
    print(
        f"E15 source {dataset}/{site}: n={len(y)}, cases={int(y.sum())}, "
        f"best epoch={best_epoch}, validation AUROC={val_auc:.4f}, "
        f"E7 linear AUROC={reference:.4f}, fit={training_time:.1f}s",
        flush=True,
    )
    return {
        "dataset": dataset, "site": site, "folder": str(folder), "meta": meta,
        "params_path": str(params_path), "source_row": _source_row_from_meta(meta),
    }


def _source_row_from_meta(meta: Mapping[str, object]) -> dict[str, object]:
    train_ids = meta.get("train_ids", [])
    validation_ids = meta.get("validation_ids", [])
    return {
        "dataset": meta.get("dataset"), "site": meta.get("site"),
        "n": len(train_ids) + len(validation_ids),
        "cases": int(meta.get("source_cases", 0)),
        "n_train": len(train_ids), "n_validation": len(validation_ids),
        "validation_cases": int(meta.get("validation_cases", 0)),
        "best_epoch": meta.get("best_epoch"),
        "validation_auroc": meta.get("validation_auroc"),
        "e7_linear_auroc": meta.get("e7_linear_auroc"),
        "linear_comparison_source": meta.get("linear_comparison_source", "cached metadata"),
        "training_time_seconds": meta.get("training_time_seconds"),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _source_row(
    cache: Mapping[str, np.ndarray], record: Mapping[str, object]
) -> dict[str, object]:
    meta = dict(record["meta"])
    if meta.get("source_cases") is None:
        prefix = f"{record['dataset']}_{record['site']}"
        y = np.asarray(cache[f"{prefix}_y"], dtype=np.int8)
        ids = np.asarray(cache[f"{prefix}_ids"])
        n_train = len(meta.get("train_ids", []))
        # Preserve the split order recorded by metadata when recovering case counts.
        id_to_y = {str(patient_id): int(label) for patient_id, label in zip(ids, y)}
        train_cases = sum(id_to_y.get(str(patient_id), 0) for patient_id in meta.get("train_ids", []))
        val_cases = sum(id_to_y.get(str(patient_id), 0) for patient_id in meta.get("validation_ids", []))
        meta["source_cases"] = train_cases + val_cases
        meta["validation_cases"] = val_cases
        meta["n_train"] = n_train
    return _source_row_from_meta(meta)


def _prepare_sources(
    cache: Mapping[str, np.ndarray], *, smoke: bool, lambda_nll: float
) -> tuple[list[dict[str, object]], list[str]]:
    references, reference_sources = _source_reference_aurocs()
    records = []
    warnings = []
    selected_sources = (("P12", "SICU"), ("P12", "CSRU")) if smoke else SOURCES
    for dataset, site in selected_sources:
        record = _train_or_load_source(
            cache, dataset, site, smoke=smoke, lambda_nll=lambda_nll,
            references=references, reference_sources=reference_sources,
        )
        # Fill cohort counts for both newly fitted and cached models, without
        # modifying their saved metadata files.
        row = _source_row(cache, record)
        record["source_row"] = row
        if record["meta"].get("source_cases") is None:
            record["meta"]["source_cases"] = int(row["cases"])
            record["meta"]["validation_cases"] = int(row["validation_cases"])
        val_auc = float(row["validation_auroc"])
        linear_auc = float(row["e7_linear_auroc"])
        if np.isfinite(linear_auc) and val_auc < linear_auc - 0.05:
            message = (
                f"E15 SANITY WARNING {dataset}/{site}: validation AUROC {val_auc:.4f} is "
                f"{linear_auc - val_auc:.4f} below linear reference AUROC {linear_auc:.4f} "
                f"({row['linear_comparison_source']})."
            )
            warnings.append(message)
            print(message, flush=True)
        records.append(record)
    return records, warnings


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if y.shape != probability.shape:
        raise ValueError("evaluation labels and predictions must match")
    clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
    return {
        "auroc": latent_filter_mv._auroc_rank(y, probability),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))),
        "brier": float(np.mean((probability - y) ** 2)),
    }


def _choose_penalized_weight(x, m, y, parameters) -> float:
    """E13 rule: run-in log loss plus the penalty 2 * (w - 1)^2."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if int(y.sum()) == 0:
        return 1.0
    losses = []
    for weight in WEIGHTS:
        probability, _ = latent_filter_mv.numpy_forward(x, m, parameters, w=float(weight))
        clipped = np.clip(probability, 1e-7, 1.0 - 1e-7)
        loss = -np.sum(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
        losses.append(float(loss + 2.0 * (float(weight) - 1.0) ** 2))
    best = min(losses)
    return float(max(weight for weight, loss in zip(WEIGHTS, losses) if loss == best))


_WORKER_CACHE: Dict[str, np.ndarray] = {}
_WORKER_PARAMS: Dict[tuple[str, str], Dict[str, np.ndarray]] = {}
_WORKER_META: Dict[tuple[str, str], dict[str, object]] = {}
_WORKER_DRAWS = 3
_WORKER_STEPS = 160
_WORKER_SMOKE = False


def _init_worker(
    cache_path: str,
    model_records: Sequence[Mapping[str, object]],
    draws: int,
    steps: int,
    smoke: bool,
) -> None:
    """Spawn-safe initializer: each worker loads the sequence cache once."""
    global _WORKER_CACHE, _WORKER_PARAMS, _WORKER_META, _WORKER_DRAWS, _WORKER_STEPS, _WORKER_SMOKE
    import torch

    torch.set_num_threads(1)
    _WORKER_CACHE = _load_cache(Path(cache_path))
    _WORKER_PARAMS = {}
    _WORKER_META = {}
    for record in model_records:
        key = (str(record["dataset"]), str(record["site"]))
        _WORKER_PARAMS[key] = _load_parameters(Path(str(record["params_path"])))
        with Path(str(record["folder"]), "meta.json").open("r", encoding="utf-8") as stream:
            _WORKER_META[key] = json.load(stream)
    _WORKER_DRAWS = int(draws)
    _WORKER_STEPS = int(steps)
    _WORKER_SMOKE = bool(smoke)


def _target_data(dataset: str, site: str, params_key: tuple[str, str]):
    prefix = f"{dataset}_{site}"
    x = np.asarray(_WORKER_CACHE[f"{prefix}_x"], dtype=np.float32)
    m = np.asarray(_WORKER_CACHE[f"{prefix}_m"], dtype=np.int8)
    y = np.asarray(_WORKER_CACHE[f"{prefix}_y"], dtype=np.int8).reshape(-1)
    ids = np.asarray(_WORKER_CACHE[f"{prefix}_ids"]).astype(str)
    meta = _WORKER_META[params_key]
    mean = np.asarray(meta["standardisation_mean"], dtype=np.float64)
    sd = np.asarray(meta["standardisation_sd"], dtype=np.float64)
    x = _standardise_target(x, mean, sd)
    return x, m, y, ids


def _pair_label(dataset: str, source: str, target: str) -> str:
    return f"{dataset} {source} to {target}"


def _metric_row(
    pair: tuple[str, str, str, str], n_run: int, draw: int, method: str,
    n_pool: int, n_eval: int, n_cases: int, y_eval: np.ndarray,
    probability: np.ndarray, weight: float,
) -> dict[str, object]:
    dataset, source, target, pair_type = pair
    return {
        "dataset": dataset, "source": source, "target": target,
        "pair_type": pair_type, "pair_label": _pair_label(dataset, source, target),
        "n_run": int(n_run), "draw": int(draw), "method": method,
        "n_pool": int(n_pool), "n_evaluation": int(n_eval), "n_cases": int(n_cases),
        **_metrics(y_eval, probability), "w": float(weight),
    }


def _run_pair(task: Mapping[str, object]) -> dict[str, object]:
    if not _WORKER_CACHE:
        raise RuntimeError("E15 worker was not initialized")
    dataset, source, target, pair_type = (
        str(task["dataset"]), str(task["source"]), str(task["target"]), str(task["pair_type"])
    )
    pair = (dataset, source, target, pair_type)
    params = _WORKER_PARAMS[(dataset, source)]
    x, m, y, ids = _target_data(dataset, target, (dataset, source))
    n = len(y)
    pool_idx, eval_idx = _target_split_indices(n, dataset, target)
    n_pool = len(pool_idx)
    x_pool, m_pool = x[pool_idx], m[pool_idx]
    x_eval, m_eval, y_eval = x[eval_idx], m[eval_idx], y[eval_idx]
    if not len(pool_idx) or not len(eval_idx):
        raise ValueError(f"Target split is empty for {dataset}/{target}")
    pool_head_seed = RUNIN_SEED + _stable_offset(f"pool_{dataset}_{source}_{target}")
    head_pool = latent_filter_mv.refit_testing_head_mv(
        x_pool, m_pool, None, params, batch_seed=pool_head_seed,
        steps=_WORKER_STEPS, lr=0.05,
    )
    head_pool_eval, _ = latent_filter_mv.numpy_forward(x_eval, m_eval, head_pool, w=1.0)
    keep_eval, _ = latent_filter_mv.numpy_forward(x_eval, m_eval, params, w=1.0)
    drop_eval, _ = latent_filter_mv.numpy_forward(x_eval, m_eval, params, w=0.0)
    sizes = SMOKE_RUN_SIZES if _WORKER_SMOKE else FULL_RUN_SIZES
    rows = []
    pair_seed_text = f"{dataset}_{source}_{target}"
    for size_index, n_run in enumerate(sizes):
        if n_pool < n_run:
            print(f"E15 skip {dataset} {source}->{target}: pool n={n_pool} < run-in n={n_run}", flush=True)
            continue
        for draw in range(_WORKER_DRAWS):
            draw_seed = RUNIN_SEED + _stable_offset(pair_seed_text) + size_index * 1009 + draw * 37
            run_idx = _draw_runin_indices(n_pool, n_run, draw_seed)
            x_run, m_run, y_run = x_pool[run_idx], m_pool[run_idx], y[pool_idx][run_idx]
            n_cases = int(y_run.sum())
            weight = _choose_penalized_weight(x_run, m_run, y_run, params)
            dial_eval, _ = latent_filter_mv.numpy_forward(x_eval, m_eval, params, w=weight)
            head_u = latent_filter_mv.refit_testing_head_mv(
                x_run, m_run, None, params, batch_seed=draw_seed + 1,
                steps=_WORKER_STEPS, lr=0.05,
            )
            head_u_eval, _ = latent_filter_mv.numpy_forward(x_eval, m_eval, head_u, w=1.0)
            if n_cases == 0:
                head_l = {name: np.asarray(params[name]).copy()
                          for name in latent_filter_mv.PARAMETER_NAMES}
            else:
                head_l = latent_filter_mv.refit_testing_head_mv(
                    x_run, m_run, y_run, params, batch_seed=draw_seed + 2,
                    steps=_WORKER_STEPS, lr=0.05,
                )
            head_l_eval, _ = latent_filter_mv.numpy_forward(x_eval, m_eval, head_l, w=1.0)
            fine = latent_filter_mv.finetune_mv(
                x_run, m_run, y_run, params, batch_seed=draw_seed + 3,
                steps=_WORKER_STEPS, lr=0.005,
            )
            fine_eval, _ = latent_filter_mv.numpy_forward(x_eval, m_eval, fine, w=1.0)
            method_predictions = (
                ("keep", keep_eval, 1.0), ("drop", drop_eval, 0.0),
                ("dial_only", dial_eval, weight), ("head_U", head_u_eval, 1.0),
                ("head_L", head_l_eval, 1.0), ("finetune", fine_eval, 1.0),
                ("head_U_pool", head_pool_eval, 1.0),
            )
            for method, prediction, method_w in method_predictions:
                rows.append(_metric_row(
                    pair, n_run, draw, method, n_pool, len(eval_idx), n_cases,
                    y_eval, prediction, method_w,
                ))
    return {"pair": pair, "rows": rows}


def _add_auroc_differences(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    indexed = [dict(row) for row in rows]
    keeps = {
        (row["dataset"], row["source"], row["target"], row["n_run"], row["draw"]): float(row["auroc"])
        for row in indexed if row["method"] == "keep"
    }
    for row in indexed:
        key = (row["dataset"], row["source"], row["target"], row["n_run"], row["draw"])
        row["auroc_diff_to_keep"] = float(row["auroc"]) - keeps[key]
    return indexed


def _summary_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple, list[Mapping[str, object]]] = {}
    group_columns = ("dataset", "source", "target", "pair_type", "pair_label", "n_run", "method")
    for row in rows:
        key = tuple(row[column] for column in group_columns)
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in grouped.items():
        record = dict(zip(group_columns, key))
        for metric in METRIC_COLUMNS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            record[f"{metric}_mean"] = float(np.nanmean(values))
            record[f"{metric}_min"] = float(np.nanmin(values))
            record[f"{metric}_max"] = float(np.nanmax(values))
        record["draws"] = len(group)
        output.append(record)
    order = {(d, s, t): i for i, (d, s, t, _) in enumerate(PAIRS)}
    method_order = {name: i for i, name in enumerate(
        ("keep", "drop", "dial_only", "head_U", "head_U_pool", "head_L", "finetune")
    )}
    output.sort(key=lambda row: (
        order.get((row["dataset"], row["source"], row["target"]), 999),
        row["n_run"], method_order.get(row["method"], 999),
    ))
    return output


def run(
    *,
    smoke: bool = False,
    train_only: bool = False,
    workers: int = 4,
    draws: int = 3,
    lambda_nll: float = 1.0,
) -> tuple[Path, Path, Path]:
    if workers < 1:
        raise ValueError("--workers must be at least 1")
    if draws < 1:
        raise ValueError("--draws must be at least 1")
    if lambda_nll < 0:
        raise ValueError("--lambda-nll must be nonnegative")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to run E15") from exc
    torch.set_num_threads(4)
    cache = _load_cache()
    model_records, sanity_warnings = _prepare_sources(
        cache, smoke=smoke, lambda_nll=lambda_nll
    )
    output_dir = ROOT / "analysis"
    prefix = "e15_smoke" if smoke else "e15"
    sources_rows = [record["source_row"] for record in model_records]
    sources_path = output_dir / f"{prefix}_sources.csv"
    _write_csv(sources_path, sources_rows, SOURCE_COLUMNS)
    if train_only:
        print(f"E15 train-only complete: wrote {sources_path}", flush=True)
        return sources_path, sources_path, sources_path
    if sanity_warnings and not smoke:
        print("E15 stopped after the source-model sanity check; pair adaptation was not run.", flush=True)
        return sources_path, sources_path, sources_path
    if sanity_warnings and smoke:
        print("E15 smoke continues past the source AUROC gate to exercise pair data handling.", flush=True)

    pairs_to_run = PAIRS[:2] if smoke else PAIRS
    tasks = [
        {"dataset": dataset, "source": source, "target": target, "pair_type": pair_type}
        for dataset, source, target, pair_type in pairs_to_run
    ]
    worker_records = [
        {key: record[key] for key in ("dataset", "site", "folder", "params_path")}
        for record in model_records
    ]
    steps = 2 if smoke else 160
    context = mp.get_context("spawn")
    with context.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(str(CACHE_PATH), worker_records, (1 if smoke else draws), steps, smoke),
    ) as pool:
        results = pool.map(_run_pair, tasks)
    rows = _add_auroc_differences([row for result in results for row in result["rows"]])
    summary_rows = _summary_rows(rows)
    runs_path = output_dir / f"{prefix}_runs.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"
    _write_csv(runs_path, rows, RUN_COLUMNS)
    summary_columns = (
        "dataset", "source", "target", "pair_type", "pair_label", "n_run", "method", "draws",
        *(f"{metric}_{stat}" for metric in METRIC_COLUMNS for stat in ("mean", "min", "max")),
    )
    _write_csv(summary_path, summary_rows, summary_columns)
    print(f"Wrote {runs_path}, {summary_path}, and {sources_path}", flush=True)
    return runs_path, summary_path, sources_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run two shift pairs with 2 epochs and 2 refit steps")
    parser.add_argument("--train-only", action="store_true", help="fit/cache the six source models and stop")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--draws", type=int, default=3)
    parser.add_argument("--lambda-nll", type=float, default=1.0)
    args = parser.parse_args(argv)
    run(
        smoke=args.smoke, train_only=args.train_only, workers=args.workers,
        draws=args.draws, lambda_nll=args.lambda_nll,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
