"""Train the two sequential GRUs for E4 (product of experts)."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

import run_grid
import sim
from models import build_recurrent_features, make_model


ROOT = Path(__file__).resolve().parent
CONFIG = "sigma1_gamma2"
SEED = 0
BIAS_MODEL_SEED = 31_337
POE_MODEL_SEED = 31_338
TEST_POLICIES = ("A", "B_match", "B_dense")


def _features(observed, value_channel_zeroed: bool = False) -> np.ndarray:
    x = np.asarray(observed["x"], dtype=np.float32)
    m = np.asarray(observed["m"], dtype=np.float32)
    features = build_recurrent_features(x, m)
    if value_channel_zeroed:
        features = features.copy()
        features[..., 0] = 0.0
    return np.ascontiguousarray(features, dtype=np.float32)


def _tensor(value, torch):
    return torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))


def _predict_logits(model, features, torch, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            outputs.append(model(features[start:start + batch_size]).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _fit_gru(train_features: np.ndarray, train_y: np.ndarray,
             val_features: np.ndarray, val_y: np.ndarray,
             empirical_mean: float, model_seed: int, max_epochs: int,
             train_offset: Optional[np.ndarray] = None,
             val_offset: Optional[np.ndarray] = None):
    """The run_grid._fit_model loop, with optional fixed precomputed logits."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(int(model_seed))
    np.random.seed(int(model_seed) % (2**32 - 1))
    x_train, y_train = _tensor(train_features, torch), _tensor(train_y, torch)
    x_val, y_val = _tensor(val_features, torch), _tensor(val_y, torch)
    offset_train = _tensor(train_offset, torch) if train_offset is not None else None
    offset_val = _tensor(val_offset, torch) if val_offset is not None else None

    model = make_model("gru", empirical_mean=empirical_mean, hidden_size=64,
                       max_length=train_features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    n_train = y_train.shape[0]
    batch_size = 256
    patience = 12
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    train_history = []
    val_history = []

    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(n_train)
        sum_loss = 0.0
        for start in range(0, n_train, batch_size):
            indices = order[start:start + batch_size]
            features = x_train.index_select(0, indices)
            targets = y_train.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            if offset_train is not None:
                logits = logits + offset_train.index_select(0, indices)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.detach()) * targets.shape[0]
        train_history.append(sum_loss / n_train)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            if offset_val is not None:
                val_logits = val_logits + offset_val
            val_loss = float(F.binary_cross_entropy_with_logits(val_logits, y_val))
        val_history.append(val_loss)
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": len(train_history),
        "train_bce": train_history,
        "val_bce": val_history,
    }


def _read_shard(path: Path, expected_keys: Sequence[str]):
    logits_path = path / "logits.npz"
    meta_path = path / "meta.json"
    if not logits_path.exists() or not meta_path.exists():
        raise RuntimeError(f"incomplete E4 output shard: {path}")
    with np.load(logits_path) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    missing = [key for key in expected_keys if key not in arrays]
    if missing:
        raise RuntimeError(f"{logits_path} is missing required arrays: {missing}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return arrays, metadata


def _write_shard(path: Path, arrays: Dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=path.parent))
    try:
        with (temp_dir / "logits.npz").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        (temp_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.rename(temp_dir, path)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _load_or_fit_bias(bias_path: Path, train_obs, train_y, val_obs, val_y,
                      test_observed: dict, test_y: np.ndarray, empirical_mean: float,
                      max_epochs: int, torch):
    required = ["train", "val", "y", *TEST_POLICIES]
    if bias_path.exists():
        arrays, metadata = _read_shard(bias_path, required)
        print(f"STEP bias SKIP wall={metadata['wall_time_seconds']:.1f}s "
              f"best_epoch={metadata['best_epoch']}", flush=True)
        return arrays

    started = time.perf_counter()
    train_features = _features(train_obs, value_channel_zeroed=True)
    val_features = _features(val_obs, value_channel_zeroed=True)
    model, fit_meta = _fit_gru(
        train_features, train_y, val_features, val_y, empirical_mean,
        BIAS_MODEL_SEED, max_epochs,
    )
    arrays = {
        "train": _predict_logits(model, _tensor(train_features, torch), torch),
        "val": _predict_logits(model, _tensor(val_features, torch), torch),
        "y": np.asarray(test_y, dtype=np.int8),
    }
    for name in TEST_POLICIES:
        features = _features(test_observed[name], value_channel_zeroed=True)
        arrays[name] = _predict_logits(model, _tensor(features, torch), torch)
    elapsed = time.perf_counter() - started
    metadata = {
        "exp": "E4", "config": CONFIG, "model": "bias_gru", "seed": SEED,
        "model_seed": BIAS_MODEL_SEED, "sigma": 1.0, "gamma": 2.0,
        "best_epoch": fit_meta["best_epoch"], "epochs_run": fit_meta["epochs_run"],
        "train_bce": fit_meta["train_bce"], "val_bce": fit_meta["val_bce"],
        "wall_time_seconds": elapsed, "torch_num_threads": 4,
        "test_policy_names": list(TEST_POLICIES),
    }
    _write_shard(bias_path, arrays, metadata)
    print(f"STEP bias DONE wall={elapsed:.1f}s best_epoch={fit_meta['best_epoch']}", flush=True)
    return arrays


def _load_or_fit_poe(poe_path: Path, bias_arrays: dict, train_obs, train_y,
                     val_obs, val_y, test_observed: dict, test_y: np.ndarray,
                     empirical_mean: float, max_epochs: int, torch):
    required = ["y", *[key for name in TEST_POLICIES for key in (f"f_{name}", f"b_{name}")]]
    if poe_path.exists():
        _, metadata = _read_shard(poe_path, required)
        print(f"STEP poe_gru SKIP wall={metadata['wall_time_seconds']:.1f}s "
              f"best_epoch={metadata['best_epoch']}", flush=True)
        return

    started = time.perf_counter()
    train_features = _features(train_obs)
    val_features = _features(val_obs)
    model, fit_meta = _fit_gru(
        train_features, train_y, val_features, val_y, empirical_mean,
        POE_MODEL_SEED, max_epochs,
        train_offset=bias_arrays["train"], val_offset=bias_arrays["val"],
    )
    arrays = {"y": np.asarray(test_y, dtype=np.int8)}
    for name in TEST_POLICIES:
        features = _features(test_observed[name])
        arrays[f"f_{name}"] = _predict_logits(model, _tensor(features, torch), torch)
        arrays[f"b_{name}"] = np.asarray(bias_arrays[name], dtype=np.float32)
    elapsed = time.perf_counter() - started
    metadata = {
        "exp": "E4", "config": CONFIG, "model": "poe_gru", "seed": SEED,
        "model_seed": POE_MODEL_SEED, "bias_model_seed": BIAS_MODEL_SEED,
        "sigma": 1.0, "gamma": 2.0,
        "best_epoch": fit_meta["best_epoch"], "epochs_run": fit_meta["epochs_run"],
        "train_bce": fit_meta["train_bce"], "val_bce": fit_meta["val_bce"],
        "wall_time_seconds": elapsed, "torch_num_threads": 4,
        "fixed_bias_logits": True, "test_policy_names": list(TEST_POLICIES),
    }
    _write_shard(poe_path, arrays, metadata)
    print(f"STEP poe_gru DONE wall={elapsed:.1f}s best_epoch={fit_meta['best_epoch']}", flush=True)


def run(out: str = "results", smoke: bool = False) -> None:
    import torch

    torch.set_num_threads(4)
    if smoke and out == "results":
        out = "results_smoke"
    out_root = run_grid._validate_out_root(Path(out))
    sizes = {"n_train": 800, "n_val": 200, "n_test": 1000, "max_epochs": 3} if smoke else {
        "n_train": 8000, "n_val": 2000, "n_test": 8000, "max_epochs": 120
    }

    train_policy, test_policies = run_grid._policy_setup("E1", 1.0, 2.0)
    (train_latent_seed, val_latent_seed, test_latent_seed,
     train_obs_seed, val_obs_seed, test_obs_seed) = run_grid._seeds_for(SEED)
    train_obs, train_y = run_grid._observed_split(
        sizes["n_train"], train_latent_seed, train_obs_seed, train_policy, 1.0
    )
    val_obs, val_y = run_grid._observed_split(
        sizes["n_val"], val_latent_seed, val_obs_seed, train_policy, 1.0
    )
    test_latent = sim.simulate_latent(sizes["n_test"], seed=test_latent_seed)
    test_y = np.asarray(test_latent["y"], dtype=np.int8)
    test_observed = {
        name: sim.observe(test_latent, policy, sigma=1.0, seed=test_obs_seed)
        for name, policy in test_policies.items()
    }
    observed_x = np.asarray(train_obs["x"], dtype=np.float32)
    observed_m = np.asarray(train_obs["m"], dtype=bool)
    empirical_mean = float(observed_x[observed_m].mean()) if observed_m.any() else 0.0

    shard_root = out_root / "E4" / CONFIG / f"seed{SEED}"
    bias_path = shard_root / "bias"
    poe_path = shard_root / "poe_gru"
    bias_arrays = _load_or_fit_bias(
        bias_path, train_obs, train_y, val_obs, val_y, test_observed, test_y,
        empirical_mean, sizes["max_epochs"], torch,
    )
    if not np.array_equal(np.asarray(bias_arrays["y"]), test_y):
        raise ValueError("saved bias logits belong to a different E4 test cohort")
    _load_or_fit_poe(
        poe_path, bias_arrays, train_obs, train_y, val_obs, val_y, test_observed,
        test_y, empirical_mean, sizes["max_epochs"], torch,
    )


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(out=args.out, smoke=args.smoke)


if __name__ == "__main__":
    main()
