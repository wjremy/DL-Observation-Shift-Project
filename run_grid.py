"""Train the specified model grid and save resumable, atomic result shards."""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import bayes
import sim
from models import build_cde_path, build_locf_delta, build_recurrent_features, make_model


ARCHITECTURES = ("gru", "grud", "cde", "transformer")
ROOT = Path(__file__).resolve().parent


def _policy_setup(exp: str, sigma: float, gamma: float) -> Tuple[sim.Policy, Dict[str, sim.Policy]]:
    """Return the training policy and named evaluation policies for a config."""
    if exp == "E1":
        train_policy = sim.informative(gamma=gamma, rate=0.3)
        test_policies = {
            "A": train_policy,
            "B_match": sim.uniform(rate=0.3),
            "B_dense": sim.uniform(rate=0.6),
        }
    else:
        train_policy = sim.uniform(rate=0.5)
        test_policies = {
            "A": train_policy,
            "B_sparse": sim.uniform(rate=0.15),
            "B_bursty": sim.bursty(rate=0.5, stay=0.9),
        }
    return train_policy, test_policies


def _config_list(exp: str, smoke: bool) -> List[Tuple[str, float, float]]:
    if exp == "E1":
        configs = [(f"sigma{sigma:g}_gamma{gamma:g}", sigma, gamma)
                   for sigma in (0.25, 0.5, 1.0, 2.0) for gamma in (1.0, 2.0)]
    else:
        configs = [("sigma0.5", 0.5, 0.0)]
    if smoke:
        # A representative middle-noise, lower-informativeness E1 config.
        return [next(c for c in configs if c[1:] == (0.5, 1.0))] if exp == "E1" else configs
    return configs


def _seeds_for(seed: int) -> Tuple[int, int, int, int, int, int]:
    """Stable, disjoint latent/observation seeds shared across architectures."""
    base = 170_000 + int(seed) * 100
    return base, base + 1, base + 2, base + 20, base + 21, base + 22


def _observed_split(n: int, latent_seed: int, obs_seed: int, policy, sigma: float):
    latent = sim.simulate_latent(n, seed=latent_seed)
    observed = sim.observe(latent, policy, sigma=sigma, seed=obs_seed)
    return observed, np.asarray(latent["y"], dtype=np.float32)


def _clean_x(observed) -> np.ndarray:
    x = np.asarray(observed["x"], dtype=np.float32)
    m = np.asarray(observed["m"], dtype=np.float32)
    return np.where(m > 0, x, 0.0).astype(np.float32, copy=False)


def _features_for(arch: str, observed) -> Tuple[np.ndarray, ...]:
    x = np.asarray(observed["x"], dtype=np.float32)
    m = np.asarray(observed["m"], dtype=np.float32)
    if arch == "gru":
        return (build_recurrent_features(x, m),)
    if arch == "grud":
        _, delta = build_locf_delta(x, m)
        return (_clean_x(observed), m, delta)
    if arch == "cde":
        return (build_cde_path(x, m),)
    if arch == "transformer":
        return (_clean_x(observed), m)
    raise ValueError(f"unknown architecture: {arch}")


def _tensorize(features: Sequence[np.ndarray], labels=None):
    import torch

    tensors = tuple(
        torch.from_numpy(np.ascontiguousarray(feature, dtype=np.float32))
        for feature in features
    )
    if labels is None:
        return tensors
    return tensors, torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))


def _forward(model, features: Sequence):
    return model(*features)


def _predict(model, features: Sequence, batch_size: int = 512) -> np.ndarray:
    import torch

    model.eval()
    n = features[0].shape[0]
    chunks = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            args = tuple(value[start:start + batch_size] for value in features)
            chunks.append(torch.sigmoid(_forward(model, args)).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _fit_model(arch: str, train_obs, train_y, val_obs, val_y, empirical_mean: float,
               seed: int, max_epochs: int):
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))

    train_features_np = _features_for(arch, train_obs)
    val_features_np = _features_for(arch, val_obs)
    train_features, y_train = _tensorize(train_features_np, train_y)
    val_features, y_val = _tensorize(val_features_np, val_y)
    model = make_model(
        arch,
        empirical_mean=empirical_mean,
        hidden_size=64,
        max_length=train_features_np[0].shape[1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    n_train = y_train.shape[0]
    batch_size = 256
    patience = 12
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    train_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(n_train)
        sum_loss = 0.0
        for start in range(0, n_train, batch_size):
            indices = order[start:start + batch_size]
            batch = tuple(value.index_select(0, indices) for value in train_features)
            targets = y_train.index_select(0, indices)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.detach()) * targets.shape[0]
        train_history.append(sum_loss / n_train)

        model.eval()
        with torch.no_grad():
            val_logits = _forward(model, val_features)
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


def _atomic_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _bayes_arrays(exp: str, sigma: float, gamma: float, seed: int, n_test: int):
    """Compute Bayes predictions without writing, for the caller's output root."""
    _, test_policies = _policy_setup(exp, sigma, gamma)
    _, _, test_latent_seed, _, _, test_obs_seed = _seeds_for(seed)
    test_latent = sim.simulate_latent(n_test, seed=test_latent_seed)
    arrays: Dict[str, np.ndarray] = {"y": np.asarray(test_latent["y"], dtype=np.int8)}
    a_policy = test_policies["A"]
    v_policy = sim.uniform(rate=0.5)
    for policy_name, policy in test_policies.items():
        observed = sim.observe(test_latent, policy, sigma=sigma, seed=test_obs_seed)
        x, m = observed["x"], observed["m"]
        arrays[f"A_rule_{policy_name}"] = np.asarray(
            bayes.bayes_predict(x, m, sigma, a_policy), dtype=np.float32
        )
        arrays[f"V_rule_{policy_name}"] = np.asarray(
            bayes.bayes_predict(x, m, sigma, v_policy), dtype=np.float32
        )
    return arrays


def _write_bayes_once(out_root: Path, exp: str, config_name: str, sigma: float,
                      gamma: float, seed: int, n_test: int) -> None:
    path = out_root / exp / config_name / f"seed{seed}" / "bayes.npz"
    if path.exists():
        return
    arrays = _bayes_arrays(exp, sigma, gamma, seed, n_test)
    _atomic_npz(path, arrays)


def _fit_task(task: Dict) -> str:
    """Spawn-safe worker: regenerate deterministic cohorts, train, and save shard."""
    import torch

    torch.set_num_threads(1)
    started = time.perf_counter()
    out_root = Path(task["out_root"])
    shard = out_root / task["exp"] / task["config"] / task["arch"] / f"seed{task['seed']}"
    if shard.exists():
        message = f"SKIP {task['exp']} {task['config']} {task['arch']} seed={task['seed']} (shard exists)"
        print(message, flush=True)
        return message

    train_policy, test_policies = _policy_setup(task["exp"], task["sigma"], task["gamma"])
    train_latent_seed, val_latent_seed, test_latent_seed, train_obs_seed, val_obs_seed, test_obs_seed = _seeds_for(task["seed"])
    train_obs, train_y = _observed_split(
        task["n_train"], train_latent_seed, train_obs_seed, train_policy, task["sigma"]
    )
    val_obs, val_y = _observed_split(
        task["n_val"], val_latent_seed, val_obs_seed, train_policy, task["sigma"]
    )

    # The same test latent cohort and observation seed pair all policies.
    test_latent = sim.simulate_latent(task["n_test"], seed=test_latent_seed)
    test_observed = {
        name: sim.observe(test_latent, policy, sigma=task["sigma"], seed=test_obs_seed)
        for name, policy in test_policies.items()
    }

    observed_x = np.asarray(train_obs["x"], dtype=np.float32)
    observed_m = np.asarray(train_obs["m"], dtype=bool)
    empirical_mean = float(observed_x[observed_m].mean()) if observed_m.any() else 0.0
    model, fit_meta = _fit_model(
        task["arch"], train_obs, train_y, val_obs, val_y, empirical_mean,
        task["model_seed"], task["max_epochs"],
    )
    predictions = {
        name: _predict(model, _tensorize(_features_for(task["arch"], observed)))
        for name, observed in test_observed.items()
    }
    predictions["y"] = np.asarray(test_latent["y"], dtype=np.int8)

    elapsed = time.perf_counter() - started
    metadata = {
        "exp": task["exp"],
        "config": task["config"],
        "arch": task["arch"],
        "seed": task["seed"],
        "sigma": task["sigma"],
        "gamma": task["gamma"] if task["exp"] == "E1" else None,
        "best_epoch": fit_meta["best_epoch"],
        "epochs_run": fit_meta["epochs_run"],
        "train_bce": fit_meta["train_bce"],
        "val_bce": fit_meta["val_bce"],
        "wall_time_seconds": elapsed,
        "test_policy_names": list(test_policies),
    }

    shard.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{shard.name}.", dir=shard.parent))
    try:
        with (temp_dir / "preds.npz").open("wb") as handle:
            np.savez_compressed(handle, **predictions)
        (temp_dir / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        # Same-volume rename makes a completed shard visible atomically.
        os.rename(temp_dir, shard)
    except BaseException:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    message = (
        f"DONE {task['exp']} {task['config']} {task['arch']} seed={task['seed']} "
        f"wall={elapsed:.1f}s best_epoch={fit_meta['best_epoch']}"
    )
    print(message, flush=True)
    return message


def _validate_out_root(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"--out must be inside the working folder {ROOT}") from exc
    return resolved


def run(exp: str, workers: int = 4, out: str = "results", smoke: bool = False) -> None:
    if exp not in ("E1", "E2", "all"):
        raise ValueError("exp must be E1, E2, or all")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if smoke and out == "results":
        out = "results_smoke"
    out_root = _validate_out_root(Path(out))
    experiments = ("E1", "E2") if exp == "all" else (exp,)
    sizes = {"n_train": 800, "n_val": 200, "n_test": 1000, "max_epochs": 5} if smoke else {
        "n_train": 8000, "n_val": 2000, "n_test": 8000, "max_epochs": 120
    }
    seeds = (0,) if smoke else (0, 1, 2)

    tasks = []
    for experiment in experiments:
        for config_name, sigma, gamma in _config_list(experiment, smoke):
            for seed in seeds:
                # Bayes files are independent of fitted architecture and saved once.
                _write_bayes_once(out_root, experiment, config_name, sigma, gamma, seed, sizes["n_test"])
                for arch in ARCHITECTURES:
                    shard = out_root / experiment / config_name / arch / f"seed{seed}"
                    if shard.exists():
                        print(
                            f"SKIP {experiment} {config_name} {arch} seed={seed} (shard exists)",
                            flush=True,
                        )
                        continue
                    task = {
                        "exp": experiment,
                        "config": config_name,
                        "sigma": sigma,
                        "gamma": gamma,
                        "arch": arch,
                        "seed": seed,
                        "model_seed": 31_337 + seed * 100 + ARCHITECTURES.index(arch),
                        "out_root": str(out_root),
                        **sizes,
                    }
                    tasks.append(task)

    if not tasks:
        return
    # Explicit spawn avoids inheriting unsafe state from the parent on Windows.
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        for _ in pool.imap_unordered(_fit_task, tasks, chunksize=1):
            pass


def main(argv: Iterable[str] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp", choices=("E1", "E2", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(args.exp, workers=args.workers, out=args.out, smoke=args.smoke)


if __name__ == "__main__":
    mp.freeze_support()
    main()

