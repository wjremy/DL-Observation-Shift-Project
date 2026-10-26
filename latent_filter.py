"""Learned discrete latent-state filter for the E5 observation-shift experiment.

The NumPy implementation is intentionally kept in lockstep with the Torch
forward pass so the recursion can be inspected and checked without Torch.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np

import bayes
import run_grid
import sim

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # NumPy reference remains usable in the supplied runtime.
    torch = None
    nn = None
    F = None


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
PARAMETER_NAMES = ("transition_logits", "initial_logits", "mu", "log_sd", "c", "u")


def _logsumexp(values: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result


def _log_softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values - _logsumexp(values, axis=axis, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _check_inputs(x: np.ndarray, m: np.ndarray, parameters: Mapping[str, np.ndarray], w: float):
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m)
    if x.ndim != 2 or m.shape != x.shape:
        raise ValueError("x and m must have matching shape [n, T]")
    if x.shape[1] < 1:
        raise ValueError("time dimension T must be positive")
    if not np.all((m == 0) | (m == 1)):
        raise ValueError("m must contain only 0/1 values")
    m = m.astype(bool, copy=False)
    if np.any(m & ~np.isfinite(x)):
        raise ValueError("observed x values must be finite")
    w = float(w)
    # w > 1 (trust the testing evidence more than at A) allowed since E14b.
    if not w >= 0.0:
        raise ValueError("w must be nonnegative")

    transition_logits = np.asarray(parameters["transition_logits"], dtype=np.float64)
    initial_logits = np.asarray(parameters["initial_logits"], dtype=np.float64)
    mu = np.asarray(parameters["mu"], dtype=np.float64)
    c = np.asarray(parameters["c"], dtype=np.float64)
    u = np.asarray(parameters["u"], dtype=np.float64)
    log_sd = float(np.asarray(parameters["log_sd"], dtype=np.float64).reshape(()))
    n_states = mu.size
    if (transition_logits.shape != (n_states, n_states)
            or initial_logits.shape != (n_states,)
            or c.shape != (n_states,) or u.shape != (n_states,)):
        raise ValueError("filter parameters have inconsistent state dimensions")
    if not np.isfinite(log_sd):
        raise ValueError("log_sd must be finite")
    return x, m, transition_logits, initial_logits, mu, log_sd, c, u, w


def numpy_forward(
    x: np.ndarray,
    m: np.ndarray,
    parameters: Mapping[str, np.ndarray],
    w: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return filtered outcome probabilities and sequence log likelihoods.

    ``x`` has shape [patients, T] with NaN at missing values; ``m`` contains
    the corresponding 0/1 mask. The mask likelihood is multiplied by ``w``.
    The transition is applied before each observation update, matching the
    reference recursion in :mod:`bayes`.
    """
    x, m, transition_logits, initial_logits, mu, log_sd, c, u, w = _check_inputs(
        x, m, parameters, w
    )
    n_patients, n_steps = x.shape
    log_transition = _log_softmax(transition_logits, axis=1)
    log_belief = np.broadcast_to(_log_softmax(initial_logits), (n_patients, mu.size)).copy()
    log_p_test = -np.logaddexp(0.0, -c)
    log_not_test = -np.logaddexp(0.0, c)
    sd = math.exp(log_sd)
    log_normal_constant = -log_sd - 0.5 * math.log(2.0 * math.pi)
    log_marginal = np.zeros(n_patients, dtype=np.float64)

    for t in range(n_steps):
        log_predicted = _logsumexp(
            log_belief[:, :, None] + log_transition[None, :, :], axis=1
        )
        observed = m[:, t]
        safe_x = np.where(observed, x[:, t], 0.0)
        residual = (safe_x[:, None] - mu[None, :]) / sd
        log_value = log_normal_constant - 0.5 * residual * residual
        log_value = np.where(observed[:, None], log_value, 0.0)
        log_test = w * (
            observed[:, None] * log_p_test[None, :]
            + (~observed)[:, None] * log_not_test[None, :]
        )
        log_unnormalized = log_predicted + log_value + log_test
        normalizer = _logsumexp(log_unnormalized, axis=1)
        log_belief = log_unnormalized - normalizer[:, None]
        log_marginal += normalizer

    posterior = np.exp(log_belief)
    probabilities = posterior @ _sigmoid(u)
    return probabilities, log_marginal


if torch is not None:
    class LatentFilter(nn.Module):
        """K-state differentiable filter with a deploy-time mask tempering knob."""

        def __init__(self, n_states: int = 32):
            super().__init__()
            if n_states < 2:
                raise ValueError("n_states must be at least 2")
            self.n_states = int(n_states)
            transition_logits = torch.zeros(n_states, n_states)
            diagonal = torch.arange(n_states)
            transition_logits[diagonal, diagonal] = 4.0
            if n_states > 1:
                transition_logits[diagonal[:-1], diagonal[1:]] = 2.0
                transition_logits[diagonal[1:], diagonal[:-1]] = 2.0
            means = torch.linspace(-2.5, 2.5, n_states)
            initial_logits = -0.5 * means.square()
            self.transition_logits = nn.Parameter(transition_logits)
            self.initial_logits = nn.Parameter(initial_logits)
            self.mu = nn.Parameter(means)
            self.log_sd = nn.Parameter(torch.tensor(0.0))
            self.c = nn.Parameter(torch.zeros(n_states))
            self.u = nn.Parameter(torch.zeros(n_states))

        def forward(self, x, m, w: float = 1.0):
            if not 0.0 <= float(w) <= 1.0:
                raise ValueError("w must be in [0, 1]")
            mask = m.to(dtype=torch.bool)
            log_transition = F.log_softmax(self.transition_logits, dim=1)
            log_belief = F.log_softmax(self.initial_logits, dim=0).expand(x.shape[0], -1)
            log_p_test = F.logsigmoid(self.c)
            log_not_test = F.logsigmoid(-self.c)
            sd = torch.exp(self.log_sd)
            log_normal_constant = -self.log_sd - 0.5 * math.log(2.0 * math.pi)
            log_marginal = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

            for t in range(x.shape[1]):
                log_predicted = torch.logsumexp(
                    log_belief.unsqueeze(2) + log_transition.unsqueeze(0), dim=1
                )
                observed = mask[:, t]
                safe_x = torch.where(observed, x[:, t], torch.zeros_like(x[:, t]))
                residual = (safe_x[:, None] - self.mu[None, :]) / sd
                log_value = log_normal_constant - 0.5 * residual.square()
                log_value = torch.where(observed[:, None], log_value, torch.zeros_like(log_value))
                observed_float = observed.to(dtype=x.dtype)[:, None]
                log_test = float(w) * (
                    observed_float * log_p_test[None, :]
                    + (1.0 - observed_float) * log_not_test[None, :]
                )
                log_unnormalized = log_predicted + log_value + log_test
                normalizer = torch.logsumexp(log_unnormalized, dim=1)
                log_belief = log_unnormalized - normalizer[:, None]
                log_marginal = log_marginal + normalizer

            posterior = torch.exp(log_belief)
            probabilities = posterior @ torch.sigmoid(self.u)
            return probabilities, log_marginal

        def numpy_parameters(self) -> Dict[str, np.ndarray]:
            return {
                name: getattr(self, name).detach().cpu().numpy().copy()
                for name in PARAMETER_NAMES
            }
else:
    class LatentFilter:  # pragma: no cover - only used to provide a clear runtime message
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to construct or train LatentFilter")


def train_filter(
    train_x: np.ndarray,
    train_m: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_m: np.ndarray,
    val_y: np.ndarray,
    *,
    n_states: int = 32,
    max_epochs: int = 200,
    batch_size: int = 512,
    patience: int = 15,
    seed: int = 31_337,
) -> Tuple[LatentFilter, Dict[str, object]]:
    """Fit the filter on A data using BCE plus per-step marginal NLL."""
    if torch is None:
        raise ImportError("PyTorch is required to train E5; NumPy forward is available")
    torch.set_num_threads(4)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))

    def tensor_data(x, m, y):
        return (
            torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
        )

    x_train, m_train, y_train = tensor_data(train_x, train_m, train_y)
    x_val, m_val, y_val = tensor_data(val_x, val_m, val_y)
    model = LatentFilter(n_states=n_states)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    n_train = y_train.shape[0]
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    train_history = []
    val_history = []
    train_bce_history = []
    val_bce_history = []
    train_nll_history = []
    val_nll_history = []

    def evaluate(x, m, y):
        model.eval()
        total_loss = total_bce = total_nll = 0.0
        with torch.no_grad():
            for start in range(0, y.shape[0], batch_size):
                stop = start + batch_size
                probabilities, log_marginal = model(x[start:stop], m[start:stop], w=1.0)
                bce = F.binary_cross_entropy(
                    probabilities.clamp(1e-7, 1.0 - 1e-7), y[start:stop]
                )
                nll = -log_marginal.mean() / x.shape[1]
                size = y[start:stop].shape[0]
                total_bce += float(bce) * size
                total_nll += float(nll) * size
                total_loss += float(bce + nll) * size
        count = int(y.shape[0])
        return total_loss / count, total_bce / count, total_nll / count

    for epoch in range(int(max_epochs)):
        model.train()
        permutation = torch.randperm(n_train)
        total_loss = total_bce = total_nll = 0.0
        for start in range(0, n_train, batch_size):
            indices = permutation[start:start + batch_size]
            xb, mb, yb = x_train[indices], m_train[indices], y_train[indices]
            optimizer.zero_grad(set_to_none=True)
            probabilities, log_marginal = model(xb, mb, w=1.0)
            bce = F.binary_cross_entropy(
                probabilities.clamp(1e-7, 1.0 - 1e-7), yb
            )
            nll = -log_marginal.mean() / x_train.shape[1]
            loss = bce + nll
            loss.backward()
            optimizer.step()
            size = yb.shape[0]
            total_bce += float(bce.detach()) * size
            total_nll += float(nll.detach()) * size
            total_loss += float(loss.detach()) * size
        train_bce = total_bce / n_train
        train_nll = total_nll / n_train
        train_history.append(total_loss / n_train)
        train_bce_history.append(train_bce)
        train_nll_history.append(train_nll)

        val_loss, val_bce, val_nll = evaluate(x_val, m_val, y_val)
        val_history.append(val_loss)
        val_bce_history.append(val_bce)
        val_nll_history.append(val_nll)
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs": len(train_history),
        "train_loss": train_history,
        "val_loss": val_history,
        "train_bce": train_bce_history,
        "val_bce": val_bce_history,
        "train_nll_per_step": train_nll_history,
        "val_nll_per_step": val_nll_history,
    }


def true_parameters(grid_size: int, sigma: float, policy) -> Dict[str, np.ndarray]:
    """Build exact simulator parameters on Bayes' discretized latent grid."""
    grid, initial, transition = bayes._grid_and_transition(int(grid_size))
    future_sd = math.sqrt(1.0 - sim.RHO ** (2 * sim.HORIZON))
    future_mean = sim.RHO ** sim.HORIZON * grid
    event_probability = 1.0 - bayes._normal_cdf(
        (sim.THRESHOLD - future_mean) / future_sd
    )
    event_probability = np.clip(event_probability, 1e-14, 1.0 - 1e-14)
    return {
        "transition_logits": np.log(transition),
        "initial_logits": np.log(initial),
        "mu": np.asarray(grid),
        "log_sd": np.asarray(math.log(float(sigma))),
        "c": policy.alpha + policy.gamma * np.asarray(grid),
        "u": np.log(event_probability) - np.log1p(-event_probability),
    }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_path = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def _predict_family(model, test_observed: Mapping[str, Mapping[str, np.ndarray]], weights):
    if torch is None:
        raise ImportError("PyTorch is required to produce learned filter predictions")
    model.eval()
    arrays: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for w in weights:
            for policy_name, observed in test_observed.items():
                x = np.asarray(observed["x"], dtype=np.float32)
                m = np.asarray(observed["m"], dtype=np.float32)
                chunks = []
                for start in range(0, len(x), 512):
                    tx = torch.from_numpy(np.ascontiguousarray(x[start:start + 512]))
                    tm = torch.from_numpy(np.ascontiguousarray(m[start:start + 512]))
                    probabilities, _ = model(tx, tm, w=float(w))
                    chunks.append(probabilities.cpu().numpy())
                arrays[f"p_w{float(w):.2f}_{policy_name}"] = np.concatenate(chunks).astype(
                    np.float32, copy=False
                )
    return arrays


def run(smoke: bool = False) -> Path:
    """Train E5 on the paired E1 seed-0 cohorts and write the result shard."""
    n_train, n_val, n_test, max_epochs = (
        (800, 200, 1000, 3) if smoke else (8000, 2000, 8000, 200)
    )
    out_root = run_grid._validate_out_root(ROOT / ("results_smoke" if smoke else "results"))
    output_dir = out_root / "E5" / "sigma1_gamma2" / "seed0" / "filter"
    expected = (output_dir / "params.npz", output_dir / "probs.npz", output_dir / "meta.json")
    if all(path.exists() for path in expected):
        print(f"SKIP {output_dir} (complete output exists)")
        return output_dir
    if torch is None:
        raise ImportError("PyTorch is required to run latent_filter.py; NumPy checks still work")

    started = time.perf_counter()
    train_policy, test_policies = run_grid._policy_setup("E1", sigma=1.0, gamma=2.0)
    train_latent_seed, val_latent_seed, test_latent_seed, train_obs_seed, val_obs_seed, test_obs_seed = (
        run_grid._seeds_for(0)
    )
    train_observed, train_y = run_grid._observed_split(
        n_train, train_latent_seed, train_obs_seed, train_policy, 1.0
    )
    val_observed, val_y = run_grid._observed_split(
        n_val, val_latent_seed, val_obs_seed, train_policy, 1.0
    )
    test_latent = sim.simulate_latent(n_test, seed=test_latent_seed)
    test_observed = {
        name: sim.observe(test_latent, policy, sigma=1.0, seed=test_obs_seed)
        for name, policy in test_policies.items()
    }

    model, fit_meta = train_filter(
        train_observed["x"], train_observed["m"], train_y,
        val_observed["x"], val_observed["m"], val_y,
        n_states=32, max_epochs=max_epochs, batch_size=512, patience=15,
        seed=31_337,
    )
    probabilities = _predict_family(model, test_observed, DEFAULT_WEIGHTS)
    probabilities["y"] = np.asarray(test_latent["y"], dtype=np.int8)
    meta = {
        "exp": "E5",
        "config": "sigma1_gamma2",
        "architecture": "latent_filter",
        "seed": 0,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "best_epoch": fit_meta["best_epoch"],
        "epochs": fit_meta["epochs"],
        "train_loss": fit_meta["train_loss"],
        "val_loss": fit_meta["val_loss"],
        "train_bce": fit_meta["train_bce"],
        "val_bce": fit_meta["val_bce"],
        "train_nll_per_step": fit_meta["train_nll_per_step"],
        "val_nll_per_step": fit_meta["val_nll_per_step"],
        "wall_time_seconds": time.perf_counter() - started,
        "test_policy_names": list(test_policies),
        "weights": DEFAULT_WEIGHTS.tolist(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(output_dir / "params.npz", model.numpy_parameters())
    _atomic_npz(output_dir / "probs.npz", probabilities)
    _atomic_json(output_dir / "meta.json", meta)
    print(
        f"DONE {output_dir} best_epoch={fit_meta['best_epoch']} "
        f"epochs={fit_meta['epochs']} wall={meta['wall_time_seconds']:.1f}s"
    )
    return output_dir


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
