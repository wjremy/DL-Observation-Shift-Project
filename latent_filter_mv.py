"""Multivariate discrete latent-state filter for E15.

The reference implementation is NumPy-only.  PyTorch is imported lazily so
the data checks and NumPy forward pass work in lightweight environments.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import numpy as np


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
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(probability) - np.log1p(-probability)


def _check_inputs(x, m, parameters: Mapping[str, np.ndarray], w: float):
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m)
    if x.ndim != 3 or m.shape != x.shape:
        raise ValueError("x and m must have matching shape [n, T, D]")
    if x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError("time and variable dimensions must be positive")
    if not np.all((m == 0) | (m == 1)):
        raise ValueError("m must contain only 0/1 values")
    mask = m.astype(bool, copy=False)
    if np.any(mask & ~np.isfinite(x)):
        raise ValueError("observed x values must be finite")
    w = float(w)
    if not 0.0 <= w <= 1.0:
        raise ValueError("w must be in [0, 1]")

    transition_logits = np.asarray(parameters["transition_logits"], dtype=np.float64)
    initial_logits = np.asarray(parameters["initial_logits"], dtype=np.float64)
    mu = np.asarray(parameters["mu"], dtype=np.float64)
    log_sd = np.asarray(parameters["log_sd"], dtype=np.float64)
    c = np.asarray(parameters["c"], dtype=np.float64)
    u = np.asarray(parameters["u"], dtype=np.float64)
    n_states = mu.shape[0] if mu.ndim == 2 else -1
    n_variables = x.shape[2]
    if (mu.shape != (n_states, n_variables)
            or transition_logits.shape != (n_states, n_states)
            or initial_logits.shape != (n_states,)
            or log_sd.shape != (n_variables,)
            or c.shape != (n_states, n_variables)
            or u.shape != (n_states,)):
        raise ValueError("filter parameters have inconsistent state or variable dimensions")
    if not all(np.all(np.isfinite(a)) for a in
               (transition_logits, initial_logits, mu, log_sd, c, u)):
        raise ValueError("filter parameters must be finite")
    return x, mask, transition_logits, initial_logits, mu, log_sd, c, u, w


def numpy_forward(
    x: np.ndarray,
    m: np.ndarray,
    params: Mapping[str, np.ndarray],
    w: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return final outcome probabilities and sequence log marginal likelihoods.

    Value and testing log likelihoods are conditionally independent across
    variables and are summed over D at each time step.
    """
    x, mask, transition_logits, initial_logits, mu, log_sd, c, u, w = _check_inputs(
        x, m, params, w
    )
    n_patients, n_steps, n_variables = x.shape
    n_states = mu.shape[0]
    log_transition = _log_softmax(transition_logits, axis=1)
    log_belief = np.broadcast_to(
        _log_softmax(initial_logits), (n_patients, n_states)
    ).copy()
    log_p_test = -np.logaddexp(0.0, -c)
    log_not_test = -np.logaddexp(0.0, c)
    sd = np.exp(log_sd)
    log_normal_constant = -log_sd - 0.5 * math.log(2.0 * math.pi)
    log_marginal = np.zeros(n_patients, dtype=np.float64)

    for t in range(n_steps):
        log_predicted = _logsumexp(
            log_belief[:, :, None] + log_transition[None, :, :], axis=1
        )
        observed = mask[:, t, :]
        safe_x = np.where(observed, x[:, t, :], 0.0)
        residual = (safe_x[:, :, None] - mu[None, :, :].transpose(0, 2, 1))
        residual = residual / sd[None, :, None]
        log_value_each = log_normal_constant[None, :, None] - 0.5 * residual * residual
        log_value = np.sum(np.where(observed[:, :, None], log_value_each, 0.0), axis=1)
        log_test_each = (
            observed[:, :, None] * log_p_test.T[None, :, :]
            + (~observed)[:, :, None] * log_not_test.T[None, :, :]
        )
        log_test = w * np.sum(log_test_each, axis=1)
        log_unnormalized = log_predicted + log_value + log_test
        normalizer = _logsumexp(log_unnormalized, axis=1)
        log_belief = log_unnormalized - normalizer[:, None]
        log_marginal += normalizer

    posterior = np.exp(log_belief)
    return posterior @ _sigmoid(u), log_marginal


def standardise(x: np.ndarray, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Standardise each variable, clipping finite values to [-5, 5]."""
    values = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != mean.size or sd.shape != mean.shape:
        raise ValueError("mean and sd must match x's final variable dimension")
    safe_sd = np.where(np.isfinite(sd) & (sd > 0.0), sd, 1.0)
    result = (values - mean) / safe_sd
    return np.clip(result, -5.0, 5.0)


def _torch_model_class():
    """Build and return the nn.Module subclass, importing Torch on demand."""
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise ImportError("PyTorch is required for LatentFilterMV") from exc

    class LatentFilterMV(nn.Module):
        """K-state differentiable filter for D conditionally independent variables."""

        def __init__(self, n_states: int = 32, n_variables: int = 1,
                     measurement_rates: np.ndarray | None = None,
                     outcome_rate: float = 0.5):
            super().__init__()
            if n_states < 2 or n_variables < 1:
                raise ValueError("n_states must be at least 2 and n_variables positive")
            self.n_states = int(n_states)
            self.n_variables = int(n_variables)
            transition_logits = torch.zeros(n_states, n_states)
            diagonal = torch.arange(n_states)
            transition_logits[diagonal, diagonal] = 4.0
            transition_logits[diagonal[:-1], diagonal[1:]] = 2.0
            transition_logits[diagonal[1:], diagonal[:-1]] = 2.0
            grid = torch.linspace(-2.5, 2.5, n_states)
            initial_logits = -0.5 * grid.square()
            means = grid[:, None].expand(n_states, n_variables).clone()
            if measurement_rates is None:
                rates = np.full(n_variables, 0.5, dtype=np.float64)
            else:
                rates = np.asarray(measurement_rates, dtype=np.float64).reshape(-1)
                if rates.shape != (n_variables,):
                    raise ValueError("measurement_rates must contain one rate per variable")
            initial_c = torch.as_tensor(_logit(rates), dtype=torch.float32)
            testing_logits = initial_c[None, :].expand(n_states, n_variables).clone()
            initial_u = float(_logit(np.asarray(outcome_rate)))
            self.transition_logits = nn.Parameter(transition_logits)
            self.initial_logits = nn.Parameter(initial_logits)
            self.mu = nn.Parameter(means)
            self.log_sd = nn.Parameter(torch.zeros(n_variables))
            self.c = nn.Parameter(testing_logits)
            self.u = nn.Parameter(torch.full((n_states,), initial_u))

        def forward(self, x, m, w: float = 1.0):
            if not 0.0 <= float(w) <= 1.0:
                raise ValueError("w must be in [0, 1]")
            if x.ndim != 3 or tuple(m.shape) != tuple(x.shape):
                raise ValueError("x and m must have matching shape [n, T, D]")
            if x.shape[2] != self.n_variables:
                raise ValueError("input variable dimension does not match model")
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
                observed = mask[:, t, :]
                safe_x = torch.where(observed, x[:, t, :], torch.zeros_like(x[:, t, :]))
                residual = (safe_x[:, :, None] - self.mu.T[None, :, :]) / sd[None, :, None]
                log_value_each = log_normal_constant[None, :, None] - 0.5 * residual.square()
                log_value = torch.where(
                    observed[:, :, None], log_value_each, torch.zeros_like(log_value_each)
                ).sum(dim=1)
                observed_float = observed.to(dtype=x.dtype)
                log_test_each = (
                    observed_float[:, :, None] * log_p_test.T[None, :, :]
                    + (1.0 - observed_float[:, :, None]) * log_not_test.T[None, :, :]
                )
                log_test = float(w) * log_test_each.sum(dim=1)
                log_unnormalized = log_predicted + log_value + log_test
                normalizer = torch.logsumexp(log_unnormalized, dim=1)
                log_belief = log_unnormalized - normalizer[:, None]
                log_marginal = log_marginal + normalizer

            probabilities = torch.exp(log_belief) @ torch.sigmoid(self.u)
            return probabilities, log_marginal

        def numpy_parameters(self) -> Dict[str, np.ndarray]:
            return {
                name: getattr(self, name).detach().cpu().numpy().copy()
                for name in PARAMETER_NAMES
            }

    LatentFilterMV.__name__ = "LatentFilterMV"
    return LatentFilterMV


def __getattr__(name: str):
    if name == "LatentFilterMV":
        return _torch_model_class()
    raise AttributeError(name)


def _auroc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(y.sum())
    negatives = y.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        stop = start + 1
        while stop < y.size and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0)
                 / (positives * negatives))


def train_filter_mv(
    train_x: np.ndarray,
    train_m: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_m: np.ndarray,
    val_y: np.ndarray,
    *,
    n_states: int = 32,
    max_epochs: int = 100,
    batch_size: int = 256,
    patience: int = 10,
    seed: int = 15_337,
    lambda_nll: float = 1.0,
) -> Tuple[object, Dict[str, object]]:
    """Fit with BCE plus lambda times per-variable, per-time marginal NLL."""
    try:
        import copy
        import torch
        from torch.nn import functional as F
    except ImportError as exc:
        raise ImportError("PyTorch is required to train LatentFilterMV") from exc
    if lambda_nll < 0:
        raise ValueError("lambda_nll must be nonnegative")
    tx = np.asarray(train_x, dtype=np.float32)
    tm = np.asarray(train_m, dtype=np.float32)
    ty = np.asarray(train_y, dtype=np.float32).reshape(-1)
    vx = np.asarray(val_x, dtype=np.float32)
    vm = np.asarray(val_m, dtype=np.float32)
    vy = np.asarray(val_y, dtype=np.float32).reshape(-1)
    if tx.ndim != 3 or tm.shape != tx.shape or len(ty) != len(tx):
        raise ValueError("training arrays must be x/m [n,T,D] and y [n]")
    if vx.ndim != 3 or vm.shape != vx.shape or vx.shape[1:] != tx.shape[1:] or len(vy) != len(vx):
        raise ValueError("validation arrays must match training T and D")
    if len(ty) == 0 or len(vy) == 0:
        raise ValueError("training and validation splits must be nonempty")

    torch.set_num_threads(4)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    model_type = _torch_model_class()
    rates = np.clip(np.mean(tm, axis=(0, 1)), 1e-6, 1.0 - 1e-6)
    model = model_type(
        n_states=n_states, n_variables=tx.shape[2], measurement_rates=rates,
        outcome_rate=float(np.mean(ty)),
    )
    x_train = torch.from_numpy(np.ascontiguousarray(tx))
    m_train = torch.from_numpy(np.ascontiguousarray(tm))
    y_train = torch.from_numpy(np.ascontiguousarray(ty))
    x_val = torch.from_numpy(np.ascontiguousarray(vx))
    m_val = torch.from_numpy(np.ascontiguousarray(vm))
    y_val = torch.from_numpy(np.ascontiguousarray(vy))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    n_train, n_steps, n_variables = tx.shape
    denominator = float(n_steps * n_variables)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    history: Dict[str, object] = {
        "train_loss": [], "val_loss": [], "train_bce": [], "val_bce": [],
        "train_nll_per_step_variable": [], "val_nll_per_step_variable": [],
        "val_auroc": [],
    }

    def evaluate():
        model.eval()
        total_loss = total_bce = total_nll = 0.0
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(y_val), batch_size):
                stop = min(start + batch_size, len(y_val))
                p, log_marginal = model(x_val[start:stop], m_val[start:stop], w=1.0)
                bce = F.binary_cross_entropy(p.clamp(1e-7, 1.0 - 1e-7), y_val[start:stop])
                nll = -log_marginal.mean() / denominator
                size = stop - start
                total_bce += float(bce) * size
                total_nll += float(nll) * size
                total_loss += float(bce + float(lambda_nll) * nll) * size
                probabilities.append(p.cpu().numpy())
        prediction = np.concatenate(probabilities)
        return (total_loss / len(y_val), total_bce / len(y_val), total_nll / len(y_val),
                _auroc_rank(vy, prediction))

    for epoch in range(int(max_epochs)):
        model.train()
        permutation = torch.randperm(n_train)
        total_loss = total_bce = total_nll = 0.0
        for start in range(0, n_train, batch_size):
            indices = permutation[start:start + batch_size]
            xb, mb, yb = x_train[indices], m_train[indices], y_train[indices]
            optimizer.zero_grad(set_to_none=True)
            p, log_marginal = model(xb, mb, w=1.0)
            bce = F.binary_cross_entropy(p.clamp(1e-7, 1.0 - 1e-7), yb)
            nll = -log_marginal.mean() / denominator
            loss = bce + float(lambda_nll) * nll
            loss.backward()
            optimizer.step()
            size = int(yb.shape[0])
            total_bce += float(bce.detach()) * size
            total_nll += float(nll.detach()) * size
            total_loss += float(loss.detach()) * size
        history["train_loss"].append(total_loss / n_train)
        history["train_bce"].append(total_bce / n_train)
        history["train_nll_per_step_variable"].append(total_nll / n_train)
        val_loss, val_bce, val_nll, val_auroc = evaluate()
        history["val_loss"].append(val_loss)
        history["val_bce"].append(val_bce)
        history["val_nll_per_step_variable"].append(val_nll)
        history["val_auroc"].append(val_auroc)
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    history["epochs"] = len(history["train_loss"])
    history["best_val_auroc"] = (
        float(history["val_auroc"][best_epoch - 1]) if best_epoch else float("nan")
    )
    return model, history


def _copy_parameters(parameters: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: np.asarray(parameters[name]).copy() for name in PARAMETER_NAMES}


def _load_model(parameters: Mapping[str, np.ndarray]):
    import torch

    mu = np.asarray(parameters["mu"])
    model_type = _torch_model_class()
    model = model_type(n_states=mu.shape[0], n_variables=mu.shape[1])
    with torch.no_grad():
        for name in PARAMETER_NAMES:
            tensor = torch.as_tensor(
                np.asarray(parameters[name]), dtype=getattr(model, name).dtype
            )
            getattr(model, name).copy_(tensor)
    return model


def refit_testing_head_mv(
    x: np.ndarray,
    m: np.ndarray,
    labels_or_None: np.ndarray | None,
    params: Mapping[str, np.ndarray],
    *,
    batch_seed: int,
    steps: int,
    lr: float,
) -> Dict[str, np.ndarray]:
    """Refit only testing logits c with value/mask NLL and optional BCE."""
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:
        raise ImportError("PyTorch is required to refit the E15 testing head") from exc
    torch.set_num_threads(1)
    torch.manual_seed(int(batch_seed))
    x_array = np.asarray(x, dtype=np.float32)
    m_array = np.asarray(m, dtype=np.float32)
    if x_array.ndim != 3 or m_array.shape != x_array.shape:
        raise ValueError("x and m must have matching shape [n,T,D]")
    n = len(x_array)
    if n == 0:
        raise ValueError("cannot refit the testing head on an empty pool")
    labels = None if labels_or_None is None else np.asarray(labels_or_None, dtype=np.float32).reshape(-1)
    if labels is not None and len(labels) != n:
        raise ValueError("labels must have one value per sequence")
    model = _load_model(params)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "c")
    optimizer = torch.optim.Adam([model.c], lr=float(lr))
    xt = torch.from_numpy(np.ascontiguousarray(x_array))
    mt = torch.from_numpy(np.ascontiguousarray(m_array))
    yt = None if labels is None else torch.from_numpy(np.ascontiguousarray(labels))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(batch_seed))
    denominator = float(x_array.shape[1] * x_array.shape[2])
    model.train()
    for _ in range(int(steps)):
        indices = torch.randint(0, n, (256,), generator=generator)
        xb, mb = xt.index_select(0, indices), mt.index_select(0, indices)
        probability, log_marginal = model(xb, mb, w=1.0)
        loss = -log_marginal.mean() / denominator
        if yt is not None:
            yb = yt.index_select(0, indices)
            bce = F.binary_cross_entropy(probability.clamp(1e-7, 1.0 - 1e-7), yb)
            loss = bce + loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.numpy_parameters()


def finetune_mv(
    x: np.ndarray,
    m: np.ndarray,
    labels: np.ndarray,
    params: Mapping[str, np.ndarray],
    *,
    batch_seed: int,
    steps: int,
    lr: float,
) -> Dict[str, np.ndarray]:
    """Fine-tune every model parameter; no positive cases leaves parameters intact."""
    labels_array = np.asarray(labels, dtype=np.float32).reshape(-1)
    if int(np.sum(labels_array)) == 0:
        return _copy_parameters(params)
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as exc:
        raise ImportError("PyTorch is required to fine-tune the E15 model") from exc
    torch.set_num_threads(1)
    torch.manual_seed(int(batch_seed))
    x_array = np.asarray(x, dtype=np.float32)
    m_array = np.asarray(m, dtype=np.float32)
    if x_array.ndim != 3 or m_array.shape != x_array.shape or len(labels_array) != len(x_array):
        raise ValueError("x, m, and labels have inconsistent shapes")
    n = len(x_array)
    if n == 0:
        return _copy_parameters(params)
    model = _load_model(params)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    xt = torch.from_numpy(np.ascontiguousarray(x_array))
    mt = torch.from_numpy(np.ascontiguousarray(m_array))
    yt = torch.from_numpy(np.ascontiguousarray(labels_array))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(batch_seed))
    denominator = float(x_array.shape[1] * x_array.shape[2])
    model.train()
    for _ in range(int(steps)):
        indices = torch.randint(0, n, (256,), generator=generator)
        xb, mb, yb = xt.index_select(0, indices), mt.index_select(0, indices), yt.index_select(0, indices)
        probability, log_marginal = model(xb, mb, w=1.0)
        bce = F.binary_cross_entropy(probability.clamp(1e-7, 1.0 - 1e-7), yb)
        nll = -log_marginal.mean() / denominator
        loss = bce + nll
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.numpy_parameters()


def check_torch_matches_numpy() -> bool:
    """Torch parity check; run explicitly in an environment with PyTorch."""
    import torch

    torch.manual_seed(1501)
    rng = np.random.default_rng(1502)
    n, t, d, k = 7, 9, 3, 8
    rates = np.array([0.2, 0.47, 0.8])
    model = _torch_model_class()(n_states=k, n_variables=d, measurement_rates=rates).double().eval()
    x = rng.normal(size=(n, t, d)).astype(np.float64)
    m = (rng.random(size=(n, t, d)) < rates[None, None, :]).astype(np.float64)
    x[m == 0] = np.nan
    params = model.numpy_parameters()
    tx, tm = torch.from_numpy(x), torch.from_numpy(m)
    with torch.no_grad():
        for weight in (0.0, 0.37, 1.0):
            torch_probability, torch_marginal = model(tx, tm, w=weight)
            numpy_probability, numpy_marginal = numpy_forward(x, m, params, w=weight)
            np.testing.assert_allclose(torch_probability.numpy(), numpy_probability, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(torch_marginal.numpy(), numpy_marginal, rtol=1e-5, atol=1e-5)
    return True


__all__ = [
    "PARAMETER_NAMES", "LatentFilterMV", "numpy_forward", "train_filter_mv",
    "refit_testing_head_mv", "finetune_mv", "standardise", "check_torch_matches_numpy",
]
