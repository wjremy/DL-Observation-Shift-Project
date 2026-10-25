"""E14 simulator extension with a hidden binary outcome-risk trait (NumPy only)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

import bayes
import sim


U_RATE = 0.3
DELTA = 1.0


@lru_cache(maxsize=1)
def outcome_threshold() -> float:
    """Solve the stationary mixture event probability for a 2% outcome rate."""
    lo, hi = -12.0, 12.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        prevalence = (
            (1.0 - U_RATE) * (1.0 - float(bayes._normal_cdf(np.asarray(mid))))
            + U_RATE * (1.0 - float(bayes._normal_cdf(np.asarray(mid - DELTA))))
        )
        if prevalence > 0.02:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def simulate_latent_u(n: int, seed: int) -> dict[str, np.ndarray]:
    """Draw the AR(1) path, independent hidden trait, and E14 outcome."""
    latent = sim.simulate_latent(n, seed)
    z = np.asarray(latent["z"], dtype=np.float32)
    z_future = np.asarray(latent["z_future"], dtype=np.float32)
    # Keep the trait stream independent of sim.simulate_latent's AR stream.
    trait_rng = np.random.default_rng([int(seed), 7])
    u = (trait_rng.random(z.shape[0]) < U_RATE).astype(np.int8)
    threshold = outcome_threshold()
    y = (z_future.astype(np.float64) + DELTA * u > threshold).astype(np.int8)
    return {"z": z, "z_future": z_future, "u": u, "y": y}


@dataclass(frozen=True)
class UPolicy:
    """Testing policy whose probability depends on latent state and trait."""

    name: str
    rate: float
    gamma: float
    beta: float
    alpha: float


def u_informative(gamma: float, beta: float, rate: float = U_RATE) -> UPolicy:
    """Construct a policy with the requested marginal rate by Gauss-Hermite."""
    gamma, beta, rate = float(gamma), float(beta), float(rate)
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must lie strictly between 0 and 1")
    if not math.isfinite(gamma) or not math.isfinite(beta):
        raise ValueError("gamma and beta must be finite")

    nodes, weights = np.polynomial.hermite.hermgauss(80)
    z = math.sqrt(2.0) * nodes
    weights = weights / math.sqrt(math.pi)

    def sigmoid(values: np.ndarray) -> np.ndarray:
        return sim._sigmoid(values)

    lo, hi = -30.0 - abs(gamma) - abs(beta), 30.0 + abs(gamma) + abs(beta)
    for _ in range(100):
        alpha = 0.5 * (lo + hi)
        marginal = float(
            (1.0 - U_RATE) * np.dot(weights, sigmoid(alpha + gamma * z))
            + U_RATE * np.dot(weights, sigmoid(alpha + gamma * z + beta))
        )
        if marginal < rate:
            lo = alpha
        else:
            hi = alpha
    alpha = 0.5 * (lo + hi)
    return UPolicy(
        name=f"u_informative_g{gamma:g}_b{beta:g}_r{rate:g}",
        rate=rate,
        gamma=gamma,
        beta=beta,
        alpha=alpha,
    )


def observe_u(
    latent: dict[str, np.ndarray], policy: UPolicy, sigma: float, seed: int
) -> dict[str, np.ndarray]:
    """Observe latent values under a trait-aware mask, matching sim.observe."""
    z = np.asarray(latent["z"], dtype=np.float64)
    u = np.asarray(latent["u"], dtype=np.float64).reshape(-1)
    if z.ndim != 2 or z.shape[1] != sim.T or u.shape != (z.shape[0],):
        raise ValueError(f"latent z/u must have shapes (n, {sim.T}) and (n,)")
    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError("sigma must be nonnegative")

    rng = np.random.default_rng(seed)
    probability = sim._sigmoid(
        policy.alpha + policy.gamma * z + policy.beta * u[:, None]
    )
    mask = rng.random(z.shape) < probability
    noisy = z + sigma * rng.standard_normal(z.shape, dtype=np.float64)
    x = np.where(mask, noisy, np.nan).astype(np.float32)
    return {"x": x, "m": mask.astype(np.int8)}


def _filter_given_u(
    x: np.ndarray,
    m: np.ndarray,
    sigma: float,
    policy: UPolicy,
    u_value: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return final grid posterior and log evidence for one fixed trait value."""
    grid, initial, transition = bayes._grid_and_transition(bayes.DEFAULT_G)
    n = x.shape[0]
    posterior = np.broadcast_to(initial, (n, initial.size)).copy()
    offset = policy.alpha + policy.beta * float(u_value)
    logits = offset + policy.gamma * grid
    log_p = -np.logaddexp(0.0, -logits)
    log_not_p = -np.logaddexp(0.0, logits)
    log_evidence = np.zeros(n, dtype=np.float64)
    tiny = np.finfo(np.float64).tiny

    for t in range(sim.T):
        predicted = posterior @ transition
        observed = m[:, t]
        likelihood = np.broadcast_to(log_not_p, predicted.shape).copy()
        if np.any(observed):
            residual = (x[observed, t, None] - grid[None, :]) / sigma
            # The Gaussian normalizing constant is common to both u branches
            # and cancels from P(u | history); this is the state-dependent part.
            likelihood[observed] = log_p[None, :] - 0.5 * residual * residual
        log_weights = np.log(np.maximum(predicted, tiny)) + likelihood
        maxima = np.max(log_weights, axis=1)
        log_normalizers = maxima + np.log(
            np.exp(log_weights - maxima[:, None]).sum(axis=1)
        )
        log_evidence += log_normalizers
        posterior = np.exp(log_weights - log_normalizers[:, None])
    return posterior, log_evidence


def bayes_predict_u(
    x: np.ndarray,
    m: np.ndarray,
    sigma: float,
    policy: UPolicy,
    threshold: float | None = None,
    delta: float = DELTA,
    return_u_probability: bool = False,
):
    """Exact grid mixture prediction integrating over the hidden trait.

    When ``return_u_probability`` is true, return ``(prediction, P(u=1|h))``.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=bool)
    if x.ndim != 2 or x.shape != m.shape or x.shape[1] != sim.T:
        raise ValueError(f"x and m must have matching shape (n, {sim.T})")
    sigma = float(sigma)
    delta = float(delta)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if np.any(m & ~np.isfinite(x)):
        raise ValueError("observed x values must be finite")
    if threshold is None:
        threshold = outcome_threshold()
    threshold = float(threshold)

    posteriors = []
    log_evidences = []
    for u_value in (0, 1):
        posterior, log_evidence = _filter_given_u(
            x, m, sigma, policy, u_value
        )
        posteriors.append(posterior)
        log_evidences.append(log_evidence)

    log_weight_0 = math.log(1.0 - U_RATE) + log_evidences[0]
    log_weight_1 = math.log(U_RATE) + log_evidences[1]
    log_total = np.logaddexp(log_weight_0, log_weight_1)
    p_u1 = np.exp(log_weight_1 - log_total)

    grid, _, _ = bayes._grid_and_transition(bayes.DEFAULT_G)
    future_sd = math.sqrt(1.0 - sim.RHO ** (2 * sim.HORIZON))
    future_mean = (sim.RHO ** sim.HORIZON) * grid
    conditional = []
    for u_value in (0, 1):
        event_given_state = 1.0 - bayes._normal_cdf(
            (threshold - delta * u_value - future_mean) / future_sd
        )
        conditional.append(posteriors[u_value] @ event_given_state)
    prediction = (1.0 - p_u1) * conditional[0] + p_u1 * conditional[1]
    prediction = np.asarray(prediction, dtype=np.float64)
    p_u1 = np.asarray(p_u1, dtype=np.float64)
    if return_u_probability:
        return prediction, p_u1
    return prediction
