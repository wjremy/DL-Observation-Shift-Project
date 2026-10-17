"""Simulation utilities for the observation-shift study (NumPy only)."""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np


T = 48
RHO = 0.95
HORIZON = 6
THRESHOLD = 1.0


def _sigmoid(x):
    """Numerically stable logistic function that accepts NumPy arrays."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


def _alpha_for_rate(gamma, rate):
    """Solve E[sigmoid(alpha + gamma Z)] = rate by Gaussian quadrature."""
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must lie strictly between 0 and 1")
    if gamma < 0.0:
        raise ValueError("gamma must be nonnegative")
    if gamma == 0.0:
        return math.log(rate / (1.0 - rate))

    # Hermite nodes integrate against exp(-x**2); sqrt(2) transforms them
    # to a standard normal.  The fixed quadrature makes policy construction
    # deterministic and avoids a costly Monte Carlo root solve.
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    z = math.sqrt(2.0) * nodes
    weights = weights / math.sqrt(math.pi)

    lo, hi = -30.0 - gamma, 30.0 + gamma
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        marginal = float(np.dot(weights, _sigmoid(mid + gamma * z)))
        if marginal < rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Policy:
    """Observation policy and its parameters."""

    name: str
    rate: float
    gamma: float = 0.0
    alpha: float = 0.0
    stay: float = 0.0


def informative(gamma, rate):
    """Return an informative policy with marginal observation rate ``rate``."""
    gamma = float(gamma)
    rate = float(rate)
    alpha = _alpha_for_rate(gamma, rate)
    return Policy(name=f"informative_g{gamma:g}_r{rate:g}", rate=rate,
                  gamma=gamma, alpha=alpha)


def _recorded_mask(z, noisy, uniforms, alpha, gamma):
    """Draw recorded-value-driven masks from fixed noise and uniforms."""
    n = z.shape[0]
    mask = np.empty((n, T), dtype=bool)
    x_last = np.zeros(n, dtype=np.float64)
    for t in range(T):
        p = _sigmoid(alpha + gamma * x_last)
        mask[:, t] = uniforms[:, t] < p
        x_last = np.where(mask[:, t], noisy[:, t], x_last)
    return mask


@lru_cache(maxsize=32)
def _alpha_for_recorded_rate(gamma, rate, sigma):
    """Calibrate recorded-policy alpha on the fixed E3 cohort and draws."""
    latent = simulate_latent(4000, seed=990001)
    z = np.asarray(latent["z"], dtype=np.float64)
    rng = np.random.default_rng(990002)
    # Common random numbers make the calibration deterministic across trials.
    noisy = z + sigma * rng.standard_normal((z.shape[0], T), dtype=np.float64)
    uniforms = rng.random((z.shape[0], T))

    lo, hi = -30.0 - gamma, 30.0 + gamma
    best_alpha = 0.5 * (lo + hi)
    best_error = float("inf")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        mask = _recorded_mask(z, noisy, uniforms, mid, gamma)
        empirical_rate = float(mask.mean())
        error = abs(empirical_rate - rate)
        if error < best_error:
            best_error = error
            best_alpha = mid
        if empirical_rate < rate:
            lo = mid
        else:
            hi = mid
    return float(best_alpha)


def recorded(gamma, rate, sigma):
    """Return a policy whose mask depends on the last recorded noisy value."""
    gamma = float(gamma)
    rate = float(rate)
    sigma = float(sigma)
    if gamma < 0.0:
        raise ValueError("gamma must be nonnegative")
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must lie strictly between 0 and 1")
    if sigma < 0.0:
        raise ValueError("sigma must be nonnegative")
    alpha = _alpha_for_recorded_rate(gamma, rate, sigma)
    return Policy(name=f"recorded_g{gamma:g}_r{rate:g}_s{sigma:g}",
                  rate=rate, gamma=gamma, alpha=alpha)


def recorded_probs(x, m, policy):
    """Return sequential recorded-policy probabilities for observed histories."""
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=bool)
    if x.ndim != 2 or x.shape != m.shape or x.shape[1] != T:
        raise ValueError(f"x and m must have matching shape (n, {T})")
    if not getattr(policy, "name", "").startswith("recorded"):
        raise ValueError("policy must be a recorded policy")
    probabilities = np.empty(x.shape, dtype=np.float64)
    x_last = np.zeros(x.shape[0], dtype=np.float64)
    for t in range(T):
        probabilities[:, t] = _sigmoid(policy.alpha + policy.gamma * x_last)
        x_last = np.where(m[:, t], x[:, t], x_last)
    return probabilities


def uniform(rate):
    """Return a Z-independent Bernoulli observation policy."""
    rate = float(rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must lie between 0 and 1")
    return Policy(name=f"uniform_{rate:g}", rate=rate)


def bursty(rate, stay):
    """Return a stationary two-state Markov observation policy.

    ``stay`` is the stationary, aggregate probability of repeating the prior
    state.  The two conditional switching probabilities are derived from the
    stationary flow constraint.  This gives equal conditional stay
    probabilities when ``rate == 0.5``.
    """
    rate = float(rate)
    stay = float(stay)
    if not 0.0 < rate < 1.0:
        raise ValueError("rate must lie strictly between 0 and 1")
    if not 0.0 <= stay <= 1.0:
        raise ValueError("stay must lie between 0 and 1")
    p10 = (1.0 - stay) / (2.0 * rate)       # 1 -> 0
    p01 = (1.0 - stay) / (2.0 * (1.0 - rate))  # 0 -> 1
    if p10 > 1.0 + 1e-12 or p01 > 1.0 + 1e-12:
        raise ValueError("requested rate/stay pair has no valid two-state chain")
    return Policy(name=f"bursty_{rate:g}_s{stay:g}", rate=rate, stay=stay)


def simulate_latent(n, seed):
    """Simulate stationary Gaussian AR(1) paths and the horizon outcome."""
    n = int(n)
    if n < 0:
        raise ValueError("n must be nonnegative")
    rng = np.random.default_rng(seed)
    total_steps = T + HORIZON
    eps = rng.standard_normal((n, total_steps), dtype=np.float64)
    z = np.empty_like(eps)
    z[:, 0] = eps[:, 0]
    innovation_sd = math.sqrt(1.0 - RHO * RHO)
    for t in range(1, total_steps):
        z[:, t] = RHO * z[:, t - 1] + innovation_sd * eps[:, t]
    z_window = z[:, :T].astype(np.float32)
    z_future = z[:, T + HORIZON - 1].astype(np.float32)
    y = (z_future > THRESHOLD).astype(np.int8)
    return {"z": z_window, "z_future": z_future, "y": y}


def observe(latent, policy, sigma, seed):
    """Draw masks and noisy values for an existing latent cohort."""
    z = np.asarray(latent["z"], dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != T:
        raise ValueError(f"latent['z'] must have shape (n, {T})")
    sigma = float(sigma)
    if sigma < 0.0:
        raise ValueError("sigma must be nonnegative")
    rng = np.random.default_rng(seed)
    n = z.shape[0]

    if policy.name.startswith("recorded"):
        # E3 draws the complete noise matrix before generating any masks.
        noisy = z + sigma * rng.standard_normal((n, T), dtype=np.float64)
        uniforms = rng.random((n, T))
        mask = _recorded_mask(z, noisy, uniforms, policy.alpha, policy.gamma)
    elif policy.name.startswith("informative"):
        p = _sigmoid(policy.alpha + policy.gamma * z)
        mask = (rng.random((n, T)) < p)
    elif policy.name.startswith("bursty"):
        # P(1->0) and P(0->1) preserve the requested stationary rate.
        p10 = (1.0 - policy.stay) / (2.0 * policy.rate)
        p01 = (1.0 - policy.stay) / (2.0 * (1.0 - policy.rate))
        mask = np.empty((n, T), dtype=bool)
        mask[:, 0] = rng.random(n) < policy.rate
        for t in range(1, T):
            u = rng.random(n)
            previous = mask[:, t - 1]
            mask[:, t] = np.where(previous, u >= p10, u < p01)
    else:
        mask = rng.random((n, T)) < policy.rate

    if not policy.name.startswith("recorded"):
        noisy = z + sigma * rng.standard_normal((n, T), dtype=np.float64)
    x = np.where(mask, noisy, np.nan).astype(np.float32)
    return {"x": x, "m": mask.astype(np.int8)}

