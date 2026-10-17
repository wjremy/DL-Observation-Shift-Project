"""Grid-based exact Bayes predictors for the observation-shift study."""

from functools import lru_cache
import math

import numpy as np

from sim import HORIZON, RHO, THRESHOLD, T


GRID_MIN = -5.5
GRID_MAX = 5.5
DEFAULT_G = 241


def _erf_array(values):
    """Apply the platform's accurate math.erf to an ndarray."""
    values = np.asarray(values, dtype=np.float64)
    flat = values.ravel()
    return np.fromiter((math.erf(float(v)) for v in flat), dtype=np.float64,
                       count=flat.size).reshape(values.shape)


def _normal_cdf(values):
    return 0.5 * (1.0 + _erf_array(np.asarray(values, dtype=np.float64) / math.sqrt(2.0)))


@lru_cache(maxsize=4)
def _grid_and_transition(G):
    """Return grid, discretized stationary prior, and row-normalized kernel."""
    G = int(G)
    if G < 3:
        raise ValueError("G must be at least 3")
    grid = np.linspace(GRID_MIN, GRID_MAX, G, dtype=np.float64)
    # Evaluate the stationary density and conditional transition density at
    # grid points, then normalize.  The common density constants and grid
    # spacing cancel in these discrete distributions.
    prior = np.exp(-0.5 * grid * grid)
    prior /= prior.sum()

    sd = math.sqrt(1.0 - RHO * RHO)
    residual = (grid[None, :] - RHO * grid[:, None]) / sd
    transition = np.exp(-0.5 * residual * residual)
    transition /= transition.sum(axis=1, keepdims=True)
    grid.setflags(write=False)
    prior.setflags(write=False)
    transition.setflags(write=False)
    return grid, prior, transition


def _is_informative(policy):
    return getattr(policy, "name", "").startswith("informative")


def _bayes_predict_grid(x, m, sigma, assumed_policy, G):
    """Grid-size implementation used by the public predictor and convergence test.

    ``assumed_policy`` controls whether the observation pattern contributes
    information about the latent state.  Any Z-independent policy (uniform or
    bursty) is treated as values-only.  ``G`` is an optional grid-size
    override used for the prescribed 241-versus-481 convergence check.
    """
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=bool)
    if x.ndim != 2 or x.shape != m.shape or x.shape[1] != T:
        raise ValueError(f"x and m must have matching shape (n, {T})")
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    grid, initial, transition = _grid_and_transition(int(G))
    n = x.shape[0]
    posterior = np.broadcast_to(initial, (n, initial.size)).copy()
    informative = _is_informative(assumed_policy)
    tiny = np.finfo(np.float64).tiny
    log_normal_scale = -math.log(sigma)

    if informative:
        gamma = float(assumed_policy.gamma)
        alpha = float(assumed_policy.alpha)
        log_p = -np.logaddexp(0.0, -(alpha + gamma * grid))
        log_not_p = -np.logaddexp(0.0, alpha + gamma * grid)

    for t in range(T):
        # Rows of the kernel are previous states; this matrix product
        # propagates every patient's posterior in one BLAS operation.
        predicted = posterior @ transition
        observed = m[:, t]
        if informative:
            log_likelihood = np.broadcast_to(log_not_p, predicted.shape).copy()
            if np.any(observed):
                residual = (x[observed, t, None] - grid[None, :]) / sigma
                log_likelihood[observed] = (
                    log_p[None, :] + log_normal_scale - 0.5 * residual * residual
                )
        else:
            log_likelihood = np.zeros_like(predicted)
            if np.any(observed):
                residual = (x[observed, t, None] - grid[None, :]) / sigma
                log_likelihood[observed] = (
                    log_normal_scale - 0.5 * residual * residual
                )

        log_weights = np.log(np.maximum(predicted, tiny)) + log_likelihood
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        posterior = np.exp(log_weights)
        posterior /= posterior.sum(axis=1, keepdims=True)

    future_sd = math.sqrt(1.0 - RHO ** (2 * HORIZON))
    future_mean = (RHO ** HORIZON) * grid
    event_given_state = 1.0 - _normal_cdf((THRESHOLD - future_mean) / future_sd)
    return np.asarray(posterior @ event_given_state, dtype=np.float64)


def bayes_predict(x, m, sigma, assumed_policy):
    """Return ``P(Y=1 | x, m)`` using the specified 241-point grid filter."""
    return _bayes_predict_grid(x, m, sigma, assumed_policy, DEFAULT_G)

