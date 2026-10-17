"""Plain-assert checks for sim.py and bayes.py; run with the pinned Python."""

import math

import numpy as np

from bayes import _bayes_predict_grid, bayes_predict
from sim import HORIZON, RHO, T, bursty, informative, observe, simulate_latent, uniform


def _normal_cdf_scalar(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _kalman_filter(x, m, sigma):
    """Exact linear Gaussian filtering moments and horizon risk."""
    n = x.shape[0]
    mean = np.zeros(n, dtype=np.float64)
    variance = np.ones(n, dtype=np.float64)
    process_var = 1.0 - RHO * RHO
    for t in range(T):
        if t:
            mean *= RHO
            variance = RHO * RHO * variance + process_var
        observed = m[:, t].astype(bool)
        if np.any(observed):
            gain = variance[observed] / (variance[observed] + sigma * sigma)
            mean[observed] += gain * (x[observed, t] - mean[observed])
            variance[observed] *= 1.0 - gain
    future_mean = (RHO ** HORIZON) * mean
    future_var = 1.0 - (RHO ** (2 * HORIZON)) + (RHO ** (2 * HORIZON)) * variance
    probability = np.fromiter(
        (_normal_cdf_scalar((mu - 1.0) / math.sqrt(var))
         for mu, var in zip(future_mean, future_var)),
        dtype=np.float64, count=n,
    )
    return mean, variance, probability


def _brier(y, p):
    return float(np.mean((np.asarray(y, dtype=np.float64) - p) ** 2))


def test_full_observation_kalman():
    sigma = 0.5
    latent = simulate_latent(160, 101)
    data = observe(latent, uniform(1.0), sigma, 202)
    p_grid = bayes_predict(data["x"], data["m"], sigma, uniform(1.0))
    mean_k, var_k, p_k = _kalman_filter(data["x"], data["m"], sigma)

    # Recover final grid posterior moments through the posterior filter helper
    # only indirectly; the prediction is checked here, and the test's moment
    # comparison below uses a numerical posterior reconstruction function.
    mean_g, var_g = _grid_final_moments(data["x"], data["m"], sigma)
    err_mean = float(np.max(np.abs(mean_g - mean_k)))
    err_var = float(np.max(np.abs(var_g - var_k)))
    err_p = float(np.max(np.abs(p_grid - p_k)))
    assert err_mean < 1e-3, err_mean
    assert err_var < 1e-3, err_var
    assert err_p < 1e-3, err_p
    print(f"Kalman full: mean={err_mean:.3g}, variance={err_var:.3g}, P(Y)={err_p:.3g}")


def _grid_final_moments(x, m, sigma, G=241):
    """Small test-only duplicate of the specified grid filtering recursion."""
    from bayes import _grid_and_transition

    grid, initial, transition = _grid_and_transition(G)
    posterior = np.broadcast_to(initial, (len(x), G)).copy()
    tiny = np.finfo(np.float64).tiny
    for t in range(T):
        predicted = posterior @ transition
        observed = m[:, t].astype(bool)
        likelihood = np.zeros_like(predicted)
        if np.any(observed):
            residual = (x[observed, t, None].astype(np.float64) - grid) / sigma
            likelihood[observed] = -0.5 * residual * residual
        log_weights = np.log(np.maximum(predicted, tiny)) + likelihood
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        posterior = np.exp(log_weights)
        posterior /= posterior.sum(axis=1, keepdims=True)
    mean = posterior @ grid
    variance = posterior @ (grid * grid) - mean * mean
    return mean, variance


def test_random_missing_kalman():
    sigma = 0.5
    latent = simulate_latent(200, 303)
    data = observe(latent, uniform(0.37), sigma, 404)
    p_grid = bayes_predict(data["x"], data["m"], sigma, uniform(0.37))
    mean_k, var_k, p_k = _kalman_filter(data["x"], data["m"], sigma)
    mean_g, var_g = _grid_final_moments(data["x"], data["m"], sigma)
    err_mean = float(np.max(np.abs(mean_g - mean_k)))
    err_var = float(np.max(np.abs(var_g - var_k)))
    err_p = float(np.max(np.abs(p_grid - p_k)))
    assert err_mean < 1e-3, err_mean
    assert err_var < 1e-3, err_var
    assert err_p < 1e-3, err_p
    print(f"Kalman missing: mean={err_mean:.3g}, variance={err_var:.3g}, P(Y)={err_p:.3g}")


def test_calibration_and_brier():
    n = 20000
    sigma = 0.5
    policy = informative(gamma=1.0, rate=0.3)
    latent = simulate_latent(n, 505)
    data = observe(latent, policy, sigma, 606)
    p_a = bayes_predict(data["x"], data["m"], sigma, policy)
    p_v = bayes_predict(data["x"], data["m"], sigma, uniform(0.5))
    y = latent["y"]

    order = np.argsort(p_a)
    deciles = np.array_split(order, 10)
    table = []
    z_scores = []
    for idx in deciles:
        observed = float(np.mean(y[idx]))
        predicted = float(np.mean(p_a[idx]))
        se = math.sqrt(max(predicted * (1.0 - predicted), 1e-12) / len(idx))
        z_score = abs(observed - predicted) / se
        table.append((predicted, observed, len(idx)))
        z_scores.append(z_score)
    max_gap = max(abs(predicted - observed) for predicted, observed, _ in table)
    max_z = max(z_scores)
    brier_a = _brier(y, p_a)
    brier_v = _brier(y, p_v)
    assert max_z < 4.5, (max_gap, max_z, table)
    assert brier_a < brier_v, (brier_a, brier_v)

    p_a_481 = _bayes_predict_grid(data["x"], data["m"], sigma, policy, G=481)
    brier_481 = _brier(y, p_a_481)
    grid_difference = abs(brier_a - brier_481)
    assert grid_difference < 1e-4, grid_difference

    print("Calibration deciles (mean prediction, observed rate): " +
          "; ".join(f"{pred:.3f}/{obs:.3f}" for pred, obs, _ in table))
    print(f"Calibration summary: max gap={max_gap:.4f}, max |z|={max_z:.2f}")
    print(f"A-data Brier (sigma=0.5, gamma=1.0): A_rule={brier_a:.6f}, V_rule={brier_v:.6f}")
    print(f"G=241 vs 481 absolute Brier difference: {grid_difference:.8g}")


def test_informative_rate_solver():
    policy = informative(gamma=2.0, rate=0.3)
    rng = np.random.default_rng(707)
    z = rng.standard_normal(500000)
    # Same stable logistic expression as the policy, evaluated directly.
    logits = policy.alpha + policy.gamma * z
    p = np.empty_like(logits)
    positive = logits >= 0
    p[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    p[~positive] = exp_logits / (1.0 + exp_logits)
    estimated_rate = float(np.mean(p))
    assert abs(estimated_rate - 0.3) < 0.003, estimated_rate
    print(f"Informative alpha rate: {estimated_rate:.5f} (target 0.30000)")


def test_bursty_rate_and_stay():
    n = 20000
    policy = bursty(rate=0.5, stay=0.9)
    latent = simulate_latent(n, 808)
    mask = observe(latent, policy, 0.5, 909)["m"].astype(bool)
    rate_hat = float(mask.mean())
    stay_hat = float(np.mean(mask[:, 1:] == mask[:, :-1]))
    assert abs(rate_hat - 0.5) < 0.01, rate_hat
    assert abs(stay_hat - 0.9) < 0.005, stay_hat
    print(f"Bursty chain: rate={rate_hat:.5f} (target 0.50000), stay={stay_hat:.5f} (target 0.90000)")


if __name__ == "__main__":
    test_full_observation_kalman()
    test_random_missing_kalman()
    test_calibration_and_brier()
    test_informative_rate_solver()
    test_bursty_rate_and_stay()
    print("All five tests passed.")

