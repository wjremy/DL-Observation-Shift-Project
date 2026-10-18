"""Pytest-free checks for the E5 latent filter."""

from __future__ import annotations

import numpy as np

import bayes
import latent_filter
import run_grid
import sim


def check_numpy_matches_bayes() -> tuple[float, float]:
    # A 101-point state grid is deliberately coarser than Bayes' 241 points.
    grid_size = 101
    n_test = 512
    train_policy, test_policies = run_grid._policy_setup("E1", sigma=1.0, gamma=2.0)
    _, _, test_latent_seed, _, _, test_obs_seed = run_grid._seeds_for(0)
    test_latent = sim.simulate_latent(n_test, seed=test_latent_seed)
    observed = sim.observe(test_latent, test_policies["A"], sigma=1.0, seed=test_obs_seed)
    parameters = latent_filter.true_parameters(grid_size, 1.0, train_policy)

    filter_a, _ = latent_filter.numpy_forward(observed["x"], observed["m"], parameters, w=1.0)
    filter_values, _ = latent_filter.numpy_forward(
        observed["x"], observed["m"], parameters, w=0.0
    )
    bayes_a = bayes.bayes_predict(observed["x"], observed["m"], 1.0, train_policy)
    bayes_values = bayes.bayes_predict(
        observed["x"], observed["m"], 1.0, sim.uniform(rate=0.5)
    )
    max_a = float(np.max(np.abs(filter_a - bayes_a)))
    max_values = float(np.max(np.abs(filter_values - bayes_values)))
    # The residual includes the 101-vs-241 state discretization difference.
    assert max_a < 0.02, max_a
    assert max_values < 0.02, max_values
    return max_a, max_values


def check_torch_matches_numpy() -> bool:
    if latent_filter.torch is None:
        return False
    torch = latent_filter.torch
    torch.manual_seed(1401)
    model = latent_filter.LatentFilter(n_states=9).double().eval()
    rng = np.random.default_rng(1402)
    x = rng.normal(size=(7, sim.T)).astype(np.float64)
    m = (rng.random(size=x.shape) < 0.31).astype(np.float64)
    x[m == 0] = np.nan
    parameters = model.numpy_parameters()
    tx = torch.from_numpy(x)
    tm = torch.from_numpy(m)
    with torch.no_grad():
        for weight in (0.0, 0.37, 1.0):
            torch_probabilities, torch_log_marginal = model(tx, tm, w=weight)
            numpy_probabilities, numpy_log_marginal = latent_filter.numpy_forward(
                x, m, parameters, w=weight
            )
            np.testing.assert_allclose(
                torch_probabilities.cpu().numpy(), numpy_probabilities, rtol=1e-9, atol=1e-10
            )
            np.testing.assert_allclose(
                torch_log_marginal.cpu().numpy(), numpy_log_marginal, rtol=1e-9, atol=1e-9
            )
    return True


def main() -> None:
    max_a, max_values = check_numpy_matches_bayes()
    print(f"NumPy filter vs Bayes max abs diff, w=1 A-rule: {max_a:.12g}")
    print(f"NumPy filter vs Bayes max abs diff, w=0 values-only: {max_values:.12g}")
    if check_torch_matches_numpy():
        print("Torch vs NumPy forward agreement: passed at w=0, 0.37, 1")
    else:
        print("Torch vs NumPy forward agreement: skipped (Torch unavailable in this runtime)")


if __name__ == "__main__":
    main()
