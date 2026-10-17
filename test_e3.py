"""Plain-assert E3 checks; run with the study's NumPy-only interpreter."""

import hashlib

import numpy as np

import sim


def _sha256(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def _check_legacy_observe_outputs():
    # These hashes were captured from sim.observe before the E3 implementation.
    latent = sim.simulate_latent(17, 54321)
    cases = (
        ("E1", sim.informative(2.0, 0.3), 1.0,
         "4deeb58b9777ac96d6556150c1e141c64e3441a2f33e22341f7c9ebe9b76e491",
         "c11f114965ae8e4a2a5130bdfdb03179750aca1f1af4e07051f53215662afe8e"),
        ("E2_uniform", sim.uniform(0.5), 0.5,
         "41436742a397cf805f69a6c45a0de965d31b9eea4af6196d381f52745e4cd025",
         "f968854118e998225a7c7950cc4610a8d8eb55131158aad0a15e40a9bf617c62"),
        ("E2_bursty", sim.bursty(0.5, 0.9), 0.5,
         "c9a5b26b010029d9c7167004fc13dcf7cdc4b6f45e1f0b38e7a3f431c85f51e8",
         "3be2ebc952289d83f9f2dbc2209355b8bdc0c247c2da260fea709109f53a37e7"),
    )
    for label, policy, sigma, expected_m, expected_x in cases:
        observed = sim.observe(latent, policy, sigma=sigma, seed=76543)
        assert _sha256(observed["m"]) == expected_m, f"{label} mask output changed"
        assert _sha256(observed["x"]) == expected_x, f"{label} value output changed"
    print("E1/E2 fixed-seed observe outputs: 3/3 baseline hashes match")


def _check_calibrated_rate(policy):
    latent = sim.simulate_latent(4000, seed=330001)
    observed = sim.observe(latent, policy, sigma=1.0, seed=330002)
    realized = float(np.asarray(observed["m"]).mean())
    assert abs(realized - 0.3) < 0.01, f"realized recorded rate {realized:.6f}"
    print(f"recorded alpha={policy.alpha:.8f}; independent realized rate={realized:.6f}")


def _check_redundancy(policy):
    latent = sim.simulate_latent(20000, seed=330003)
    observed = sim.observe(latent, policy, sigma=1.0, seed=330004)
    x = np.asarray(observed["x"], dtype=np.float64)
    m = np.asarray(observed["m"], dtype=bool)
    z = np.asarray(latent["z"], dtype=np.float64)

    last = np.zeros(len(m), dtype=np.float64)
    x_last_rows = []
    z_rows = []
    mask_rows = []
    for t in range(sim.T):
        if t >= 5:
            x_last_rows.append(last.copy())
            z_rows.append(z[:, t].copy())
            mask_rows.append(m[:, t].astype(np.float64))
        last = np.where(m[:, t], x[:, t], last)
    x_last = np.concatenate(x_last_rows)
    z_t = np.concatenate(z_rows)
    mask_t = np.concatenate(mask_rows)
    design = np.column_stack((np.ones(len(mask_t)), x_last, z_t))
    # The mask is Bernoulli with a logistic link, so fit that conditional mean.
    coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(40):
        linear = np.clip(design @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weights = probability * (1.0 - probability)
        gradient = design.T @ (mask_t - probability)
        information = (design.T * weights) @ design
        step = np.linalg.solve(information, gradient)
        coefficients += step
        if np.max(np.abs(step)) < 1e-10:
            break
    assert abs(float(coefficients[2])) < 0.01, f"z_t coefficient {coefficients[2]:.6f}"
    assert float(coefficients[1]) > 0.1, f"x_last coefficient not clearly positive: {coefficients[1]:.6f}"
    print(f"redundancy logistic regression: x_last={coefficients[1]:.6f}, z_t={coefficients[2]:.6f}")


def _check_recorded_probabilities(policy):
    latent = sim.simulate_latent(128, seed=330005)
    observed = sim.observe(latent, policy, sigma=1.0, seed=330006)
    x = np.asarray(observed["x"], dtype=np.float64)
    m = np.asarray(observed["m"], dtype=bool)
    expected = np.empty(x.shape, dtype=np.float64)
    last = np.zeros(len(m), dtype=np.float64)
    for t in range(sim.T):
        expected[:, t] = 1.0 / (1.0 + np.exp(-(policy.alpha + policy.gamma * last)))
        last = np.where(m[:, t], x[:, t], last)
    actual = sim.recorded_probs(x, m, policy)
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-14)
    print("recorded_probs sequential reconstruction: exact to 1e-14")


def main():
    _check_legacy_observe_outputs()
    policy = sim.recorded(2.0, 0.3, 1.0)
    _check_calibrated_rate(policy)
    _check_redundancy(policy)
    _check_recorded_probabilities(policy)
    print("all E3 assertions passed")


if __name__ == "__main__":
    main()
