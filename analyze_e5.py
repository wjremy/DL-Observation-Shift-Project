"""Summarize the E5 latent-filter tempering curve against E1 references."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sim


ROOT = Path(__file__).resolve().parent
CONFIG = "sigma1_gamma2"
POLICIES = ("B_match", "B_dense")
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(probabilities) - np.log1p(-probabilities)


def _brier(y: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != y.shape:
        raise ValueError(f"prediction shape {probabilities.shape} does not match y shape {y.shape}")
    return float(np.mean((y - probabilities) ** 2))


def _check_y(name: str, candidate: np.ndarray, reference: np.ndarray) -> None:
    candidate = np.asarray(candidate)
    if candidate.shape != reference.shape or not np.array_equal(candidate, reference):
        raise ValueError(f"test cohort mismatch: {name} y does not match E1 Bayes y")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def analyze(out_path: Path | None = None) -> pd.DataFrame:
    e1_dir = ROOT / "results" / "E1" / CONFIG
    filter_dir = ROOT / "results" / "E5" / CONFIG / "seed0" / "filter"
    bayes = _load_npz(e1_dir / "seed0" / "bayes.npz")
    learned = _load_npz(filter_dir / "probs.npz")
    params = _load_npz(filter_dir / "params.npz")
    with np.load(e1_dir / "gru" / "seed0" / "preds.npz") as archive:
        plain = {key: np.asarray(archive[key]) for key in archive.files}

    y = np.asarray(bayes["y"])
    _check_y("E5 filter", learned["y"], y)
    _check_y("plain E1 GRU", plain["y"], y)
    for policy in ("A", *POLICIES):
        for rule in ("A_rule", "V_rule"):
            if f"{rule}_{policy}" not in bayes:
                raise ValueError(f"E1 Bayes output is missing {rule}_{policy}")
        if policy not in plain:
            raise ValueError(f"plain E1 GRU is missing test policy {policy}")
        if policy not in ("A", *POLICIES) or f"p_w0.50_{policy}" not in learned:
            raise ValueError(f"E5 filter output is missing test policy {policy}")

    rows = []
    for target_policy in POLICIES:
        for w in WEIGHTS:
            blended = {}
            for policy in ("A", target_policy):
                log_a = _logit(bayes[f"A_rule_{policy}"])
                log_v = _logit(bayes[f"V_rule_{policy}"])
                blended[policy] = _sigmoid(log_v + float(w) * (log_a - log_v))
            rows.append({
                "family": "exact_blend",
                "w": float(w),
                "test_policy": target_policy,
                "brier_A": _brier(y, blended["A"]),
                "brier_B": _brier(y, blended[target_policy]),
            })

        for w in WEIGHTS:
            prediction_a = learned[f"p_w{float(w):.2f}_A"]
            prediction_b = learned[f"p_w{float(w):.2f}_{target_policy}"]
            rows.append({
                "family": "filter",
                "w": float(w),
                "test_policy": target_policy,
                "brier_A": _brier(y, prediction_a),
                "brier_B": _brier(y, prediction_b),
            })

        rows.append({
            "family": "plain_gru",
            "w": np.nan,
            "test_policy": target_policy,
            "brier_A": _brier(y, plain["A"]),
            "brier_B": _brier(y, plain[target_policy]),
        })

    frame = pd.DataFrame(rows, columns=["family", "w", "test_policy", "brier_A", "brier_B"])
    if out_path is None:
        out_path = ROOT / "analysis" / "e5_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    with pd.option_context("display.max_rows", 120, "display.width", 120):
        print(frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Wrote {out_path}")

    policy = sim.informative(gamma=2.0, rate=0.3)
    mu = np.asarray(params["mu"], dtype=np.float64)
    learned_testing = _sigmoid(np.asarray(params["c"], dtype=np.float64))
    true_testing = _sigmoid(policy.alpha + policy.gamma * mu)
    curve = pd.DataFrame({
        "state": np.arange(mu.size),
        "mu_k": mu,
        "sigmoid_c_k": learned_testing,
        "true_sigmoid_alpha_gamma_mu_k": true_testing,
    })
    print("\nLearned testing-head curve vs true A-policy curve:")
    print(curve.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    return frame


if __name__ == "__main__":
    analyze()
