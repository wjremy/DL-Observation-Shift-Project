"""Summarize exact-rule blends, E4 PoE outputs, and the plain E1 GRU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CONFIG = "sigma1_gamma2"
POLICIES = ("B_match", "B_dense")
EXACT_WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
POE_WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.1), 1)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_x = np.exp(logits[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(p) - np.log1p(-p)


def _brier(y: np.ndarray, probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError(f"prediction shape {p.shape} does not match y shape {y.shape}")
    return float(np.mean((y - p) ** 2))


def _check_y(name: str, candidate: np.ndarray, reference: np.ndarray) -> None:
    candidate = np.asarray(candidate)
    if candidate.shape != reference.shape or not np.array_equal(candidate, reference):
        raise ValueError(
            f"test cohort mismatch: {name} y does not exactly match "
            "results/E1/sigma1_gamma2/seed0/bayes.npz y"
        )


def analyze(out_path: Path | None = None) -> pd.DataFrame:
    e1_dir = ROOT / "results" / "E1" / CONFIG
    e4_dir = ROOT / "results" / "E4" / CONFIG / "seed0"
    with np.load(e1_dir / "seed0" / "bayes.npz") as archive:
        bayes = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(e4_dir / "bias" / "logits.npz") as archive:
        bias = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(e4_dir / "poe_gru" / "logits.npz") as archive:
        poe = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(e1_dir / "gru" / "seed0" / "preds.npz") as archive:
        plain = {key: np.asarray(archive[key]) for key in archive.files}

    y = np.asarray(bayes["y"])
    _check_y("E4 bias", bias["y"], y)
    _check_y("E4 PoE", poe["y"], y)
    _check_y("plain E1 GRU", plain["y"], y)
    for name in ("A", *POLICIES):
        if f"A_rule_{name}" not in bayes or f"V_rule_{name}" not in bayes:
            raise ValueError(f"E1 Bayes output is missing rules for test policy {name}")
        if name not in bias:
            raise ValueError(f"E4 bias output is missing test policy {name}")
        if name not in plain:
            raise ValueError(f"plain E1 GRU output is missing test policy {name}")
        if f"f_{name}" not in poe or f"b_{name}" not in poe:
            raise ValueError(f"E4 PoE output is missing logits for test policy {name}")

    rows = []
    for target_policy in POLICIES:
        for w in EXACT_WEIGHTS:
            predictions = {}
            for name in ("A", target_policy):
                la = _logit(bayes[f"A_rule_{name}"])
                lv = _logit(bayes[f"V_rule_{name}"])
                predictions[name] = _sigmoid(lv + w * (la - lv))
            rows.append({
                "family": "exact_blend", "w": float(w), "test_policy": target_policy,
                "brier_A": _brier(y, predictions["A"]),
                "brier_B": _brier(y, predictions[target_policy]),
            })

        for w in POE_WEIGHTS:
            predictions = {}
            for name in ("A", target_policy):
                predictions[name] = _sigmoid(poe[f"f_{name}"] + w * poe[f"b_{name}"])
            rows.append({
                "family": "poe", "w": float(w), "test_policy": target_policy,
                "brier_A": _brier(y, predictions["A"]),
                "brier_B": _brier(y, predictions[target_policy]),
            })

        bias_predictions = {name: _sigmoid(bias[name]) for name in ("A", target_policy)}
        rows.append({
            "family": "bias", "w": np.nan, "test_policy": target_policy,
            "brier_A": _brier(y, bias_predictions["A"]),
            "brier_B": _brier(y, bias_predictions[target_policy]),
        })
        rows.append({
            "family": "plain_gru", "w": np.nan, "test_policy": target_policy,
            "brier_A": _brier(y, plain["A"]),
            "brier_B": _brier(y, plain[target_policy]),
        })

    frame = pd.DataFrame(rows, columns=["family", "w", "test_policy", "brier_A", "brier_B"])
    if out_path is None:
        out_path = ROOT / "analysis" / "e4_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    with pd.option_context("display.max_rows", 100, "display.width", 120):
        print(frame.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Wrote {out_path}")
    return frame


if __name__ == "__main__":
    analyze()
