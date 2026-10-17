"""Summarize E3 prediction shards (NumPy and pandas only)."""

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import sim


EPS = 1e-7
BOOTSTRAP_RESAMPLES = 1000
POLICY_NAMES = ("A", "B_match", "B_dense")


def _clip_probability(values):
    return np.clip(np.asarray(values, dtype=np.float64), EPS, 1.0 - EPS)


def _risks(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = _clip_probability(p)
    return {
        "brier": float(np.mean((y - p) ** 2)),
        "bce": float(np.mean(-y * np.log(p) - (1.0 - y) * np.log1p(-p))),
    }


def _logit(p):
    p = _clip_probability(p)
    return np.log(p) - np.log1p(-p)


def _gap_since_last(m):
    """Number of unobserved steps after the final observation; T if none."""
    m = np.asarray(m, dtype=bool)
    n = m.shape[0]
    indices = np.arange(sim.T, dtype=np.int64)[None, :]
    last = np.max(np.where(m, indices, -1), axis=1)
    return np.where(last < 0, sim.T, sim.T - 1 - last).astype(np.float64)


def _shortcut_probe(p_model, p_v, m, gap):
    response = _logit(p_model) - _logit(p_v)
    design = np.column_stack((
        np.ones(len(response)),
        np.asarray(m, dtype=np.float64).sum(axis=1) / sim.T,
        np.asarray(gap, dtype=np.float64) / sim.T,
    ))
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    fitted = design @ coefficients
    residual_sum = float(np.sum((response - fitted) ** 2))
    total_sum = float(np.sum((response - response.mean()) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    return coefficients, r_squared


def _bootstrap_ci(differences, rng, resamples=BOOTSTRAP_RESAMPLES):
    differences = np.asarray(differences, dtype=np.float64).ravel()
    n = len(differences)
    draws = np.empty(resamples, dtype=np.float64)
    for b in range(resamples):
        indices = rng.integers(0, n, size=n)
        draws[b] = differences[indices].mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _seed_from_path(path, meta):
    if "seed" in meta:
        return int(meta["seed"])
    for part in reversed(path.parts):
        match = re.fullmatch(r"seed[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot determine seed from {path}")


def _regenerate_test_observations(seed, sigma, gamma, n_test):
    """Recreate the paired test histories from the runner's stable seeds."""
    base = 170_000 + int(seed) * 100
    latent = sim.simulate_latent(n_test, seed=base + 2)
    policies = {
        "A": sim.recorded(gamma=gamma, rate=0.3, sigma=sigma),
        "B_match": sim.uniform(rate=0.3),
        "B_dense": sim.uniform(rate=0.6),
    }
    observed = {
        name: sim.observe(latent, policy, sigma=sigma, seed=base + 22)
        for name, policy in policies.items()
    }
    return latent, observed


def analyze_e3(results_dir="results/E3", output_path="analysis/e3_summary.csv",
               bootstrap_resamples=BOOTSTRAP_RESAMPLES):
    root = Path(results_dir).resolve()
    output = Path(output_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"E3 results directory does not exist: {root}")
    pred_paths = sorted(root.rglob("preds.npz"))
    if not pred_paths:
        raise FileNotFoundError(f"No preds.npz shards found under {root}")

    rows = []
    v_rule_rows = set()
    for preds_path in pred_paths:
        shard = preds_path.parent
        meta_path = shard / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing meta.json next to {preds_path}")
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        sigma = float(meta["sigma"])
        gamma = float(meta["gamma"])
        seed = _seed_from_path(shard, meta)
        with np.load(preds_path, allow_pickle=False) as data:
            y = np.asarray(data["y"], dtype=np.int8).ravel()
            model_predictions = {
                name: np.asarray(data[name], dtype=np.float64).ravel()
                for name in POLICY_NAMES
            }

        latent, observed = _regenerate_test_observations(seed, sigma, gamma, len(y))
        if not np.array_equal(y, np.asarray(latent["y"], dtype=np.int8)):
            raise ValueError(f"Saved outcomes do not match regenerated cohort: {preds_path}")

        config_dir = shard.parent.parent
        bayes_path = config_dir / f"seed{seed}" / "bayes.npz"
        if not bayes_path.is_file():
            raise FileNotFoundError(f"Missing Bayes predictions: {bayes_path}")
        with np.load(bayes_path, allow_pickle=False) as bayes_data:
            v_predictions = {
                name: np.asarray(bayes_data[f"V_rule_{name}"], dtype=np.float64).ravel()
                for name in POLICY_NAMES
            }
            a_predictions = {
                name: np.asarray(bayes_data[f"A_rule_{name}"], dtype=np.float64).ravel()
                for name in POLICY_NAMES
            }
        for name in POLICY_NAMES:
            if not np.array_equal(a_predictions[name], v_predictions[name]):
                if not np.allclose(a_predictions[name], v_predictions[name], rtol=0, atol=1e-7):
                    raise ValueError(f"E3 A_rule and V_rule differ for {name} in {bayes_path}")
            if len(v_predictions[name]) != len(y):
                raise ValueError(f"Prediction length mismatch in {bayes_path}")

        arch = str(meta["arch"])
        config = str(meta.get("config", shard.parent.parent.name))
        risks_v = {name: _risks(y, v_predictions[name]) for name in POLICY_NAMES}
        risks_model = {name: _risks(y, model_predictions[name]) for name in POLICY_NAMES}
        loss_differences = {
            name: (y.astype(np.float64) - model_predictions[name]) ** 2
                 - (y.astype(np.float64) - v_predictions[name]) ** 2
            for name in POLICY_NAMES
        }
        s_value = float(loss_differences["A"].mean())
        seed_tag = f"{config}|{arch}|{seed}"
        s_ci = _bootstrap_ci(
            loss_differences["A"],
            np.random.default_rng(7331 + sum(map(ord, seed_tag + "|S"))),
            bootstrap_resamples,
        )

        for policy_name in POLICY_NAMES:
            m = np.asarray(observed[policy_name]["m"], dtype=bool)
            gap = _gap_since_last(m)
            coeffs, probe_r2 = _shortcut_probe(
                model_predictions[policy_name], v_predictions[policy_name], m, gap
            )
            if policy_name == "A":
                x_value = np.nan
                x_ci = (np.nan, np.nan)
            else:
                x_value = float(loss_differences[policy_name].mean())
                x_ci = _bootstrap_ci(
                    loss_differences[policy_name],
                    np.random.default_rng(7331 + sum(map(ord, seed_tag + "|" + policy_name))),
                    bootstrap_resamples,
                )
            rows.append({
                "config": config,
                "seed": seed,
                "arch": arch,
                "test_policy": policy_name,
                "brier": risks_model[policy_name]["brier"],
                "bce": risks_model[policy_name]["bce"],
                "V_rule_brier": risks_v[policy_name]["brier"],
                "V_rule_bce": risks_v[policy_name]["bce"],
                "S": s_value,
                "S_ci_low": s_ci[0],
                "S_ci_high": s_ci[1],
                "X_B": x_value,
                "X_B_ci_low": x_ci[0],
                "X_B_ci_high": x_ci[1],
                "probe_intercept": float(coeffs[0]),
                "probe_n_obs_coef": float(coeffs[1]),
                "probe_gap_coef": float(coeffs[2]),
                "probe_r2": probe_r2,
                "observed_rate": float(m.mean()),
            })

            # Include one exact values-only Bayes row per test policy.
            v_key = (config, seed, policy_name)
            if v_key not in v_rule_rows:
                v_rule_rows.add(v_key)
                rows.append({
                    "config": config,
                    "seed": seed,
                    "arch": "V_rule",
                    "test_policy": policy_name,
                    "brier": risks_v[policy_name]["brier"],
                    "bce": risks_v[policy_name]["bce"],
                    "V_rule_brier": risks_v[policy_name]["brier"],
                    "V_rule_bce": risks_v[policy_name]["bce"],
                    "S": 0.0,
                    "S_ci_low": 0.0,
                    "S_ci_high": 0.0,
                    "X_B": 0.0 if policy_name != "A" else np.nan,
                    "X_B_ci_low": 0.0 if policy_name != "A" else np.nan,
                    "X_B_ci_high": 0.0 if policy_name != "A" else np.nan,
                    "probe_intercept": 0.0,
                    "probe_n_obs_coef": 0.0,
                    "probe_gap_coef": 0.0,
                    "probe_r2": np.nan,
                    "observed_rate": float(m.mean()),
                })

    summary = pd.DataFrame(rows).sort_values(
        ["config", "seed", "test_policy", "arch"], kind="stable"
    ).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/E3", help="E3 results directory")
    parser.add_argument("--out", default="analysis/e3_summary.csv", help="summary CSV path")
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    analyze_e3(args.results, args.out, args.bootstrap_resamples)


if __name__ == "__main__":
    main()
