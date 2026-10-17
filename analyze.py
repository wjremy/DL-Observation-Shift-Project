"""Summarize simulation shards written by run_grid.py (NumPy and pandas)."""

import argparse
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


EPS = 1e-7
BOOTSTRAP_RESAMPLES = 1000


def _clip_probability(p):
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)


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


def _probe(y, p_model, p_v, p_a):
    response = _logit(p_model)
    values_logit = _logit(p_v)
    design = np.column_stack((np.ones(len(response)), values_logit,
                              _logit(p_a) - values_logit))
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    fitted = design @ coefficients
    residual_sum = float(np.sum((response - fitted) ** 2))
    total_sum = float(np.sum((response - response.mean()) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
    return float(coefficients[1]), float(coefficients[2]), r_squared


def _config_label(path_config, config_meta):
    if isinstance(config_meta, dict):
        # Keep config identity readable and stable across seeds.
        fields = [(key, config_meta[key]) for key in sorted(config_meta)
                  if key not in {"name", "exp", "experiment", "seed"}]
        if fields:
            return ",".join(f"{key}={config_meta[key]}" for key, _ in fields)
        if config_meta.get("name"):
            return str(config_meta["name"])
    if config_meta is not None and not isinstance(config_meta, (dict, list)):
        return str(config_meta)
    return str(path_config)


def _extract_num(pattern, values):
    for value in values:
        if value is None:
            continue
        match = re.search(pattern, str(value), flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return np.nan


def _seed_id(meta, shard_dir):
    if "seed" in meta:
        return int(meta["seed"])
    for part in reversed(shard_dir.parts):
        match = re.fullmatch(r"seed[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError(f"Cannot determine seed for shard {shard_dir}")


def _bayes_path(root, shard_dir, exp, path_config, seed, all_bayes):
    candidates = [
        root / exp / path_config / f"seed{seed}" / "bayes.npz",
        root / exp / path_config / f"seed_{seed}" / "bayes.npz",
        root / exp / path_config / f"bayes_seed{seed}.npz",
        root / exp / path_config / f"seed{seed}_bayes.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    seed_token = f"seed{seed}"
    scored = []
    for candidate in all_bayes:
        rel_parts = candidate.relative_to(root).parts
        score = 0
        if exp in rel_parts:
            score += 4
        if path_config in rel_parts:
            score += 4
        if any(part.lower() in {seed_token, f"seed_{seed}", f"seed-{seed}"}
               for part in rel_parts) or seed_token in candidate.stem.lower():
            score += 4
        # Prefer paths structurally close to the model shard.
        common = 0
        for left, right in zip(shard_dir.parts, candidate.parent.parts):
            if left != right:
                break
            common += 1
        score += common
        if score >= 8:
            scored.append((score, candidate))
    if scored:
        scored.sort(key=lambda item: (-item[0], len(item[1].parts)))
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            raise ValueError(f"Ambiguous bayes.npz match for {shard_dir}: {scored[:2]}")
        return scored[0][1]
    raise FileNotFoundError(f"No matching bayes.npz found for {shard_dir}")


def _bayes_prediction(archive, rule, policy):
    keys = list(archive.files)
    candidates = [f"{rule}_{policy}", f"{rule}__{policy}",
                  f"{rule}-{policy}", f"{policy}_{rule}",
                  f"{rule}{policy}"]
    for key in candidates:
        if key in keys:
            return np.asarray(archive[key], dtype=np.float64)
    normalized = lambda name: re.sub(r"[^a-z0-9]+", "", name.lower())
    target = normalized(f"{rule}_{policy}")
    for key in keys:
        if key.lower() in {"y", "outcome"}:
            continue
        if normalized(key) == target:
            return np.asarray(archive[key], dtype=np.float64)
    raise KeyError(f"Missing {rule} prediction for policy {policy}; keys={keys}")


def _aggregate_seed_rows(frame, keys, metrics):
    rows = []
    for group_key, group in frame.groupby(keys, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(keys, group_key))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[metric] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_seed_min"] = float(values.min()) if len(values) else np.nan
            row[f"{metric}_seed_max"] = float(values.max()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_contrasts(loss_arrays, e1_seed_rows, resamples=BOOTSTRAP_RESAMPLES):
    """Patient-paired model contrasts, preserving seed and policy pairing."""
    archs = sorted(e1_seed_rows["arch"].dropna().unique().tolist())
    configs = sorted(e1_seed_rows["config"].dropna().unique().tolist())
    records = []
    for config in configs:
        config_rows = e1_seed_rows[e1_seed_rows["config"] == config]
        policies = sorted(config_rows["test_policy"].unique().tolist())
        for policy in policies:
            if policy != "A" and policy not in {"B_match", "B_dense"}:
                continue
            metric = "S" if policy == "A" else "X_B"
            for arch1, arch2 in itertools.combinations(archs, 2):
                seed_differences = []
                for seed in sorted(config_rows["seed"].unique()):
                    key1 = (config, int(seed), arch1, policy)
                    key2 = (config, int(seed), arch2, policy)
                    if key1 not in loss_arrays or key2 not in loss_arrays:
                        continue
                    loss1, loss2 = loss_arrays[key1], loss_arrays[key2]
                    if loss1.shape != loss2.shape:
                        raise ValueError(f"Unpaired patient arrays for {key1} and {key2}")
                    seed_differences.append(loss1 - loss2)
                if not seed_differences:
                    continue
                point = float(np.mean([diff.mean() for diff in seed_differences]))
                rng = np.random.default_rng(
                    481516 + sum(ord(ch) for ch in f"{config}|{policy}|{arch1}|{arch2}|{metric}")
                )
                draws = np.empty(resamples, dtype=np.float64)
                for b in range(resamples):
                    per_seed = []
                    for diff in seed_differences:
                        indices = rng.integers(0, len(diff), size=len(diff))
                        per_seed.append(float(np.mean(diff[indices])))
                    draws[b] = float(np.mean(per_seed))
                records.append({
                    "config": config, "test_policy": policy, "metric": metric,
                    "arch_1": arch1, "arch_2": arch2,
                    "mean_difference_arch1_minus_arch2": point,
                    "bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                    "bootstrap_ci_high": float(np.quantile(draws, 0.975)),
                    "bootstrap_resamples": resamples,
                })
    return pd.DataFrame(records)


def analyze(results_dir="results", out_dir=None, bootstrap_resamples=BOOTSTRAP_RESAMPLES):
    root = Path(results_dir).resolve()
    output = Path(out_dir).resolve() if out_dir else Path("analysis").resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")
    output.mkdir(parents=True, exist_ok=True)
    all_bayes = list(root.rglob("bayes.npz"))
    shard_paths = sorted(root.rglob("preds.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No preds.npz shards found under {root}")

    e1_records = []
    e2_records = []
    loss_arrays = {}

    for preds_path in shard_paths:
        shard_dir = preds_path.parent
        meta_path = shard_dir / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing meta.json next to {preds_path}")
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)

        rel_parts = preds_path.relative_to(root).parts
        exp = str(meta.get("exp", meta.get("experiment", rel_parts[0] if rel_parts else "")))
        if exp not in {"E1", "E2"}:
            # Paths specified by the study are results/E1/... and results/E2/....
            exp = next((part for part in rel_parts if part in {"E1", "E2"}), exp)
        if exp not in {"E1", "E2"}:
            raise ValueError(f"Cannot determine E1/E2 for shard {preds_path}")
        path_config = rel_parts[1] if len(rel_parts) > 1 else str(meta.get("config", "default"))
        config_meta = meta.get("config")
        config = _config_label(path_config, config_meta)
        seed = _seed_id(meta, shard_dir)
        arch = str(meta.get("arch", rel_parts[-3] if len(rel_parts) >= 3 else "unknown"))
        sigma = _extract_num(r"sigma\s*[_=:-]?\s*([0-9]+(?:\.[0-9]+)?)", [
            meta.get("sigma"), config, path_config, json.dumps(config_meta, sort_keys=True)
            if isinstance(config_meta, dict) else config_meta,
        ])
        gamma = _extract_num(r"gamma\s*[_=:-]?\s*([0-9]+(?:\.[0-9]+)?)", [
            meta.get("gamma"), config, path_config, json.dumps(config_meta, sort_keys=True)
            if isinstance(config_meta, dict) else config_meta,
        ])
        bayes_file = _bayes_path(root, shard_dir, exp, path_config, seed, all_bayes)

        with np.load(preds_path, allow_pickle=False) as model_data:
            if "y" not in model_data.files:
                raise KeyError(f"{preds_path} has no y array")
            y = np.asarray(model_data["y"], dtype=np.int8).ravel()
            policies = [key for key in model_data.files if key != "y"]
            model_predictions = {policy: np.asarray(model_data[policy], dtype=np.float64).ravel()
                                 for policy in policies}
        with np.load(bayes_file, allow_pickle=False) as bayes_data:
            p_a = {policy: _bayes_prediction(bayes_data, "A_rule", policy).ravel()
                   for policy in policies}
            p_v = {policy: _bayes_prediction(bayes_data, "V_rule", policy).ravel()
                   for policy in policies}

        if any(len(p) != len(y) for mapping in (model_predictions, p_a, p_v)
               for p in mapping.values()):
            raise ValueError(f"Prediction length mismatch in {preds_path}")
        if "A" not in policies:
            raise ValueError(f"Shard {preds_path} has no A test predictions")

        risk_a_a = _risks(y, p_a["A"])
        risk_a_v = _risks(y, p_v["A"])
        risk_a_model = _risks(y, model_predictions["A"])
        gain = risk_a_v["brier"] - risk_a_a["brier"]
        source_excess = risk_a_model["brier"] - risk_a_a["brier"]
        captured = ((risk_a_v["brier"] - risk_a_model["brier"]) / gain
                    if abs(gain) > 1e-15 else np.nan)
        b_values, b_mask, probe_r2 = _probe(
            y, model_predictions["A"], p_v["A"], p_a["A"]
        )

        for policy in policies:
            risk_model = _risks(y, model_predictions[policy])
            risk_b_a = _risks(y, p_a[policy])
            risk_b_v = _risks(y, p_v[policy])
            is_target = policy != "A"
            target_excess = risk_model["brier"] - risk_b_v["brier"]
            transferred_excess = risk_model["brier"] - risk_b_a["brier"]
            justified_cost = risk_b_a["brier"] - risk_b_v["brier"]

            common = {
                "experiment": exp, "config": config, "sigma": sigma, "gamma": gamma,
                "arch": arch, "seed": seed, "test_policy": policy,
            }
            if exp == "E1":
                row = dict(common)
                row.update({
                    "G": gain,
                    "G_bce": risk_a_v["bce"] - risk_a_a["bce"],
                    "J_B": justified_cost if is_target else np.nan,
                    "J_B_bce": risk_b_a["bce"] - risk_b_v["bce"] if is_target else np.nan,
                    "R_A_A_rule_brier": risk_a_a["brier"],
                    "R_A_V_rule_brier": risk_a_v["brier"],
                    "R_A_A_rule_bce": risk_a_a["bce"],
                    "R_A_V_rule_bce": risk_a_v["bce"],
                    "R_A_model_brier": risk_a_model["brier"],
                    "R_A_model_bce": risk_a_model["bce"],
                    "S": source_excess,
                    "S_bce": risk_a_model["bce"] - risk_a_a["bce"],
                    "R_B_A_rule_brier": risk_b_a["brier"],
                    "R_B_V_rule_brier": risk_b_v["brier"],
                    "R_B_A_rule_bce": risk_b_a["bce"],
                    "R_B_V_rule_bce": risk_b_v["bce"],
                    "R_B_model_brier": risk_model["brier"],
                    "R_B_model_bce": risk_model["bce"],
                    "X_B": transferred_excess if is_target else np.nan,
                    "X_B_bce": risk_model["bce"] - risk_b_a["bce"] if is_target else np.nan,
                    "excess_over_B_optimum": target_excess if is_target else np.nan,
                    "excess_over_B_optimum_bce": risk_model["bce"] - risk_b_v["bce"] if is_target else np.nan,
                    "b_values": b_values, "b_mask": b_mask, "probe_r2": probe_r2,
                    "mask_gain_captured": captured,
                })
                e1_records.append(row)
            else:
                e2_records.append({
                    **common,
                    "R_model_brier": risk_model["brier"],
                    "R_V_rule_brier": risk_b_v["brier"],
                    "excess_brier": target_excess,
                    "R_model_bce": risk_model["bce"],
                    "R_V_rule_bce": risk_b_v["bce"],
                    "excess_bce": risk_model["bce"] - risk_b_v["bce"],
                })
            loss_arrays[(config, seed, arch, policy)] = (
                np.asarray(y, dtype=np.float64) - model_predictions[policy]
            ) ** 2

    e1_seed = pd.DataFrame(e1_records)
    e2_seed = pd.DataFrame(e2_records)
    e1_metrics = [
        "G", "G_bce", "J_B", "J_B_bce", "R_A_A_rule_brier", "R_A_V_rule_brier",
        "R_A_A_rule_bce", "R_A_V_rule_bce", "R_A_model_brier",
        "R_A_model_bce", "S", "S_bce", "R_B_A_rule_brier", "R_B_V_rule_brier",
        "R_B_A_rule_bce", "R_B_V_rule_bce", "R_B_model_brier",
        "R_B_model_bce", "X_B", "X_B_bce", "excess_over_B_optimum",
        "excess_over_B_optimum_bce", "b_values",
        "b_mask", "probe_r2", "mask_gain_captured",
    ]
    e2_metrics = ["R_model_brier", "R_V_rule_brier", "excess_brier",
                  "R_model_bce", "R_V_rule_bce", "excess_bce"]
    group_keys = ["experiment", "config", "sigma", "gamma", "arch", "test_policy"]
    e1_summary = _aggregate_seed_rows(e1_seed, group_keys, e1_metrics) if not e1_seed.empty else pd.DataFrame()
    e2_summary = _aggregate_seed_rows(e2_seed, group_keys, e2_metrics) if not e2_seed.empty else pd.DataFrame()
    contrasts = _bootstrap_contrasts(loss_arrays, e1_seed, bootstrap_resamples) if not e1_seed.empty else pd.DataFrame()

    e1_summary.to_csv(output / "e1_summary.csv", index=False)
    e2_summary.to_csv(output / "e2_summary.csv", index=False)
    contrasts.to_csv(output / "contrasts.csv", index=False)

    headline = {
        "n_e1_shards": int(e1_seed[["config", "arch", "seed"]].drop_duplicates().shape[0]) if not e1_seed.empty else 0,
        "n_e2_shards": int(e2_seed[["config", "arch", "seed"]].drop_duplicates().shape[0]) if not e2_seed.empty else 0,
        "e1_mean_G": float(e1_seed["G"].mean()) if not e1_seed.empty else None,
        "e1_mean_J_B": float(e1_seed["J_B"].mean()) if not e1_seed.empty else None,
        "e1_mean_b_mask": float(e1_seed["b_mask"].mean()) if not e1_seed.empty else None,
        "e1_mean_b_values": float(e1_seed["b_values"].mean()) if not e1_seed.empty else None,
        "e2_mean_excess_brier": float(e2_seed["excess_brier"].mean()) if not e2_seed.empty else None,
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(headline, handle, indent=2, allow_nan=False)
    return e1_summary, e2_summary, contrasts, headline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="root results directory")
    parser.add_argument("--out", default=None, help="output directory (default: ./analysis)")
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    _, _, _, headline = analyze(args.results, args.out, args.bootstrap_resamples)
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()

