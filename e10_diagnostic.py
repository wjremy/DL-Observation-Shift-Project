"""E10 direct transfer diagnostic: pattern versus physiology across site pairs."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from e6_cohort import RAW as P19_RAW, read_hospital
from e6_dial import PENALTY_GRID, fit_with_cv, point_metrics, sigmoid
from e7_dial import _build_source_features


ROOT = Path(__file__).resolve().parent
E7_CACHE = ROOT / "data" / "e7_cache.npz"
E6_CACHE = ROOT / "data" / "e6_cache.npz"
ANALYSIS = ROOT / "analysis"
SPLIT_SEED = 10102026
BOOTSTRAP_SEED = 10102027
BOOTSTRAP_N = 500
FEATURE_SETS = ("pattern", "physiology", "combined")
EPS = 1e-12


def _load_e7_sites() -> tuple[dict, dict]:
    if not E7_CACHE.is_file():
        raise FileNotFoundError(f"Missing E7 cache: {E7_CACHE}")
    with np.load(E7_CACHE, allow_pickle=False) as z:
        cache = {key: z[key].copy() for key in z.files}
    meta = json.loads(str(cache["meta_json"].item()))
    if meta.get("cache_version") != 1:
        raise ValueError("Unexpected E7 cache version")
    sites: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for dataset in ("P19", "P12"):
        sites[dataset] = {}
        for site in cache[f"{dataset}_sites"].astype(str).tolist():
            prefix = f"{dataset}_{site}"
            sites[dataset][site] = {
                "raw_phys": cache[f"{prefix}_raw_phys"].astype(np.float64),
                "raw_pattern": cache[f"{prefix}_raw_pattern"].astype(np.float64),
                "y": cache[f"{prefix}_y"].astype(np.uint8),
                "ids": cache[f"{prefix}_patient_ids"].astype(str),
            }
    return sites, meta


def _build_whole_hospitals() -> tuple[dict, dict]:
    """Rebuild raw E6 cohorts with the exact E6 inclusion and feature rules."""
    out = {}
    exclusions = {}
    for hospital, folder in (("A", "training_setA"), ("B", "training_setB")):
        values = read_hospital(P19_RAW / folder, hospital)
        raw_phys, raw_pattern, y, ids, site_exclusions = values
        out[f"{hospital}-all"] = {
            "raw_phys": raw_phys.astype(np.float64),
            "raw_pattern": raw_pattern.astype(np.float64),
            "y": y.astype(np.uint8),
            "ids": np.asarray(ids, dtype=str),
        }
        exclusions[hospital] = site_exclusions

    # Check that rebuilding followed the same E6 eligible-cohort counts.
    if E6_CACHE.is_file():
        with np.load(E6_CACHE, allow_pickle=False) as z:
            e6_meta = json.loads(str(z["meta_json"].item()))
        expected = e6_meta.get("hospital_sizes", {})
        actual = {h: len(out[f"{h}-all"]["y"]) for h in ("A", "B")}
        if expected and actual != expected:
            raise ValueError(f"Rebuilt E6 cohort counts differ from e6_cache: {actual} vs {expected}")
    return out, exclusions


def _site_splits(sites: dict[str, dict], dataset_index: int) -> dict[str, np.ndarray]:
    split_map = {}
    for site_index, (site, values) in enumerate(sites.items()):
        n = len(values["y"])
        if n < 20:
            raise ValueError(f"Site too small for E10 split: {site}, n={n}")
        rng = np.random.default_rng(SPLIT_SEED + dataset_index * 1000 + site_index)
        order = rng.permutation(n)
        n_fit = n // 2
        split_map[site] = {"fit": order[:n_fit], "eval": order[n_fit:]}
        if len(np.unique(values["y"][order[:n_fit]])) < 2:
            raise ValueError(f"One outcome class in fit half: {site}")
    return split_map


def _feature_matrix(raw_phys: np.ndarray, raw_pattern: np.ndarray, n_dynamic: int,
                    prep: dict, feature_set: str) -> np.ndarray:
    medians = np.asarray(prep["impute_medians"], dtype=np.float64)
    imputed = np.where(np.isfinite(raw_phys), raw_phys, medians[None, :])
    physiology = np.column_stack((
        imputed[:, :n_dynamic],
        imputed[:, :n_dynamic] ** 2,
        imputed[:, n_dynamic:],
    ))
    phys_mean = np.asarray(prep["phys_mean"], dtype=np.float64)
    phys_scale = np.asarray(prep["phys_scale"], dtype=np.float64)
    pat_mean = np.asarray(prep["pattern_mean"], dtype=np.float64)
    pat_scale = np.asarray(prep["pattern_scale"], dtype=np.float64)
    x_phys = (physiology - phys_mean) / phys_scale
    x_pattern = (raw_pattern - pat_mean) / pat_scale
    if feature_set == "physiology":
        return x_phys
    if feature_set == "pattern":
        return x_pattern
    if feature_set == "combined":
        return np.column_stack((x_phys, x_pattern))
    raise ValueError(f"Unknown feature set: {feature_set}")


def _rank_groups(scores: np.ndarray) -> tuple[np.ndarray, int]:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    group_id_sorted = np.repeat(np.arange(len(starts), dtype=np.int32), np.diff(np.r_[starts, len(scores)]))
    group_id = np.empty(len(scores), dtype=np.int32)
    group_id[order] = group_id_sorted
    return group_id, len(starts)


def _weighted_auc(y: np.ndarray, counts: np.ndarray, group_id: np.ndarray,
                  n_groups: int) -> float:
    positive_n = float(np.dot(counts, y))
    negative_n = float(np.sum(counts) - positive_n)
    if positive_n <= 0 or negative_n <= 0:
        return float("nan")
    positive = np.bincount(group_id, weights=counts * y, minlength=n_groups)
    negative = np.bincount(group_id, weights=counts * (1 - y), minlength=n_groups)
    negative_before = np.cumsum(negative) - negative
    concordant = np.sum(positive * (negative_before + 0.5 * negative))
    return float(concordant / (positive_n * negative_n))


def _paired_gap_bootstrap(y: np.ndarray, local_p: np.ndarray, transferred_p: np.ndarray,
                          seed: int) -> dict[str, object]:
    """Percentile intervals for paired local-minus-transfer AUROC and LL gaps."""
    y = np.asarray(y, dtype=np.uint8)
    probs = np.column_stack((local_p, transferred_p)).astype(np.float64)
    probs = np.clip(probs, EPS, 1.0 - EPS)
    n = len(y)
    rank_info = [_rank_groups(probs[:, j]) for j in range(2)]
    losses = -(y[:, None] * np.log(probs) + (1 - y[:, None]) * np.log1p(-probs))
    ll_gaps = np.empty(BOOTSTRAP_N, dtype=np.float64)
    auc_gaps = np.full(BOOTSTRAP_N, np.nan, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for b in range(BOOTSTRAP_N):
        sample = rng.integers(0, n, size=n, dtype=np.int32)
        counts = np.bincount(sample, minlength=n).astype(np.float64)
        ll = counts @ losses / n
        ll_gaps[b] = ll[1] - ll[0]  # transferred minus local
        aucs = [_weighted_auc(y, counts, group_id, n_groups)
                for group_id, n_groups in rank_info]
        auc_gaps[b] = aucs[0] - aucs[1]  # local minus transferred
    finite_auc = auc_gaps[np.isfinite(auc_gaps)]
    return {
        "auroc_gap_ci_low": float(np.percentile(finite_auc, 2.5)) if len(finite_auc) else float("nan"),
        "auroc_gap_ci_high": float(np.percentile(finite_auc, 97.5)) if len(finite_auc) else float("nan"),
        "log_loss_gap_ci_low": float(np.percentile(ll_gaps, 2.5)),
        "log_loss_gap_ci_high": float(np.percentile(ll_gaps, 97.5)),
        "bootstrap_auc_valid": int(len(finite_auc)),
    }


def _coverage(dataset: str, site: str, values: dict, raw_names: list[str],
              n_dynamic: int) -> tuple[list[dict], dict[str, float]]:
    raw_phys = values["raw_phys"]
    n = len(raw_phys)
    rows = []
    measurement_rates: dict[str, float] = {}
    imputed_shares: dict[str, float] = {}
    for j, name in enumerate(raw_names):
        if j < n_dynamic:
            # E7 pattern columns are recent variables, then ever-measured variables.
            rate = float(np.mean(values["raw_pattern"][:, n_dynamic + j]))
            variable_type = "time_varying"
        else:
            # Static covariates have availability, not repeated window measurements.
            rate = float(np.isfinite(raw_phys[:, j]).mean())
            variable_type = "static"
        imputed = float((~np.isfinite(raw_phys[:, j])).mean())
        measurement_rates[name] = rate
        imputed_shares[name] = imputed
        rows.append({
            "dataset": dataset, "site": site, "variable": name,
            "variable_type": variable_type, "n": n,
            "measurement_rate": rate, "imputed_share": imputed,
        })
    return rows, {
        "mean_imputed_share_physiology": float(np.mean(list(imputed_shares.values()))),
        "measurement_rates": measurement_rates,
        "imputed_shares": imputed_shares,
    }


def _pair_group(dataset: str, source: str, target: str) -> str:
    if dataset == "P12":
        return "P12"
    if source.endswith("-all"):
        return "P19 whole hospital"
    src_hosp, src_unit = source.split("-")
    tgt_hosp, tgt_unit = target.split("-")
    if src_hosp == tgt_hosp:
        return "P19 other unit, same hospital"
    if src_unit == tgt_unit:
        return "P19 same unit, other hospital"
    return "P19 other unit, other hospital"


def _make_pairs() -> list[tuple[str, str, str]]:
    pairs = []
    p19_units = ("A-MICU", "A-SICU", "B-MICU", "B-SICU")
    for source in p19_units:
        for target in p19_units:
            if source != target:
                pairs.append(("P19", source, target))
    pairs.extend((("P19", "A-all", "B-all"), ("P19", "B-all", "A-all")))
    p12_sites = ("CCU", "CSRU", "MICU", "SICU")
    for source in p12_sites:
        for target in p12_sites:
            if source != target:
                pairs.append(("P12", source, target))
    return pairs


def main() -> None:
    started = time.perf_counter()
    unit_sites, e7_meta = _load_e7_sites()
    whole_sites, e6_rebuild_exclusions = _build_whole_hospitals()
    sites = {"P19": dict(unit_sites["P19"]), "P12": dict(unit_sites["P12"])}
    sites["P19"].update(whole_sites)

    p19_names = e7_meta["P19"]["raw_phys_names"]
    p12_names = e7_meta["P12"]["raw_phys_names"]
    raw_names_by_ds = {"P19": p19_names, "P12": p12_names}
    dynamic_by_ds = {"P19": len(p19_names) - 3, "P12": len(p12_names) - 4}
    dataset_index = {"P19": 0, "P12": 1}

    split_maps = {}
    fit_records = {}
    site_rows = []
    coverage_rows = []
    coverage_stats = {}

    for dataset in ("P19", "P12"):
        dataset_sites = sites[dataset]
        split_maps[dataset] = _site_splits(dataset_sites, dataset_index[dataset])
        n_dynamic = dynamic_by_ds[dataset]
        raw_names = raw_names_by_ds[dataset]
        for site, values in dataset_sites.items():
            coverage, stats = _coverage(dataset, site, values, raw_names, n_dynamic)
            coverage_rows.extend(coverage)
            coverage_stats[(dataset, site)] = stats
            fit_idx = split_maps[dataset][site]["fit"]
            eval_idx = split_maps[dataset][site]["eval"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                x_all, x_phys, prep = _build_source_features(
                    values["raw_phys"], values["raw_pattern"], n_dynamic, fit_idx
                )
            n_phys = x_phys.shape[1]
            x_pattern = x_all[:, n_phys:]
            x_by_feature = {
                "pattern": x_pattern,
                "physiology": x_phys,
                "combined": x_all,
            }
            fit_records[(dataset, site)] = {"prep": prep, "models": {}, "eval_idx": eval_idx}
            for feature_set in FEATURE_SETS:
                x = x_by_feature[feature_set]
                y = values["y"]
                fit_started = time.perf_counter()
                beta, penalty, cv = fit_with_cv(
                    x[fit_idx], y[fit_idx],
                    seed=SPLIT_SEED + dataset_index[dataset] * 1000 + list(dataset_sites).index(site) * 10
                         + FEATURE_SETS.index(feature_set),
                )
                local_logits = beta[0] + x[eval_idx] @ beta[1:]
                local_p = sigmoid(local_logits)
                local_metrics = point_metrics(y[eval_idx], local_p)
                fit_records[(dataset, site)]["models"][feature_set] = {
                    "beta": beta, "penalty": penalty, "cv": cv,
                    "local_metrics": local_metrics,
                }
                site_rows.append({
                    "dataset": dataset, "site": site, "feature_set": feature_set,
                    "n": len(y), "fit_n": len(fit_idx), "evaluation_n": len(eval_idx),
                    "outcome_cases": int(y.sum()), "evaluation_cases": int(y[eval_idx].sum()),
                    "selected_penalty": penalty,
                    "local_auroc": local_metrics["auroc"],
                    "mean_imputed_share_physiology": stats["mean_imputed_share_physiology"],
                    "cv_log_loss": {str(k): v for k, v in cv.items()},
                })
                elapsed = time.perf_counter() - fit_started
                print(f"{dataset}/{site} {feature_set}: penalty={penalty:g}, local AUROC={local_metrics['auroc']:.4f}, fit/CV={elapsed:.1f}s", flush=True)

    pair_rows = []
    pairs = _make_pairs()
    for pair_index, (dataset, source, target) in enumerate(pairs):
        source_values = sites[dataset][source]
        target_values = sites[dataset][target]
        source_fit = split_maps[dataset][source]["fit"]
        target_eval = split_maps[dataset][target]["eval"]
        n_dynamic = dynamic_by_ds[dataset]
        source_prep = fit_records[(dataset, source)]["prep"]
        target_prep = fit_records[(dataset, target)]["prep"]
        x_target_by_feature = {
            feature_set: _feature_matrix(
                target_values["raw_phys"], target_values["raw_pattern"],
                n_dynamic, source_prep, feature_set,
            )
            for feature_set in FEATURE_SETS
        }
        x_target_local_by_feature = {
            feature_set: _feature_matrix(
                target_values["raw_phys"], target_values["raw_pattern"],
                n_dynamic, target_prep, feature_set,
            )
            for feature_set in FEATURE_SETS
        }
        source_train_ids = source_values["ids"][source_fit]
        if np.intersect1d(source_train_ids, target_values["ids"]).size:
            raise AssertionError(f"Source fit patients overlap target: {dataset} {source}->{target}")

        source_cov = coverage_stats[(dataset, source)]
        target_cov = coverage_stats[(dataset, target)]
        dynamic_names = raw_names_by_ds[dataset][:n_dynamic]
        all_names = raw_names_by_ds[dataset]
        mean_rate_diff = float(np.mean([
            abs(source_cov["measurement_rates"][name] - target_cov["measurement_rates"][name])
            for name in dynamic_names
        ]))
        mean_imputed_diff = float(np.mean([
            abs(source_cov["imputed_shares"][name] - target_cov["imputed_shares"][name])
            for name in all_names
        ]))

        for feature_index, feature_set in enumerate(FEATURE_SETS):
            model = fit_records[(dataset, source)]["models"][feature_set]
            beta = model["beta"]
            local_model = fit_records[(dataset, target)]["models"][feature_set]
            y_eval = target_values["y"][target_eval]
            local_p = sigmoid(
                local_model["beta"][0]
                + x_target_local_by_feature[feature_set][target_eval] @ local_model["beta"][1:]
            )
            transferred_p = sigmoid(
                beta[0] + x_target_by_feature[feature_set][target_eval] @ beta[1:]
            )
            local_metrics = point_metrics(y_eval, local_p)
            transferred_metrics = point_metrics(y_eval, transferred_p)
            bootstrap = _paired_gap_bootstrap(
                y_eval, local_p, transferred_p,
                seed=BOOTSTRAP_SEED + pair_index * 10 + feature_index,
            )
            pair_rows.append({
                "dataset": dataset, "source": source, "target": target,
                "pair": f"{source}->{target}", "pair_group": _pair_group(dataset, source, target),
                "feature_set": feature_set,
                "target_evaluation_n": len(y_eval), "target_evaluation_cases": int(y_eval.sum()),
                "local_auroc": local_metrics["auroc"],
                "transferred_auroc": transferred_metrics["auroc"],
                "auroc_gap": local_metrics["auroc"] - transferred_metrics["auroc"],
                "auroc_gap_ci_low": bootstrap["auroc_gap_ci_low"],
                "auroc_gap_ci_high": bootstrap["auroc_gap_ci_high"],
                "local_log_loss": local_metrics["log_loss"],
                "transferred_log_loss": transferred_metrics["log_loss"],
                "log_loss_gap": transferred_metrics["log_loss"] - local_metrics["log_loss"],
                "log_loss_gap_ci_low": bootstrap["log_loss_gap_ci_low"],
                "log_loss_gap_ci_high": bootstrap["log_loss_gap_ci_high"],
                "mean_measurement_rate_abs_diff": mean_rate_diff,
                "mean_imputed_share_abs_diff": mean_imputed_diff,
                "source_selected_penalty": model["penalty"],
                "target_local_selected_penalty": local_model["penalty"],
                "bootstrap_auc_valid": bootstrap["bootstrap_auc_valid"],
            })
        print(f"{dataset} {source}->{target}: coverage Δ rate={mean_rate_diff:.3f}, imputed={mean_imputed_diff:.3f}", flush=True)

    pair_df = pd.DataFrame.from_records(pair_rows)
    site_df = pd.DataFrame.from_records(site_rows)
    coverage_df = pd.DataFrame.from_records(coverage_rows)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    paths = {
        "gaps": ANALYSIS / "e10_gaps.csv",
        "sites": ANALYSIS / "e10_sites.csv",
        "coverage": ANALYSIS / "e10_coverage.csv",
        "summary": ANALYSIS / "e10_summary.json",
    }
    pair_df.to_csv(paths["gaps"], index=False)
    site_df.to_csv(paths["sites"], index=False)
    coverage_df.to_csv(paths["coverage"], index=False)

    phys = pair_df[pair_df["feature_set"] == "physiology"]
    x = phys["mean_imputed_share_abs_diff"].to_numpy(dtype=np.float64)
    y_gap = phys["auroc_gap"].to_numpy(dtype=np.float64)
    pearson = float(np.corrcoef(x, y_gap)[0, 1]) if np.std(x) > 0 and np.std(y_gap) > 0 else float("nan")
    # pandas' method="spearman" delegates to SciPy, which is not installed in
    # the requested NumPy/pandas-only runtime; rank then use Pearson directly.
    spearman = float(pd.Series(x).rank().corr(pd.Series(y_gap).rank()))
    group_summary = []
    for group, group_df in pair_df.groupby("pair_group", sort=False):
        pats = group_df[group_df["feature_set"] == "pattern"]["auroc_gap"]
        phys_gaps = group_df[group_df["feature_set"] == "physiology"]["auroc_gap"]
        group_summary.append({
            "pair_group": group, "n_pairs": int(len(pats)),
            "mean_pattern_auroc_gap": float(pats.mean()),
            "mean_physiology_auroc_gap": float(phys_gaps.mean()),
        })
    summary = {
        "spec": "E10",
        "split_seed_base": SPLIT_SEED,
        "split_seed_rule": "base + dataset_index*1000 + site_index; default_rng permutation; first floor(n/2) fit, remainder evaluation",
        "bootstrap_seed_base": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_N,
        "bootstrap_interval": "percentile 2.5th and 97.5th; resample target evaluation patients paired for local and transferred predictions",
        "metric_gap_conventions": {
            "auroc_gap": "local AUROC - transferred AUROC",
            "log_loss_gap": "transferred log loss - local log loss",
        },
        "pair_counts": {"all": len(pairs), "P19_unit": 12, "P19_whole_hospital": 2, "P12": 12},
        "feature_sets": list(FEATURE_SETS),
        "penalty_grid_average_bce": PENALTY_GRID.tolist(),
        "preprocessing": "Each source/local fit uses median imputation and physiology/pattern mean/std from that site's fit half only; physiology expands dynamic values with squares before standardization; source preprocessing applied to target for transfer.",
        "whole_hospital_data_source": "Rebuilt A-all/B-all raw unimputed feature arrays in memory using e6_cohort.read_hospital and its E6 cohort rules; e6_cache.npz contained transformed, imputed X matrices but no raw unimputed feature arrays.",
        "whole_hospital_rebuild": {
            "e6_cache_expected_site_sizes": json.loads(str(np.load(E6_CACHE, allow_pickle=False)["meta_json"].item())).get("hospital_sizes", {}) if E6_CACHE.is_file() else {},
            "rebuilt_site_sizes": {site: len(values["y"]) for site, values in whole_sites.items()},
            "exclusions": e6_rebuild_exclusions,
        },
        "coverage_definitions": {
            "dynamic_measurement_rate": "mean E7 ever-measured indicator for time-varying raw physiology variables",
            "static_measurement_rate": "finite raw-value availability for static covariates",
            "imputed_share": "fraction of patients with a missing raw physiology value, i.e. median-imputed by the model",
            "pair_mean_measurement_rate_abs_diff": "mean absolute source-target difference over dynamic variables",
            "pair_mean_imputed_share_abs_diff": "mean absolute source-target difference over all raw physiology variables, including static covariates",
        },
        "coverage_physiology_gap_association": {
            "pearson_mean_imputed_share_difference_vs_physiology_auroc_gap": pearson,
            "spearman_mean_imputed_share_difference_vs_physiology_auroc_gap": spearman,
            "n_pairs": int(len(phys)),
        },
        "group_summary": group_summary,
        "site_results": site_rows,
        "pair_results": pair_rows,
        "outputs": [str(paths[k].relative_to(ROOT)) for k in ("gaps", "sites", "coverage", "summary")],
        "elapsed_analysis_seconds": float(time.perf_counter() - started),
    }
    with paths["summary"].open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print("\nE10 pair diagnostic: pair | group | pattern AUROC gap [95% CI] | physiology AUROC gap [95% CI] | combined AUROC gap [95% CI] | mean |Δ measurement rate| | mean |Δ imputed share|", flush=True)
    by_pair = pair_df.pivot(index=["dataset", "pair", "pair_group", "mean_measurement_rate_abs_diff", "mean_imputed_share_abs_diff"], columns="feature_set")
    for (dataset, pair, group, rate_diff, imputed_diff), row in by_pair.iterrows():
        def fmt(feature: str) -> str:
            return (f"{row[('auroc_gap', feature)]:+.3f}"
                    f"[{row[('auroc_gap_ci_low', feature)]:+.3f},"
                    f"{row[('auroc_gap_ci_high', feature)]:+.3f}]")
        print(f"{dataset} {pair} | {group} | {fmt('pattern')} | {fmt('physiology')} | {fmt('combined')} | {rate_diff:.3f} | {imputed_diff:.3f}", flush=True)

    print("\nMean AUROC gaps by group: group | pattern | physiology", flush=True)
    for row in group_summary:
        print(f"{row['pair_group']} | {row['mean_pattern_auroc_gap']:+.3f} | {row['mean_physiology_auroc_gap']:+.3f}", flush=True)

    pattern_shift = pair_df[(pair_df["feature_set"] == "pattern") & (pair_df["auroc_gap_ci_low"] > 0)]["pair"].tolist()
    physiology_shift = pair_df[(pair_df["feature_set"] == "physiology") & (pair_df["auroc_gap_ci_low"] > 0)]["pair"].tolist()
    shift_rule = "positive AUROC transfer gap with percentile 95% interval excluding zero"
    print("\nInterpretation:", flush=True)
    print(f"The testing pattern's predictive meaning shows a clear transfer loss in {', '.join(pattern_shift) if pattern_shift else 'no pair under the stated rule'} ({shift_rule}).", flush=True)
    print(f"Physiology shows a clear transfer loss in {', '.join(physiology_shift) if physiology_shift else 'no pair under the stated rule'} ({shift_rule}).", flush=True)
    print(f"Across the 26 pairs, mean absolute imputed-share difference has Pearson r={pearson:+.3f} and Spearman rho={spearman:+.3f} with the physiology AUROC gap; runtime {summary['elapsed_analysis_seconds']:.1f}s.", flush=True)
    print(f"\nWrote {paths['gaps']}, {paths['sites']}, {paths['coverage']}, {paths['summary']}", flush=True)


if __name__ == "__main__":
    main()
