"""E7 source-specific logistic fits and trust-dial comparisons across sites."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from e6_dial import PENALTY_GRID, auc_from_ranks, fit_with_cv, point_metrics, sigmoid


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "e7_cache.npz"
ANALYSIS = ROOT / "analysis"
W_GRID = np.round(np.arange(0.0, 1.5001, 0.05), 2)
SPLIT_SEED = 71019
TUNING_SEED = 71119
TUNING_REPEATS = 100
EPS = 1e-12


def _build_source_features(raw_phys: np.ndarray, raw_pattern: np.ndarray, n_dynamic: int,
                           train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit imputation/scaling using source train rows, then transform all rows."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(raw_phys[train_idx], axis=0)
    all_missing = ~np.isfinite(medians)
    medians[all_missing] = 0.0
    imputed = np.where(np.isfinite(raw_phys), raw_phys, medians[None, :])
    physiology = np.column_stack((imputed[:, :n_dynamic], imputed[:, :n_dynamic] ** 2,
                                  imputed[:, n_dynamic:]))
    phys_mean = np.mean(physiology[train_idx], axis=0)
    phys_scale = np.std(physiology[train_idx], axis=0)
    pat_mean = np.mean(raw_pattern[train_idx], axis=0)
    pat_scale = np.std(raw_pattern[train_idx], axis=0)
    phys_scale[~np.isfinite(phys_scale) | (phys_scale < 1e-12)] = 1.0
    pat_scale[~np.isfinite(pat_scale) | (pat_scale < 1e-12)] = 1.0
    x_phys = (physiology - phys_mean) / phys_scale
    x_pattern = (raw_pattern - pat_mean) / pat_scale
    metadata = {
        "impute_medians": medians.tolist(),
        "all_missing_source_training_features": np.flatnonzero(all_missing).astype(int).tolist(),
        "phys_mean": phys_mean.tolist(), "phys_scale": phys_scale.tolist(),
        "pattern_mean": pat_mean.tolist(), "pattern_scale": pat_scale.tolist(),
    }
    return np.column_stack((x_phys, x_pattern)), x_phys, metadata


def _metrics_from_logits(y: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    return point_metrics(y, sigmoid(logits))


def _evaluate_pair(y: np.ndarray, base: np.ndarray, pattern_part: np.ndarray,
                   dataset: str, source: str, target: str):
    logits = base[:, None] + pattern_part[:, None] * W_GRID[None, :]
    probs = sigmoid(logits)
    curve_rows = []
    for j, w in enumerate(W_GRID):
        met = point_metrics(y, probs[:, j])
        curve_rows.append({
            "dataset": dataset, "source": source, "target": target, "w": float(w),
            "log_loss": met["log_loss"], "brier": met["brier"], "auroc": met["auroc"],
        })
    ll = np.asarray([r["log_loss"] for r in curve_rows])
    best_ix = int(np.argmin(ll))
    ix0 = int(np.flatnonzero(np.isclose(W_GRID, 0.0))[0])
    ix1 = int(np.flatnonzero(np.isclose(W_GRID, 1.0))[0])
    best_w = float(W_GRID[best_ix])

    rng = np.random.default_rng(TUNING_SEED + sum((i + 1) * ord(c) for i, c in enumerate(dataset + source + target)))
    differences = []
    chosen_ws = []
    interior = []
    n = len(y)
    n_tune = n // 2
    for split in range(TUNING_REPEATS):
        order = rng.permutation(n)
        tune, evaluation = order[:n_tune], order[n_tune:]
        tune_losses = np.mean(
            np.logaddexp(0.0, logits[tune]) - y[tune, None] * logits[tune], axis=0
        )
        dial_ix = int(np.argmin(tune_losses))
        end_ixs = [ix0, ix1]
        end_losses = [float(tune_losses[ix]) for ix in end_ixs]
        end_ix = end_ixs[int(np.argmin(end_losses))]
        dial_met = point_metrics(y[evaluation], probs[evaluation, dial_ix])
        end_met = point_metrics(y[evaluation], probs[evaluation, end_ix])
        differences.append({
            "log_loss": dial_met["log_loss"] - end_met["log_loss"],
            "auroc": dial_met["auroc"] - end_met["auroc"],
        })
        chosen_w = float(W_GRID[dial_ix])
        chosen_ws.append(chosen_w)
        interior.append(0.05 < chosen_w < 0.95)
    ll_diff = np.asarray([d["log_loss"] for d in differences])
    auc_diff = np.asarray([d["auroc"] for d in differences])
    pair = {
        "dataset": dataset, "source": source, "target": target,
        "target_n": int(n), "target_cases": int(y.sum()),
        "best_w_full_target": best_w,
        "log_loss_w0": float(ll[ix0]), "log_loss_w1": float(ll[ix1]),
        "log_loss_best": float(ll[best_ix]),
        "brier_w0": float(curve_rows[ix0]["brier"]),
        "brier_w1": float(curve_rows[ix1]["brier"]),
        "brier_best": float(curve_rows[best_ix]["brier"]),
        "auroc_w0": float(curve_rows[ix0]["auroc"]),
        "auroc_w1": float(curve_rows[ix1]["auroc"]),
        "auroc_best": float(curve_rows[best_ix]["auroc"]),
        "mean_chosen_w_tuned": float(np.mean(chosen_ws)),
        "share_interior_w_tuned": float(np.mean(interior)),
        "dial_minus_end_log_loss_mean": float(np.mean(ll_diff)),
        "dial_minus_end_log_loss_p10": float(np.percentile(ll_diff, 10)),
        "dial_minus_end_log_loss_p90": float(np.percentile(ll_diff, 90)),
        "dial_minus_end_auroc_mean": float(np.mean(auc_diff)),
        "dial_minus_end_auroc_p10": float(np.percentile(auc_diff, 10)),
        "dial_minus_end_auroc_p90": float(np.percentile(auc_diff, 90)),
        "tuning_repeats": TUNING_REPEATS,
    }
    return pair, curve_rows


def main() -> None:
    started = time.perf_counter()
    if not CACHE.is_file():
        raise FileNotFoundError(f"Run e7_sites.py first; missing {CACHE}")
    with np.load(CACHE, allow_pickle=False) as z:
        cache = {key: z[key].copy() for key in z.files}
    meta = json.loads(str(cache["meta_json"].item()))
    if meta.get("cache_version") != 1:
        raise ValueError("Unexpected E7 cache version")
    p19_meta, p12_meta = meta["P19"], meta["P12"]
    if (p19_meta.get("feature_window_iculos") != [1, 24]
            or p19_meta.get("outcome_window_iculos") != [25, 72]
            or not p19_meta.get("features_use_iculos_1_24_only")):
        raise ValueError("P19 cache does not certify E6 feature/outcome windows")
    if (p12_meta.get("feature_window_hours") != [0, 48]
            or p12_meta.get("recent_pattern_window_hours") != [40, 48]
            or not p12_meta.get("features_use_hours_0_48_only")
            or not p12_meta.get("weight_static_at_00_00_only")):
        raise ValueError("P12 cache does not certify E7 feature windows/static Weight handling")
    result_pairs = []
    result_curves = []
    result_sites = []
    fit_records = {}
    target_source_overlap_checks = 0

    for ds_ix, dataset in enumerate(("P19", "P12")):
        ds_meta = meta[dataset]
        sites = cache[f"{dataset}_sites"].astype(str).tolist()
        n_dynamic = len(ds_meta["raw_phys_names"]) - (3 if dataset == "P19" else 4)
        if dataset == "P19":
            n_dynamic = len(ds_meta["raw_phys_names"]) - 3
        dataset_arrays = {}
        for site in sites:
            prefix = f"{dataset}_{site}"
            dataset_arrays[site] = {
                "raw_phys": cache[f"{prefix}_raw_phys"].astype(np.float64),
                "raw_pattern": cache[f"{prefix}_raw_pattern"].astype(np.float64),
                "y": cache[f"{prefix}_y"].astype(np.uint8),
                "ids": cache[f"{prefix}_patient_ids"].astype(str),
            }

        for site_ix, source in enumerate(sites):
            arrays = dataset_arrays[source]
            n = len(arrays["y"])
            if n < 10:
                raise ValueError(f"Source site {dataset}/{source} is too small: n={n}")
            split_rng = np.random.default_rng(SPLIT_SEED + ds_ix * 1000 + site_ix)
            permutation = split_rng.permutation(n)
            n_train = int(np.floor(0.8 * n))
            train_idx, holdout_idx = permutation[:n_train], permutation[n_train:]
            X, X_phys, prep = _build_source_features(
                arrays["raw_phys"], arrays["raw_pattern"], n_dynamic, train_idx
            )
            y = arrays["y"]
            if len(np.unique(y[train_idx])) < 2:
                raise ValueError(f"Source training split has one outcome class: {dataset}/{source}")
            cv_seed = SPLIT_SEED + 10000 + ds_ix * 1000 + site_ix
            fit_started = time.perf_counter()
            print(f"{dataset} source {source}: fit/CV n_train={len(train_idx):,}, features={X.shape[1]}", flush=True)
            beta, penalty, cv = fit_with_cv(X[train_idx], y[train_idx], seed=cv_seed)
            n_phys = X_phys.shape[1]
            h_logits = beta[0] + X_phys[holdout_idx] @ beta[1:1+n_phys]
            p_holdout = ((X[holdout_idx, n_phys:]) @ beta[1+n_phys:])
            holdout_metrics = _metrics_from_logits(y[holdout_idx], h_logits + p_holdout)
            fit_records[(dataset, source)] = {
                "beta": beta, "prep": prep, "n_phys": n_phys,
                "train_idx": train_idx, "holdout_idx": holdout_idx,
                "penalty": penalty, "cv": cv,
                "source_holdout_metrics": holdout_metrics,
            }
            fit_elapsed = time.perf_counter() - fit_started
            result_sites.append({
                "dataset": dataset, "site": source, "n": n,
                "outcome_cases": int(y.sum()), "outcome_rate": float(y.mean()),
                "source_train_n": len(train_idx), "source_holdout_n": len(holdout_idx),
                "selected_penalty": penalty,
                "source_holdout_auroc_w1": holdout_metrics["auroc"],
                "source_holdout_brier_w1": holdout_metrics["brier"],
                "source_holdout_log_loss_w1": holdout_metrics["log_loss"],
                "source_training_cases": int(y[train_idx].sum()),
                "source_holdout_cases": int(y[holdout_idx].sum()),
                "cv_log_loss": {str(k): v for k, v in cv.items()},
                "preprocessing_all_missing_raw_indices": prep["all_missing_source_training_features"],
            })
            print(f"{dataset} source {source}: penalty={penalty:g}, holdout AUROC={holdout_metrics['auroc']:.4f}, log loss={holdout_metrics['log_loss']:.4f}, fit={fit_elapsed/60:.1f} min", flush=True)

        for source in sites:
            fitted = fit_records[(dataset, source)]
            beta = fitted["beta"]
            n_phys = fitted["n_phys"]
            prep = fitted["prep"]
            medians = np.asarray(prep["impute_medians"], dtype=np.float64)
            phys_mean = np.asarray(prep["phys_mean"], dtype=np.float64)
            phys_scale = np.asarray(prep["phys_scale"], dtype=np.float64)
            pat_mean = np.asarray(prep["pattern_mean"], dtype=np.float64)
            pat_scale = np.asarray(prep["pattern_scale"], dtype=np.float64)
            for target in sites:
                if target == source:
                    continue
                target_arrays = dataset_arrays[target]
                source_train_ids = dataset_arrays[source]["ids"][fitted["train_idx"]]
                overlap = np.intersect1d(source_train_ids, target_arrays["ids"])
                if len(overlap):
                    raise AssertionError(
                        f"Target patients leaked into source training: {dataset}/{source}->{target}; "
                        f"examples={overlap[:3].tolist()}"
                    )
                target_source_overlap_checks += 1
                raw_phys = target_arrays["raw_phys"]
                imputed = np.where(np.isfinite(raw_phys), raw_phys, medians[None, :])
                physiology = np.column_stack((imputed[:, :n_dynamic], imputed[:, :n_dynamic] ** 2,
                                              imputed[:, n_dynamic:]))
                x_phys = (physiology - phys_mean) / phys_scale
                x_pattern = (target_arrays["raw_pattern"] - pat_mean) / pat_scale
                base = beta[0] + x_phys @ beta[1:1+n_phys]
                pattern_part = x_pattern @ beta[1+n_phys:]
                pair, curve = _evaluate_pair(target_arrays["y"], base, pattern_part, dataset, source, target)
                pair["source_penalty"] = fitted["penalty"]
                pair["source_holdout_log_loss"] = fitted["source_holdout_metrics"]["log_loss"]
                pair["source_holdout_auroc"] = fitted["source_holdout_metrics"]["auroc"]
                result_pairs.append(pair)
                result_curves.extend(curve)
                print(f"{dataset} pair {source} -> {target}: best w={pair['best_w_full_target']:.2f}; tuned dial-end LL={pair['dial_minus_end_log_loss_mean']:+.5f} [{pair['dial_minus_end_log_loss_p10']:+.5f}, {pair['dial_minus_end_log_loss_p90']:+.5f}]", flush=True)

    pairs_df = pd.DataFrame.from_records(result_pairs)
    curves_df = pd.DataFrame.from_records(result_curves)
    sites_df = pd.DataFrame.from_records(result_sites)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pairs_path = ANALYSIS / "e7_pairs.csv"
    curves_path = ANALYSIS / "e7_curves.csv"
    sites_path = ANALYSIS / "e7_sites.csv"
    summary_path = ANALYSIS / "e7_summary.json"
    pairs_df.to_csv(pairs_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    sites_df.to_csv(sites_path, index=False)
    summary = {
        "spec": "E7",
        "cache_version": meta["cache_version"],
        "split_seed": SPLIT_SEED,
        "tuning_seed_base": TUNING_SEED,
        "tuning_repeats": TUNING_REPEATS,
        "target_source_training_overlap_checks": target_source_overlap_checks,
        "target_source_training_patient_id_overlaps": 0,
        "horizon_checks": {
            "P19_features_iculos_1_24_only": True,
            "P19_outcome_iculos_25_72": True,
            "P12_features_hours_0_48_only": True,
            "P12_recent_pattern_hours_40_48": True,
            "P12_weight_static_at_00_00_only": True,
        },
        "penalty_grid_average_bce": PENALTY_GRID.tolist(),
        "w_grid": W_GRID.tolist(),
        "source_preprocessing": "per source site; medians and means/std from source 80% training split only",
        "site_counts": sites_df[["dataset", "site", "n", "outcome_cases"]].to_dict(orient="records"),
        "pair_results": result_pairs,
        "data_checks": {k: meta[k] for k in ("P19", "P12")},
        "outputs": [str(p.relative_to(ROOT)) for p in (pairs_path, curves_path, sites_path)],
        "elapsed_analysis_seconds": float(time.perf_counter() - started),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print("\nE7 sites: dataset | site | n | cases", flush=True)
    for row in result_sites:
        print(f"{row['dataset']} | {row['site']} | {row['n']} | {row['outcome_cases']}", flush=True)
    print("\nE7 directed pairs: dataset | source | target | best_w | LL(w=0) | LL(w=1) | LL(best) | mean tuned w | interior share | dial-end LL mean [p10,p90] | dial-end AUROC mean [p10,p90]", flush=True)
    for row in result_pairs:
        print(
            f"{row['dataset']} | {row['source']} | {row['target']} | {row['best_w_full_target']:.2f} | "
            f"{row['log_loss_w0']:.4f} | {row['log_loss_w1']:.4f} | {row['log_loss_best']:.4f} | "
            f"{row['mean_chosen_w_tuned']:.2f} | {row['share_interior_w_tuned']:.2f} | "
            f"{row['dial_minus_end_log_loss_mean']:+.5f} [{row['dial_minus_end_log_loss_p10']:+.5f},{row['dial_minus_end_log_loss_p90']:+.5f}] | "
            f"{row['dial_minus_end_auroc_mean']:+.4f} [{row['dial_minus_end_auroc_p10']:+.4f},{row['dial_minus_end_auroc_p90']:+.4f}]",
            flush=True,
        )
    best_ws = np.asarray([r["best_w_full_target"] for r in result_pairs])
    interior_full = np.mean((best_ws > 0.05) & (best_ws < 0.95))
    mean_gain = np.mean([max(0.0, min(r["log_loss_w0"], r["log_loss_w1"]) - r["log_loss_best"]) for r in result_pairs])
    tuned_gain = -np.mean([r["dial_minus_end_log_loss_mean"] for r in result_pairs])
    data_issues = {
        "P19_exclusions": meta["P19"]["exclusions"],
        "P12_checks": meta["P12"]["exclusions_and_missing_static"],
        "source_missing_features": int(sum(len(r["preprocessing_all_missing_raw_indices"]) for r in result_sites)),
    }
    print("\nSummary:", flush=True)
    print(f"The full-target best w is strictly interior (0.05, 0.95) in {interior_full:.1%} of pairs; the 100-split tuned dial selects an interior w in {np.mean([r['share_interior_w_tuned'] for r in result_pairs]):.1%} of pair-splits.", flush=True)
    print(f"Relative to the better fixed end, the full-target best dial gains {mean_gain:.5f} log-loss units per patient on average; in 100-split evaluation the tuned dial minus end mean is {np.mean([r['dial_minus_end_log_loss_mean'] for r in result_pairs]):+.5f}.", flush=True)
    print(f"Data checks: P19 exclusions={data_issues['P19_exclusions']}; P12 missing/invalid checks={data_issues['P12_checks']}; source fits with wholly missing raw features={data_issues['source_missing_features']}.", flush=True)
    print(f"E7 analysis runtime: {summary['elapsed_analysis_seconds']/60:.1f} min", flush=True)
    print(f"Wrote {pairs_path}, {curves_path}, {sites_path}, {summary_path}", flush=True)


if __name__ == "__main__":
    main()
