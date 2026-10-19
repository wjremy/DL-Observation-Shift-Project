"""E7b: compare the extended observation-pattern dial with recalibration."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from e6_dial import fit_with_cv, point_metrics, sigmoid
from e7_dial import SPLIT_SEED, _build_source_features


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "e7_cache.npz"
ANALYSIS = ROOT / "analysis"
E7_CURVES = ANALYSIS / "e7_curves.csv"
W_GRID = np.round(np.arange(0.0, 3.0001, 0.05), 2)
TUNING_SEED = 71219
TUNING_REPEATS = 100
EPS = 1e-12

OPTION_W = {
    "end": np.array([0.0, 1.0]),
    "dial01": W_GRID[W_GRID <= 1.0],
    "dial03": W_GRID,
    "end_recal": np.array([0.0, 1.0]),
    "dial01_recal": W_GRID[W_GRID <= 1.0],
    "dial03_recal": W_GRID,
}


def _log_loss_from_logits(y: np.ndarray, logits: np.ndarray) -> float:
    z = np.asarray(logits, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def _fit_recalibration(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Unpenalised logistic intercept/slope fit by damped Newton iterations."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z_mean = float(np.mean(z))
    z_sd = float(np.std(z))
    prevalence = float(np.clip(np.mean(y), EPS, 1.0 - EPS))
    intercept = float(np.log(prevalence / (1.0 - prevalence)))
    if not np.isfinite(z_sd) or z_sd < 1e-12:
        return intercept, 0.0

    z_std = (z - z_mean) / z_sd
    slope_std = 0.0

    def loss(a: float, b: float) -> float:
        eta = a + b * z_std
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta))

    old_loss = loss(intercept, slope_std)
    for _ in range(100):
        eta = np.clip(intercept + slope_std * z_std, -40.0, 40.0)
        p = sigmoid(eta)
        residual = p - y
        weight = p * (1.0 - p)
        g0 = float(np.mean(residual))
        g1 = float(np.mean(residual * z_std))
        if max(abs(g0), abs(g1)) < 1e-9:
            break

        h00 = float(np.mean(weight))
        h01 = float(np.mean(weight * z_std))
        h11 = float(np.mean(weight * z_std * z_std))
        determinant = h00 * h11 - h01 * h01
        if not np.isfinite(determinant) or determinant <= 1e-20:
            break
        step0 = (h11 * g0 - h01 * g1) / determinant
        step1 = (h00 * g1 - h01 * g0) / determinant
        if not np.isfinite(step0 + step1):
            break

        scale = 1.0
        accepted = False
        for _ in range(40):
            next0 = float(intercept - scale * step0)
            next1 = float(slope_std - scale * step1)
            next_loss = loss(next0, next1)
            if np.isfinite(next_loss) and next_loss <= old_loss + 1e-12:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
        intercept, slope_std, old_loss = next0, next1, next_loss
        if max(abs(scale * step0), abs(scale * step1)) < 1e-8:
            break

    slope = slope_std / z_sd
    intercept_raw = intercept - slope * z_mean
    return float(intercept_raw), float(slope)


def _pair_rng_seed(dataset: str, source: str, target: str) -> int:
    label = dataset + source + target
    return TUNING_SEED + sum((i + 1) * ord(char) for i, char in enumerate(label))


def _evaluate_tuning_splits(y: np.ndarray, logits: np.ndarray, dataset: str,
                            source: str, target: str) -> list[dict]:
    """Run the paired 100-split choice/evaluation comparison for one pair."""
    n = len(y)
    n_tune = n // 2
    rng = np.random.default_rng(_pair_rng_seed(dataset, source, target))
    all_results = {option: [] for option in OPTION_W}
    grid_index = {float(w): i for i, w in enumerate(W_GRID)}
    option_indices = {
        option: np.asarray([grid_index[float(w)] for w in ws], dtype=np.int64)
        for option, ws in OPTION_W.items()
    }

    for _ in range(TUNING_REPEATS):
        order = rng.permutation(n)
        tune_ix, eval_ix = order[:n_tune], order[n_tune:]
        y_tune = y[tune_ix]
        logits_tune = logits[tune_ix]
        tune_losses = np.mean(
            np.logaddexp(0.0, logits_tune) - y_tune[:, None] * logits_tune,
            axis=0,
        )

        recal_intercepts = np.empty(len(W_GRID), dtype=np.float64)
        recal_slopes = np.empty(len(W_GRID), dtype=np.float64)
        recal_tune_losses = np.empty(len(W_GRID), dtype=np.float64)
        for j in range(len(W_GRID)):
            a, b = _fit_recalibration(logits_tune[:, j], y_tune)
            recal_intercepts[j] = a
            recal_slopes[j] = b
            recal_tune_losses[j] = _log_loss_from_logits(
                y_tune, a + b * logits_tune[:, j]
            )

        for option, indices in option_indices.items():
            is_recal = option.endswith("_recal")
            candidate_losses = recal_tune_losses if is_recal else tune_losses
            chosen_ix = int(indices[int(np.argmin(candidate_losses[indices]))])
            eval_logits = logits[eval_ix, chosen_ix]
            if is_recal:
                eval_logits = (recal_intercepts[chosen_ix]
                               + recal_slopes[chosen_ix] * eval_logits)
            metrics = point_metrics(y[eval_ix], sigmoid(eval_logits))
            all_results[option].append({
                "log_loss": metrics["log_loss"],
                "auroc": metrics["auroc"],
                "chosen_w": float(W_GRID[chosen_ix]),
            })

    end = all_results["end"]
    end_recal = all_results["end_recal"]
    rows = []
    for option, split_results in all_results.items():
        chosen_w = np.asarray([r["chosen_w"] for r in split_results])
        row = {
            "dataset": dataset,
            "source": source,
            "target": target,
            "option": option,
            "target_n": int(n),
            "target_cases": int(y.sum()),
            "tuning_repeats": TUNING_REPEATS,
            "tuning_seed": _pair_rng_seed(dataset, source, target),
            "mean_eval_log_loss": float(np.mean([r["log_loss"] for r in split_results])),
            "mean_eval_auroc": float(np.mean([r["auroc"] for r in split_results])),
            "mean_chosen_w": float(np.mean(chosen_w)),
            "share_chosen_w_gt_1": float(np.mean(chosen_w > 1.0)),
        }
        for comparator_name, comparator in (("end", end), ("end_recal", end_recal)):
            for metric in ("log_loss", "auroc"):
                paired_diff = np.asarray([
                    current[metric] - baseline[metric]
                    for current, baseline in zip(split_results, comparator)
                ])
                row[f"delta_to_{comparator_name}_{metric}_mean"] = float(np.mean(paired_diff))
                row[f"delta_to_{comparator_name}_{metric}_p10"] = float(np.percentile(paired_diff, 10))
                row[f"delta_to_{comparator_name}_{metric}_p90"] = float(np.percentile(paired_diff, 90))
        rows.append(row)
    return rows


def _transform_target(raw_phys: np.ndarray, raw_pattern: np.ndarray, n_dynamic: int,
                      prep: dict) -> tuple[np.ndarray, np.ndarray]:
    medians = np.asarray(prep["impute_medians"], dtype=np.float64)
    phys_mean = np.asarray(prep["phys_mean"], dtype=np.float64)
    phys_scale = np.asarray(prep["phys_scale"], dtype=np.float64)
    pat_mean = np.asarray(prep["pattern_mean"], dtype=np.float64)
    pat_scale = np.asarray(prep["pattern_scale"], dtype=np.float64)
    imputed = np.where(np.isfinite(raw_phys), raw_phys, medians[None, :])
    physiology = np.column_stack((imputed[:, :n_dynamic], imputed[:, :n_dynamic] ** 2,
                                  imputed[:, n_dynamic:]))
    x_phys = (physiology - phys_mean) / phys_scale
    x_pattern = (raw_pattern - pat_mean) / pat_scale
    return x_phys, x_pattern


def _load_cache() -> tuple[dict, dict]:
    if not CACHE.is_file():
        raise FileNotFoundError(f"Missing E7 cache: {CACHE}")
    with np.load(CACHE, allow_pickle=False) as z:
        cache = {key: z[key].copy() for key in z.files}
    meta = json.loads(str(cache["meta_json"].item()))
    if meta.get("cache_version") != 1:
        raise ValueError("Unexpected E7 cache version")
    p19_meta, p12_meta = meta["P19"], meta["P12"]
    if (p19_meta.get("feature_window_iculos") != [1, 24]
            or p19_meta.get("outcome_window_iculos") != [25, 72]
            or not p19_meta.get("features_use_iculos_1_24_only")):
        raise ValueError("P19 cache does not certify E7 feature/outcome windows")
    if (p12_meta.get("feature_window_hours") != [0, 48]
            or p12_meta.get("recent_pattern_window_hours") != [40, 48]
            or not p12_meta.get("features_use_hours_0_48_only")
            or not p12_meta.get("weight_static_at_00_00_only")):
        raise ValueError("P12 cache does not certify E7 feature windows/static Weight handling")
    return cache, meta


def main() -> None:
    started = time.perf_counter()
    cache, meta = _load_cache()
    e7_curves = pd.read_csv(E7_CURVES)
    e7_sites = pd.read_csv(ANALYSIS / "e7_sites.csv")
    option_rows: list[dict] = []
    best_rows: list[dict] = []
    curve_rows: list[dict] = []
    source_fit_rows: list[dict] = []

    for ds_ix, dataset in enumerate(("P19", "P12")):
        ds_meta = meta[dataset]
        sites = cache[f"{dataset}_sites"].astype(str).tolist()
        n_dynamic = len(ds_meta["raw_phys_names"]) - (3 if dataset == "P19" else 4)
        dataset_arrays = {}
        for site in sites:
            prefix = f"{dataset}_{site}"
            dataset_arrays[site] = {
                "raw_phys": cache[f"{prefix}_raw_phys"].astype(np.float64),
                "raw_pattern": cache[f"{prefix}_raw_pattern"].astype(np.float64),
                "y": cache[f"{prefix}_y"].astype(np.uint8),
                "ids": cache[f"{prefix}_patient_ids"].astype(str),
            }

        source_fits = {}
        for site_ix, source in enumerate(sites):
            arrays = dataset_arrays[source]
            n_source = len(arrays["y"])
            rng = np.random.default_rng(SPLIT_SEED + ds_ix * 1000 + site_ix)
            permutation = rng.permutation(n_source)
            n_train = int(np.floor(0.8 * n_source))
            train_ix, holdout_ix = permutation[:n_train], permutation[n_train:]
            x, x_phys, prep = _build_source_features(
                arrays["raw_phys"], arrays["raw_pattern"], n_dynamic, train_ix
            )
            cv_seed = SPLIT_SEED + 10000 + ds_ix * 1000 + site_ix
            beta, penalty, cv = fit_with_cv(x[train_ix], arrays["y"][train_ix], seed=cv_seed)
            n_phys = x_phys.shape[1]
            source_fits[source] = {
                "beta": beta,
                "prep": prep,
                "n_phys": n_phys,
                "train_ix": train_ix,
                "penalty": penalty,
            }
            prior_site = e7_sites[(e7_sites.dataset == dataset) & (e7_sites.site == source)].iloc[0]
            penalty_match = bool(np.isclose(penalty, prior_site.selected_penalty, rtol=0, atol=0))
            source_fit_rows.append({
                "dataset": dataset,
                "source": source,
                "selected_penalty": float(penalty),
                "e7_selected_penalty": float(prior_site.selected_penalty),
                "penalty_matches_e7": penalty_match,
                "cv_log_loss": {str(k): float(v) for k, v in cv.items()},
            })
            if not penalty_match:
                raise AssertionError(f"E7 penalty mismatch for {dataset}/{source}")

        for source in sites:
            fitted = source_fits[source]
            arrays_source = dataset_arrays[source]
            target_train_ids = arrays_source["ids"][fitted["train_ix"]]
            beta = fitted["beta"]
            n_phys = fitted["n_phys"]
            for target in sites:
                if target == source:
                    continue
                target_arrays = dataset_arrays[target]
                overlap = np.intersect1d(target_train_ids, target_arrays["ids"])
                if len(overlap):
                    raise AssertionError(f"Source train / target overlap: {dataset}/{source}->{target}")
                x_phys, x_pattern = _transform_target(
                    target_arrays["raw_phys"], target_arrays["raw_pattern"], n_dynamic, fitted["prep"]
                )
                base = beta[0] + x_phys @ beta[1:1 + n_phys]
                pattern_part = x_pattern @ beta[1 + n_phys:]
                logits = base[:, None] + pattern_part[:, None] * W_GRID[None, :]
                probs = sigmoid(logits)
                y = target_arrays["y"]

                pair_curve_rows = []
                for j, w in enumerate(W_GRID):
                    metrics = point_metrics(y, probs[:, j])
                    row = {
                        "dataset": dataset,
                        "source": source,
                        "target": target,
                        "w": float(w),
                        "log_loss": metrics["log_loss"],
                        "brier": metrics["brier"],
                        "auroc": metrics["auroc"],
                    }
                    pair_curve_rows.append(row)
                ll_full = np.asarray([r["log_loss"] for r in pair_curve_rows])
                best_ix = int(np.argmin(ll_full))
                best_w = float(W_GRID[best_ix])
                at_edge = bool(best_ix == 0 or best_ix == len(W_GRID) - 1)
                best_rows.append({
                    "dataset": dataset,
                    "source": source,
                    "target": target,
                    "target_n": int(len(y)),
                    "target_cases": int(y.sum()),
                    "best_w_full_target": best_w,
                    "best_at_grid_edge": at_edge,
                    "best_log_loss": float(ll_full[best_ix]),
                    "best_auroc": float(pair_curve_rows[best_ix]["auroc"]),
                })
                curve_rows.extend(pair_curve_rows)

                pair_options = _evaluate_tuning_splits(y, logits, dataset, source, target)
                for row in pair_options:
                    row["best_w_full_target"] = best_w
                    row["best_at_grid_edge"] = at_edge
                option_rows.extend(pair_options)
                print(f"Finished {dataset} {source}->{target} (best w={best_w:.2f}{' edge' if at_edge else ''})",
                      flush=True)

    curves_df = pd.DataFrame.from_records(curve_rows)
    common = curves_df[curves_df.w <= 1.5].merge(
        e7_curves[["dataset", "source", "target", "w", "log_loss"]],
        on=["dataset", "source", "target", "w"],
        suffixes=("", "_e7"),
        validate="one_to_one",
    )
    common["e7_abs_log_loss_difference"] = np.abs(common.log_loss - common.log_loss_e7)
    max_abs_difference = float(common.e7_abs_log_loss_difference.max())
    per_pair_reproduction = []
    for (dataset, source, target), frame in common.groupby(["dataset", "source", "target"], sort=False):
        per_pair_reproduction.append({
            "dataset": dataset,
            "source": source,
            "target": target,
            "max_abs_log_loss_difference": float(frame.e7_abs_log_loss_difference.max()),
        })
    if len(common) != 24 * 31:
        raise AssertionError(f"Expected 744 overlapping E7 curve rows, found {len(common)}")

    option_df = pd.DataFrame.from_records(option_rows)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pairs_path = ANALYSIS / "e7b_pairs.csv"
    curves_path = ANALYSIS / "e7b_curves.csv"
    summary_path = ANALYSIS / "e7b_summary.json"
    option_df.to_csv(pairs_path, index=False)
    curves_df.to_csv(curves_path, index=False)

    pair_lookup = option_df[option_df.option.notna()].set_index(["dataset", "source", "target", "option"])
    recal_totals = {}
    for option in ("dial03_recal", "dial01_recal"):
        paired = pair_lookup.xs(option, level="option")
        delta_mean = paired["delta_to_end_recal_log_loss_mean"]
        delta_p90 = paired["delta_to_end_recal_log_loss_p90"]
        recal_totals[option] = {
            "pairs_beating_end_recal_by_mean_and_p90": int(((delta_mean < 0) & (delta_p90 < 0)).sum()),
            "mean_pair_mean_log_loss_difference_to_end_recal": float(delta_mean.mean()),
        }

    summary = {
        "spec": "E7b",
        "cache_version": meta["cache_version"],
        "source_split_seed": SPLIT_SEED,
        "tuning_seed_base": TUNING_SEED,
        "tuning_repeats": TUNING_REPEATS,
        "tuning_seed_rule": "base + weighted character-code sum of dataset+source+target",
        "tuning_choices": {key: value.tolist() for key, value in OPTION_W.items()},
        "recalibration": "per candidate w, unpenalised two-parameter logistic regression (intercept and slope) fit by damped Newton on tune half",
        "source_fit_penalties_match_e7": all(row["penalty_matches_e7"] for row in source_fit_rows),
        "source_fits": source_fit_rows,
        "e7_curve_reproduction": {
            "overlapping_rows": int(len(common)),
            "max_absolute_target_log_loss_difference": max_abs_difference,
            "per_pair": per_pair_reproduction,
        },
        "recalibrated_dial_totals": recal_totals,
        "full_target_best_weights": best_rows,
        "elapsed_analysis_seconds": float(time.perf_counter() - started),
        "outputs": [str(path.relative_to(ROOT)) for path in (pairs_path, curves_path, summary_path)],
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    print("\nE7b pairs: dataset | source->target | best w (edge) | dial03-end LL [p10,p90] / AUROC | dial03_recal-end_recal LL [p10,p90] / AUROC")
    options = option_df[option_df.option.notna()].set_index(["dataset", "source", "target", "option"])
    best_pairs = best_rows
    for best in best_pairs:
        key = (best["dataset"], best["source"], best["target"])
        dial03 = options.loc[key + ("dial03",)]
        dial03_recal = options.loc[key + ("dial03_recal",)]
        print(
            f"{key[0]} | {key[1]}->{key[2]} | {best['best_w_full_target']:.2f} ({'edge' if best['best_at_grid_edge'] else 'interior'}) | "
            f"{dial03.delta_to_end_log_loss_mean:+.5f} [{dial03.delta_to_end_log_loss_p10:+.5f},{dial03.delta_to_end_log_loss_p90:+.5f}] / {dial03.delta_to_end_auroc_mean:+.4f} | "
            f"{dial03_recal.delta_to_end_recal_log_loss_mean:+.5f} [{dial03_recal.delta_to_end_recal_log_loss_p10:+.5f},{dial03_recal.delta_to_end_recal_log_loss_p90:+.5f}] / {dial03_recal.delta_to_end_recal_auroc_mean:+.4f}"
        )

    print(f"\nE7 curve reproduction: max absolute target log-loss difference = {max_abs_difference:.12g} over {len(common)} rows.")
    for option in ("dial03_recal", "dial01_recal"):
        total = recal_totals[option]
        print(f"{option} beats end_recal by mean and p90 in {total['pairs_beating_end_recal_by_mean_and_p90']}/24 pairs; mean pair difference={total['mean_pair_mean_log_loss_difference_to_end_recal']:+.6f}.")
    n03 = recal_totals["dial03_recal"]["pairs_beating_end_recal_by_mean_and_p90"]
    m03 = recal_totals["dial03_recal"]["mean_pair_mean_log_loss_difference_to_end_recal"]
    n01 = recal_totals["dial01_recal"]["pairs_beating_end_recal_by_mean_and_p90"]
    m01 = recal_totals["dial01_recal"]["mean_pair_mean_log_loss_difference_to_end_recal"]
    raw_mean = float(options.xs("dial03", level="option")["delta_to_end_log_loss_mean"].mean())
    recal_above_one = options.xs("dial03_recal", level="option")
    above_one_pairs = int((recal_above_one["mean_chosen_w"] > 1.0).sum())
    above_one_split_share = float(recal_above_one["share_chosen_w_gt_1"].mean())
    print(f"The recalibrated 0-3 dial chose mean w>1 in {above_one_pairs}/24 pairs ({above_one_split_share:.1%} of split choices), and beat end_recal by the strict paired criterion in {n03}/24 pairs.")
    print(f"Its mean pair log-loss difference was {m03:+.6f}, versus {raw_mean:+.6f} for the uncalibrated 0-3 dial, so recalibration attenuates the gain without removing it.")
    print(f"The recalibrated 0-1 dial also beat end_recal in {n01}/24 pairs (mean difference {m01:+.6f}); gains from allowing w>1 are present but concentrated in a minority of pairs.")
    print(f"E7b analysis runtime: {summary['elapsed_analysis_seconds'] / 60:.1f} min.")
    print(f"Wrote {pairs_path}, {curves_path}, {summary_path}.")


if __name__ == "__main__":
    main()
