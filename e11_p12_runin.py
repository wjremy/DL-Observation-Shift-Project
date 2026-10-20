"""E11: safe dial tuning with realistic run-ins on the 2012 mortality pairs.

This is the pre-specified E11 design from SPEC.md. It reuses E7's P12 source
splits, feature preparation and penalised logistic fits, plus E8/E9's run-in,
recalibration, tuning and metric helpers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import e8_runin as e8
import e9_fixed_tuning as e9
from e6_dial import fit_with_cv
from e7_dial import SPLIT_SEED, _build_source_features
from e7b_recal import _transform_target


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "e7_cache.npz"
E7_SITES = ROOT / "analysis" / "e7_sites.csv"
ANALYSIS = ROOT / "analysis"
SUMMARY_PATH = ANALYSIS / "e11_summary.csv"
JSON_PATH = ANALYSIS / "e11_summary.json"

W_GRID = e8.W_GRID
W_ONE_INDEX = e8.W_ONE_INDEX
RUNIN_SIZES = e8.RUNIN_SIZES
N_DRAWS = 200
RUNIN_SEED = 111020  # New fixed seed, distinct from E8 and E9.
MIN_EVALUATION_N = e8.MIN_EVALUATION_N
RECAL_W_PENALTY = 2.0
AUC_W_PENALTY = 0.02
METHODS = ("keep", "end_recal", "dial_recal", "dial_recal_shrink", "dial_auc_shrink")
SUBGROUP_PAIRS = {
    ("CCU", "SICU"),
    ("CSRU", "CCU"),
    ("CSRU", "SICU"),
    ("SICU", "CSRU"),
}


def _load_cache() -> tuple[dict[str, np.ndarray], dict]:
    if not CACHE.is_file():
        raise FileNotFoundError(f"Missing E7 cache: {CACHE}")
    with np.load(CACHE, allow_pickle=False) as z:
        cache = {key: z[key].copy() for key in z.files}
    meta = json.loads(str(cache["meta_json"].item()))
    if meta.get("cache_version") != 1:
        raise ValueError("Unexpected E7 cache version")
    p12_meta = meta["P12"]
    if (p12_meta.get("feature_window_hours") != [0, 48]
            or p12_meta.get("recent_pattern_window_hours") != [40, 48]
            or not p12_meta.get("features_use_hours_0_48_only")
            or not p12_meta.get("weight_static_at_00_00_only")):
        raise ValueError("P12 cache does not certify E7 feature windows/static Weight handling")
    return cache, meta


def _fit_p12_source_models(cache: dict[str, np.ndarray], meta: dict,
                           sites: list[str]) -> tuple[dict[str, dict], list[dict], dict[str, dict]]:
    """Recreate E7's P12 source fits exactly and verify each selected penalty."""
    n_dynamic = len(meta["P12"]["raw_phys_names"]) - 4
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for site in sites:
        prefix = f"P12_{site}"
        arrays[site] = {
            "raw_phys": cache[f"{prefix}_raw_phys"].astype(np.float64),
            "raw_pattern": cache[f"{prefix}_raw_pattern"].astype(np.float64),
            "y": cache[f"{prefix}_y"].astype(np.uint8),
            "ids": cache[f"{prefix}_patient_ids"].astype(str),
        }

    e7_sites = pd.read_csv(E7_SITES)
    fits: dict[str, dict] = {}
    rows: list[dict] = []
    dataset_index = 1  # E7 loops over (P19, P12), so P12 is dataset index 1.
    for site_ix, source in enumerate(sites):
        source_arrays = arrays[source]
        n_source = len(source_arrays["y"])
        split_seed = SPLIT_SEED + dataset_index * 1000 + site_ix
        permutation = np.random.default_rng(split_seed).permutation(n_source)
        n_train = int(np.floor(0.8 * n_source))
        train_ix = permutation[:n_train]
        x, x_phys, prep = _build_source_features(
            source_arrays["raw_phys"], source_arrays["raw_pattern"], n_dynamic, train_ix
        )
        y_train = source_arrays["y"][train_ix]
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"P12 source training split has one outcome class: {source}")
        cv_seed = SPLIT_SEED + 10000 + dataset_index * 1000 + site_ix
        beta, penalty, cv = fit_with_cv(x[train_ix], y_train, seed=cv_seed)
        matches = e7_sites[(e7_sites.dataset == "P12") & (e7_sites.site == source)]
        if len(matches) != 1:
            raise ValueError(f"Expected one E7 site row for P12/{source}; found {len(matches)}")
        e7_penalty = float(matches.iloc[0].selected_penalty)
        penalty_matches = bool(np.isclose(penalty, e7_penalty, rtol=0.0, atol=0.0))
        if not penalty_matches:
            raise AssertionError(
                f"E7 penalty mismatch for P12/{source}: refit {penalty} vs E7 {e7_penalty}"
            )
        n_phys = x_phys.shape[1]
        fits[source] = {
            "beta": beta,
            "prep": prep,
            "n_phys": n_phys,
            "train_ix": train_ix,
            "penalty": float(penalty),
        }
        rows.append({
            "dataset": "P12",
            "source": source,
            "source_split_seed": split_seed,
            "cv_seed": cv_seed,
            "selected_penalty": float(penalty),
            "e7_selected_penalty": e7_penalty,
            "penalty_matches_e7": penalty_matches,
            "cv_log_loss": {str(k): float(v) for k, v in cv.items()},
            "source_train_n": int(n_train),
        })
    return fits, rows, arrays


def _run_pair(source: str, target: str, base: np.ndarray, pattern: np.ndarray,
              y: np.ndarray, rng: np.random.Generator) -> tuple[list[dict], list[dict], list[dict]]:
    """Run E9's tuning methods with E11's endpoint-recalibration method."""
    rows: list[dict] = []
    skipped: list[dict] = []
    runin_counts: list[dict] = []
    n_target = len(y)
    logits_all = base[:, None] + pattern[:, None] * W_GRID[None, :]
    all_indices = np.arange(len(W_GRID), dtype=np.int64)
    end_indices = np.asarray([0, W_ONE_INDEX], dtype=np.int64)

    for n_run, months in RUNIN_SIZES:
        if n_run > n_target - MIN_EVALUATION_N:
            skipped.append({
                "source": source,
                "target": target,
                "n_run": n_run,
                "runin_months": months,
                "target_n": n_target,
                "evaluation_n_if_run": n_target - n_run,
                "minimum_evaluation_n": MIN_EVALUATION_N,
                "reason": "run-in would leave fewer than 500 evaluation patients",
            })
            continue

        zero_case_count = 0
        zero_control_count = 0
        for draw in range(1, N_DRAWS + 1):
            order = rng.permutation(n_target)
            run_ix = order[:n_run]
            eval_ix = order[n_run:]
            y_run = y[run_ix]
            n_cases = int(y_run.sum())
            n_controls = n_run - n_cases
            no_class_pair = n_cases == 0 or n_controls == 0
            zero_case_count += int(n_cases == 0)
            zero_control_count += int(n_controls == 0)
            logits_run = logits_all[run_ix]
            logits_eval = logits_all[eval_ix]

            recal_a, recal_s, recal_losses = e8._fit_recalibration_grid(logits_run, y_run)
            recal_shrink_losses = recal_losses + RECAL_W_PENALTY * (W_GRID - 1.0) ** 2
            runin_auc = e9._auc_curve_from_ranks(y_run, logits_run)
            auc_shrink_score = runin_auc - AUC_W_PENALTY * (W_GRID - 1.0) ** 2

            chosen = {
                "keep": W_ONE_INDEX,
                "end_recal": e8._choose_index(recal_losses, end_indices, no_class_pair),
                "dial_recal": e8._choose_index(recal_losses, all_indices, no_class_pair),
                "dial_recal_shrink": e8._choose_index(
                    recal_shrink_losses, all_indices, no_class_pair
                ),
                "dial_auc_shrink": e9._choose_max_index(
                    auc_shrink_score, all_indices, no_class_pair
                ),
            }
            recalibrated_methods = set(METHODS) - {"keep"}
            recal_params = {
                method: (float(recal_a[chosen[method]]), float(recal_s[chosen[method]]))
                for method in recalibrated_methods
            }
            for method in METHODS:
                idx = chosen[method]
                eval_logits = logits_eval[:, idx]
                if method in recal_params:
                    a, slope = recal_params[method]
                    eval_logits = a + slope * eval_logits
                auroc, log_loss = e8._metric_row(y[eval_ix], eval_logits)
                rows.append({
                    "dataset": "P12",
                    "source": source,
                    "target": target,
                    "pair_type": "subgroup" if (source, target) in SUBGROUP_PAIRS else "other",
                    "n_run": n_run,
                    "runin_months": months,
                    "draw": draw,
                    "method": method,
                    "auroc": auroc,
                    "log_loss": log_loss,
                    "chosen_w": float(W_GRID[idx]),
                    "runin_cases": n_cases,
                })

        runin_counts.append({
            "source": source,
            "target": target,
            "pair_type": "subgroup" if (source, target) in SUBGROUP_PAIRS else "other",
            "n_run": n_run,
            "runin_months": months,
            "draws": N_DRAWS,
            "zero_case_draws": zero_case_count,
            "zero_case_share": zero_case_count / N_DRAWS,
            "zero_control_draws": zero_control_count,
            "zero_control_share": zero_control_count / N_DRAWS,
        })
    return rows, skipped, runin_counts


def _summarise(draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["source", "target", "pair_type", "n_run", "runin_months"]
    for (source, target, pair_type, n_run, months), frame in draws.groupby(group_cols, sort=False):
        by_method = {
            method: frame[frame.method == method].sort_values("draw")
            for method in METHODS
        }
        for method in METHODS:
            current = by_method[method]
            row = {
                "dataset": "P12",
                "source": source,
                "target": target,
                "pair_type": pair_type,
                "n_run": int(n_run),
                "runin_months": int(months),
                "method": method,
                "draws": int(len(current)),
                "mean_auroc": float(current.auroc.mean()),
                "mean_log_loss": float(current.log_loss.mean()),
                "mean_chosen_w": float(current.chosen_w.mean()),
                "p10_chosen_w": float(current.chosen_w.quantile(0.10)),
                "p90_chosen_w": float(current.chosen_w.quantile(0.90)),
                "mean_runin_cases": float(current.runin_cases.mean()),
                "zero_case_draws": int((current.runin_cases == 0).sum()),
                "zero_case_share": float((current.runin_cases == 0).mean()),
            }
            current_draws = current.draw.to_numpy()
            for comparator in ("keep", "end_recal"):
                baseline = by_method[comparator]
                if not np.array_equal(current_draws, baseline.draw.to_numpy()):
                    raise AssertionError(
                        f"Unpaired draws for {source}->{target}, n_run={n_run}"
                    )
                for metric in ("auroc", "log_loss"):
                    difference = (current[metric].to_numpy(dtype=np.float64)
                                  - baseline[metric].to_numpy(dtype=np.float64))
                    gained = difference > 0 if metric == "auroc" else difference < 0
                    row[f"delta_to_{comparator}_{metric}_mean"] = float(np.mean(difference))
                    row[f"delta_to_{comparator}_{metric}_p10"] = float(np.percentile(difference, 10))
                    row[f"delta_to_{comparator}_{metric}_p90"] = float(np.percentile(difference, 90))
                    row[f"delta_to_{comparator}_{metric}_share_gain"] = float(np.mean(gained))
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _primary_result(summary: pd.DataFrame) -> dict:
    three_month = summary[summary.n_run == 162]
    if set(zip(three_month.source, three_month.target)) != set(
            (s, t) for s in ("CCU", "CSRU", "MICU", "SICU")
            for t in ("CCU", "CSRU", "MICU", "SICU") if s != t):
        raise AssertionError("The 3-month primary results must include all 12 P12 pairs")
    pair_results = []
    for source, target in sorted(set(zip(three_month.source, three_month.target))):
        frame = three_month[(three_month.source == source) & (three_month.target == target)]
        shrink = frame[frame.method == "dial_recal_shrink"].iloc[0]
        auc_shrink = frame[frame.method == "dial_auc_shrink"].iloc[0]
        pair_results.append({
            "source": source,
            "target": target,
            "dial_recal_shrink_minus_keep_auroc": {
                "mean": float(shrink.delta_to_keep_auroc_mean),
                "p10": float(shrink.delta_to_keep_auroc_p10),
                "p90": float(shrink.delta_to_keep_auroc_p90),
                "share_gain": float(shrink.delta_to_keep_auroc_share_gain),
                "mean_runin_deaths": float(shrink.mean_runin_cases),
                "mean_chosen_w": float(shrink.mean_chosen_w),
            },
            "dial_recal_shrink_minus_end_recal_auroc_mean": float(
                shrink.delta_to_end_recal_auroc_mean
            ),
            "dial_auc_shrink_minus_keep_auroc_mean": float(
                auc_shrink.delta_to_keep_auroc_mean
            ),
        })
    return {
        "n_run": 162,
        "runin_months": 3,
        "dial_recal_shrink_minus_keep_auroc_mean_over_12_pairs": float(np.mean([
            r["dial_recal_shrink_minus_keep_auroc"]["mean"] for r in pair_results
        ])),
        "pairs": pair_results,
    }


def _fmt_pair_means(summary: pd.DataFrame, method: str, comparator: str) -> str:
    part = summary[(summary.n_run == 162) & (summary.method == method)]
    values = []
    for row in part.sort_values(["source", "target"]).itertuples():
        values.append(f"{row.source}->{row.target} {getattr(row, f'delta_to_{comparator}_auroc_mean'):+.4f}")
    return ", ".join(values)


def _print_report(summary: pd.DataFrame, runtime: float) -> None:
    primary = _primary_result(summary)
    three_month = summary[summary.n_run == 162]
    shrink = three_month[three_month.method == "dial_recal_shrink"].sort_values(
        ["source", "target"]
    )
    detail = "; ".join(
        f"{row.source}->{row.target} {row.delta_to_keep_auroc_mean:+.4f} "
        f"[{row.delta_to_keep_auroc_p10:+.4f},{row.delta_to_keep_auroc_p90:+.4f}] "
        f"gain={row.delta_to_keep_auroc_share_gain:.0%}, deaths={row.mean_runin_cases:.1f}, "
        f"w={row.mean_chosen_w:.2f}"
        for row in shrink.itertuples()
    )
    print(
        f"Primary at 3 months: dial_recal_shrink−keep AUROC mean over 12 pairs="
        f"{primary['dial_recal_shrink_minus_keep_auroc_mean_over_12_pairs']:+.4f}; "
        f"per pair mean [p10,p90], gain share, run-in deaths, chosen w: {detail}.",
        flush=True,
    )
    print(
        "At 3 months, dial_recal_shrink−end_recal AUROC means: "
        f"{_fmt_pair_means(summary, 'dial_recal_shrink', 'end_recal')}; "
        "dial_auc_shrink−keep AUROC means: "
        f"{_fmt_pair_means(summary, 'dial_auc_shrink', 'keep')}.",
        flush=True,
    )
    subgroup = summary[(summary.pair_type == "subgroup")
                       & (summary.method == "dial_recal_shrink")].sort_values(
                           ["runin_months", "source", "target"]
                       )
    subgroup_by_month = []
    for months, frame in subgroup.groupby("runin_months", sort=True):
        mean = frame.delta_to_keep_auroc_mean.mean()
        subgroup_by_month.append(f"{int(months)}mo {mean:+.4f}")
    print(
        f"Seed {RUNIN_SEED}; four named subgroup pairs (CCU→SICU, CSRU→CCU, CSRU→SICU, SICU→CSRU), "
        "dial_recal_shrink−keep AUROC mean across the four pairs by run-in size: "
        f"{', '.join(subgroup_by_month)}; runtime {runtime:.1f}s ({runtime / 60:.1f}min).",
        flush=True,
    )


def main() -> None:
    started = time.perf_counter()
    cache, meta = _load_cache()
    sites = cache["P12_sites"].astype(str).tolist()
    if set(sites) != {"CCU", "CSRU", "MICU", "SICU"} or len(sites) != 4:
        raise ValueError(f"Expected the four P12 sites; found {sites}")
    fits, source_fit_rows, arrays = _fit_p12_source_models(cache, meta, sites)

    pair_order = [(source, target) for source in sites for target in sites if source != target]
    if len(pair_order) != 12 or SUBGROUP_PAIRS - set(pair_order):
        raise AssertionError("Unexpected P12 site ordering/pair count")
    rng = np.random.default_rng(RUNIN_SEED)
    draw_rows: list[dict] = []
    skipped_sizes: list[dict] = []
    runin_counts: list[dict] = []
    n_dynamic = len(meta["P12"]["raw_phys_names"]) - 4

    for source, target in pair_order:
        fitted = fits[source]
        source_train_ids = arrays[source]["ids"][fitted["train_ix"]]
        target_arrays = arrays[target]
        overlap = np.intersect1d(source_train_ids, target_arrays["ids"])
        if len(overlap):
            raise AssertionError(
                f"Source train / target overlap: P12/{source}->{target}; "
                f"examples={overlap[:3].tolist()}"
            )
        x_phys, x_pattern = _transform_target(
            target_arrays["raw_phys"], target_arrays["raw_pattern"],
            n_dynamic, fitted["prep"],
        )
        beta = fitted["beta"]
        n_phys = fitted["n_phys"]
        base = beta[0] + x_phys @ beta[1:1 + n_phys]
        pattern = x_pattern @ beta[1 + n_phys:]
        pair_rows, pair_skips, pair_counts = _run_pair(
            source, target, base, pattern, target_arrays["y"], rng
        )
        draw_rows.extend(pair_rows)
        skipped_sizes.extend(pair_skips)
        runin_counts.extend(pair_counts)

    draws = pd.DataFrame.from_records(draw_rows)
    if not np.all(draws.loc[draws.method == "keep", "chosen_w"].to_numpy() == 1.0):
        raise AssertionError("E11 keep must remain fixed at w=1")
    summary = _summarise(draws)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    primary_result = _primary_result(summary)
    summary_json = {
        "spec": "E11",
        "cache_version": meta["cache_version"],
        "dataset": "P12",
        "sites": sites,
        "source_split_seed_base": SPLIT_SEED + 1000,
        "source_fits": source_fit_rows,
        "source_fit_penalties_match_e7": all(row["penalty_matches_e7"] for row in source_fit_rows),
        "pairs": [list(pair) for pair in pair_order],
        "subgroup_pairs": [list(pair) for pair in sorted(SUBGROUP_PAIRS)],
        "runin_sizes": [{"n_run": n, "months": months} for n, months in RUNIN_SIZES],
        "draws_per_pair_size": N_DRAWS,
        "runin_rng_seed": RUNIN_SEED,
        "runin_sampling": "one NumPy default_rng seeded once; target patients sampled without replacement; evaluation is every remaining target patient",
        "min_evaluation_n": MIN_EVALUATION_N,
        "methods": list(METHODS),
        "dial_grid": W_GRID.tolist(),
        "recalibration": "E8 penalised intercept/slope fit: summed logistic loss + a^2 + (s-1)^2; dial_recal_shrink adds 2*(w-1)^2",
        "auc_tuning": "dial_auc_shrink maximises run-in AUROC - 0.02*(w-1)^2; applies E8 penalised recalibration at selected w",
        "tie_break": "ties and run-ins with no case or no non-case choose w=1",
        "primary_result": primary_result,
        "summary": summary.to_dict(orient="records"),
        "zero_case_runins": runin_counts,
        "skipped_sizes": skipped_sizes,
        "outputs": [str(path.relative_to(ROOT)) for path in (SUMMARY_PATH, JSON_PATH)],
    }
    summary_json["elapsed_analysis_seconds"] = float(time.perf_counter() - started)
    with JSON_PATH.open("w", encoding="utf-8") as out:
        json.dump(summary_json, out, indent=2, allow_nan=True)
    _print_report(summary, float(summary_json["elapsed_analysis_seconds"]))


if __name__ == "__main__":
    main()
