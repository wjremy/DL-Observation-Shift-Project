"""E9: E8's realistic run-in design with ranking-focused corrected tuning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import e8_runin as e8
from e6_dial import fit_with_cv
from e7_dial import SPLIT_SEED, _build_source_features
from e7b_recal import _transform_target


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis"
SUMMARY_PATH = ANALYSIS / "e9_summary.csv"
JSON_PATH = ANALYSIS / "e9_summary.json"

W_GRID = e8.W_GRID
W_ONE_INDEX = e8.W_ONE_INDEX
RUNIN_SIZES = e8.RUNIN_SIZES
N_DRAWS = e8.N_DRAWS
MIN_EVALUATION_N = e8.MIN_EVALUATION_N
PRIMARY_PAIRS = e8.PRIMARY_PAIRS
METHODS = ("keep", "dial_recal", "dial_recal_shrink", "dial_auc", "dial_auc_shrink")
RUNIN_SEED = 91020
RECAL_W_PENALTY = 2.0
AUC_W_PENALTY = 0.02


def _auc_curve_from_ranks(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Compute AUROC for every candidate column, retaining E8's tie handling."""
    y = np.asarray(y, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.full(scores.shape[1], np.nan, dtype=np.float64)

    order = np.argsort(scores, axis=0, kind="mergesort")
    sorted_scores = np.take_along_axis(scores, order, axis=0)
    sorted_y = y[order]
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)[:, None]
    rank_sums = np.sum(sorted_y * ranks, axis=0, dtype=np.float64)
    auc = (rank_sums - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    tied_columns = np.flatnonzero(np.any(sorted_scores[1:] == sorted_scores[:-1], axis=0))
    for column in tied_columns:
        auc[column] = e8._auc_from_ranks_fast(y, scores[:, column])
    return auc


def _choose_max_index(values: np.ndarray, candidate_indices: np.ndarray,
                      force_w_one: bool = False) -> int:
    if force_w_one:
        return W_ONE_INDEX
    candidate_values = np.asarray(values, dtype=np.float64)[candidate_indices]
    maximum = float(np.max(candidate_values))
    tolerance = 1e-12 * max(1.0, abs(maximum))
    tied = candidate_indices[np.abs(candidate_values - maximum) <= tolerance]
    return int(tied[np.argmin(np.abs(W_GRID[tied] - 1.0))])


def _run_pair(source: str, target: str, base: np.ndarray, pattern: np.ndarray,
              y: np.ndarray, rng: np.random.Generator,
              pair_type: str) -> tuple[list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    skipped: list[dict] = []
    runin_counts: list[dict] = []
    n_target = len(y)
    logits_all = base[:, None] + pattern[:, None] * W_GRID[None, :]
    all_indices = np.arange(len(W_GRID), dtype=np.int64)

    for n_run, months in RUNIN_SIZES:
        if n_run > n_target - MIN_EVALUATION_N:
            skipped.append({
                "source": source, "target": target, "pair_type": pair_type,
                "n_run": n_run, "runin_months": months,
                "target_n": n_target, "evaluation_n_if_run": n_target - n_run,
                "minimum_evaluation_n": MIN_EVALUATION_N,
                "reason": "run-in would leave fewer than 500 evaluation patients",
            })
            print(f"Skipped {source}->{target} n_run={n_run}: evaluation n would be "
                  f"{n_target - n_run} (<500).", flush=True)
            continue

        zero_case_count = 0
        for draw in range(1, N_DRAWS + 1):
            # Same sampling path as E8: one shared RNG, permutation without replacement,
            # then evaluate every patient outside the run-in.
            order = rng.permutation(n_target)
            run_ix = order[:n_run]
            eval_ix = order[n_run:]
            y_run = y[run_ix]
            n_cases = int(y_run.sum())
            n_controls = n_run - n_cases
            zero_case_count += int(n_cases == 0)
            logits_run = logits_all[run_ix]
            logits_eval = logits_all[eval_ix]

            # E8's un-recalibrated log loss is retained only as a reference in E8;
            # E9's calibration-tuned dial methods use this exact penalised path.
            recal_a, recal_s, recal_losses = e8._fit_recalibration_grid(logits_run, y_run)
            no_cases = n_cases == 0
            no_class_pair = no_cases or n_controls == 0
            recal_shrink_losses = recal_losses + RECAL_W_PENALTY * (W_GRID - 1.0) ** 2
            runin_auc = _auc_curve_from_ranks(y_run, logits_run)
            auc_shrink_score = runin_auc - AUC_W_PENALTY * (W_GRID - 1.0) ** 2

            chosen = {
                "keep": W_ONE_INDEX,
                # This call and the objective above exactly match E8's dial_recal definition.
                "dial_recal": e8._choose_index(recal_losses, all_indices, no_cases),
                "dial_recal_shrink": e8._choose_index(
                    recal_shrink_losses, all_indices, no_cases
                ),
                "dial_auc": _choose_max_index(runin_auc, all_indices, no_class_pair),
                "dial_auc_shrink": _choose_max_index(
                    auc_shrink_score, all_indices, no_class_pair
                ),
            }

            recalibrated_methods = ("dial_recal", "dial_recal_shrink",
                                    "dial_auc", "dial_auc_shrink")
            recal_params = {
                method: (float(recal_a[chosen[method]]), float(recal_s[chosen[method]]))
                for method in recalibrated_methods
            }
            for method in METHODS:
                idx = chosen[method]
                eval_logits = logits_eval[:, idx]
                if method in recal_params:
                    a, s = recal_params[method]
                    eval_logits = a + s * eval_logits
                auroc, log_loss = e8._metric_row(y[eval_ix], eval_logits)
                rows.append({
                    "dataset": "P19", "source": source, "target": target,
                    "pair_type": pair_type, "n_run": n_run,
                    "runin_months": months, "draw": draw,
                    "method": method, "auroc": auroc, "log_loss": log_loss,
                    "chosen_w": float(W_GRID[idx]), "runin_cases": n_cases,
                })

        runin_counts.append({
            "source": source, "target": target, "pair_type": pair_type,
            "n_run": n_run, "runin_months": months,
            "draws": N_DRAWS, "zero_case_draws": zero_case_count,
            "zero_case_share": zero_case_count / N_DRAWS,
        })
        print(f"Finished {source}->{target} n_run={n_run} ({months} months); "
              f"zero-case draws={zero_case_count}/{N_DRAWS}.", flush=True)
    return rows, skipped, runin_counts


def _summarise(draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["source", "target", "pair_type", "n_run", "runin_months"]
    for (source, target, pair_type, n_run, months), frame in draws.groupby(group_cols, sort=False):
        by_method = {
            method: frame[frame["method"] == method].sort_values("draw")
            for method in METHODS
        }
        baseline = by_method["keep"]
        for method in METHODS:
            current = by_method[method]
            row = {
                "dataset": "P19", "source": source, "target": target,
                "pair_type": pair_type, "n_run": int(n_run),
                "runin_months": int(months), "method": method,
                "draws": int(len(current)),
                "mean_auroc": float(current["auroc"].mean()),
                "mean_log_loss": float(current["log_loss"].mean()),
                "mean_chosen_w": float(current["chosen_w"].mean()),
                "p10_chosen_w": float(current["chosen_w"].quantile(0.10)),
                "p90_chosen_w": float(current["chosen_w"].quantile(0.90)),
                "mean_runin_cases": float(current["runin_cases"].mean()),
                "zero_case_draws": int((current["runin_cases"] == 0).sum()),
                "zero_case_share": float((current["runin_cases"] == 0).mean()),
            }
            if not np.array_equal(current["draw"].to_numpy(), baseline["draw"].to_numpy()):
                raise AssertionError(f"Unpaired draws for {source}->{target}, n_run={n_run}")
            for metric in ("auroc", "log_loss"):
                difference = (current[metric].to_numpy(dtype=np.float64)
                              - baseline[metric].to_numpy(dtype=np.float64))
                direction = difference > 0 if metric == "auroc" else difference < 0
                row[f"delta_to_keep_{metric}_mean"] = float(np.mean(difference))
                row[f"delta_to_keep_{metric}_p10"] = float(np.percentile(difference, 10))
                row[f"delta_to_keep_{metric}_p90"] = float(np.percentile(difference, 90))
                row[f"delta_to_keep_{metric}_share_gain"] = float(np.mean(direction))
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _fit_source_models(cache: dict[str, np.ndarray], meta: dict,
                       sites: list[str]) -> tuple[dict[str, dict], list[dict], dict[str, dict]]:
    n_dynamic = len(meta["P19"]["raw_phys_names"]) - 3
    arrays: dict[str, dict] = {}
    for site in sites:
        prefix = f"P19_{site}"
        arrays[site] = {
            "raw_phys": cache[f"{prefix}_raw_phys"].astype(np.float64),
            "raw_pattern": cache[f"{prefix}_raw_pattern"].astype(np.float64),
            "y": cache[f"{prefix}_y"].astype(np.uint8),
            "ids": cache[f"{prefix}_patient_ids"].astype(str),
        }

    e7_sites = pd.read_csv(e8.E7_SITES)
    source_fits: dict[str, dict] = {}
    fit_rows: list[dict] = []
    for site_ix, source in enumerate(sites):
        source_arrays = arrays[source]
        n_source = len(source_arrays["y"])
        split_rng = np.random.default_rng(SPLIT_SEED + site_ix)
        permutation = split_rng.permutation(n_source)
        n_train = int(np.floor(0.8 * n_source))
        train_ix = permutation[:n_train]
        x, x_phys, prep = _build_source_features(
            source_arrays["raw_phys"], source_arrays["raw_pattern"], n_dynamic, train_ix
        )
        if len(np.unique(source_arrays["y"][train_ix])) < 2:
            raise ValueError(f"Source training split has one outcome class: P19/{source}")
        cv_seed = SPLIT_SEED + 10000 + site_ix
        beta, penalty, cv = fit_with_cv(x[train_ix], source_arrays["y"][train_ix], seed=cv_seed)
        e7_row = e7_sites[(e7_sites.dataset == "P19") & (e7_sites.site == source)].iloc[0]
        e7_penalty = float(e7_row.selected_penalty)
        penalty_matches = bool(np.isclose(penalty, e7_penalty, rtol=0.0, atol=0.0))
        if not penalty_matches:
            raise AssertionError(f"E7 penalty mismatch for P19/{source}: {penalty} vs {e7_penalty}")
        n_phys = x_phys.shape[1]
        source_fits[source] = {
            "beta": beta, "prep": prep, "n_phys": n_phys,
            "train_ix": train_ix, "penalty": float(penalty),
        }
        fit_rows.append({
            "dataset": "P19", "source": source,
            "selected_penalty": float(penalty), "e7_selected_penalty": e7_penalty,
            "penalty_matches_e7": penalty_matches,
            "cv_log_loss": {str(k): float(v) for k, v in cv.items()},
            "source_train_n": int(len(train_ix)),
        })
        print(f"Fitted source {source}: penalty={penalty:g} (matches E7).", flush=True)
    return source_fits, fit_rows, arrays


def _fmt_primary(row: pd.Series) -> str:
    return (f"{row.delta_to_keep_auroc_mean:+.4f} "
            f"[{row.delta_to_keep_auroc_p10:+.4f},{row.delta_to_keep_auroc_p90:+.4f}] "
            f"gain={row.delta_to_keep_auroc_share_gain:.0%} w={row.mean_chosen_w:.2f}")


def _print_report(summary: pd.DataFrame, pair_order: list[tuple[str, str]],
                  runin_counts: list[dict], runtime: float) -> None:
    primary_methods = METHODS[1:]
    three_month = summary[summary.n_run == 162]
    primary_three_month = three_month[three_month.pair_type == "primary"]
    print("\nPRIMARY, 3 months: AUROC delta to keep [p10,p90]; gain share; mean w")
    print("pair | " + " | ".join(primary_methods))
    for source, target in PRIMARY_PAIRS:
        cells = []
        for method in primary_methods:
            row = primary_three_month[(primary_three_month.source == source)
                                      & (primary_three_month.target == target)
                                      & (primary_three_month.method == method)].iloc[0]
            cells.append(_fmt_primary(row))
        print(f"{source}->{target} | " + " | ".join(cells))

    print("\nPRIMARY, all run-in sizes: mean AUROC delta / gain share / mean w")
    print("pair | months | " + " | ".join(primary_methods))
    for source, target in PRIMARY_PAIRS:
        for n_run, months in RUNIN_SIZES:
            part = summary[(summary.source == source) & (summary.target == target)
                           & (summary.n_run == n_run)]
            if part.empty:
                print(f"{source}->{target} | {months} | skipped")
                continue
            cells = []
            for method in primary_methods:
                row = part[part.method == method].iloc[0]
                cells.append(f"{row.delta_to_keep_auroc_mean:+.4f}/"
                             f"{row.delta_to_keep_auroc_share_gain:.0%}/"
                             f"{row.mean_chosen_w:.2f}")
            print(f"{source}->{target} | {months} | " + " | ".join(cells))

    print("\nSECONDARY, 3 months: AUROC delta to keep [p10,p90]; gain share; mean w")
    print("pair | dial_recal_shrink | dial_auc_shrink")
    for source, target in pair_order:
        if (source, target) in set(PRIMARY_PAIRS):
            continue
        part = three_month[(three_month.source == source) & (three_month.target == target)]
        if part.empty:
            print(f"{source}->{target} | skipped")
            continue
        cells = []
        for method in ("dial_recal_shrink", "dial_auc_shrink"):
            method_row = part[part.method == method]
            cells.append(_fmt_primary(method_row.iloc[0]) if not method_row.empty else "skipped")
        print(f"{source}->{target} | " + " | ".join(cells))

    primary_means = primary_three_month[
        primary_three_month.method.isin(primary_methods)
    ].groupby("method")["delta_to_keep_auroc_mean"].mean()
    best_method = str(primary_means.idxmax())
    best_gain = float(primary_means.max())
    mean_zero_share = float(np.mean([row["zero_case_share"] for row in runin_counts
                                     if row["pair_type"] == "primary" and row["n_run"] == 162]))
    print(f"At three months, {best_method} had the largest mean AUROC change across primary pairs ({best_gain:+.4f}); the tables report each pair's uncertainty and gain share.")
    print(f"Primary three-month run-ins averaged {mean_zero_share:.1%} zero-case draws; all reported AUROC comparisons use the remaining target patients.")
    print(f"E9 runtime: {runtime / 60:.1f} min ({runtime:.1f} s); wrote {SUMMARY_PATH} and {JSON_PATH}.")


def report_existing() -> None:
    summary = pd.read_csv(SUMMARY_PATH)
    with JSON_PATH.open("r", encoding="utf-8") as source:
        saved = json.load(source)
    sites = e8._load_cache()[0]["P19_sites"].astype(str).tolist()
    pair_order = [(source, target) for source in sites for target in sites if source != target]
    _print_report(summary, pair_order, saved["zero_case_runins"],
                  float(saved["elapsed_analysis_seconds"]))


def main() -> None:
    started = time.perf_counter()
    cache, meta = e8._load_cache()
    sites = cache["P19_sites"].astype(str).tolist()
    source_fits, source_fit_rows, arrays = _fit_source_models(cache, meta, sites)

    pair_order = [(source, target) for source in sites for target in sites if source != target]
    if len(pair_order) != 12 or not set(PRIMARY_PAIRS).issubset(set(pair_order)):
        raise AssertionError("Unexpected P19 site ordering/pair count")
    primary_set = set(PRIMARY_PAIRS)
    rng = np.random.default_rng(RUNIN_SEED)
    draw_rows: list[dict] = []
    skipped_sizes: list[dict] = []
    runin_counts: list[dict] = []

    for source, target in pair_order:
        fitted = source_fits[source]
        source_ids = arrays[source]["ids"][fitted["train_ix"]]
        target_arrays = arrays[target]
        overlap = np.intersect1d(source_ids, target_arrays["ids"])
        if len(overlap):
            raise AssertionError(f"Source train / target overlap: P19/{source}->{target}")
        x_phys, x_pattern = _transform_target(
            target_arrays["raw_phys"], target_arrays["raw_pattern"],
            len(meta["P19"]["raw_phys_names"]) - 3, fitted["prep"],
        )
        beta = fitted["beta"]
        n_phys = fitted["n_phys"]
        base = beta[0] + x_phys @ beta[1:1 + n_phys]
        pattern = x_pattern @ beta[1 + n_phys:]
        pair_type = "primary" if (source, target) in primary_set else "secondary"
        pair_rows, pair_skips, pair_counts = _run_pair(
            source, target, base, pattern, target_arrays["y"], rng, pair_type
        )
        draw_rows.extend(pair_rows)
        skipped_sizes.extend(pair_skips)
        runin_counts.extend(pair_counts)

    draws = pd.DataFrame.from_records(draw_rows)
    if not np.all(draws.loc[draws.method == "keep", "chosen_w"].to_numpy() == 1.0):
        raise AssertionError("E9 keep did not remain at E8's fixed w=1")
    summary = _summarise(draws)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    summary_json = {
        "spec": "E9",
        "cache_version": meta["cache_version"],
        "dataset": "P19",
        "source_split_seed": SPLIT_SEED,
        "source_fit_penalties_match_e7": all(row["penalty_matches_e7"] for row in source_fit_rows),
        "source_fits": source_fit_rows,
        "primary_pairs": [list(pair) for pair in PRIMARY_PAIRS],
        "secondary_pairs": [list(pair) for pair in pair_order if pair not in primary_set],
        "runin_sizes": [{"n_run": n, "months": months} for n, months in RUNIN_SIZES],
        "draws_per_pair_size": N_DRAWS,
        "runin_rng_seed": RUNIN_SEED,
        "runin_sampling": "single NumPy default_rng seeded once; target patients sampled without replacement; evaluation is every remaining target patient",
        "min_evaluation_n": MIN_EVALUATION_N,
        "methods": list(METHODS),
        "dial_grid": W_GRID.tolist(),
        "e8_definition_checks": {
            "keep_is_fixed_at_w1": True,
            "dial_recal_reuses_e8_fit_and_choice_helpers": True,
            "note": "keep uses E8's w=1 grid index; dial_recal calls e8._fit_recalibration_grid and e8._choose_index with E8's no-case override",
        },
        "recalibration": "E8 penalised intercept/slope fit: summed logistic loss + a^2 + (s-1)^2; dial_recal_shrink adds 2*(w-1)^2",
        "auc_tuning": "run-in AUROC by E8 rank helper; dial_auc_shrink maximises AUROC - 0.02*(w-1)^2",
        "tie_break": "closest w to 1; dial_auc variants force w=1 when run-in has no cases or no controls; recalibration dials force w=1 with no cases, matching E8",
        "zero_case_runins": runin_counts,
        "skipped_sizes": skipped_sizes,
        "summary": summary.to_dict(orient="records"),
        "outputs": [str(path.relative_to(ROOT)) for path in (SUMMARY_PATH, JSON_PATH)],
    }
    summary_json["elapsed_analysis_seconds"] = float(time.perf_counter() - started)
    with JSON_PATH.open("w", encoding="utf-8") as out:
        json.dump(summary_json, out, indent=2, allow_nan=True)

    _print_report(summary, pair_order, runin_counts,
                  float(summary_json["elapsed_analysis_seconds"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-existing", action="store_true",
                        help="print the report from completed E9 summary files")
    args = parser.parse_args()
    if args.report_existing:
        report_existing()
    else:
        main()
