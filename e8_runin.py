"""E8: realistic unit-level run-in simulation for the P19 trust dial."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from e6_dial import fit_with_cv, sigmoid
from e7_dial import SPLIT_SEED, _build_source_features
from e7b_recal import _transform_target


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "e7_cache.npz"
ANALYSIS = ROOT / "analysis"
E7_SITES = ANALYSIS / "e7_sites.csv"
DRAW_PATH = ANALYSIS / "e8_runin_draws.csv"
SUMMARY_PATH = ANALYSIS / "e8_runin_summary.csv"
JSON_PATH = ANALYSIS / "e8_summary.json"

W_GRID = np.round(np.arange(0.0, 3.0001, 0.05), 2)
W_ONE_INDEX = int(np.flatnonzero(np.isclose(W_GRID, 1.0))[0])
RUNIN_SIZES = ((54, 1), (108, 2), (162, 3), (325, 6), (650, 12), (1300, 24))
N_DRAWS = 200
RUNIN_SEED = 81019
MIN_EVALUATION_N = 500
METHODS = ("keep", "end", "dial", "end_recal", "dial_recal")
PRIMARY_PAIRS = (
    ("A-MICU", "B-MICU"),
    ("B-MICU", "A-MICU"),
    ("A-SICU", "B-SICU"),
    ("B-SICU", "A-SICU"),
)
EPS = 1e-12


def _load_cache() -> tuple[dict[str, np.ndarray], dict]:
    if not CACHE.is_file():
        raise FileNotFoundError(f"Missing E7 cache: {CACHE}")
    with np.load(CACHE, allow_pickle=False) as z:
        cache = {key: z[key].copy() for key in z.files}
    meta = json.loads(str(cache["meta_json"].item()))
    if meta.get("cache_version") != 1:
        raise ValueError("Unexpected E7 cache version")
    p19_meta = meta["P19"]
    if (p19_meta.get("feature_window_iculos") != [1, 24]
            or p19_meta.get("outcome_window_iculos") != [25, 72]
            or not p19_meta.get("features_use_iculos_1_24_only")):
        raise ValueError("P19 cache does not certify E7 feature/outcome windows")
    return cache, meta


def _fit_recalibration_grid(logits: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Penalised logistic intercept/slope fits, vectorised over candidate logits.

    For each column z, minimise sum(log loss(sigmoid(a + s*z))) +
    a**2 + (s - 1)**2, starting at (a, s) = (0, 1).
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = np.zeros(z.shape[1], dtype=np.float64)
    s = np.ones(z.shape[1], dtype=np.float64)

    def objective(a_value: np.ndarray, s_value: np.ndarray) -> np.ndarray:
        eta = np.clip(a_value[None, :] + s_value[None, :] * z, -40.0, 40.0)
        return (np.sum(np.logaddexp(0.0, eta) - y[:, None] * eta, axis=0)
                + a_value * a_value + (s_value - 1.0) ** 2)

    old_loss = objective(a, s)
    for _ in range(100):
        eta = np.clip(a[None, :] + s[None, :] * z, -40.0, 40.0)
        p = sigmoid(eta)
        residual = p - y[:, None]
        weight = p * (1.0 - p)
        g0 = np.sum(residual, axis=0) + 2.0 * a
        g1 = np.sum(residual * z, axis=0) + 2.0 * (s - 1.0)
        if float(np.max(np.maximum(np.abs(g0), np.abs(g1)))) < 1e-8:
            break

        h00 = np.sum(weight, axis=0) + 2.0
        h01 = np.sum(weight * z, axis=0)
        h11 = np.sum(weight * z * z, axis=0) + 2.0
        determinant = h00 * h11 - h01 * h01
        step0 = (h11 * g0 - h01 * g1) / determinant
        step1 = (h00 * g1 - h01 * g0) / determinant
        finite = np.isfinite(step0) & np.isfinite(step1)
        if not np.any(finite):
            break

        scale = np.ones(z.shape[1], dtype=np.float64)
        next_a = a.copy()
        next_s = s.copy()
        next_loss = old_loss.copy()
        pending = finite.copy()
        for _ in range(40):
            if not np.any(pending):
                break
            candidate_a = a - scale * step0
            candidate_s = s - scale * step1
            candidate_loss = objective(candidate_a, candidate_s)
            accepted = pending & np.isfinite(candidate_loss) & (candidate_loss <= old_loss + 1e-12)
            next_a[accepted] = candidate_a[accepted]
            next_s[accepted] = candidate_s[accepted]
            next_loss[accepted] = candidate_loss[accepted]
            pending[accepted] = False
            scale[pending] *= 0.5
        if np.array_equal(next_a, a) and np.array_equal(next_s, s):
            break
        max_step = float(np.max(np.maximum(np.abs(next_a - a), np.abs(next_s - s))))
        a, s, old_loss = next_a, next_s, next_loss
        if max_step < 1e-8:
            break

    return a, s, old_loss


def _choose_index(losses: np.ndarray, candidate_indices: np.ndarray,
                  force_w_one: bool = False) -> int:
    if force_w_one:
        return W_ONE_INDEX
    candidate_losses = np.asarray(losses, dtype=np.float64)[candidate_indices]
    minimum = float(np.min(candidate_losses))
    tolerance = 1e-12 * max(1.0, abs(minimum))
    tied = candidate_indices[np.abs(candidate_losses - minimum) <= tolerance]
    # E8 resolves exact run-in ties in favour of w=1.
    return int(tied[np.argmin(np.abs(W_GRID[tied] - 1.0))])


def _auc_from_ranks_fast(y: np.ndarray, scores: np.ndarray) -> float:
    """E6's Mann-Whitney AUROC formula with vectorised tied-score groups."""
    y = np.asarray(y, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    ends = np.r_[starts[1:], len(y)]
    group_positives = np.add.reduceat(sorted_y, starts)
    average_ranks = (starts + 1 + ends) / 2.0
    rank_sum = float(np.dot(group_positives, average_ranks))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _metric_row(y: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    p = np.clip(sigmoid(logits), EPS, 1.0 - EPS)
    auroc = _auc_from_ranks_fast(y, p)
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))
    return auroc, log_loss


def _run_pair(source: str, target: str, base: np.ndarray, pattern: np.ndarray,
              y: np.ndarray, rng: np.random.Generator,
              pair_type: str) -> tuple[list[dict], list[dict], list[dict]]:
    draw_rows: list[dict] = []
    skip_rows: list[dict] = []
    zero_rows: list[dict] = []
    n_target = len(y)
    logits_all = base[:, None] + pattern[:, None] * W_GRID[None, :]
    end_indices = np.asarray([0, W_ONE_INDEX], dtype=np.int64)
    dial_indices = np.arange(len(W_GRID), dtype=np.int64)

    for n_run, months in RUNIN_SIZES:
        if n_run > n_target - MIN_EVALUATION_N:
            skip_rows.append({
                "source": source, "target": target, "pair_type": pair_type,
                "n_run": n_run, "runin_months": months,
                "target_n": n_target, "evaluation_n_if_run": n_target - n_run,
                "minimum_evaluation_n": MIN_EVALUATION_N,
                "reason": "run-in would leave fewer than 500 evaluation patients",
            })
            print(f"Skipped {source}->{target} n_run={n_run}: target n={n_target} would leave {n_target - n_run} evaluation patients (<500).", flush=True)
            continue

        zero_count = 0
        for draw in range(1, N_DRAWS + 1):
            order = rng.permutation(n_target)
            run_ix = order[:n_run]
            eval_ix = order[n_run:]
            y_run = y[run_ix]
            n_cases = int(y_run.sum())
            zero_count += int(n_cases == 0)
            logits_run = logits_all[run_ix]
            logits_eval = logits_all[eval_ix]

            run_losses = np.sum(
                np.logaddexp(0.0, logits_run) - y_run[:, None] * logits_run,
                axis=0,
            )
            # With no observed cases, E8 treats the run-in choice as tied and keeps w=1.
            no_cases = n_cases == 0
            chosen = {
                "keep": W_ONE_INDEX,
                "end": _choose_index(run_losses, end_indices, no_cases),
                "dial": _choose_index(run_losses, dial_indices, no_cases),
            }

            recal_a, recal_s, recal_losses = _fit_recalibration_grid(logits_run, y_run)
            chosen["end_recal"] = _choose_index(recal_losses, end_indices, no_cases)
            chosen["dial_recal"] = _choose_index(recal_losses, dial_indices, no_cases)

            recal_params: dict[str, tuple[float, float]] = {}
            for method in ("end_recal", "dial_recal"):
                idx = chosen[method]
                recal_params[method] = (float(recal_a[idx]), float(recal_s[idx]))

            for method in METHODS:
                idx = chosen[method]
                eval_logits = logits_eval[:, idx]
                if method in recal_params:
                    a, s = recal_params[method]
                    eval_logits = a + s * eval_logits
                auroc, log_loss = _metric_row(y[eval_ix], eval_logits)
                draw_rows.append({
                    "dataset": "P19", "source": source, "target": target,
                    "pair_type": pair_type, "n_run": n_run,
                    "runin_months": months, "draw": draw,
                    "method": method, "auroc": auroc, "log_loss": log_loss,
                    "chosen_w": float(W_GRID[idx]), "runin_cases": n_cases,
                })

        zero_rows.append({
            "source": source, "target": target, "pair_type": pair_type,
            "n_run": n_run, "runin_months": months,
            "draws": N_DRAWS, "zero_case_draws": zero_count,
            "zero_case_share": zero_count / N_DRAWS,
        })
        print(f"Finished {source}->{target} n_run={n_run} ({months} months); zero-case draws={zero_count}/{N_DRAWS}.", flush=True)

    return draw_rows, skip_rows, zero_rows


def _summarise(draws: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (source, target, pair_type, n_run, months), frame in draws.groupby(
            ["source", "target", "pair_type", "n_run", "runin_months"], sort=False):
        by_method = {
            method: frame[frame["method"] == method].sort_values("draw")
            for method in METHODS
        }
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
                "mean_runin_cases": float(current["runin_cases"].mean()),
                "zero_case_draws": int((current["runin_cases"] == 0).sum()),
                "zero_case_share": float((current["runin_cases"] == 0).mean()),
            }
            for comparator in ("keep", "end_recal"):
                baseline = by_method[comparator]
                if not np.array_equal(current["draw"].to_numpy(), baseline["draw"].to_numpy()):
                    raise AssertionError(f"Unpaired draws for {source}->{target}, n_run={n_run}")
                for metric in ("auroc", "log_loss"):
                    difference = (current[metric].to_numpy(dtype=np.float64)
                                  - baseline[metric].to_numpy(dtype=np.float64))
                    direction = difference > 0 if metric == "auroc" else difference < 0
                    row[f"delta_to_{comparator}_{metric}_mean"] = float(np.mean(difference))
                    row[f"delta_to_{comparator}_{metric}_p10"] = float(np.percentile(difference, 10))
                    row[f"delta_to_{comparator}_{metric}_p90"] = float(np.percentile(difference, 90))
                    row[f"delta_to_{comparator}_{metric}_share_gain"] = float(np.mean(direction))
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def _fmt_delta(summary: pd.DataFrame, source: str, target: str, n_run: int,
               method: str, comparator: str = "keep") -> str:
    row = summary[(summary.source == source) & (summary.target == target)
                  & (summary.n_run == n_run) & (summary.method == method)].iloc[0]
    return (f"{row[f'delta_to_{comparator}_auroc_mean']:+.4f}"
            f" [{row[f'delta_to_{comparator}_auroc_p10']:+.4f},"
            f"{row[f'delta_to_{comparator}_auroc_p90']:+.4f}]"
            f"; gain={row[f'delta_to_{comparator}_auroc_share_gain']:.0%}")


def main() -> None:
    started = time.perf_counter()
    cache, meta = _load_cache()
    sites = cache["P19_sites"].astype(str).tolist()
    n_dynamic = len(meta["P19"]["raw_phys_names"]) - 3
    e7_sites = pd.read_csv(E7_SITES)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for site in sites:
        prefix = f"P19_{site}"
        arrays[site] = {
            "raw_phys": cache[f"{prefix}_raw_phys"].astype(np.float64),
            "raw_pattern": cache[f"{prefix}_raw_pattern"].astype(np.float64),
            "y": cache[f"{prefix}_y"].astype(np.uint8),
            "ids": cache[f"{prefix}_patient_ids"].astype(str),
        }

    source_fits: dict[str, dict] = {}
    source_fit_rows: list[dict] = []
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
        source_fit_rows.append({
            "dataset": "P19", "source": source, "selected_penalty": float(penalty),
            "e7_selected_penalty": e7_penalty, "penalty_matches_e7": penalty_matches,
            "cv_log_loss": {str(k): float(v) for k, v in cv.items()},
            "source_train_n": int(len(train_ix)),
        })
        print(f"Fitted source {source}: penalty={penalty:g} (matches E7).", flush=True)

    pair_order = [(s, t) for s in sites for t in sites if s != t]
    primary_set = set(PRIMARY_PAIRS)
    if not primary_set.issubset(set(pair_order)) or len(pair_order) != 12:
        raise AssertionError("Unexpected P19 site ordering/pair count")

    rng = np.random.default_rng(RUNIN_SEED)
    all_draw_rows: list[dict] = []
    skipped_sizes: list[dict] = []
    zero_case_rows: list[dict] = []
    for source, target in pair_order:
        fitted = source_fits[source]
        source_ids = arrays[source]["ids"][fitted["train_ix"]]
        target_arrays = arrays[target]
        overlap = np.intersect1d(source_ids, target_arrays["ids"])
        if len(overlap):
            raise AssertionError(f"Source train / target overlap: P19/{source}->{target}")
        x_phys, x_pattern = _transform_target(
            target_arrays["raw_phys"], target_arrays["raw_pattern"], n_dynamic, fitted["prep"]
        )
        beta = fitted["beta"]
        n_phys = fitted["n_phys"]
        base = beta[0] + x_phys @ beta[1:1 + n_phys]
        pattern = x_pattern @ beta[1 + n_phys:]
        pair_type = "primary" if (source, target) in primary_set else "secondary"
        pair_rows, pair_skips, pair_zero = _run_pair(
            source, target, base, pattern, target_arrays["y"], rng, pair_type
        )
        all_draw_rows.extend(pair_rows)
        skipped_sizes.extend(pair_skips)
        zero_case_rows.extend(pair_zero)

    draws = pd.DataFrame.from_records(all_draw_rows)
    summary = _summarise(draws)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    draws.to_csv(DRAW_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    json_summary = {
        "spec": "E8",
        "cache_version": meta["cache_version"],
        "dataset": "P19",
        "source_split_seed": SPLIT_SEED,
        "source_fit_penalties_match_e7": all(r["penalty_matches_e7"] for r in source_fit_rows),
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
        "recalibration": "per candidate logit, minimise summed logistic loss + a^2 + (s-1)^2 by Newton from a=0,s=1",
        "tie_break": "w closest to 1; zero-case run-ins explicitly choose w=1",
        "zero_case_runins": zero_case_rows,
        "skipped_sizes": skipped_sizes,
        "outputs": [str(p.relative_to(ROOT)) for p in (DRAW_PATH, SUMMARY_PATH, JSON_PATH)],
    }
    json_summary["elapsed_analysis_seconds"] = float(time.perf_counter() - started)
    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, allow_nan=True)

    print("\nSelected source penalties (all match E7): " + ", ".join(
        f"{row['source']}={row['selected_penalty']:g}" for row in source_fit_rows
    ))
    print("\nPrimary pairs at n_run=162 (3 months): pair | mean cases | dial_recal-end_recal AUROC | dial_recal-keep AUROC")
    for source, target in PRIMARY_PAIRS:
        row = summary[(summary.source == source) & (summary.target == target)
                      & (summary.n_run == 162) & (summary.method == "dial_recal")].iloc[0]
        print(f"{source}->{target} | {row.mean_runin_cases:.2f} | "
              f"{_fmt_delta(summary, source, target, 162, 'dial_recal', 'end_recal')} | "
              f"{_fmt_delta(summary, source, target, 162, 'dial_recal', 'keep')}")
    print("\nPrimary pairs across sizes: AUROC difference to keep (mean); columns end / dial / end_recal / dial_recal")
    for source, target in PRIMARY_PAIRS:
        for n_run, months in RUNIN_SIZES:
            raw_part = summary[(summary.source == source) & (summary.target == target)
                               & (summary.n_run == n_run)]
            if raw_part.empty:
                print(f"{source}->{target} {months:>2}mo (n={n_run:>4}): skipped (<500 evaluation patients)")
                continue
            part = raw_part.set_index("method")
            means = [part.loc[m, "delta_to_keep_auroc_mean"] for m in ("end", "dial", "end_recal", "dial_recal")]
            print(f"{source}->{target} {months:>2}mo (n={n_run:>4}): "
                  + " / ".join(f"{value:+.4f}" for value in means))
    print("\nSecondary pairs at n_run=162 (3 months): pair | mean cases | dial_recal-keep AUROC [p10,p90] | gain | zero-case draws")
    for source, target in pair_order:
        if (source, target) in primary_set:
            continue
        row = summary[(summary.source == source) & (summary.target == target)
                      & (summary.n_run == 162) & (summary.method == "dial_recal")].iloc[0]
        print(f"{source}->{target} | {row.mean_runin_cases:.2f} | "
              f"{_fmt_delta(summary, source, target, 162, 'dial_recal', 'keep')} | "
              f"{int(row.zero_case_draws)}/{N_DRAWS}")

    # Three compact findings, followed by the measured runtime.
    primary_162 = summary[(summary.pair_type == "primary") & (summary.n_run == 162)
                          & (summary.method == "dial_recal")]
    mean_gain_keep = float(primary_162["delta_to_keep_auroc_mean"].mean())
    mean_gain_end_recal = float(primary_162["delta_to_end_recal_auroc_mean"].mean())
    mean_zero_share = float(primary_162["zero_case_share"].mean())
    print(f"Across the four primary pairs, mean dial_recal AUROC change versus keep was {mean_gain_keep:+.4f}; versus end_recal it was {mean_gain_end_recal:+.4f}.")
    print(f"At three months, primary run-ins averaged {mean_zero_share:.1%} zero-case draws, with each pair's count shown above.")
    print(f"Source penalties matched E7 for all four P19 sites, and {len(skipped_sizes)} pair-size combinations were skipped under the 500-patient evaluation rule.")
    runtime = json_summary["elapsed_analysis_seconds"]
    print(f"E8 analysis runtime: {runtime / 60:.1f} min ({runtime:.1f} s).")
    print(f"Wrote {DRAW_PATH}, {SUMMARY_PATH}, {JSON_PATH}.")


if __name__ == "__main__":
    main()
