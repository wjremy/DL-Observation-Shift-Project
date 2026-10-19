"""E6 analysis: L2 logistic dial, paired bootstrap, run-in adaptation, descriptives."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "e6_cache.npz"
ANALYSIS = ROOT / "analysis"
PENALTY_GRID = np.asarray([0.001, 0.003, 0.01, 0.03, 0.1, 0.3], dtype=np.float64)
W_GRID = np.round(np.arange(0.0, 1.5001, 0.1), 1)
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 66019
RUNIN_SEED = 66100
EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def logistic_loss(y: np.ndarray, logits: np.ndarray) -> float:
    z = np.clip(logits, -40.0, 40.0)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def fit_logistic(x: np.ndarray, y: np.ndarray, penalty: float, init: np.ndarray | None = None,
                 max_iter: int = 60) -> np.ndarray:
    """Newton/IRLS for average BCE + penalty/2 * ||slopes||^2."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = x.shape
    design = np.empty((n, p + 1), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1:] = x
    reg = np.full(p + 1, penalty, dtype=np.float64)
    reg[0] = 1e-10  # numerical stabilization; intercept is otherwise unpenalized.
    beta = np.zeros(p + 1, dtype=np.float64) if init is None else np.array(init, dtype=np.float64, copy=True)
    beta[0] = np.clip(beta[0], -30, 30)

    def objective(b: np.ndarray) -> float:
        z = np.clip(design @ b, -40.0, 40.0)
        return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * penalty * np.dot(b[1:], b[1:]))

    old_loss = objective(beta)
    for _ in range(max_iter):
        z = np.clip(design @ beta, -40.0, 40.0)
        prob = sigmoid(z)
        weight = np.maximum(prob * (1.0 - prob), 1e-10)
        grad = (design.T @ (prob - y)) / n + reg * beta
        hess = (design.T @ (design * weight[:, None])) / n
        hess.flat[:: p + 2] += reg
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=1e-10)[0]
        if not np.all(np.isfinite(step)):
            break
        if float(np.max(np.abs(grad))) < 2e-7:
            break
        scale = 1.0
        accepted = False
        for _ in range(30):
            candidate = beta - scale * step
            candidate[0] = np.clip(candidate[0], -35.0, 35.0)
            loss = objective(candidate)
            if loss <= old_loss + 1e-12:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
        beta, old_loss = candidate, loss
        if float(np.max(np.abs(scale * step))) < 1e-7:
            break
    return beta


def stratified_folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(k)]
    for cls in (0, 1):
        indices = np.flatnonzero(y == cls)
        indices = rng.permutation(indices)
        for j, part in enumerate(np.array_split(indices, k)):
            fold_parts[j].append(part)
    return [np.concatenate(parts) for parts in fold_parts]


def choose_penalty(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, dict[float, float]]:
    folds = stratified_folds(y, 5, seed)
    losses = np.zeros(len(PENALTY_GRID), dtype=np.float64)
    for fold_id, valid in enumerate(folds):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[valid] = False
        train = np.flatnonzero(train_mask)
        init = None
        for li, penalty in enumerate(PENALTY_GRID):
            beta = fit_logistic(x[train], y[train], float(penalty), init=init)
            init = beta
            losses[li] += logistic_loss(y[valid], np.column_stack((np.ones(len(valid)), x[valid])) @ beta) / len(folds)
    result = {float(p): float(loss) for p, loss in zip(PENALTY_GRID, losses)}
    return float(PENALTY_GRID[int(np.argmin(losses))]), result


def fit_with_cv(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, float, dict[float, float]]:
    penalty, cv = choose_penalty(x, y, seed)
    return fit_logistic(x, y, penalty), penalty, cv


def auc_from_ranks(y: np.ndarray, scores: np.ndarray) -> float:
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
    rank_sums = 0.0
    for start, end in zip(starts, ends):
        avg_rank = (start + 1 + end) / 2.0
        rank_sums += float(sorted_y[start:end].sum()) * avg_rank
    return (rank_sums - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return {
        "auroc": auc_from_ranks(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
    }


def bootstrap_curve(y: np.ndarray, predictions: np.ndarray, seed: int, n_boot: int = BOOTSTRAP_N):
    """Paired patient bootstrap; rank groups are precomputed for efficient AUROC."""
    y = np.asarray(y, dtype=np.uint8)
    predictions = np.asarray(predictions, dtype=np.float64)
    n, m = predictions.shape
    rank_info = []
    for j in range(m):
        order = np.argsort(predictions[:, j], kind="mergesort")
        ordered = predictions[order, j]
        group_starts = np.r_[0, np.flatnonzero(np.diff(ordered) != 0) + 1]
        group_id_sorted = np.repeat(np.arange(len(group_starts), dtype=np.int32), np.diff(np.r_[group_starts, n]))
        group_id = np.empty(n, dtype=np.int32)
        group_id[order] = group_id_sorted
        rank_info.append((group_id, len(group_starts)))

    out = {key: np.full((n_boot, m), np.nan, dtype=np.float64) for key in ("auroc", "brier", "log_loss")}
    p_clip = np.clip(predictions, EPS, 1.0 - EPS)
    loss_values = -(y[:, None] * np.log(p_clip) + (1 - y[:, None]) * np.log1p(-p_clip))
    brier_values = (p_clip - y[:, None]) ** 2
    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        sample = rng.integers(0, n, size=n, dtype=np.int32)
        counts = np.bincount(sample, minlength=n).astype(np.float64)
        pos_n = float(np.dot(counts, y))
        neg_n = n - pos_n
        out["brier"][b] = counts @ brier_values / n
        out["log_loss"][b] = counts @ loss_values / n
        if pos_n <= 0 or neg_n <= 0:
            continue
        positive_weights = counts * y
        negative_weights = counts * (1 - y)
        for j, (group_id, n_groups) in enumerate(rank_info):
            group_pos = np.bincount(group_id, weights=positive_weights, minlength=n_groups)
            group_neg = np.bincount(group_id, weights=negative_weights, minlength=n_groups)
            negatives_before = np.cumsum(group_neg) - group_neg
            concordant = np.sum(group_pos * (negatives_before + 0.5 * group_neg))
            out["auroc"][b, j] = concordant / (pos_n * neg_n)
    return {k: np.nanpercentile(v, [2.5, 97.5], axis=0) for k, v in out.items()}


def calibrate_intercept(base_logit: np.ndarray, y: np.ndarray, initial: float = 0.0) -> float:
    z = np.asarray(base_logit, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    delta = float(initial)
    for _ in range(100):
        p = sigmoid(z + delta)
        grad = float(np.mean(p - y))
        hess = float(np.mean(np.maximum(p * (1 - p), 1e-10)))
        step = np.clip(grad / hess, -5.0, 5.0)
        delta_new = float(np.clip(delta - step, -35.0, 35.0))
        if abs(delta_new - delta) < 1e-8:
            return delta_new
        delta = delta_new
    return delta


def summarize_runin(X: np.ndarray, y: np.ndarray, b_pool: np.ndarray, b_test: np.ndarray,
                    w_logits_pool: np.ndarray, w_logits_test: np.ndarray, x_test: np.ndarray,
                    y_test: np.ndarray, full_beta: np.ndarray, seed: int) -> pd.DataFrame:
    requested_sizes = [100, 250, 500, 1000, 2500, 5000]
    sizes = [n for n in requested_sizes if n <= len(b_pool)]
    if len(b_pool) not in sizes:
        sizes.append(len(b_pool))
    missing_sizes = [n for n in requested_sizes if n > len(b_pool)]
    if missing_sizes:
        print(f"Run-in sizes unavailable without reusing B-test patients: {missing_sizes}; pool={len(b_pool):,}", flush=True)
    method_names = ["w_tuned", "intercept_slope", "w_plus_intercept", "full_refit", "w0_fixed", "w1_fixed"]
    # fixed source-model references use b0 + X beta, with pattern coefficient dialed.
    records = []
    rng = np.random.default_rng(seed)
    n_repeats = 50
    grid_logits_pool = np.column_stack([w_logits_pool[:, wi] for wi in range(len(W_GRID))])
    grid_logits_test = np.column_stack([w_logits_test[:, wi] for wi in range(len(W_GRID))])
    base1_pool = grid_logits_pool[:, int(np.argmin(np.abs(W_GRID - 1.0)))]
    base1_test = grid_logits_test[:, int(np.argmin(np.abs(W_GRID - 1.0)))]
    fixed_test_probs = {
        "w0_fixed": sigmoid(grid_logits_test[:, 0]),
        "w1_fixed": sigmoid(base1_test),
    }

    for size_ix, n_run in enumerate(sizes):
        reps = [np.arange(len(b_pool))] * n_repeats if n_run == len(b_pool) else [rng.choice(len(b_pool), size=n_run, replace=False) for _ in range(n_repeats)]
        for rep_ix, local_idx in enumerate(reps):
            rows = []
            y_run = y[b_pool[local_idx]]
            n_cases = int(y_run.sum())
            run_logits = grid_logits_pool[local_idx]
            run_losses = np.mean(np.logaddexp(0.0, run_logits) - y_run[:, None] * run_logits, axis=0)
            best_w_ix = int(np.argmin(run_losses))
            p_wtest = sigmoid(grid_logits_test[:, best_w_ix])
            rows.append(("w_tuned", p_wtest))

            # Conventional intercept+slope recalibration of the w=1 source-model logit.
            calibrator = fit_logistic(base1_pool[local_idx, None], y_run, penalty=1e-8)
            cal_logits = calibrator[0] + calibrator[1] * base1_test
            rows.append(("intercept_slope", sigmoid(cal_logits)))

            best_combo_loss = np.inf
            best_combo_p = None
            for wi, w in enumerate(W_GRID):
                delta = calibrate_intercept(grid_logits_pool[local_idx, wi], y_run)
                adjusted = grid_logits_test[:, wi] + delta
                zrun = run_logits[:, wi] + delta
                loss = logistic_loss(y_run, zrun)
                if loss < best_combo_loss:
                    best_combo_loss = loss
                    best_combo_p = sigmoid(adjusted)
            rows.append(("w_plus_intercept", best_combo_p))

            # Re-tune the same L2 grid by 5-fold CV on each run-in sample, then refit it.
            cv_seed = seed + size_ix * 1000 + (0 if n_run == len(b_pool) else rep_ix)
            refit_beta, _, _ = fit_with_cv(X[b_pool[local_idx]], y_run, cv_seed)
            p_refit = sigmoid(np.column_stack((np.ones(len(y_test)), x_test)) @ refit_beta)
            rows.append(("full_refit", p_refit))
            rows.append(("w0_fixed", fixed_test_probs["w0_fixed"]))
            rows.append(("w1_fixed", fixed_test_probs["w1_fixed"]))

            for method, probs in rows:
                met = point_metrics(y_test, probs)
                records.append({
                    "n_run": int(n_run), "repeat": rep_ix, "method": method,
                    "n_cases": n_cases, "selected_w": float(W_GRID[best_w_ix]) if method == "w_tuned" else np.nan,
                    "auroc": met["auroc"], "log_loss": met["log_loss"],
                })
        print(f"Run-in complete: n={n_run:,}; repeats={n_repeats}; cases per sample mean={np.mean([r['n_cases'] for r in records if r['n_run']==n_run and r['method']=='w_tuned']):.1f}", flush=True)

    raw = pd.DataFrame.from_records(records)
    summaries = []
    for (n_run, method), group in raw.groupby(["n_run", "method"], sort=True):
        summaries.append({
            "n_run": int(n_run), "method": method,
            "n_cases_mean": float(group["n_cases"].mean()),
            "n_cases_p10": float(group["n_cases"].quantile(0.1)),
            "n_cases_p90": float(group["n_cases"].quantile(0.9)),
            "selected_w_mean": float(group["selected_w"].mean()) if group["selected_w"].notna().any() else np.nan,
            "selected_w_p10": float(group["selected_w"].quantile(0.1)) if group["selected_w"].notna().any() else np.nan,
            "selected_w_p90": float(group["selected_w"].quantile(0.9)) if group["selected_w"].notna().any() else np.nan,
            "auroc_mean": float(group["auroc"].mean()),
            "auroc_p10": float(group["auroc"].quantile(0.1)),
            "auroc_p90": float(group["auroc"].quantile(0.9)),
            "log_loss_mean": float(group["log_loss"].mean()),
            "log_loss_p10": float(group["log_loss"].quantile(0.1)),
            "log_loss_p90": float(group["log_loss"].quantile(0.9)),
        })
    return pd.DataFrame(summaries), raw


def main() -> None:
    start = time.perf_counter()
    if not CACHE.exists():
        raise FileNotFoundError(f"Run e6_cohort.py first; missing {CACHE}")
    with np.load(CACHE, allow_pickle=False) as z:
        X = z["X"].astype(np.float64)
        X_phys = z["X_phys"].astype(np.float64)
        X_pattern = z["X_pattern"].astype(np.float64)
        y = z["y"].astype(np.uint8)
        a_train, a_test = z["a_train"].astype(int), z["a_test"].astype(int)
        b_test, b_pool = z["b_test"].astype(int), z["b_pool"].astype(int)
        meta = json.loads(str(z["meta_json"].item()))
        pattern_names = z["pattern_names"].astype(str).tolist()

    if meta.get("feature_window_iculos") != [1, 24] or meta.get("outcome_window_iculos") != [25, 72]:
        raise ValueError("Cache window metadata does not match E6; rebuild the cache")
    X_train, y_train = X[a_train], y[a_train]
    print(f"Fitting source logistic models: A train n={len(a_train):,}, features={X.shape[1]}", flush=True)
    beta, selected_penalty, cv = fit_with_cv(X_train, y_train, seed=60101)
    beta_phys, phys_penalty, cv_phys = fit_with_cv(X_phys[a_train], y_train, seed=60102)
    beta_pattern, pattern_penalty, cv_pattern = fit_with_cv(X_pattern[a_train], y_train, seed=60103)
    print(f"Selected L2 penalties: combined={selected_penalty:g}, physiology={phys_penalty:g}, pattern={pattern_penalty:g}", flush=True)

    n_phys = X_phys.shape[1]
    x_a_test, x_b_test = X[a_test], X[b_test]
    phys_a_test, phys_b_test = X_phys[a_test], X_phys[b_test]
    pattern_a_test, pattern_b_test = X_pattern[a_test], X_pattern[b_test]
    b0, bphys, bpattern = beta[0], beta[1:1+n_phys], beta[1+n_phys:]
    base_a = b0 + x_a_test[:, :n_phys] @ bphys
    base_b = b0 + x_b_test[:, :n_phys] @ bphys
    pattern_part_a = x_a_test[:, n_phys:] @ bpattern
    pattern_part_b = x_b_test[:, n_phys:] @ bpattern
    logits_a = np.column_stack([base_a + w * pattern_part_a for w in W_GRID])
    logits_b = np.column_stack([base_b + w * pattern_part_b for w in W_GRID])
    probs_a, probs_b = sigmoid(logits_a), sigmoid(logits_b)

    # Paired patient bootstrap intervals for the B curve, with shared resamples across w.
    print(f"Bootstrapping B-test w curve ({BOOTSTRAP_N} paired resamples)...", flush=True)
    boot = bootstrap_curve(y[b_test], probs_b, BOOTSTRAP_SEED)
    w_rows = []
    for j, w in enumerate(W_GRID):
        ma = point_metrics(y[a_test], probs_a[:, j])
        mb = point_metrics(y[b_test], probs_b[:, j])
        w_rows.append({
            "model": "combined_w", "w": float(w), "penalty": selected_penalty,
            "auroc_A_test": ma["auroc"], "brier_A_test": ma["brier"], "log_loss_A_test": ma["log_loss"],
            "auroc_B": mb["auroc"], "auroc_B_low": boot["auroc"][0, j], "auroc_B_high": boot["auroc"][1, j],
            "brier_B": mb["brier"], "brier_B_low": boot["brier"][0, j], "brier_B_high": boot["brier"][1, j],
            "log_loss_B": mb["log_loss"], "log_loss_B_low": boot["log_loss"][0, j], "log_loss_B_high": boot["log_loss"][1, j],
        })
    ref_preds = {
        "physiology_only": (sigmoid(np.column_stack((np.ones(len(a_test)), phys_a_test)) @ beta_phys),
                            sigmoid(np.column_stack((np.ones(len(b_test)), phys_b_test)) @ beta_phys), phys_penalty),
        "pattern_only": (sigmoid(np.column_stack((np.ones(len(a_test)), pattern_a_test)) @ beta_pattern),
                         sigmoid(np.column_stack((np.ones(len(b_test)), pattern_b_test)) @ beta_pattern), pattern_penalty),
    }
    for name, (pa, pb, penalty) in ref_preds.items():
        ma, mb = point_metrics(y[a_test], pa), point_metrics(y[b_test], pb)
        w_rows.append({
            "model": name, "w": np.nan, "penalty": penalty,
            "auroc_A_test": ma["auroc"], "brier_A_test": ma["brier"], "log_loss_A_test": ma["log_loss"],
            "auroc_B": mb["auroc"], "auroc_B_low": np.nan, "auroc_B_high": np.nan,
            "brier_B": mb["brier"], "brier_B_low": np.nan, "brier_B_high": np.nan,
            "log_loss_B": mb["log_loss"], "log_loss_B_low": np.nan, "log_loss_B_high": np.nan,
        })
    w_df = pd.DataFrame(w_rows)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    w_df.to_csv(ANALYSIS / "e6_wcurve.csv", index=False)

    # In-sample vs held-out A diagnostic; the training AUROC is intentionally apparent performance.
    train_logits = np.column_stack((np.ones(len(a_train)), X[a_train])) @ beta
    train_auc = auc_from_ranks(y[a_train], train_logits)
    a_test_w1_auc = point_metrics(y[a_test], probs_a[:, int(np.argmin(np.abs(W_GRID - 1.0)))])["auroc"]
    print(f"A AUROC diagnostic: apparent train={train_auc:.4f}; held-out A test (w=1)={a_test_w1_auc:.4f}", flush=True)
    print("Leakage check: cache declares features ICULOS 1..24 and labels 25..72; feature extraction masks rows to 1..24.", flush=True)

    # Run-in choices on a single fixed B test cohort.
    x_pool = X[b_pool]
    pool_base = b0 + x_pool[:, :n_phys] @ bphys
    pool_pattern = x_pool[:, n_phys:] @ bpattern
    pool_logits = np.column_stack([pool_base + w * pool_pattern for w in W_GRID])
    w1_refit = np.column_stack((np.ones(len(b_test)), x_b_test)) @ beta
    w0_fixed = logits_b[:, 0]
    # (kept as explicit arrays for direct comparison/diagnostics)
    assert np.allclose(w1_refit, logits_b[:, int(np.argmin(np.abs(W_GRID - 1.0)))], atol=1e-8)

    runin_summary, runin_raw = summarize_runin(
        X, y, b_pool, b_test, pool_logits, logits_b, x_b_test, y[b_test], beta, RUNIN_SEED
    )
    runin_summary.to_csv(ANALYSIS / "e6_runin.csv", index=False)

    # Per-variable ever/recent measurement rates and pattern-only discrimination by hospital.
    v = len(pattern_names) // 2
    pmean = np.asarray(meta["pattern_mean"], dtype=np.float64)
    pscale = np.asarray(meta["pattern_scale"], dtype=np.float64)
    pattern_unscaled = X_pattern * pscale + pmean
    hosp_a = np.arange(len(y)) < int(meta["hospital_sizes"]["A"])
    hosp_b = ~hosp_a
    desc = []
    for j in range(v):
        desc.append({
            "variable": pattern_names[j].removeprefix("recent_"),
            "measurement_rate_17_24_A": float(pattern_unscaled[hosp_a, j].mean()),
            "measurement_rate_17_24_B": float(pattern_unscaled[hosp_b, j].mean()),
            "measurement_rate_1_24_A": float(pattern_unscaled[hosp_a, v+j].mean()),
            "measurement_rate_1_24_B": float(pattern_unscaled[hosp_b, v+j].mean()),
            "pattern_only_auroc_A": np.nan,
            "pattern_only_auroc_B": np.nan,
        })
    p_auc_a = auc_from_ranks(y[a_test], logit(ref_preds["pattern_only"][0]))
    beta_pattern_b, pattern_penalty_b, cv_pattern_b = fit_with_cv(X_pattern[b_pool], y[b_pool], seed=60104)
    p_auc_b = auc_from_ranks(y[b_test], np.column_stack((np.ones(len(b_test)), X_pattern[b_test])) @ beta_pattern_b)
    desc.append({
        "variable": "ALL_PATTERN_FEATURES",
        "measurement_rate_17_24_A": np.nan, "measurement_rate_17_24_B": np.nan,
        "measurement_rate_1_24_A": np.nan, "measurement_rate_1_24_B": np.nan,
        "pattern_only_auroc_A": p_auc_a, "pattern_only_auroc_B": p_auc_b,
    })
    desc_df = pd.DataFrame(desc)
    desc_df.to_csv(ANALYSIS / "e6_descriptives.csv", index=False)

    # Compact machine-readable headline summary.
    selected_w = w_df[(w_df["model"] == "combined_w")].set_index("w")
    headline_w = {}
    for w in (0.0, 0.5, 1.0):
        row = selected_w.loc[w]
        headline_w[str(w)] = {k: float(row[k]) for k in (
            "auroc_A_test", "brier_A_test", "log_loss_A_test", "auroc_B", "auroc_B_low",
            "auroc_B_high", "brier_B", "brier_B_low", "brier_B_high", "log_loss_B",
            "log_loss_B_low", "log_loss_B_high",
        )}
    best_row = w_df[w_df["model"] == "combined_w"].sort_values("log_loss_B").iloc[0]
    runin_points = {}
    for n_run in (250, 1000):
        sub = runin_summary[runin_summary["n_run"] == n_run]
        runin_points[str(n_run)] = sub.set_index("method").to_dict(orient="index")
    summary = {
        "cohort": {
            "hospital_sizes": meta["hospital_sizes"], "hospital_outcome_rates": meta["hospital_outcome_rates"],
            "split_sizes": meta["split_sizes"], "split_outcome_rates": meta["split_outcome_rates"],
            "exclusions": meta["exclusions"],
        },
        "seeds": {"split": meta["split_seed"], "bootstrap": BOOTSTRAP_SEED, "runin": RUNIN_SEED},
        "penalty_grid_average_bce": PENALTY_GRID.tolist(),
        "selected_penalties": {"combined": selected_penalty, "physiology": phys_penalty, "pattern": pattern_penalty},
        "pattern_only_within_B_penalty": pattern_penalty_b,
        "cv_log_loss": {
            "combined": {str(k): v for k, v in cv.items()},
            "physiology": {str(k): v for k, v in cv_phys.items()},
            "pattern": {str(k): v for k, v in cv_pattern.items()},
        },
        "w_curve_w0_0.5_1": headline_w,
        "best_w_on_B_by_log_loss": {"w": float(best_row["w"]), "log_loss": float(best_row["log_loss_B"]),
                                    "auroc": float(best_row["auroc_B"]), "brier": float(best_row["brier_B"])},
        "runin_n250_1000": runin_points,
        "pattern_only_auroc": {"A_test": p_auc_a, "B_test": p_auc_b},
        "A_auroc_apparent_train_w1": train_auc,
        "A_auroc_heldout_test_w1": a_test_w1_auc,
        "features_use_iculos_1_24_only": True,
        "outcomes_use_iculos_25_72_only": True,
        "preprocessing_limitation": "EtCO2 had no measured values in A train; its undefined A median was set to 0.",
        "elapsed_analysis_seconds": float(time.perf_counter() - start),
        "cohort_build_seconds": meta.get("build_seconds"),
        "requested_runin_sizes": [100, 250, 500, 1000, 2500, 5000, "full_pool"],
        "available_runin_sizes": sorted(set(runin_summary["n_run"].astype(int).tolist())),
    }
    with (ANALYSIS / "e6_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)
    print(f"Best B log-loss dial: w={best_row['w']:.1f}, log loss={best_row['log_loss_B']:.4f}", flush=True)
    print(f"Pattern-only AUROC: A={p_auc_a:.4f}, B={p_auc_b:.4f}", flush=True)
    print(f"E6 analysis runtime: {summary['elapsed_analysis_seconds']/60:.1f} min", flush=True)
    print(f"Wrote {ANALYSIS / 'e6_wcurve.csv'}, {ANALYSIS / 'e6_runin.csv'}, {ANALYSIS / 'e6_descriptives.csv'}, {ANALYSIS / 'e6_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
