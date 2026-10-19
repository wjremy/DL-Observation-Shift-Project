"""Build and cache the E6 PhysioNet two-hospital cohort and feature matrices."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw_physionet2019"
CACHE = ROOT / "data" / "e6_cache.npz"
VITAL_LABS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
DEMOGRAPHICS = ["Age", "Gender", "HospAdmTime"]
RAW_COLUMNS = VITAL_LABS + DEMOGRAPHICS + ["ICULOS", "SepsisLabel"]
SPLIT_SEED = 20261019
CACHE_VERSION = 1


def last_finite(values: np.ndarray, rows: np.ndarray) -> float:
    v = values[rows]
    good = np.flatnonzero(np.isfinite(v))
    return float(v[good[-1]]) if len(good) else np.nan


def read_hospital(folder: Path, hospital: str):
    paths = sorted(folder.glob("*.psv"))
    raw_phys, pattern, outcomes, patient_ids = [], [], [], []
    exclusions = {"missing_hours_1_24": 0, "early_sepsis": 0}
    start = time.perf_counter()
    n_v = len(VITAL_LABS)
    for i, path in enumerate(paths, 1):
        df = pd.read_csv(path, sep="|", usecols=RAW_COLUMNS, dtype=np.float32, na_values=["NaN"])
        icu = df["ICULOS"].to_numpy(dtype=np.float32, copy=False)
        labels = df["SepsisLabel"].to_numpy(dtype=np.float32, copy=False)
        in_window = np.isfinite(icu) & (icu >= 1) & (icu <= 24)
        hours = icu[in_window].astype(np.int16)
        if not np.all(np.isin(np.arange(1, 25), hours)):
            exclusions["missing_hours_1_24"] += 1
            continue
        if np.any(labels[in_window] == 1):
            exclusions["early_sepsis"] += 1
            continue

        physiology = np.full(n_v + len(DEMOGRAPHICS), np.nan, dtype=np.float64)
        observed = np.zeros((int(in_window.sum()), n_v), dtype=bool)
        for j, col in enumerate(VITAL_LABS):
            vals = df[col].to_numpy(dtype=np.float32, copy=False)
            physiology[j] = last_finite(vals, in_window)
            observed[:, j] = np.isfinite(vals[in_window])
        for j, col in enumerate(DEMOGRAPHICS):
            physiology[n_v + j] = last_finite(df[col].to_numpy(dtype=np.float32, copy=False), in_window)
        recent_rows = icu[in_window] >= 17
        pattern.append(np.concatenate((
            observed[recent_rows].any(axis=0).astype(np.float64),
            observed.any(axis=0).astype(np.float64),
            [np.log1p(int(observed.sum()))],
        )))

        # Hours after 24 contribute only to the outcome, never to features.
        outcome_window = np.isfinite(icu) & (icu >= 25) & (icu <= 72)
        outcomes.append(int(np.any(labels[outcome_window] == 1)))
        raw_phys.append(physiology)
        patient_ids.append(f"{hospital}:{path.stem}")
        if i % 5000 == 0:
            print(f"{hospital}: read {i:,}/{len(paths):,} patients ({time.perf_counter()-start:.1f}s)", flush=True)
    print(
        f"{hospital}: files={len(paths):,}, eligible={len(outcomes):,}, "
        f"excluded={exclusions}, read_time={time.perf_counter()-start:.1f}s",
        flush=True,
    )
    return np.asarray(raw_phys), np.asarray(pattern), np.asarray(outcomes, dtype=np.uint8), patient_ids, exclusions


def standardize(x: np.ndarray, train_idx: np.ndarray):
    mean = np.mean(x[train_idx], axis=0)
    scale = np.std(x[train_idx], axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return ((x - mean) / scale).astype(np.float32), mean, scale


def report(meta: dict) -> None:
    print("E6 eligible cohort (24-hour landmark; early sepsis excluded)", flush=True)
    for h in ("A", "B"):
        print(f"Hospital {h}: n={meta['hospital_sizes'][h]:,}, outcome rate={meta['hospital_outcome_rates'][h]:.4f}", flush=True)
    print("Fixed splits:", json.dumps(meta["split_sizes"], sort_keys=True), flush=True)
    print("Split outcome rates:", json.dumps(meta["split_outcome_rates"], sort_keys=True), flush=True)


def build_cache() -> None:
    build_started = time.perf_counter()
    if not (RAW / "training_setA").is_dir() or not (RAW / "training_setB").is_dir():
        raise FileNotFoundError(f"Expected both hospital folders below {RAW}")
    results = {}
    for h, folder in (("A", "training_setA"), ("B", "training_setB")):
        results[h] = read_hospital(RAW / folder, h)
    raw = np.vstack((results["A"][0], results["B"][0]))
    pattern_raw = np.vstack((results["A"][1], results["B"][1]))
    y = np.concatenate((results["A"][2], results["B"][2]))
    all_ids = results["A"][3] + results["B"][3]
    ranges = {"A": (0, len(results["A"][2])), "B": (len(results["A"][2]), len(y))}
    a0, a1 = ranges["A"]
    b0, b1 = ranges["B"]

    rng = np.random.default_rng(SPLIT_SEED)
    a_perm = rng.permutation(np.arange(a0, a1))
    n_a_train = int(np.floor(0.8 * len(a_perm)))
    a_train, a_test = a_perm[:n_a_train], a_perm[n_a_train:]
    b_perm = rng.permutation(np.arange(b0, b1))
    if len(b_perm) < 13000:
        raise ValueError(f"Eligible B cohort too small for 8,000 test and 5,000 run-in: {len(b_perm):,}")
    b_test, b_pool = b_perm[:8000], b_perm[8000:]

    # A training-part medians fill never-observed values, including demographics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(raw[a_train], axis=0)
    raw_names = VITAL_LABS + DEMOGRAPHICS
    all_missing_in_a_train = [
        name for j, name in enumerate(raw_names)
        if not np.isfinite(raw[a_train, j]).any()
    ]
    medians[~np.isfinite(medians)] = 0.0
    if all_missing_in_a_train:
        print(f"No A-training measurements for {all_missing_in_a_train}; using 0 fallback for undefined median.", flush=True)
    imputed = np.where(np.isfinite(raw), raw, medians[None, :])
    n_v = len(VITAL_LABS)
    phys_raw = np.column_stack((imputed[:, :n_v], imputed[:, :n_v] ** 2, imputed[:, n_v:]))
    phys_names = VITAL_LABS + [f"{name}^2" for name in VITAL_LABS] + DEMOGRAPHICS
    pattern_names = (
        [f"recent_{name}" for name in VITAL_LABS]
        + [f"ever_{name}" for name in VITAL_LABS]
        + ["log1p_total_measurements_1_24"]
    )
    x_phys, phys_mean, phys_scale = standardize(phys_raw, a_train)
    x_pattern, pattern_mean, pattern_scale = standardize(pattern_raw, a_train)
    x_all = np.column_stack((x_phys, x_pattern)).astype(np.float32)

    sizes = {"A": int(a1 - a0), "B": int(b1 - b0)}
    rates = {"A": float(y[a0:a1].mean()), "B": float(y[b0:b1].mean())}
    split_sizes = {"A_train": len(a_train), "A_test": len(a_test), "B_test": len(b_test), "B_runin_pool": len(b_pool)}
    split_rates = {
        "A_train": float(y[a_train].mean()), "A_test": float(y[a_test].mean()),
        "B_test": float(y[b_test].mean()), "B_runin_pool": float(y[b_pool].mean()),
    }
    meta = {
        "cache_version": CACHE_VERSION,
        "split_seed": SPLIT_SEED,
        "feature_window_iculos": [1, 24],
        "outcome_window_iculos": [25, 72],
        "hospital_sizes": sizes,
        "hospital_outcome_rates": rates,
        "split_sizes": split_sizes,
        "split_outcome_rates": split_rates,
        "exclusions": {h: results[h][4] for h in ("A", "B")},
        "all_missing_in_A_train": all_missing_in_a_train,
        "impute_medians": medians.tolist(),
        "phys_mean": phys_mean.tolist(), "phys_scale": phys_scale.tolist(),
        "pattern_mean": pattern_mean.tolist(), "pattern_scale": pattern_scale.tolist(),
        "all_ids": all_ids,
        "build_seconds": float(time.perf_counter() - build_started),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, X=x_all, X_phys=x_phys, X_pattern=x_pattern, y=y,
        a_train=a_train.astype(np.int32), a_test=a_test.astype(np.int32),
        b_test=b_test.astype(np.int32), b_pool=b_pool.astype(np.int32),
        hosp=np.r_[np.zeros(a1-a0, dtype=np.uint8), np.ones(b1-b0, dtype=np.uint8)],
        phys_names=np.asarray(phys_names, dtype="U64"),
        pattern_names=np.asarray(pattern_names, dtype="U64"),
        meta_json=np.asarray(json.dumps(meta, separators=(",", ":")), dtype="U"),
    )
    report(meta)
    print(f"Cached matrices: {CACHE} ({CACHE.stat().st_size / 1024**2:.1f} MiB)", flush=True)


def main() -> None:
    if CACHE.exists():
        with np.load(CACHE, allow_pickle=False) as z:
            meta = json.loads(str(z["meta_json"].item()))
        if meta.get("cache_version") == CACHE_VERSION:
            report(meta)
            print(f"Using existing cache: {CACHE}", flush=True)
            return
    build_cache()


if __name__ == "__main__":
    main()
