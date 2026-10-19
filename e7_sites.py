"""Build E7 PhysioNet site cohorts with raw (unimputed, unscaled) features."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
P19_RAW = ROOT / "data" / "raw_physionet2019"
P12_RAW = ROOT / "data" / "raw_physionet2012"
CACHE = ROOT / "data" / "e7_cache.npz"

P19_VITAL_LABS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]
P19_DEMOGRAPHICS = ["Age", "Gender", "HospAdmTime"]
P19_COLUMNS = P19_VITAL_LABS + P19_DEMOGRAPHICS + ["ICULOS", "SepsisLabel", "Unit1", "Unit2"]
P12_STATIC = ["RecordID", "Age", "Gender", "Height", "ICUType", "Weight"]
P12_SITE_NAMES = {1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"}
P19_SITE_NAMES = [("A-MICU", "A", "MICU"), ("A-SICU", "A", "SICU"),
                  ("B-MICU", "B", "MICU"), ("B-SICU", "B", "SICU")]
CACHE_VERSION = 1


def _last_finite(values: np.ndarray, rows: np.ndarray) -> float:
    v = values[rows]
    good = np.flatnonzero(np.isfinite(v))
    return float(v[good[-1]]) if len(good) else np.nan


def build_p19() -> dict:
    started = time.perf_counter()
    if not (P19_RAW / "training_setA").is_dir() or not (P19_RAW / "training_setB").is_dir():
        raise FileNotFoundError(f"Expected P19 training_setA and training_setB below {P19_RAW}")
    groups = {name: {"phys": [], "pattern": [], "y": [], "ids": []} for name, _, _ in P19_SITE_NAMES}
    exclusions = {"missing_hours_1_24": 0, "early_sepsis": 0, "neither_unit": 0,
                  "ambiguous_both_units": 0}
    file_counts = {"A": 0, "B": 0}
    for hospital, folder_name in (("A", "training_setA"), ("B", "training_setB")):
        paths = sorted((P19_RAW / folder_name).glob("*.psv"))
        file_counts[hospital] = len(paths)
        for i, path in enumerate(paths, 1):
            df = pd.read_csv(path, sep="|", usecols=P19_COLUMNS, dtype=np.float32, na_values=["NaN"])
            icu = df["ICULOS"].to_numpy(dtype=np.float32, copy=False)
            labels = df["SepsisLabel"].to_numpy(dtype=np.float32, copy=False)
            feature_rows = np.isfinite(icu) & (icu >= 1) & (icu <= 24)
            hours = icu[feature_rows].astype(np.int16)
            if not np.all(np.isin(np.arange(1, 25), hours)):
                exclusions["missing_hours_1_24"] += 1
                continue
            if np.any(labels[feature_rows] == 1):
                exclusions["early_sepsis"] += 1
                continue

            # Unit indicators are static in practice; use the first row in the
            # feature window where at least one indicator is recorded.
            u1 = df["Unit1"].to_numpy(dtype=np.float32, copy=False)
            u2 = df["Unit2"].to_numpy(dtype=np.float32, copy=False)
            unit_rows = np.flatnonzero(feature_rows & (np.isfinite(u1) | np.isfinite(u2)))
            if not len(unit_rows):
                exclusions["neither_unit"] += 1
                continue
            row = int(unit_rows[0])
            one = np.isfinite(u1[row]) and u1[row] == 1
            two = np.isfinite(u2[row]) and u2[row] == 1
            if not one and not two:
                exclusions["neither_unit"] += 1
                continue
            if one and two:
                exclusions["ambiguous_both_units"] += 1
            unit = "MICU" if one else "SICU"
            site = f"{hospital}-{unit}"

            physiology = np.full(len(P19_VITAL_LABS) + len(P19_DEMOGRAPHICS), np.nan, dtype=np.float64)
            observed = np.zeros((int(feature_rows.sum()), len(P19_VITAL_LABS)), dtype=bool)
            for j, col in enumerate(P19_VITAL_LABS):
                vals = df[col].to_numpy(dtype=np.float32, copy=False)
                physiology[j] = _last_finite(vals, feature_rows)
                observed[:, j] = np.isfinite(vals[feature_rows])
            for j, col in enumerate(P19_DEMOGRAPHICS):
                vals = df[col].to_numpy(dtype=np.float32, copy=False)
                physiology[len(P19_VITAL_LABS) + j] = _last_finite(vals, feature_rows)
            recent_rows = icu[feature_rows] >= 17
            pattern = np.concatenate((
                observed[recent_rows].any(axis=0).astype(np.float64),
                observed.any(axis=0).astype(np.float64),
                [np.log1p(int(observed.sum()))],
            ))
            outcome_rows = np.isfinite(icu) & (icu >= 25) & (icu <= 72)
            y = int(np.any(labels[outcome_rows] == 1))
            groups[site]["phys"].append(physiology)
            groups[site]["pattern"].append(pattern)
            groups[site]["y"].append(y)
            groups[site]["ids"].append(f"{hospital}:{path.stem}")
            if i % 5000 == 0:
                print(f"P19 {hospital}: read {i:,}/{len(paths):,} files", flush=True)

    out = {}
    for name, values in groups.items():
        out[name] = {
            "raw_phys": np.asarray(values["phys"], dtype=np.float64).reshape((-1, len(P19_VITAL_LABS) + len(P19_DEMOGRAPHICS))),
            "raw_pattern": np.asarray(values["pattern"], dtype=np.float64).reshape((-1, 2 * len(P19_VITAL_LABS) + 1)),
            "y": np.asarray(values["y"], dtype=np.uint8),
            "patient_ids": np.asarray(values["ids"], dtype="U40"),
        }
    info = {
        "file_counts": file_counts,
        "exclusions": exclusions,
        "feature_window_iculos": [1, 24],
        "outcome_window_iculos": [25, 72],
        "features_use_iculos_1_24_only": True,
        "phys_names": P19_VITAL_LABS + [f"{v}^2" for v in P19_VITAL_LABS] + P19_DEMOGRAPHICS,
        "raw_phys_names": P19_VITAL_LABS + P19_DEMOGRAPHICS,
        "pattern_names": ([f"recent_{v}" for v in P19_VITAL_LABS]
                           + [f"ever_{v}" for v in P19_VITAL_LABS]
                           + ["log1p_total_measurements_1_24"]),
        "build_seconds": time.perf_counter() - started,
    }
    return out, info


def _parse_time(value: str) -> float:
    parts = str(value).split(":")
    if len(parts) != 2:
        return np.nan
    try:
        return float(parts[0]) + float(parts[1]) / 60.0
    except ValueError:
        return np.nan


def build_p12() -> dict:
    started = time.perf_counter()
    if not P12_RAW.is_dir():
        raise FileNotFoundError(f"Expected P12 directory {P12_RAW}")
    records = []
    dynamic_vars = set()
    problems = {"missing_outcome": 0, "missing_icu_type": 0, "invalid_icu_type": 0,
                "missing_static_age": 0, "missing_static_gender": 0,
                "missing_static_height": 0, "missing_static_weight": 0}
    for set_letter in ("a", "b", "c"):
        folder = P12_RAW / f"set-{set_letter}"
        outcome_path = P12_RAW / f"Outcomes-{set_letter}.txt"
        if not folder.is_dir() or not outcome_path.is_file():
            raise FileNotFoundError(f"Missing P12 input for set {set_letter}")
        outcome_df = pd.read_csv(outcome_path)
        outcome_map = dict(zip(outcome_df["RecordID"].astype(int), outcome_df["In-hospital_death"].astype(int)))
        paths = sorted(folder.glob("*.txt"))
        for i, path in enumerate(paths, 1):
            record_id = int(path.stem)
            if record_id not in outcome_map:
                problems["missing_outcome"] += 1
                continue
            df = pd.read_csv(path, dtype={"Time": str, "Parameter": str})
            df["hour"] = df["Time"].map(_parse_time)
            # The feature dictionary is the cohort-wide set of time-varying
            # variables; values and measurement patterns are still windowed.
            dynamic_vars.update(
                df.loc[~df["Parameter"].isin(P12_STATIC), "Parameter"].dropna().astype(str).tolist()
            )
            static = {}
            for key in ("Age", "Gender", "Height", "ICUType", "Weight"):
                vals = pd.to_numeric(df.loc[(df["Time"] == "00:00") & (df["Parameter"] == key), "Value"], errors="coerce")
                val = float(vals.iloc[0]) if len(vals) else np.nan
                if key != "ICUType" and val == -1:
                    val = np.nan
                static[key] = val
            icu_value = static["ICUType"]
            if not np.isfinite(icu_value):
                problems["missing_icu_type"] += 1
                continue
            site_type = int(icu_value)
            if site_type not in P12_SITE_NAMES:
                problems["invalid_icu_type"] += 1
                continue
            for key in ("Age", "Gender", "Height", "Weight"):
                if not np.isfinite(static[key]):
                    problems[f"missing_static_{key.lower()}"] += 1
            within = np.isfinite(df["hour"].to_numpy(dtype=np.float64)) & (df["hour"].to_numpy(dtype=np.float64) >= 0) & (df["hour"].to_numpy(dtype=np.float64) <= 48)
            dyn_df = df.loc[within & ~df["Parameter"].isin(P12_STATIC)]
            by_var: dict[str, tuple[float, bool, bool, float]] = {}
            for parameter, part in dyn_df.groupby("Parameter", sort=False):
                name = str(parameter)
                if name in P12_STATIC:
                    continue
                hours = part["hour"].to_numpy(dtype=np.float64)
                values = pd.to_numeric(part["Value"], errors="coerce").to_numpy(dtype=np.float64)
                finite = np.isfinite(values)
                valid = np.flatnonzero(finite)
                if len(valid):
                    latest_hour = np.max(hours[valid])
                    latest_rows = valid[hours[valid] == latest_hour]
                    last = float(values[latest_rows[-1]])
                else:
                    last = np.nan
                ever = len(part) > 0
                recent = bool(np.any((hours >= 40.0) & (hours <= 48.0)))
                count = float(len(part))
                by_var[name] = (last, recent, ever, count)
            records.append({
                "set": set_letter,
                "record_id": record_id,
                "patient_id": f"{set_letter}:{record_id}",
                "site": P12_SITE_NAMES[site_type],
                "static": static,
                "dynamic": by_var,
                "y": int(outcome_map[record_id]),
            })
            if i % 1000 == 0:
                print(f"P12 set-{set_letter}: read {i:,}/{len(paths):,} stays", flush=True)

    variables = sorted(dynamic_vars)
    sites = sorted(P12_SITE_NAMES.values())
    output = {}
    phys_names = variables + [f"{v}^2" for v in variables] + ["Age", "Gender", "Height", "Weight"]
    pattern_names = ([f"recent_{v}" for v in variables] + [f"ever_{v}" for v in variables]
                     + ["log1p_total_measurements_0_48"])
    for site in sites:
        selected = [r for r in records if r["site"] == site]
        n, nv = len(selected), len(variables)
        raw_phys = np.full((n, nv + 4), np.nan, dtype=np.float64)
        raw_pattern = np.zeros((n, 2 * nv + 1), dtype=np.float64)
        y = np.empty(n, dtype=np.uint8)
        patient_ids = []
        for i, record in enumerate(selected):
            counts = 0.0
            for j, name in enumerate(variables):
                last, recent, ever, count = record["dynamic"].get(name, (np.nan, False, False, 0.0))
                raw_phys[i, j] = last
                raw_pattern[i, j] = float(recent)
                raw_pattern[i, nv + j] = float(ever)
                counts += count
            raw_pattern[i, -1] = np.log1p(counts)
            for j, key in enumerate(("Age", "Gender", "Height", "Weight")):
                raw_phys[i, nv + j] = record["static"][key]
            y[i] = record["y"]
            patient_ids.append(record["patient_id"])
        output[site] = {"raw_phys": raw_phys, "raw_pattern": raw_pattern, "y": y,
                        "patient_ids": np.asarray(patient_ids, dtype="U24")}
    info = {
        "file_counts": {letter: sum(1 for _ in (P12_RAW / f"set-{letter}").glob("*.txt")) for letter in "abc"},
        "included_records": len(records),
        "exclusions_and_missing_static": problems,
        "feature_window_hours": [0, 48],
        "recent_pattern_window_hours": [40, 48],
        "features_use_hours_0_48_only": True,
        "phys_names": phys_names,
        "raw_phys_names": variables + ["Age", "Gender", "Height", "Weight"],
        "pattern_names": pattern_names,
        "time_varying_variables": variables,
        "weight_static_at_00_00_only": True,
        "build_seconds": time.perf_counter() - started,
    }
    return output, info


def _pack_dataset(arrays: dict, names: list[str], dataset: str, payload: dict) -> None:
    for site, values in arrays.items():
        prefix = f"{dataset}_{site}"
        payload[f"{prefix}_raw_phys"] = values["raw_phys"].astype(np.float32)
        payload[f"{prefix}_raw_pattern"] = values["raw_pattern"].astype(np.float32)
        payload[f"{prefix}_y"] = values["y"].astype(np.uint8)
        payload[f"{prefix}_patient_ids"] = values["patient_ids"]
    payload[f"{dataset}_sites"] = np.asarray(list(arrays), dtype="U16")


def main() -> None:
    started = time.perf_counter()
    p19, p19_info = build_p19()
    p12, p12_info = build_p12()
    payload = {}
    _pack_dataset(p19, list(p19), "P19", payload)
    _pack_dataset(p12, list(p12), "P12", payload)
    metadata = {"cache_version": CACHE_VERSION, "P19": p19_info, "P12": p12_info,
                "cache_built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_build_seconds": time.perf_counter() - started}
    payload["P19_phys_names"] = np.asarray(p19_info["phys_names"], dtype="U64")
    payload["P19_raw_phys_names"] = np.asarray(p19_info["raw_phys_names"], dtype="U64")
    payload["P19_pattern_names"] = np.asarray(p19_info["pattern_names"], dtype="U64")
    payload["P12_phys_names"] = np.asarray(p12_info["phys_names"], dtype="U64")
    payload["P12_raw_phys_names"] = np.asarray(p12_info["raw_phys_names"], dtype="U96")
    payload["P12_pattern_names"] = np.asarray(p12_info["pattern_names"], dtype="U96")
    payload["meta_json"] = np.asarray(json.dumps(metadata, separators=(",", ":")), dtype="U")
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **payload)
    print("E7 cohorts (n, outcome cases)", flush=True)
    for ds, arrays in (("P19", p19), ("P12", p12)):
        for site, values in arrays.items():
            print(f"{ds} {site}: n={len(values['y']):,}, cases={int(values['y'].sum()):,}", flush=True)
    print(f"P19 exclusions: {p19_info['exclusions']}", flush=True)
    print(f"P12 data checks: {p12_info['exclusions_and_missing_static']}", flush=True)
    print(f"P12 time-varying variables={len(p12_info['time_varying_variables'])}", flush=True)
    print(f"Cached raw feature matrices: {CACHE} ({CACHE.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    print(f"E7 cohort build runtime: {metadata['total_build_seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
