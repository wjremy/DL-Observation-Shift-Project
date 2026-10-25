"""Build the hourly, unscaled sequence cache for experiment E15.

The cohort and outcome labels are taken from ``data/e7_cache.npz``. Raw
records are read only for those cached patient ids, in the same order.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import e7_sites as e7


ROOT = Path(__file__).resolve().parent
E7_CACHE = ROOT / "data" / "e7_cache.npz"
E15_CACHE = ROOT / "data" / "e15_cache.npz"
SUMMARY_PATH = ROOT / "out" / "e15" / "e15_sequences_summary.json"
P19_T = 24
P12_T = 48


def _load_e7() -> dict[str, np.ndarray]:
    if not E7_CACHE.is_file():
        raise FileNotFoundError(f"Expected E7 cache at {E7_CACHE}")
    with np.load(E7_CACHE, allow_pickle=False) as cache:
        return {name: cache[name] for name in cache.files}


def _invalid_values(values: np.ndarray, variables: list[str], dataset: str) -> np.ndarray:
    """Return the finite cells rejected by the E15 physiological cleaning."""
    invalid = np.zeros(values.shape, dtype=bool)
    finite = np.isfinite(values)
    for j, name in enumerate(variables):
        if name.casefold() != "baseexcess":
            invalid[:, j] |= finite[:, j] & (values[:, j] < 0)
        if name.casefold() == "ph":
            invalid[:, j] |= finite[:, j] & ((values[:, j] < 6.5) | (values[:, j] > 8.0))
        if dataset == "P12":
            invalid[:, j] |= finite[:, j] & (values[:, j] == -1)
    return invalid


def _hourly_last_valid(hours: np.ndarray, values: np.ndarray, n_hours: int) -> tuple[np.ndarray, np.ndarray]:
    """For each hour-variable pair, retain its last finite value and mask."""
    n_variables = values.shape[1]
    x = np.full((n_hours, n_variables), np.nan, dtype=np.float32)
    m = np.zeros((n_hours, n_variables), dtype=np.int8)
    row, col = np.nonzero(np.isfinite(values))
    if len(row) == 0:
        return x, m
    flat_key = hours[row].astype(np.int64) * n_variables + col
    last_row = np.full(n_hours * n_variables, -1, dtype=np.int64)
    np.maximum.at(last_row, flat_key, row)
    chosen = np.flatnonzero(last_row >= 0)
    x.reshape(-1)[chosen] = values[last_row[chosen], chosen % n_variables].astype(np.float32)
    m.reshape(-1)[chosen] = 1
    return x, m


def _p19_unit(site: str, ids: np.ndarray, expected_y: np.ndarray,
              variables: list[str], mismatch_list: list[str], removed: dict[str, int],
              progress: dict[str, int]) -> dict[str, np.ndarray]:
    n, d = len(ids), len(variables)
    x = np.full((n, P19_T, d), np.nan, dtype=np.float32)
    m = np.zeros((n, P19_T, d), dtype=np.int8)
    y = np.empty(n, dtype=np.int8)
    for i, patient_id in enumerate(ids):
        hospital, stem = str(patient_id).split(":", 1)
        path = e7.P19_RAW / f"training_set{hospital}" / f"{stem}.psv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached P19 patient {patient_id}: {path}")
        df = pd.read_csv(path, sep="|", usecols=e7.P19_COLUMNS, dtype=np.float32,
                         na_values=["NaN"])
        icu = df["ICULOS"].to_numpy(dtype=np.float32, copy=False)
        labels = df["SepsisLabel"].to_numpy(dtype=np.float32, copy=False)
        outcome_rows = np.isfinite(icu) & (icu >= 25) & (icu <= 72)
        y[i] = int(np.any(labels[outcome_rows] == 1))
        if y[i] != int(expected_y[i]):
            mismatch_list.append(f"P19 {patient_id}: E7={int(expected_y[i])}, raw={int(y[i])}")

        in_window = np.isfinite(icu) & (icu >= 1) & (icu <= 24)
        hours = icu[in_window].astype(np.int16) - 1
        values = df.loc[in_window, variables].to_numpy(dtype=np.float64, copy=True)
        invalid = _invalid_values(values, variables, "P19")
        for j, name in enumerate(variables):
            removed[name] += int(invalid[:, j].sum())
        values[invalid] = np.nan
        x[i], m[i] = _hourly_last_valid(hours, values, P19_T)

        progress["count"] += 1
        if progress["count"] % 2000 == 0:
            print(f"P19: read {progress['count']:,}/{progress['total']:,} cached patient files", flush=True)
    return {"x": x, "m": m, "y": y, "ids": ids.copy()}


def _p12_outcomes() -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for letter in ("a", "b", "c"):
        outcome_path = e7.P12_RAW / f"Outcomes-{letter}.txt"
        outcomes = pd.read_csv(outcome_path)
        result[letter] = dict(zip(
            outcomes["RecordID"].astype(int), outcomes["In-hospital_death"].astype(int)
        ))
    return result


def _p12_site(site: str, ids: np.ndarray, expected_y: np.ndarray, variables: list[str],
              variable_index: dict[str, int], outcomes: dict[str, dict[int, int]],
              mismatch_list: list[str], removed: dict[str, int], observed_variables: set[str],
              progress: dict[str, int]) -> dict[str, np.ndarray]:
    n, d = len(ids), len(variables)
    x = np.full((n, P12_T, d), np.nan, dtype=np.float32)
    m = np.zeros((n, P12_T, d), dtype=np.int8)
    y = np.empty(n, dtype=np.int8)
    static = set(e7.P12_STATIC)

    for i, patient_id in enumerate(ids):
        letter, record_string = str(patient_id).split(":", 1)
        record_id = int(record_string)
        path = e7.P12_RAW / f"set-{letter}" / f"{record_id}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached P12 patient {patient_id}: {path}")
        df = pd.read_csv(path, dtype={"Time": str, "Parameter": str})
        params = df["Parameter"].to_numpy(dtype=object)
        dynamic = np.asarray([isinstance(p, str) and p not in static for p in params], dtype=bool)
        for name in pd.unique(params[dynamic]):
            if isinstance(name, str):
                observed_variables.add(name)
                if name not in variable_index:
                    raise ValueError(
                        f"P12 variable {name!r} in cached patient {patient_id} is absent from "
                        "E7's P12_raw_phys_names; cannot safely build the fixed-width cache"
                    )

        time_hours = df["Time"].map(e7._parse_time).to_numpy(dtype=np.float64)
        numeric_values = pd.to_numeric(df["Value"], errors="coerce").to_numpy(dtype=np.float64)
        within = dynamic & np.isfinite(time_hours) & (time_hours >= 0) & (time_hours <= 48)
        row_ids = np.flatnonzero(within)
        if len(row_ids):
            names = params[row_ids].astype(str)
            cols = np.fromiter((variable_index[name] for name in names), dtype=np.int32, count=len(names))
            values = numeric_values[row_ids]
            # Apply each variable's rules (base excess is the sole negative exception).
            invalid = np.zeros(len(values), dtype=bool)
            finite = np.isfinite(values)
            for j, name in enumerate(names):
                if name.casefold() != "baseexcess":
                    invalid[j] |= finite[j] and values[j] < 0
                if name.casefold() == "ph":
                    invalid[j] |= finite[j] and (values[j] < 6.5 or values[j] > 8.0)
                invalid[j] |= finite[j] and values[j] == -1
            for j, name in enumerate(names):
                if invalid[j]:
                    removed[name] += 1
            values[invalid] = np.nan
            hour_values = np.floor(time_hours[row_ids]).astype(np.int16)
            hour_values[hour_values == 48] = 47
            flat_key = hour_values.astype(np.int64) * d + cols
            valid = np.isfinite(values)
            if np.any(valid):
                valid_rows = np.flatnonzero(valid)
                last_row = np.full(P12_T * d, -1, dtype=np.int64)
                np.maximum.at(last_row, flat_key[valid_rows], valid_rows)
                chosen = np.flatnonzero(last_row >= 0)
                x[i].reshape(-1)[chosen] = values[last_row[chosen]].astype(np.float32)
                m[i].reshape(-1)[chosen] = 1

        if record_id not in outcomes.get(letter, {}):
            raise KeyError(f"No P12 outcome for cached patient {patient_id}")
        y[i] = outcomes[letter][record_id]
        if y[i] != int(expected_y[i]):
            mismatch_list.append(f"P12 {patient_id}: E7={int(expected_y[i])}, raw={int(y[i])}")

        progress["count"] += 1
        if progress["count"] % 2000 == 0:
            print(f"P12: read {progress['count']:,}/{progress['total']:,} cached patient files", flush=True)
    return {"x": x, "m": m, "y": y, "ids": ids.copy()}


def _pool_skewness(values: np.ndarray) -> float:
    if len(values) < 3:
        return float("nan")
    return float(pd.Series(values, copy=False).skew())


def _log_variables(dataset_sites: dict[str, dict[str, np.ndarray]], variables: list[str],
                   pooling_sites: list[str]) -> list[str]:
    log_names = []
    for j, name in enumerate(variables):
        chunks = [site_data["x"][:, :, j][site_data["m"][:, :, j] == 1]
                  for site_name in pooling_sites
                  for site_data in [dataset_sites[site_name]]]
        chunks = [chunk for chunk in chunks if len(chunk)]
        if not chunks:
            continue
        pooled = np.concatenate(chunks).astype(np.float64, copy=False)
        if np.all(pooled > 0) and _pool_skewness(pooled) > 2:
            log_names.append(name)
    return log_names


def _apply_logs(dataset_sites: dict[str, dict[str, np.ndarray]], variables: list[str],
                log_names: list[str]) -> None:
    for name in log_names:
        j = variables.index(name)
        for site_data in dataset_sites.values():
            mask = site_data["m"][:, :, j] == 1
            site_data["x"][:, :, j][mask] = np.log1p(site_data["x"][:, :, j][mask])


def _site_summary(site_data: dict[str, np.ndarray], variables: list[str]) -> dict:
    rates = site_data["m"].mean(axis=(0, 1), dtype=np.float64)
    order_high = sorted(range(len(variables)), key=lambda j: (-rates[j], variables[j]))[:10]
    order_low = sorted(range(len(variables)), key=lambda j: (rates[j], variables[j]))[:10]
    n = len(site_data["y"])
    return {
        "n": int(n),
        "outcome_rate": float(site_data["y"].mean()) if n else None,
        "mean_measurement_rate": float(site_data["m"].mean()) if n else None,
        "most_measured_variables": [
            {"variable": variables[j], "measurement_rate": float(rates[j])} for j in order_high
        ],
        "least_measured_variables": [
            {"variable": variables[j], "measurement_rate": float(rates[j])} for j in order_low
        ],
    }


def _verify_nan_masks(all_sites: dict[str, dict[str, np.ndarray]]) -> None:
    for key, site_data in all_sites.items():
        if not np.array_equal(np.isnan(site_data["x"]), site_data["m"] == 0):
            raise AssertionError(f"{key}: x is not NaN exactly where m == 0")
        if not np.all(np.isfinite(site_data["x"][site_data["m"] == 1])):
            raise AssertionError(f"{key}: non-finite value has m == 1")


def main() -> None:
    started = time.perf_counter()
    e7_cache = _load_e7()
    mismatches: list[str] = []
    removed_p19 = {name: 0 for name in e7.P19_VITAL_LABS}
    p19_ids = {
        site: e7_cache[f"P19_{site}_patient_ids"]
        for site, _, _ in e7.P19_SITE_NAMES
    }
    p19_units: dict[str, dict[str, np.ndarray]] = {}
    p19_count = sum(len(values) for values in p19_ids.values())
    p19_progress = {"count": 0, "total": p19_count}
    for site, _, _ in e7.P19_SITE_NAMES:
        p19_units[site] = _p19_unit(
            site, p19_ids[site], e7_cache[f"P19_{site}_y"], e7.P19_VITAL_LABS,
            mismatches, removed_p19, p19_progress,
        )

    # Site unions deliberately preserve the E15 specification's unit order.
    p19_sites: dict[str, dict[str, np.ndarray]] = dict(p19_units)
    for hospital in ("A", "B"):
        unit_sites = (f"{hospital}-MICU", f"{hospital}-SICU")
        p19_sites[f"{hospital}-all"] = {
            field: np.concatenate([p19_units[site][field] for site in unit_sites], axis=0)
            for field in ("x", "m", "y", "ids")
        }

    p12_candidate_names = [str(name) for name in e7_cache["P12_raw_phys_names"]
                           if str(name) not in e7.P12_STATIC]
    p12_variables = sorted(set(p12_candidate_names))
    p12_index = {name: j for j, name in enumerate(p12_variables)}
    observed_p12_variables: set[str] = set()
    removed_p12: dict[str, int] = {name: 0 for name in p12_variables}
    p12_ids = {site: e7_cache[f"P12_{site}_patient_ids"] for site in e7.P12_SITE_NAMES.values()}
    p12_outcomes = _p12_outcomes()
    p12_count = sum(len(values) for values in p12_ids.values())
    p12_progress = {"count": 0, "total": p12_count}
    p12_sites: dict[str, dict[str, np.ndarray]] = {}
    for site in e7.P12_SITE_NAMES.values():
        p12_sites[site] = _p12_site(
            site, p12_ids[site], e7_cache[f"P12_{site}_y"], p12_variables, p12_index,
            p12_outcomes, mismatches, removed_p12, observed_p12_variables, p12_progress,
        )

    if mismatches:
        for mismatch in mismatches:
            print(f"OUTCOME MISMATCH: {mismatch}", flush=True)
        raise AssertionError(f"Found {len(mismatches)} E7/raw outcome mismatches")
    if observed_p12_variables != set(p12_variables):
        absent = sorted(set(p12_variables) - observed_p12_variables)
        p12_variables = sorted(observed_p12_variables)
        keep = [p12_index[name] for name in p12_variables]
        for site_data in p12_sites.values():
            site_data["x"] = site_data["x"][:, :, keep]
            site_data["m"] = site_data["m"][:, :, keep]
        removed_p12 = {name: removed_p12[name] for name in p12_variables}
        print(f"P12 variables absent from the included E7 raw records (excluded from cache): {absent}")
    else:
        p12_variables = sorted(observed_p12_variables)

    p19_log = _log_variables(p19_sites, e7.P19_VITAL_LABS, list(p19_units))
    p12_log = _log_variables(p12_sites, p12_variables, list(p12_sites))
    _apply_logs(p19_sites, e7.P19_VITAL_LABS, p19_log)
    _apply_logs(p12_sites, p12_variables, p12_log)

    all_sites = {**{f"P19_{site}": data for site, data in p19_sites.items()},
                 **{f"P12_{site}": data for site, data in p12_sites.items()}}
    _verify_nan_masks(all_sites)

    payload: dict[str, np.ndarray] = {}
    for dataset, site_data, variables, log_names, n_hours in (
        ("P19", p19_sites, e7.P19_VITAL_LABS, p19_log, P19_T),
        ("P12", p12_sites, p12_variables, p12_log, P12_T),
    ):
        payload[f"{dataset}_variables"] = np.asarray(variables, dtype="U96")
        payload[f"{dataset}_log_variables"] = np.asarray(log_names, dtype="U96")
        payload[f"{dataset}_T"] = np.asarray(n_hours, dtype=np.int16)
        for site, arrays in site_data.items():
            key = f"{dataset}_{site}"
            payload[f"{key}_x"] = arrays["x"].astype(np.float32, copy=False)
            payload[f"{key}_m"] = arrays["m"].astype(np.int8, copy=False)
            payload[f"{key}_y"] = arrays["y"].astype(np.int8, copy=False)
            payload[f"{key}_ids"] = arrays["ids"]

    summaries = {
        "P19": {
            "D": len(e7.P19_VITAL_LABS), "T": P19_T, "log_variables": p19_log,
            "removed_values_by_variable": removed_p19,
            "sites": {site: _site_summary(data, e7.P19_VITAL_LABS)
                      for site, data in p19_sites.items()},
        },
        "P12": {
            "D": len(p12_variables), "T": P12_T, "log_variables": p12_log,
            "removed_values_by_variable": removed_p12,
            "sites": {site: _site_summary(data, p12_variables)
                      for site, data in p12_sites.items()},
        },
        "runtime_seconds": None,
    }
    summaries["runtime_seconds"] = time.perf_counter() - started

    E15_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(E15_CACHE, **payload)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("E15 hourly sequence cache summary", flush=True)
    for dataset in ("P19", "P12"):
        info = summaries[dataset]
        print(f"{dataset}: D={info['D']}, T={info['T']}, log_variables={info['log_variables']}", flush=True)
        for site, stats in info["sites"].items():
            high = ", ".join(f"{item['variable']}={item['measurement_rate']:.3f}"
                              for item in stats["most_measured_variables"])
            low = ", ".join(f"{item['variable']}={item['measurement_rate']:.3f}"
                             for item in stats["least_measured_variables"])
            print(
                f"  {dataset}_{site}: n={stats['n']:,}, outcome_rate={stats['outcome_rate']:.4f}, "
                f"mean_measurement_rate={stats['mean_measurement_rate']:.4f}\n"
                f"    most measured: {high}\n    least measured: {low}",
                flush=True,
            )
    print(f"Cleaning removals P19: {removed_p19}", flush=True)
    print(f"Cleaning removals P12: {removed_p12}", flush=True)
    print(f"Wrote {E15_CACHE} ({E15_CACHE.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    print(f"Wrote {SUMMARY_PATH}", flush=True)
    print(f"Runtime: {summaries['runtime_seconds'] / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
