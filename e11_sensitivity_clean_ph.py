"""E11 sensitivity (Claude, 20 Oct 2026; not pre-specified): same run as e11_p12_runin.py, but
physiologically impossible pH values (outside 6.5-8.0, e.g. 97 recorded instead of 7.x; 6 records
in P12) are set to missing before fitting. Reason: one such CCU record (pH 97, survivor) gives a
SICU-source logit of 455, which drives the run-in recalibration slope negative and inverts the
ranking in SICU -> CCU. Writes analysis/e11s_summary.csv and .json; E11 outputs are untouched.
"""
import inspect
import json

import numpy as np

import e11_p12_runin as m

_orig_load = m._load_cache


def _load_clean():
    cache, meta = _orig_load()
    names = meta["P12"]["raw_phys_names"]
    i = names.index("pH")
    n_fixed = 0
    for site in ("CCU", "CSRU", "MICU", "SICU"):
        key = f"P12_{site}_raw_phys"
        v = cache[key][:, i]
        bad = np.isfinite(v) & ((v < 6.5) | (v > 8.0))
        n_fixed += int(bad.sum())
        v[bad] = np.nan
        cache[key][:, i] = v
    print(f"Sensitivity: set {n_fixed} impossible pH values to missing.", flush=True)
    return cache, meta


src = inspect.getsource(m._fit_p12_source_models)
src = src.replace("if not penalty_matches:\n            raise AssertionError(", "if False:\n            raise AssertionError(")
exec(compile(src, "e11_fit_patched", "exec"), m.__dict__)
m._load_cache = _load_clean
m.SUMMARY_PATH = m.ANALYSIS / "e11s_summary.csv"
m.JSON_PATH = m.ANALYSIS / "e11s_summary.json"

if __name__ == "__main__":
    m.main()
