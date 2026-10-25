Implemented [e15_sequences.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/e15_sequences.py). The run completed in **4.91 minutes**, and wrote [e15_cache.npz](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/data/e15_cache.npz) and [e15_sequences_summary.json](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/out/e15/e15_sequences_summary.json).

All E7 IDs and outcomes matched; the cache-wide NaN/mask check passed. I also compared three raw values for two patients in each dataset against the cache. No outcome mismatches or departures from the spec.

**Cache summary:** P19 has D=34, T=24; P12 has D=36, T=48.

| Site | n | Outcome rate | Mean measurement rate |
|---|---:|---:|---:|
| P19 A-MICU | 3,065 | 3.43% | 18.27% |
| P19 A-SICU | 1,470 | 1.70% | 23.05% |
| P19 A-all | 4,535 | 2.87% | 19.82% |
| P19 B-MICU | 5,018 | 1.75% | 18.78% |
| P19 B-SICU | 4,844 | 2.68% | 21.39% |
| P19 B-all | 9,862 | 2.21% | 20.06% |
| P12 CCU | 1,769 | 12.66% | 17.72% |
| P12 CSRU | 2,530 | 4.86% | 20.70% |
| P12 MICU | 4,293 | 19.85% | 18.18% |
| P12 SICU | 3,408 | 14.91% | 20.74% |

Log variables: P19 `AST, BUN, Alkalinephos, Creatinine, Bilirubin_direct, Glucose, Lactate, Magnesium, Potassium, Bilirubin_total, TroponinI, PTT, WBC`; P12 `ALP, ALT, AST, Creatinine, Glucose, K, TroponinT`. Cleaning removals are recorded per variable in the JSON; P19 had none, while P12 removals included 167 NIDiasABP, 59 Temp, 57 DiasABP, and 21 pH values.

**Command run:**

```powershell
$py = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'; & $py e15_sequences.py
```