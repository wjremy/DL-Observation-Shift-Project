**E10 completed.** The successful run took **150.1 seconds**. AUROC gap is local minus transferred, so a positive gap means the transferred model performed worse. Intervals are paired 95% bootstrap intervals over target evaluation patients (500 resamples).

The E7 cache supplied raw arrays for unit and P12 sites. E6’s cache had only imputed/scaled matrices, so A-all and B-all were rebuilt in memory with E6’s cohort rules; their sizes matched the cache. Coverage differences are mean absolute differences: measurement rates across time-varying variables, imputed shares across all raw physiology variables, including static covariates.

| Pair | Group | Pattern AUROC gap [95% CI] | Physiology AUROC gap [95% CI] | Combined AUROC gap [95% CI] | \|Δ rate\| | \|Δ imputed\| |
|---|---|---:|---:|---:|---:|---:|
| P19 A-MICU→A-SICU | Other unit, same hospital | +0.064 [−0.194, +0.352] | −0.119 [−0.359, +0.103] | +0.005 [−0.199, +0.299] | 0.120 | 0.110 |
| P19 A-MICU→B-MICU | Same unit, other hospital | +0.028 [−0.043, +0.106] | +0.096 [+0.001, +0.185] | +0.088 [+0.008, +0.173] | 0.200 | 0.184 |
| P19 A-MICU→B-SICU | Other unit, other hospital | +0.004 [−0.037, +0.047] | +0.023 [−0.103, +0.136] | +0.003 [−0.061, +0.060] | 0.149 | 0.137 |
| P19 A-SICU→A-MICU | Other unit, same hospital | +0.135 [+0.040, +0.240] | +0.013 [−0.083, +0.105] | +0.084 [+0.006, +0.174] | 0.120 | 0.110 |
| P19 A-SICU→B-MICU | Other unit, other hospital | +0.065 [−0.066, +0.169] | +0.030 [−0.069, +0.116] | +0.058 [−0.037, +0.167] | 0.258 | 0.237 |
| P19 A-SICU→B-SICU | Same unit, other hospital | +0.030 [−0.063, +0.124] | +0.019 [−0.068, +0.107] | +0.047 [−0.065, +0.148] | 0.193 | 0.178 |
| P19 B-MICU→A-MICU | Same unit, other hospital | +0.030 [−0.045, +0.106] | +0.068 [−0.047, +0.194] | +0.101 [+0.001, +0.195] | 0.200 | 0.184 |
| P19 B-MICU→A-SICU | Other unit, other hospital | +0.060 [−0.286, +0.412] | −0.020 [−0.288, +0.184] | +0.025 [−0.271, +0.352] | 0.258 | 0.237 |
| P19 B-MICU→B-SICU | Other unit, same hospital | +0.000 [−0.038, +0.035] | +0.015 [−0.079, +0.099] | +0.022 [−0.032, +0.082] | 0.095 | 0.088 |
| P19 B-SICU→A-MICU | Other unit, other hospital | +0.076 [+0.020, +0.144] | +0.060 [−0.064, +0.171] | +0.095 [+0.026, +0.172] | 0.149 | 0.137 |
| P19 B-SICU→A-SICU | Same unit, other hospital | +0.020 [−0.219, +0.317] | +0.092 [−0.313, +0.440] | +0.065 [−0.162, +0.401] | 0.193 | 0.178 |
| P19 B-SICU→B-MICU | Other unit, same hospital | +0.007 [−0.034, +0.054] | +0.024 [−0.109, +0.129] | +0.015 [−0.048, +0.076] | 0.095 | 0.088 |
| P19 A-all→B-all | Whole hospital | +0.031 [−0.006, +0.068] | +0.044 [−0.001, +0.091] | +0.046 [+0.001, +0.090] | 0.180 | 0.165 |
| P19 B-all→A-all | Whole hospital | +0.060 [+0.018, +0.109] | +0.005 [−0.054, +0.060] | +0.066 [+0.016, +0.117] | 0.180 | 0.165 |
| P12 CCU→CSRU | P12 | +0.025 [−0.011, +0.067] | +0.023 [−0.022, +0.068] | +0.028 [−0.003, +0.059] | 0.171 | 0.162 |
| P12 CCU→MICU | P12 | +0.004 [−0.012, +0.021] | +0.049 [+0.032, +0.065] | +0.057 [+0.040, +0.073] | 0.066 | 0.070 |
| P12 CCU→SICU | P12 | +0.034 [+0.005, +0.070] | +0.037 [+0.015, +0.060] | +0.056 [+0.029, +0.082] | 0.088 | 0.085 |
| P12 CSRU→CCU | P12 | +0.071 [+0.033, +0.115] | +0.003 [−0.035, +0.040] | +0.032 [−0.007, +0.074] | 0.171 | 0.162 |
| P12 CSRU→MICU | P12 | +0.020 [−0.001, +0.041] | +0.055 [+0.038, +0.074] | +0.063 [+0.046, +0.080] | 0.187 | 0.188 |
| P12 CSRU→SICU | P12 | +0.042 [+0.006, +0.077] | +0.031 [+0.010, +0.052] | +0.043 [+0.019, +0.068] | 0.108 | 0.111 |
| P12 MICU→CCU | P12 | +0.021 [−0.008, +0.059] | −0.019 [−0.052, +0.012] | −0.026 [−0.054, +0.001] | 0.066 | 0.070 |
| P12 MICU→CSRU | P12 | +0.008 [−0.023, +0.040] | +0.040 [+0.006, +0.080] | +0.031 [+0.000, +0.062] | 0.187 | 0.188 |
| P12 MICU→SICU | P12 | −0.014 [−0.041, +0.012] | +0.024 [−0.001, +0.053] | +0.013 [−0.010, +0.039] | 0.088 | 0.084 |
| P12 SICU→CCU | P12 | +0.044 [−0.003, +0.090] | +0.010 [−0.017, +0.041] | +0.002 [−0.037, +0.037] | 0.088 | 0.085 |
| P12 SICU→CSRU | P12 | +0.183 [+0.110, +0.249] | +0.005 [−0.020, +0.035] | +0.044 [+0.005, +0.088] | 0.108 | 0.111 |
| P12 SICU→MICU | P12 | +0.009 [−0.012, +0.029] | +0.044 [+0.025, +0.064] | +0.042 [+0.023, +0.063] | 0.088 | 0.084 |

**Mean AUROC gaps by group**

| Group | Pattern | Physiology |
|---|---:|---:|
| P19 other unit, same hospital | +0.052 | −0.017 |
| P19 same unit, other hospital | +0.027 | +0.069 |
| P19 other unit, other hospital | +0.051 | +0.023 |
| P19 whole hospital | +0.046 | +0.025 |
| P12 | +0.037 | +0.025 |

Using positive AUROC gaps whose 95% intervals exclude zero as the flag, the pattern shows a transfer loss in P19 A-SICU→A-MICU, P19 B-SICU→A-MICU, P19 B-all→A-all, and P12 CCU→SICU, CSRU→CCU, CSRU→SICU, and SICU→CSRU.  
Physiology shows a transfer loss in P19 A-MICU→B-MICU and P12 CCU→MICU, CCU→SICU, CSRU→MICU, CSRU→SICU, MICU→CSRU, and SICU→MICU.  
Across 26 pairs, imputed-share difference has a weak positive association with the physiology AUROC gap (Pearson r = 0.217; Spearman ρ = 0.222); runtime was 150.1 seconds.

**Files**

- [e10_diagnostic.py](</C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/e10_diagnostic.py>)
- [e10_gaps.csv](</C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/analysis/e10_gaps.csv>)
- [e10_sites.csv](</C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/analysis/e10_sites.csv>)
- [e10_coverage.csv](</C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/analysis/e10_coverage.csv>)
- [e10_summary.json](</C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/analysis/e10_summary.json>)
- [e10_figure.py](</C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/figures/e10_figure.py>)

The figure script passed a Python syntax check; I did not render the PNG, as requested.