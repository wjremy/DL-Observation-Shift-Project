## Files written

[sim.py](/C:/Users/wesre/Documents/Claude/obs-shift-sim/sim.py), [bayes.py](/C:/Users/wesre/Documents/Claude/obs-shift-sim/bayes.py), [test_sim_bayes.py](/C:/Users/wesre/Documents/Claude/obs-shift-sim/test_sim_bayes.py), [analyze.py](/C:/Users/wesre/Documents/Claude/obs-shift-sim/analyze.py), [plot.py](/C:/Users/wesre/Documents/Claude/obs-shift-sim/plot.py), and [bayes_reference.csv](/C:/Users/wesre/Documents/Claude/obs-shift-sim/bayes_reference.csv). I did not edit `models.py` or `run_grid.py`.

## Checks

All five tests passed using the specified Python interpreter. Kalman maximum errors were `1.97e-11`/`3.11e-12`/`9.97e-13` for full observation and `6.66e-7`/`8.90e-7`/`1.06e-9` with missingness (mean/variance/P(Y)). Calibration’s maximum decile gap was `0.0258` (`2.82` standard errors). On A data, Brier was `0.099521` for A_rule and `0.102924` for V_rule; the G=241 versus 481 difference was `2.45e-13`. The fake-tree analyzer check passed (6 E1 rows, 3 E2 rows, 3 contrasts); its temporary tree was deleted. `plot.py` passed `py_compile`.

## Reference results

Each row uses 20,000 patients; values are G / J_B_match / J_B_dense.

| σ | γ | G | J_B_match | J_B_dense |
|---:|---:|---:|---:|---:|
| 0.25 | 1 | 0.0023073 | 0.0032805 | 0.0003409 |
| 0.25 | 2 | 0.0071722 | 0.0077650 | 0.0016718 |
| 0.5 | 1 | 0.0035651 | 0.0040501 | 0.0009804 |
| 0.5 | 2 | 0.0087280 | 0.0093508 | 0.0035126 |
| 1 | 1 | 0.0060927 | 0.0063451 | 0.0036268 |
| 1 | 2 | 0.0122020 | 0.0122723 | 0.0103664 |
| 2 | 1 | 0.0104379 | 0.0085687 | 0.0211809 |
| 2 | 2 | 0.0169749 | 0.0129568 | 0.0267731 |

No spec deviations. One open detail: `SPEC.md` doesn’t fix the exact Bayes NPZ key spelling; the analyzer supports rule-policy keys such as `A_rule_A` and was checked against that format.