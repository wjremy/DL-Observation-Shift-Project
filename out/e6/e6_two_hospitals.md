## E6 results

The eligible cohort has **10,038 A patients** (3.04% outcome rate) and **13,843 B patients** (2.08%). Splits: A train/test **8,030/2,008**; B test/run-in pool **8,000/5,843**.

| w | A-test AUROC / Brier / log loss | B AUROC [95% CI] | B Brier [95% CI] | B log loss [95% CI] |
|---:|---|---|---|---|
| 0 | .615 / .0265 / .1232 | .632 [.584, .678] | .01990 [.01710, .02288] | .10008 [.08939, .11213] |
| .5 | .734 / .0261 / .1174 | .673 [.626, .716] | .01983 [.01706, .02280] | .09854 [.08806, .11045] |
| 1 | .744 / .0257 / .1145 | .683 [.639, .723] | .01989 [.01721, .02276] | .09822 [.08799, .10963] |

The lowest B-test log loss is at **w=.9: .098165**, only .000054 below w=1; the intervals overlap broadly. Apparent A-train AUROC (.783) exceeds A-test (.744), as expected. Pattern-only within-hospital AUROC is .729 in A and .687 in B.

Run-in cases averaged **5.6 [2.9–8.0]** at n=250 and **22.5 [16.8–28.1]** at n=1,000. Entries below are B-test mean AUROC [10–90%] / log loss [10–90%].

| Method | n=250 | n=1,000 |
|---|---|---|
| Choose w | .670 [.632,.684] / .0992 [.0982,.1001] | .681 [.679,.684] / .0986 [.0982,.0996] |
| Intercept + slope | .647 [.647,.683] / .1031 [.0978,.1117] | .683 [.683,.683] / .0980 [.0970,.0993] |
| Choose w + intercept | .672 [.632,.683] / .1000 [.0974,.1023] | .682 [.681,.684] / .0979 [.0971,.0988] |
| Full refit | .596 [.421,.680] / .1206 [.1002,.1799] | .665 [.627,.693] / .0991 [.0954,.1064] |
| Fixed w=0 | .632 / .1001 | .632 / .1001 |
| Fixed w=1 | .683 / .0982 | .683 / .0982 |

## Files and checks

Created [e6_cohort.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/e6_cohort.py), [e6_dial.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/e6_dial.py), and [e6_figure.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/figures/e6_figure.py). Results: [w curve](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/analysis/e6_wcurve.csv), [run-in](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/analysis/e6_runin.csv), [descriptives](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/analysis/e6_descriptives.csv), [summary](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/analysis/e6_summary.json). The cache is gitignored. Total runtime was about **11.2 minutes**.

A future-hour perturbation check confirmed features are unchanged by edits after hour 24. The figure script was syntax-checked only, as instructed; the PNG was not generated.

SPEC.md did not specify the penalty candidates, so I used a fixed average-BCE L2 grid from .001 to .3. EtCO₂ had no A-training measurements; its undefined median used the zero fallback. Low event counts make small run-ins noisy. Git also reports missing tracked E1–E4 result shards; I did not modify those.