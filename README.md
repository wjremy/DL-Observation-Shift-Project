# DL: Observation Shift Project

Do deep time-series models rely on the observation pattern (who gets measured, and when) more than the data justify, and how do they cope when the observation policy changes between hospitals? A simulation study with exact Bayes benchmarks, building on Sarah Martinson's proposal "Deep Time-Series Models under Observation Shift".

## Start here

The project has moved on from the original simulation ranking: it now studies a per-hospital trust dial on the testing pattern. Read [docs/MAP.md](docs/MAP.md) first.

| File | What it holds |
|---|---|
| [docs/MAP.md](docs/MAP.md) | Overview: current question and claims, how we got here, evidence map, next steps. |
| [docs/FUTURE.md](docs/FUTURE.md) | Directions recorded for later (adaptation without outcomes). |
| [STATUS.md](STATUS.md) | Current state: what is running, done, open, and results so far. Update it with every change. |
| [docs/ARGUMENT.md](docs/ARGUMENT.md) | Why the study is designed this way (the objection to a naive robustness ranking and how the design answers it). |
| [SPEC.md](SPEC.md) | Binding specification: data-generating process, Bayes rules, models, experiments, metrics, file formats. |
| [CLAUDE.md](CLAUDE.md) | Working rules for AI-assisted sessions in this folder. |
| [docs/prior_codex_runs/](docs/prior_codex_runs/) | Reports of two earlier studies (17 Oct 2026, Codex app). |

## Questions

- Q1 Justified cost: how much does the exact A-optimal predictor lose when applied unchanged under policy B?
- Q2 Excess reliance: do GRU, GRU-D, Neural CDE and a Transformer, trained end to end on values and masks under A, lose more than that on B (pattern used instead of values) or less (pattern under-learned)?
- Q3 Redundancy: how do Q1 and Q2 change as the values get noisier and the pattern's extra information grows?
- Q4 Sampling coverage: with uninformative observation in both hospitals, how much does each architecture lose when only the density or burstiness of sampling changes?

## Code

| File | Role |
|---|---|
| `sim.py` | Latent AR(1) state, outcome, observation policies (informative, uniform, bursty). |
| `bayes.py` | Exact Bayes predictors by grid forward filter (checked against a Kalman filter). |
| `models.py` | GRU, GRU-D, Neural CDE, Transformer (PyTorch) and their feature builders. |
| `run_grid.py` | Runs the experiment grid in parallel, one resumable shard per fit. |
| `analyze.py`, `plot.py` | Summary tables, architecture contrasts, figures. |
| `test_sim_bayes.py`, `test_models.py` | Tests. |
| `bayes_reference.csv` | Bayes mask gain G and justified cost J_B per configuration (20,000 patients). |

## Run

Python 3.12 with numpy, pandas and matplotlib; PyTorch (CPU build is enough):

```
python -m pip install numpy pandas matplotlib torch
python test_sim_bayes.py
python test_models.py
python run_grid.py --exp all --workers 4 --out results
python analyze.py
python plot.py
```

`--workers` should equal the number of physical cores (each fit uses one thread). `run_grid.py` skips finished shards, so an interrupted run can be restarted with the same command. `--smoke` runs a two-minute end-to-end check.

## Working together

- Pull before you start, commit and push small steps with clear messages.
- Record decisions and their reasons in STATUS.md (Decisions) and, for design changes, in docs/ARGUMENT.md and SPEC.md.
- Do not change SPEC.md silently: a change of the design is a decision and gets a dated line in STATUS.md.
