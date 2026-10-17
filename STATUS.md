# Status

Update this file with every change: what started, what finished, what was decided. Newest first within each section.

## Running now

- Full grid (E1 informative, 96 fits; E2 coverage, 12 fits), started 17 Oct 2026 17:20 on Wesley's laptop: `python run_grid.py --exp all --workers 4 --out results`, log `run_full.log`. Exact Bayes files were written by 17:34; fits follow. Next step: `python analyze.py`, `python plot.py`, review, write results below.

## Open

- Rename the local folder from `obs-shift-sim` to `DL - Observation Shift Project` once no job runs inside it (Windows names cannot contain ':').
- After the grid: decide whether effects are large enough to separate architectures. The Bayes mask gain is small in this simulator (see below); if model estimation noise swamps it, add configurations with stronger informativeness (gamma 3-4) or rarer testing.
- Optional: real-data arm (MIMIC-IV). Needs credentialed PhysioNet access, which a team member must apply for; on real data report mask gain on A against cost on B (no Bayes rules available).

## Results so far

Exact Bayes reference (`bayes_reference.csv`, 20,000 patients, Brier; base Brier about 0.10):

| sigma | gamma | G (mask gain on A) | J_B match (B: uniform 0.3) | J_B dense (B: uniform 0.6) |
|---:|---:|---:|---:|---:|
| 0.25 | 1 | 0.0023 | 0.0033 | 0.0003 |
| 0.25 | 2 | 0.0072 | 0.0078 | 0.0017 |
| 0.5 | 1 | 0.0036 | 0.0041 | 0.0010 |
| 0.5 | 2 | 0.0087 | 0.0094 | 0.0035 |
| 1 | 1 | 0.0061 | 0.0063 | 0.0036 |
| 1 | 2 | 0.0122 | 0.0123 | 0.0104 |
| 2 | 1 | 0.0104 | 0.0086 | 0.0212 |
| 2 | 2 | 0.0170 | 0.0130 | 0.0268 |

Reading: the observation pattern adds little beyond the values when values are precise (near-redundant), more as they get noisy. With noisy values, dense uniform testing in B hurts the transferred A rule most: it reads "tested often" as "sick", and in B everyone is tested often.

## Done

- 17 Oct 2026: repository pushed to https://github.com/wjremy/DL-Observation-Shift-Project (private).
- 17 Oct 2026: smoke run and all tests passed with real PyTorch (feature builders, forward passes, empty histories, one-step loss reduction, end-to-end smoke). Bayes filter matches Kalman to 1e-6; grid resolution 241 vs 481 changes Brier by 2e-13.
- 17 Oct 2026: code written by two GPT-6 Luna agents from SPEC.md (briefs in `tasks/`, reports in `out/`), reviewed by Claude.
- 17 Oct 2026: design fixed in SPEC.md, following docs/ARGUMENT.md.

## Decisions

- 17 Oct 2026: simulation first, not MIMIC: exact Bayes benchmarks exist only in simulation, and MIMIC needs credentialed access.
- 17 Oct 2026: models are measured against the exact A_rule and V_rule, not ranked by B score (docs/ARGUMENT.md, step 6).
- 17 Oct 2026: CPU only (laptop has Intel UHD 620, no CUDA); 4 workers with one thread each.
