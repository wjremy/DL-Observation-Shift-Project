# Status

Update this file with every change: what started, what finished, what was decided. Newest first within each section.

## Running now

- E15 code (25 Oct 2026): written by Luna (`e15_sequences.py`, `latent_filter_mv.py`, `e15_real_pairs.py`, `figures/e15_figure.py`; briefs `tasks/e15/`, answers `out/e15/`). Cache built by the agent (`data/e15_cache.npz`, 4.9 min; ids and outcomes equal E7's; P19 D 34, T 24; P12 D 36, T 48). Torch-NumPy parity check passed (Claude). Linear comparison AUROC from `analysis/e7_sites.csv` `source_holdout_auroc_w1` (P19 whole hospitals: `analysis/e10_sites.csv` `local_auroc`). Smoke test passed (25 Oct). Source models trained (`python e15_real_pairs.py --train-only`, log `out/e15/e15_train.log`, `analysis/e15_sources.csv`): **source AUROC gate failed for 5 of 6 sources**, validation AUROC vs E7 linear: P12 CCU 0.697 vs 0.801, CSRU 0.806 vs 0.928, MICU 0.712 vs 0.830, SICU 0.688 vs 0.852, P19 B-all 0.601 vs 0.689; only P19 A-all passes (0.717 vs 0.721). Pair runs not started (SPEC). Small departure found in review: standardisation uses all source patients, not only the training split. MICU fit shows 22,757 s wall time (laptop probably slept; other fits 3-6 min). Diagnostic (log `out/e15/e15_lambda_check.log`): a smaller NLL weight helps only partly. Validation AUROC at lambda 0.1 / 0.01 / 0: CSRU 0.782 / 0.821 / 0.837 (linear 0.928), CCU 0.726 / 0.719 / 0.744 (linear 0.801). Even the purely discriminative fit stays 0.06-0.09 below, so the bottleneck is the model: the outcome is read only from 32 discrete states, and the state-space model has no static variables, while E7's linear model uses age, sex, height and weight (P19: age, sex, hospital admission time). Waiting for Wesley's decision on the model (stopped; nothing running).


## Proposal (22 Oct 2026)

- 23 Oct: compute section rewritten (one GTX 1080 Ti on snowflake plus the laptop; no Euler; cost table removed; measured CPU times kept; time one site first, reduce draws or pairs if needed). Page limit confirmed by Wesley on Moodle: 2 pages excluding references. The official course template is `ethdl2026` (zip from Moodle; same layout as icml2026, requires `\icmlcorrespondingauthor` with the `accepted` option); compiled with it, the main text ends on page 2. Overleaf still uses icml2026; switching is two lines plus the style files.
- Teammate review (23 Oct): (1) the proposal drifted from the topic (compare architectures) to one model; proposed fix: E3 compares how GRU, GRU-D, Neural CDE, Transformer and the state-space model adapt. (2) Compute section wrong: no Euler access; the team has one GTX 1080 Ti on snowflake (now in CLAUDE.md). (3) Page limit claimed 1 page; the course text Wesley pasted says at most 2 pages excluding references; to be clarified. (4) Rung order to update after E12/E13 (unlabelled testing-head refit first).

- Course template received (ICML 2026 style, "Project 14"): the proposal now lives in `docs/proposal/icml/main.tex` + `bibliography.bib` (all DOIs checked on Crossref), and supersedes `docs/proposal/proposal.tex`. Length checked with a stand-in style using ICML's page geometry (the real icml2026.sty is not on this laptop): main text ends on page 2 with some room; confirm in Overleaf. Advisor and topic go in the first-column footnote via \printAffiliationsAndNotice. Still to fill: surnames and ETH usernames (Daniele, Rafael, Meds, Wesley Remy), corresponding author (optional).

- Course-format proposal drafted around the ladder: `docs/proposal/proposal.tex` / `.pdf` (2 pages plus references). Sections follow the course requirements: authors, advisor and topic; idea and research question; literature overview (with preliminary results); experiment plan with a compute table (about 45 CPU-hours, CPU only). The older `proposal_trust_dial.tex` stays as the 19 Oct version.
- Placeholders to fill: team members' full names and ETH usernames; the course LaTeX template (layout is a stand-in; swap the preamble, keep the body). TA assumed Sarah Martinson, topic her "Deep Time-Series Models under Observation Shift"; confirm.
- E2 dataset to be determined (Wesley, 22 Oct): needs several hospitals with hospital IDs and a preserved testing pattern; the eICU demo is useless (hospital IDs removed); main candidate eICU-CRD (credentialed). Compute table assumes about 8 sites, 25 pairs.
- Model (Wesley, 22 Oct): the proposal uses the latent state-space model (E5, extended to many variables) as the main model; the two-branch additive network is dropped. Rungs act on the testing head's log-likelihood (weights) or its parameters (re-learn). Compute re-estimated at about 100 CPU-hours (source fits about 15 min as in E5; adaptation about 20 s per run, not yet measured).
- Compute figures are estimates: rungs 3-4 at about 15 s per fine-tune have not been measured yet.
- Literature batch lit12-14 done (Luna xhigh with web search, 507 s, Codex 17% of 5-hour window, 10% of 7-day; briefs `tasks/lit7/`, answers `out/lit/lit12-14_*.md`); reviewed, key references checked, summary in docs/GAP.md.

## Open

- Team: everyone applies for PhysioNet credentialing (CITI course plus approval, 1-2 weeks) for eICU-CRD; see docs/DATA.md.

- Review `analysis/contrasts.csv` (bootstrap intervals for architecture differences) and the figures.
- Possible quick follow-ups (minutes, not a grid): stronger informativeness (gamma 3-4) to enlarge the effects; a CDE with longer training to test whether its under-learning of the pattern is a prior or slow convergence.
- Real-data arm: PhysioNet Challenge 2019 sepsis data (open access, CC BY 4.0, 42 MB, two hospital systems A and B). Proposed: train the structured model at A, re-learn the testing head on B's unlabelled records, evaluate on B's labels. Downloaded 18 Oct (docs/DATA.md). Group meeting deck (19 Oct): https://claude.ai/artifact/CtdgYzJg2qEK2n7Y7pX6nE
- Literature search done (19 Oct): `docs/GAP.md`. Closest papers (Futoma 2021, Rockenschaub 2024, CLIO) read in full and summarised there.
- Earlier optional item: real-data arm (MIMIC-IV). Needs credentialed PhysioNet access, which a team member must apply for; on real data report mask gain on A against cost on B (no Bayes rules available).

## Results so far

### E14: a shift only outcomes can reveal (25 Oct 2026; exploratory, quick; SPEC.md E14; `analysis/e14_runs.csv`, `analysis/e14_summary.csv`, `analysis/e14_shift_size.csv`, `figures/e14_meaning_shift.png`; log `out/e14/e14_run.log`)

Run: `python e14_meaning_shift.py --train-only` (A models: gain 31 min, lose 21 min; training outcome rate 1.9%), then `--workers 4 --draws 2`. Code by Luna, reviewed by Claude. Mean regret x1e-4 at weeks 4 / 13 / 52 / 104:

| Direction (keep AUROC -> oracle AUROC) | keep | head_U | dial only | head_L | fine-tune lr 0.05 | fine-tune lr 0.005 |
|---|---|---|---|---|---|---|
| gain: B also tests on the hidden risk factor (0.896 -> 0.920; 0.5 / 7 / 27 / 54 cases) | 10.5 | 10.8 / 10.5 / 10.4 / 10.4 | 10.6 / 10.5 / 10.5 / 10.5 | 10.6 / 10.7 / 10.4 / 10.3 | 15 / 31 / 17 / 9.9 | 13 / 12 / 11 / 10.4 |
| lose: A tested on it, B does not (0.845 -> 0.845; 1 / 4 / 17 / 42 cases) | 2.4 | 3.5 / 3.5 / 3.2 / 3.3 | 2.4 / 2.4 / 2.4 / 2.3 | 2.5 / 2.5 / 2.8 / 3.2 | 117 / 29 / 6.2 / 1.6 | 4.5 / 3.1 / 1.7 / 0.9 |

- `gain` (as predicted): the shift is real (keep 4x the regret of the testing-only shifts of E13, AUROC 0.024 below the optimum), and no rung without outcomes touches it: head_U neither helps nor harms. The labelled light rungs (dial, head_L) do not help either; only full fine-tuning at lr 0.05 recovers part of it after 2 years (AUROC 0.908 at 54 cases, half the gap), and it harms before. Reading: when B's doctors test on something the model cannot see, B's records cannot reveal it and one ICU's run-in cannot fix it within 2 years; it can at most be detected.
- `lose` (prediction wrong): there is almost nothing to fix. The oracle at B cannot infer the trait (P(u=1|h) = 0.3 for everyone), and the A model does not misfire: check by Claude on 5,000 new patients, the lose A model at B gives the same mean prediction to u = 1 and u = 0 patients (0.0197 each) and its regret (3.5e-4) equals the gain A model's at A (3.2e-4), i.e. the model's own baseline error. At A the same model does use the trait (mean prediction 0.031 vs 0.014; AUROC 0.914 vs oracle 0.936). Losing meaning is harmless here because tests at B simply look like an average patient's. head_U is slightly worse than keep (+1e-4). Fine-tuning at lr 0.005 reduces the model's own baseline error after 1-2 years (0.9e-4 at 104 weeks).
- Together with E13: the shift that hurts and that the light rungs cannot fix is testing that carries information the model cannot see. The reframing "which kind of shift" holds for the simulator: testing-only shifts -> head_U from week 1; shifts only outcomes reveal -> nothing works within a realistic run-in, fine-tuning after years.

### E13: which rung when, over weeks of deployment (23 Oct 2026; exploratory; SPEC.md E13; `analysis/e13_runs.csv`, `analysis/e13_summary.csv`, `figures/e13_rungs_over_time.png`; log `out/e13/e13_run.log`)

Run: `python e13_rungs_over_time.py --train-only`, then `--workers 4 --lr 0.05 --steps 160` (160 jobs, 5 draws, 25 patients a week, outcome about 2%, so about 0.5 cases a week). Mean regret (Brier minus B optimum, x1e-4) at weeks 1 / 13 / 104 (0.6 / 7 / 52 outcome cases):

| B policy | keep | dial only | head_U | head_U + dial | fine-tune |
|---|---|---|---|---|---|
| less selective | 2.5 / 2.5 / 2.5 | 2.3 / 1.9 / 1.5 | 1.7 / 1.3 / 1.3 | 1.7 / 1.4 / 1.3 | 52 / 23 / 3.7 |
| more selective | 3.4 / 3.4 / 3.4 | 3.4 / 3.4 / 3.4 | 2.1 / 2.2 / 2.1 | 2.1 / 2.1 / 2.1 | 51 / 7.4 / 4.1 |
| random, same rate | 11.6 / 11.6 / 11.6 | 7.7 / 1.4 / 1.0 | 1.5 / 1.0 / 0.9 | 1.5 / 1.0 / 0.9 | 47 / 14 / 2.5 |
| random, twice as often | 9.3 / 9.3 / 9.3 | 8.8 / 4.2 / 2.5 | 2.8 / 2.3 / 2.4 | 2.8 / 2.3 / 2.4 | 56 / 14 / 8.8 |

- The expected staircase (each rung with its own peak) did not appear. head_U is best or tied from week 1 under all four policies, because it needs no outcomes: 25 unlabelled patients already bring it close to its floor, and it improves only slightly after week 4.
- The dial on top of head_U adds nothing (w about 0.95; differences at or below 1e-5). The dial alone catches up with head_U only under random testing at the same rate, after about 13 weeks (7 cases).
- Full fine-tuning is the worst method at every week up to 2 years (52 cases), 10-50x the regret of head_U in the first 2 months; it approaches keep only at 26-104 weeks. Caveat: it uses the same lr 0.05 / 160 steps on all parameters, without early stopping, so this baseline may be handicapped; a smaller learning rate is the obvious check.
- head_U's remaining regret (1-2.5e-4) is about what keep has under the mildest shift, i.e. probably the A model's own error rather than leftover shift.
- Limit: this simulator shifts only the testing policy, never what a test means for the outcome, so there is nothing for outcome-based rungs to fix. The ladder's upper rungs can only show their value in a simulator where B also changes the link between state and outcome, or between tests, or on real data.
- **E13b, fine-tune at smaller learning rates (24 Oct 2026; quick; Wesley asked):** `python e13b_finetune_lr.py --lrs 0.05 0.005 0.001 --workers 4 --steps 160` (wrapper by Claude that calls only E13's own functions; same A model, streams, batch seeds and objective; weeks 4/13/52/104, draws 0-1, 96 fits, about 35 min; log `out/e13/e13b_quick.log`, `analysis/e13b_quick_runs.csv`). lr 0.05 reproduces `e13_runs.csv` exactly (32 rows, difference 0). Mean regret x1e-4 at weeks 4 / 13 / 52 / 104 (about 2 / 9 / 28 / 55 cases), lr 0.001 | 0.005 | 0.05, against head_U and keep (same draws):
  - less selective: 2.2/1.8/2.8/2.8 | 6.6/5.4/2.4/2.3 | 37/23/6.2/2.9; head_U 1.3, keep 2.5
  - more selective: 5.3/5.0/4.1/3.7 | 17/7.0/3.8/3.3 | 128/9.5/1.9/2.0; head_U 2.1-2.3, keep 3.4
  - random, same rate: 12.5/11.9/12.0/12.0 | 10.5/8.8/9.1/9.0 | 39/19/6.5/1.2; head_U 0.9-1.0, keep 11.6
  - random, twice as often: 9.3/9.3/9.4/9.4 | 9.3/7.6/7.3/7.3 | 38/12/6.7/8.3; head_U 2.4, keep 9.3
  - Reading: the learning rate trades early harm against late gain. A small rate removes the early damage (lr 0.001 stays at about keep) but then learns almost nothing in 160 steps; the large rate harms for the first 3 months and catches up only after 1-2 years. No rate beats head_U within 2 years except a near-tie under "more selective" (lr 0.05, 1.9-2.0 against 2.1-2.2, 2 draws). The E13 conclusion stands: in this simulator, full fine-tuning does not pay off within realistic run-ins. Not tried: more steps at small rates, or early stopping (needs held-out labels the run-in does not have).
- Figure fix (Claude, 23 Oct): log y-axis (fine-tuning hid all other curves), no best-method shading (head_U and head_U + dial are tied, so the shading flipped on noise), week labels on all panels.

### E12: re-learning the testing head at B without outcomes (23 Oct 2026; exploratory; SPEC.md E12; `analysis/e12_*.csv`)

Code by Luna (`e12_unlabelled_head.py`), reviewed by Claude. E5 filter trained at A (informative gamma 2); 4 B policies; test cohort 8,000; 3 draws. Regret = Brier minus exact-optimum Brier at B (mean over draws):

| Method (data) | B_less (gamma 1) | B_more (gamma 3.5) | B_match (uniform 0.3) | B_dense (uniform 0.6) |
|---|---|---|---|---|
| keep (w = 1) | 0.0019 | 0.0006 | 0.0146 | 0.0125 |
| drop (w = 0) | 0.0053 | 0.0161 | 0.0003 | 0.0003 |
| dial_L, 30 labelled (about 5 cases) | 0.0016 | 0.0058 (max 0.016) | 0.0029 | 0.0004 |
| dial_L, 500 labelled | 0.0009 | 0.0007 | 0.0006 | 0.0004 |
| head_L, 30 labelled | 0.0010 | 0.0008 | 0.0096 | 0.0052 |
| head_U, 200 unlabelled | 0.0005 | 0.0004 | 0.0092 | 0.0048 |
| head_U, 2,000 unlabelled | 0.0006 | 0.0003 | 0.0092 | 0.0048 |

- When B's testing still depends on the state (B_less, B_more), re-learning the testing head without outcomes is the best method: within 0.0003-0.0006 of the optimum from only 200 unlabelled patients, better than the dial at every run-in size, and it handles the more selective policy, where the dial with a small run-in can do harm.
- When B tests uniformly, head_U and head_L remove only about two thirds of keep's loss and stay far behind drop and the dial (0.005-0.009 against 0.0003). The refitted testing curves show why: with lr 0.01 and 80 steps each logit moves at most about 0.8, and the A head is very steep (probabilities 0.05-0.93), so the refit stopped halfway. Checked in E12b with longer refits.
- Size barely matters for head_U (200 vs 2,000 patients alike): the testing pattern of 200 patients x 48 hours carries far more information about 32 testing probabilities than a handful of outcome cases.

**E12b (longer refits: lr 0.05, 160 steps; `analysis/e12_b_*.csv`):** the halfway refits were the cause. head_U now reaches regret 0.0002-0.0004 at all four B policies from 200 unlabelled patients (B_less 0.0002, B_more 0.0004, B_match 0.0003, B_dense 0.0003-0.0004): as good as the best alternative everywhere (drop on uniform testing, 0.0003) and better than every labelled method on informative testing. head_L with small run-ins now overfits (30 patients: B_match 0.0040, B_more 0.0022) and only matches head_U at 500 labelled patients. The refitted testing curves need not be flat under uniform testing (range 0.16-0.63 for B_match) while predictions are near-optimal: the learned states are not the simulator's states (as in E5). Reading: in this simulator, re-learning the testing head from B's records without outcomes is the best single rung from 200 patients on; outcomes add nothing here because the shift is only in testing. Settings lr 0.05 / 160 steps carried into E13.

### E11: safe dial tuning with realistic run-ins on the 2012 mortality pairs (20 Oct 2026; pre-specified, SPEC.md E11; `analysis/e11_summary.*`, `figures/e11_runin.png`; sensitivity `e11_sensitivity_clean_ph.py`, `analysis/e11s_summary.csv`)

Code by Luna (`e11_p12_runin.py`), reviewed by Claude. 3-month run-in = 162 patients, about 8-32 deaths.

- **Pre-specified primary result:** `dial_recal_shrink` minus `keep`, AUROC at 3 months, mean over 12 pairs: **-0.009**. Entirely due to SICU to CCU (-0.129, 10th percentile -0.57).
- Cause (found by Claude): one CCU record has pH 97 (a recording error; 6 impossible pH values in P12). With the SICU-source model its logit is 455; the run-in recalibration then turns its slope negative and inverts the ranking. A data error, not the dial.
- **Sensitivity (not pre-specified; impossible pH values, outside 6.5-8, set to missing):** mean over 12 pairs at 3 months: `dial_recal_shrink` +0.0015, `end_recal` +0.0015, `dial_auc_shrink` +0.0047, `dial_recal` -0.0005. SICU to CCU becomes -0.0006; the other pairs barely change.
- Where E10 found the pattern's meaning shifted, the dial gains, growing with the run-in: CSRU to SICU `dial_recal_shrink` +0.007 (3 months), +0.022 (12), +0.026 (24); `dial_auc_shrink` +0.022 already at 3 months; CCU to SICU +0.006 / +0.009 / +0.011 (`dial_auc_shrink` +0.015 at 3 months). Where it did not shift (MICU as source), the safe dial stays within 0.001 of `keep`.
- Against the binary choice with recalibration, the safe dial is equal on average; the AUROC-tuned dial is better (+0.0047 vs +0.0015).
- Lesson: recalibration needs outlier protection (clip logits or clean impossible values) before the slope is estimated.

### E10: which part shifted per pair (20 Oct 2026; SPEC.md E10; `analysis/e10_*`, `figures/e10_gaps.png`)

Code by Luna (`e10_diagnostic.py`), reviewed by Claude. Transfer gap = AUROC of a model fitted on half of the target minus AUROC of the source model, on the target's other half (95% paired bootstrap). The local models are fitted on few cases at the small sites, so gaps are noisy and, if anything, underestimated.

- Realistic sepsis pairs (same unit, other hospital): physiology gaps are larger than pattern gaps (means +0.069 vs +0.027); the only clear gap is physiology in A-MICU to B-MICU (+0.096). Pattern gaps are small and never clearly above zero. Confirms the reading of E7-E9: here the testing pattern's meaning largely transferred, physiology did not.
- The pattern's meaning clearly shifted (interval above zero) in 7 of 26 pairs: four P12 pairs involving the cardiac surgery and surgical ICUs (largest SICU to CSRU +0.183), two P19 pairs within or towards A-MICU, and whole hospital B to A (+0.060). Physiology clearly shifted in 7 pairs, mostly P12.
- Coverage (difference in imputed share) is only weakly related to the physiology gap (r = 0.22).
- A pattern gap does not imply that a single dial can close it: the dial scales all pattern terms together, while a shift in which tests matter needs the pattern part re-learned per variable (docs/FUTURE.md).

### E9: E8 with corrected tuning (20 Oct 2026; exploratory, SPEC.md E9; `analysis/e9_summary.*`, `figures/e9_runin.png`)

Code by Luna (`e9_fixed_tuning.py`, reuses E8 helpers), reviewed by Claude. AUROC difference to `keep` (w = 1) on the 4 primary pairs:

| Method | 3 months (about 3-6 cases) | 12 months | 24 months |
|---|---|---|---|
| `dial_recal` (E8) | -0.026, +0.002, -0.006, -0.015 | -0.011, +0.016, -0.007, -0.015 | +0.008, +0.020, -0.008, n/a |
| `dial_recal_shrink` | -0.001, +0.001, -0.001, -0.001 | -0.001, +0.004, -0.002, -0.001 | +0.006, +0.007, -0.004, n/a |
| `dial_auc` | -0.013, -0.001, -0.005, -0.005 | +0.004, +0.008, -0.006, -0.001 | +0.012, +0.013, -0.008, n/a |

(pair order: A-MICU to B-MICU, B-MICU to A-MICU, A-SICU to B-SICU, B-SICU to A-SICU)

- Shrinking w towards 1 makes tuning safe: within 0.002 AUROC of the trained model at every realistic run-in size, with narrow ranges (3 months: 10-90% within about -0.017 to +0.011 in the worst pair). Tuning by run-in AUROC is noisier and only helps from about 12 months.
- No method gives a meaningful gain at a realistic single-ICU run-in. Gains appear only with 1-2 years of one ICU, in the two medical-ICU pairs (+0.004 to +0.020); the surgical pairs never gain.
- Reading: with corrected tuning the dial is safe but, in these data, useful only with long run-ins. The limit is the number of cases a single ICU collects, not the tuning method.

### E8: the dial with a realistic unit-level run-in (19 Oct 2026; pre-specified in SPEC.md E8; `analysis/e8_*`, `figures/e8_runin.png`)

Code by Luna (`e8_runin.py`), reviewed by Claude: same source fits as E7 (penalties match), run-ins drawn without replacement from the target unit, evaluation on all other target patients, 200 draws per size, ties and zero-case run-ins resolved towards w = 1. Runtime 3 min.

**Primary result (pre-specified; 4 same-ICU-type cross-hospital pairs, 3-month run-in of 162 patients, about 3-6 sepsis cases):** no tuning method beats doing nothing. AUROC of `dial_recal` minus `keep` (w = 1, untuned): A-MICU to B-MICU -0.025, B-MICU to A-MICU -0.002, A-SICU to B-SICU -0.007, B-SICU to A-SICU -0.016 (mean -0.012). `dial_recal` does beat `end_recal` (+0.014, +0.014, +0.001, +0.003; mean +0.008), and it is the least harmful tuning method at every size.

- Choosing by run-in log loss hurts ranking, even with large run-ins: `end` and `dial` lose 0.03-0.09 AUROC against `keep` in the MICU pairs and get worse as the run-in grows. Log loss rewards settings that fix calibration (the source model is too confident at the target, and lowering w shrinks the predictions), which costs ranking. Recalibration takes over the calibration job and removes most of this loss.
- Only one pair gains with a long run-in: B-MICU to A-MICU, `dial_recal` +0.009 (6 months), +0.015 (12), +0.020 (24) against `keep`.
- Correction of E7b: its AUROC gains of the dial were measured against `end_recal`, which itself loses ranking; against doing nothing they mostly disappear.
- Reading: at realistic run-in sizes the safest choice is to keep the model as trained; per-unit tuning of the dial does not pay off in these data. Lessons for the design: tune for the metric that matters (or shrink w towards 1), and separate calibration from the dial.

### E7b: dial above 1 versus recalibration (19 Oct 2026; `analysis/e7b_*`)

Implemented in `e7b_recal.py`, reusing E7's source splits, preprocessing and CV fits. On the extended 0-3 grid, the best weight is above 1 in 6/24 pairs (maximum 1.90), with no optimum at the upper grid edge. The w=0..1.5 target log-loss curve reproduces E7 to maximum absolute difference 9.7e-17 over 744 points; all eight selected source penalties also match E7.

- On 100 paired target tune/evaluation splits (new seed base 71219), the uncalibrated 0-3 dial minus end has mean pair log-loss difference -0.000996. After per-candidate, unpenalised intercept/slope calibration on the tune half, dial03_recal beats end_recal by a negative mean and negative 90th percentile in 5/24 pairs; the mean pair difference is -0.000732. The recalibrated 0-3 option chooses mean w > 1 in 14/24 pairs and does so in 52.6% of split choices overall.
- The recalibrated 0-1 dial meets the same paired criterion in 3/24 pairs, with mean difference -0.000375. Calibration attenuates the average dial advantage but does not remove it; the remaining gains are concentrated in a minority of pairs. Runtime was 4.9 min using the Codex NumPy/pandas runtime.
- Claude's review (19 Oct): code reproduces E7 exactly (max difference 1e-16). Reading: (1) the large above-1 gains of the cardiac-surgery (CSRU) source pairs were mostly calibration: after recalibration CSRU to MICU and CSRU to SICU gain nothing (-0.0001, +0.0003); CSRU to CCU keeps -0.0032. (2) With recalibration the slope can shrink, so w then acts as the weight of the pattern relative to physiology; the recalibrated dial picks w near 2.7-2.9 in most P19 pairs (close to the new grid edge), i.e. it shifts weight from physiology to the pattern. (3) Its AUROC gains over end_recal are positive on average (mean +0.009; above +0.005 in 11 of 24 pairs; three of the four same-unit cross-hospital sepsis pairs +0.02 to +0.036), but the 10-90% ranges mostly include zero, so this is weak evidence. Net: allowing w above 1 is fine, but its apparent large gains were calibration; the dial's real advantage over the binary choice in these open data is small and limited to a minority of pairs.
- Artifacts: `e7b_recal.py`, `analysis/e7b_pairs.csv` (144 pair-option rows), `analysis/e7b_curves.csv` (1,464 rows), and `analysis/e7b_summary.json`.

### E7: the dial across 24 open site pairs (19 Oct 2026; SPEC.md E7; `analysis/e7_*.csv`, `figures/e7_pairs.png`)

Code by Luna (`e7_sites.py`, `e7_dial.py`), reviewed by Claude: P19 features hours 1-24, outcome 25-72; P12 features hours 0-48; imputation and scaling from the source's 80% training split; no source-training patient in any target (checked). Runtime about 5 min. Sites: P19 A-MICU 3,065 (105 sepsis cases), A-SICU 1,470 (25), B-MICU 5,018 (88), B-SICU 4,844 (130); P12 CCU 1,769 (224 deaths), CSRU 2,530 (123), MICU 4,293 (852), SICU 3,408 (508).

- Where the best w sits (whole target, grid 0-1.5): at 0 in 5 pairs, strictly inside (0.05-0.95) in 10, at 1 in 3, above 1 in 6. The optimum is spread over the whole scale; no single end and no fixed value is right.
- Fair test (100 splits; w and the better end both chosen on one half of the target, judged on the other): the dial has lower log loss than the better end in 13 of 24 pairs, mean difference -0.0011 per patient. Pairs with an interior optimum gain little (mean -0.0005; largest SICU to CCU -0.0027, A-MICU to B-SICU -0.0013); most of the average comes from pairs wanting w above 1 (mean -0.0037, all with CSRU as source). Where the best end is right, the dial loses a little from tuning noise (up to +0.0007, worst with 25 cases at A-SICU).
- Shift size (19 Oct; `figures/e7_shift_size.py` by Claude, `analysis/e7_shift_size.csv`, `figures/e7_shift_size.png`): per pair, the curvature c of target log loss in w measures how much the trust weight matters at all. The best dial beats the better end by about c * d^2, where d is the distance of the optimum from the nearest of 0 and 1 (the quadratic prediction matches the observed gain in most pairs). So a gain needs both a large shift and an optimum away from the ends. The realistic sepsis pairs (same ICU type, other hospital) have small curvature (0.001-0.004) and gains near zero; the largest gains (0.005-0.010, 1.3-2.6% of log loss) are the P12 pairs from the cardiac surgery ICU, whose optimum lies above 1. Rank correlation of curvature with gain: 0.45 (best possible dial), 0.14 (tuned).
- Caveat 1 (tested in E7b): scaling the pattern term also changes prediction spread, so part of the above-1 gain is calibration. Tune-half intercept/slope recalibration reduces the average 0-3 gain from -0.000996 to -0.000732 log-loss per patient, but dial03_recal still beats end_recal under the strict paired criterion in 5/24 pairs; the largest uncalibrated CSRU-to-SICU gain disappears after recalibration.
- Caveat 2: the P12 sites are units of one hospital (2001-08) and P19 sites mix hospitals and units, so differences include case mix, not only testing policy.

### E6: the dial on two real hospitals (19 Oct 2026; PhysioNet 2019, A = Beth Israel, B = Emory; `analysis/e6_*.csv`, `analysis/e6_summary.json`, `figures/e6_two_hospitals.png`)

Code by Luna (`e6_cohort.py`, `e6_dial.py`), reviewed by Claude: features use ICU hours 1-24 only (agent also checked by perturbing later hours), outcome hours 25-72, A's medians and scaling, the dial scales only the pattern coefficients of one fitted model. Cohort: A 10,038 patients (3.0% outcome), B 13,843 (2.1%); A train/test 8,030/2,008; B test 8,000, run-in pool 5,843. Runtime about 11 min.

| | A-test AUROC | B AUROC [95% CI] | B log loss |
|---|---|---|---|
| w = 0 (Futoma's "fit with, deploy without") | 0.615 | 0.632 [0.584, 0.678] | 0.1001 |
| w = 0.5 | 0.734 | 0.673 | 0.0985 |
| w = 1 (keep the pattern) | 0.744 | 0.683 [0.639, 0.723] | 0.0982 |
| physiology-only model | 0.630 | 0.664 | 0.0992 |
| pattern-only model | 0.729 | 0.649 | 0.0989 |

- For this pair the best dial is near 1 (log loss minimum at w = 0.8-1.0, flat; AUROC flat from 0.9 to 1.3). The intermediate value does not win here; dropping the pattern does clearly worse, and even worse than a model fitted on physiology alone. Futoma found the opposite for Beth Israel models moved to Memphis (dropping was best). Same source hospital, opposite best end: the best weight depends on the target, which is the case for setting it per hospital rather than choosing 0 or 1 once.
- The testing pattern stays informative at B (pattern-only AUROC fitted within each hospital: 0.729 at A, 0.687 at B).
- Caveat: much of the shift is coverage of the data export, not clinical policy: HCO3 and chloride are recorded for 96% of A patients but 2% and 8% at B; DBP 56% versus 100%; troponin 3% versus 32%. Imputed medians also carry some missingness signal into the physiology terms, so w = 0 does not remove the pattern completely.
- Run-in (50 draws per size, B-test AUROC mean): choosing w is stable from about 1,000 patients (about 22 cases; chosen w 10-90% range 0.69-1.41) and reaches 0.681 against 0.6835 for fixed w = 1. Because the optimum is near 1 here, choosing w adds nothing over keeping w = 1 for this pair. A full refit loses below 1,000 patients (0.596 at 250, 0.665 at 1,000) but wins from about 2,500 (53 cases): 0.698 at 2,500, 0.714 on the whole pool of 5,843 (log loss 0.0955 and 0.0945 against 0.097-0.098). So one-parameter adaptation is the small-run-in tool; with a few thousand labelled patients a full refit is better for a linear model.
- Anchor (docs/GAP.md): documented sepsis silent run-ins last about 6 weeks to 3 months; Epic's model at Tufts collected about 1,000 sepsis cases in 77 days. The crossover (somewhere between 22 and 53 cases) lies well within a typical run-in for a large hospital, so for this linear model a refit is feasible there; the dial matters for smaller hospitals, short run-ins, or models too large to refit on a few thousand patients.

### Dial versus recalibration (19 Oct 2026; E5 filter, sigma 1, gamma 2, seed 0)

Recalibration and per-site thresholds only rescale the output, so they cannot change the ranking of patients (AUC). Observation shift damages the ranking: patients who happen to be tested often at B are pushed up. Even an optimistic logistic recalibration (intercept and slope fitted on B's evaluation data itself) leaves AUC unchanged and barely helps Brier. The dial changes the ranking:

| | B_match AUC | B_match Brier | B_dense AUC | B_dense Brier |
|---|---|---|---|---|
| no adaptation (w = 1) | 0.643 | 0.1286 | 0.734 | 0.1204 |
| w = 1 + best recalibration | 0.643 | 0.1275 | 0.734 | 0.1166 |
| dial w = 0.5 | 0.721 | 0.1233 | 0.795 | 0.1117 |
| dial w = 0.3 | 0.750 | 0.1200 | 0.807 | 0.1092 |

(AUC at A, w = 1: 0.824.) This is the answer to 'per-site thresholds and recalibration already exist': they act on the output and cannot fix a shift that reorders patients.

### E5: a learned model with a separate testing head reaches the exact trade-off (18 Oct 2026; sigma 1, gamma 2, seed 0; `analysis/e5_summary.csv`, `figures/filter_sigma1_gamma2.png`)

A 32-state latent filter (learned dynamics, value head, testing head and outcome head; trained on A only, 908 s), tempered at deployment by scaling the testing-head evidence with w. Brier scores:

| | A | B_match | B_dense |
|---|---|---|---|
| perfect A model (exact, w = 1) | 0.1020 | 0.1282 | 0.1179 |
| exact blend, w = 0.5 | 0.1039 | 0.1192 | 0.1093 |
| plain GRU | 0.1031 | 0.1287 | 0.1277 |
| filter, w = 1 | 0.1024 | 0.1286 | 0.1204 |
| filter, w = 0.3 | 0.1041 | 0.1200 | 0.1092 |
| filter, w = 0 | 0.1144 | 0.1149 | 0.1076 |

- The tempered filter follows the exact curve closely on both B policies (figure). At the same error at A, it is within about 0.001 of the exact curve at B. Its w scale differs (w = 0.3 for the filter is about w = 0.5 for the exact blend), so w has to be chosen on the model's own curve, not copied from theory.
- Even untempered (w = 1), the filter beats the plain GRU at A and is much better under dense testing (0.120 against 0.128): it counts tests through a likelihood, so it does not extrapolate wildly beyond A's range of counts.
- The learned testing head does not match the true testing curve state by state (its states are arranged differently from the simulator's), yet the predictions are near-exact. This matters for re-learning the testing head at B, which assumes the rest of the model is identified well enough; to be tested.
- Expected in advance: the simulator is itself a latent-state model, so this is a proof of principle that the right lever works when learned, not evidence for real data.

### E4: product-of-experts discounting fails (18 Oct 2026; GRU, sigma 1, gamma 2, seed 0; `analysis/e4_summary.csv`, `figures/poe_sigma1_gamma2.png`)

The question was whether a standard shortcut-debiasing method can move a network onto the exact trade-off curve. Brier scores, for the exact rule, the plain GRU and the best debiased model:

| | A | B_match | B_dense |
|---|---|---|---|
| perfect A model | 0.1020 | 0.1282 | 0.1179 |
| exact blend, w = 0.5 | 0.1039 | 0.1192 | 0.1093 |
| plain GRU | 0.1031 | 0.1287 | 0.1277 |
| debiased GRU, best w (0.7) | 0.1077 | 0.1270 | 0.1382 |
| pattern-only bias model | 0.1063 | 0.1446 | 0.1821 |

- The method does not work. At B_match, its best point gives up 0.005 at A to save 0.002 at B. The exact curve saves 0.009 at B for 0.002 at A. At B_dense, every w is worse at B than the plain GRU.
- Why: the model that sees only who was tested already scores 0.106 at A, almost as good as the full model. When testing tracks sickness this strongly, the pattern and the values are two views of the same hidden state, not two separate signals. Product of experts assumes the shortcut is separate from the real signal, so subtracting the pattern removes most of what the main model would use (f alone: Brier 0.25). This is the redundancy problem again, from the other side.
- The exact curve is reachable only by discounting the pattern where it enters, as evidence about the hidden state. A model with an explicit observation process (state-space or neural filter with a testing likelihood that can be tempered) would do that by construction. Not tested.

### Explainer figure and a new observation (18 Oct 2026)

`figures/tradeoff_sigma1_gamma2.png` (script `figures/tradeoff_figure.py`) plots error at A against error at B for sigma 1, gamma 2 (3-seed means). The perfect model of A and its error at B form a dashed reference line; the gap to the model that ignores who is tested is the price of learning A. Made for the email to Sarah.
With `--curve`, the plot also shows the blend of the two exact rules (discounting the testing pattern by a factor w). This trade-off is very lopsided: near the perfect A model, discounting the pattern a little costs almost nothing at A and saves much of the price at B. The reason: A's error is flat at its optimum (second order), while B's error changes linearly (first order). This is general, not specific to this simulation. All four networks lie far above that curve. A possible project question: can mask-reliance regularisation reach the curve? Not tested yet.

### E3: exact redundancy (17 Oct 2026; 4 fits, seed 0, 8,000 test patients; `analysis/e3_summary.csv`)

Testing depends only on the last recorded value (sigma 1, gamma 2, rate 0.3), so the pattern adds nothing beyond the recorded history. The Bayes rule ignores it, and the justified transfer cost is exactly zero: every loss on B is unjustified. The V_rule's Brier is 0.117 on A, 0.115 on B_match and 0.107 on B_dense.

| arch | S on A | X on B_match [95% CI] | X on B_dense |
|---|---|---|---|
| GRU-D | 0.0014 | 0.012 [0.010, 0.014] | 0.005 |
| GRU | 0.0017 | 0.030 [0.026, 0.033] | 0.011 |
| Transformer | 0.0027 | 0.007 [0.005, 0.008] | 0.004 |
| CDE | 0.0035 | 0.006 [0.004, 0.007] | 0.010 |

- Under exact redundancy all four models still lose on B, by 0.006-0.030. That is 2-5x more than any excess in E1, where the pattern was informative and models tracked the transferred Bayes rule within about 0.002. So shortcut use appears precisely when A does not pin the rule down.
- The architectures differ five-fold, and the order is not the order of fit on A: GRU-D fits A best but loses twice as much as the Transformer and CDE on B_match, and GRU loses most. Under B_dense the CDE drops from best to worst. A ranking by B performance therefore ranks architectural priors for extrapolating off A's support, and that ranking depends on which shift B is.
- Caveats: one seed; the intervals cover patients, not training seeds. The linear shortcut probe (residual logit on count and gap) explains only 2-9% of the residual and is not informative here. The run took 18 min, not 10 (the CDE fit took 1,060 s).

### Full grid (17 Oct 2026; 108 fits, 3 seeds, 8,000 test patients; Brier; tables in `analysis/`, figures `analysis/*.png`)

Symbols (SPEC.md): G = Bayes mask gain on A; S = model's excess over the A-optimal rule on A; J = justified cost (A-optimal rule applied to B, minus B's optimum); X = model's B loss minus the transferred A-optimal rule's B loss (X > 0: loses more than perfect A learning would; X < 0: loses less); b_values, b_mask = probe weights (exact Bayes has 1, 1).

1. No substitution under a matched-rate policy change (B_match). GRU, GRU-D and Transformer have b_mask 0.9-1.13 and b_values 0.83-1.01, and X_match within about ±0.002 in every setting: they learn the pattern about as much as the data warrant and lose almost exactly the justified cost J (0.004-0.014). Their B loss is what perfect A learning would also suffer.
2. The CDE looks "robust" because it under-learns the pattern, which is the original objection to a naive robustness ranking, here measured. b_mask 0.44-0.67 in every setting, X_match negative (down to -0.0025), but the largest source excess S (0.003-0.010, three to ten times the others; it grows with sigma, where the pattern matters more).
3. Dense uniform testing (B_dense) is where models differ beyond the justified cost. GRU, GRU-D and CDE lose 0.002-0.010 more than the transferred A rule; they extrapolate badly to observation counts outside the training range. The Transformer loses less than the A rule at sigma ≥ 1 (X_dense -0.002 to -0.012) with a small S (0.002-0.003): attention pooling is roughly insensitive to the number of tokens, so "tested more often" does not push its prediction up as much. That is an architectural prior that happens to match this shift, not robustness learned from A.
4. Sampling coverage (E2, uninformative testing): every model's excess over Bayes grows from 0.001-0.003 on A to 0.004-0.006 on sparse or bursty sampling. GRU-D is best (0.0037-0.0042), the CDE gets no advantage from its continuous-time design (0.0049-0.0057).

Caveats: effects are small in absolute terms (base Brier about 0.10), because the pattern adds little beyond the values in this simulator; contrasts between architectures with bootstrap intervals are in `analysis/contrasts.csv` and have not yet been reviewed. Some Transformer fits stopped early (best epoch about 8); their S stays small, so finding 3 is not an under-fitting artefact.

### Exact Bayes reference (before the grid)

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

- E13 (which rung when) implemented by Luna (`tasks/e13/`, `out/e13/`), reviewed, run (about 3 h on the laptop) and analysed (23 Oct).

- E5 (learned latent filter with tempered testing head) implemented by Luna (`tasks/e5_filter.md`, `out/e5_filter.md`), reviewed (numpy filter reproduces Bayes to 4e-9; torch matches numpy), run and analysed (18 Oct).
- E4 (product-of-experts discounting) implemented by Luna (`tasks/e4_poe.md`, `out/e4_poe.md`), reviewed, run and analysed (18 Oct).
- E3 (exact redundancy) implemented by Luna (`tasks/e3_recorded.md`, `out/e3_recorded.md`), reviewed, run and analysed (17 Oct).
- 17 Oct 2026: full grid finished (108 fits, 215 min). Bug fixed in `run_grid.py` `_bayes_arrays`: it unpacked the seed tuple in the wrong order, so Bayes predictions were computed on the training cohort's latent paths, not the test patients. All 27 `bayes.npz` files recomputed on the right cohort (y arrays now match the model shards); analysis rerun. Local folder renamed to `DL - Observation Shift Project`.
- 17 Oct 2026: repository pushed to https://github.com/wjremy/DL-Observation-Shift-Project (private).
- 17 Oct 2026: smoke run and all tests passed with real PyTorch (feature builders, forward passes, empty histories, one-step loss reduction, end-to-end smoke). Bayes filter matches Kalman to 1e-6; grid resolution 241 vs 481 changes Brier by 2e-13.
- 17 Oct 2026: code written by two GPT-6 Luna agents from SPEC.md (briefs in `tasks/`, reports in `out/`), reviewed by Claude.
- 17 Oct 2026: design fixed in SPEC.md, following docs/ARGUMENT.md.

## Decisions

- 25 Oct 2026 (Wesley): after E12-E13 (head_U best at every run-in length), test the reframing "which kind of shift, not which rung": E14 simulates a shift only outcomes reveal (B's doctors test on a hidden risk factor) and its reverse; E15 runs head_U on real pairs with and without a pattern shift (E10), first run of the state-space model on real data.

- 22 Oct 2026 (Wesley): the method becomes a ladder of adaptations of the testing part (global dial, dial per test group, re-learn testing part, full refit), chosen by run-in size and kind of shift; answers the objection that one dial cannot capture all observation shift (docs/ARGUMENT.md section 10). Proposal to be rewritten in the course format around it.

- 22 Oct 2026 (Wesley): proposal (docs/proposal/proposal_trust_dial.tex), preliminary binary case now states that keeping the policy won for Beth Israel to Emory because there was barely any observation shift there (E10 A-all to B-all pattern gap +0.031, 95% CI -0.006 to +0.068). PDF rebuilt.

- 19 Oct 2026 (Wesley): primary transfer = same ICU type in another hospital (most realistic deployment); primary metric AUROC; E7/E7b count as exploratory; E8 tests realistic run-in sizes with a design fixed before running (SPEC.md E8).

- 19 Oct 2026 (Wesley): the project studies the dial only; adaptation without outcomes is future work (docs/FUTURE.md). Use open data without sign-in for more hospital pairs first; eICU later. Project map in docs/MAP.md.

- 17 Oct 2026: simulation first, not MIMIC: exact Bayes benchmarks exist only in simulation, and MIMIC needs credentialed access.
- 17 Oct 2026: models are measured against the exact A_rule and V_rule, not ranked by B score (docs/ARGUMENT.md, step 6).
- 17 Oct 2026: CPU only (laptop has Intel UHD 620, no CUDA); 4 workers with one thread each.
