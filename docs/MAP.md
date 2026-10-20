# Project map (read this first; updated 19 Oct 2026)

How the project got here, what it now claims, where each piece of evidence lives, and what comes next. STATUS.md holds the running log and full results; this file is the overview a new session or teammate needs to continue.

## 1. The question now

**A trust dial for observation shift.** A model trained at hospital A learns A's testing policy (which tests are ordered, and when). At hospital B the policy differs. We give the model one weight w on the evidence from the testing pattern (w = 0 ignore it, w = 1 trust it as at A) and set w per hospital on the small labelled sample of a silent run-in.

Scope decision (Wesley, 19 Oct): the project studies the dial only. Adaptation without outcomes and the full adaptation ladder are future work (docs/FUTURE.md).

Claims (wording as in the group post, docs/proposal_post_2026-10-19.md):
1. Keep or remove the learned policy depends on the target hospital, so it should be set per hospital from run-in data, not fixed by a general rule (Futoma et al. 2021 used a general rule).
2. Whether the best setting lies strictly between 0 and 1, which a binary choice cannot reach. Theory and simulation predict it; still to be shown on real hospitals.
3. How many run-in cases each option needs (binary choice, dial, full refit), compared with the cases a sepsis run-in yields.

## 2. How we got here (one line per step)

1. Started from Sarah Martinson's proposal: rank GRU, GRU-D, Neural CDE, Transformer by their drop from A to B.
2. Objection (docs/ARGUMENT.md): the drop mostly measures how well a model learned A; judge it against the drop of the exact Bayes model of A. Simulation grid E1/E2 and redundancy test E3 confirmed it.
3. The trade-off between error at A and at B is lopsided (flat at A's optimum, first order at B), so partial trust can beat both ends (figures/tradeoff_*.png).
4. A product-of-experts discount fails (E4); a latent-state model with a separate testing head reaches the exact trade-off when its testing log-likelihood is scaled by w (E5). Recalibration keeps the ranking; the dial changes it (AUC 0.64 to 0.75 at B in simulation).
5. Literature (docs/GAP.md): prior work compares only all-or-nothing (Futoma 2021, Rockenschaub 2024); nobody tunes a per-hospital weight or measures its data needs.
6. Real data E6 (PhysioNet 2019, Beth Israel to Emory, linear model): best w near 1; removing the pattern costs 0.05 AUROC; one-parameter update beats a full refit up to about 20 cases, refit wins from about 50.
7. Run-in literature: sepsis silent runs last about 6 weeks to 3 months; one ICU yields about 2-6 sepsis cases in 1-3 months (docs/GAP.md).
8. E7 (24 open site pairs): the best w is spread over 0 to above 1 (interior in 10 of 24); a tuned dial beats the better end in 13 of 24, small gains, partly calibration (STATUS.md E7).
9. E7b: the large gains above w = 1 were mostly calibration. E8 (pre-specified, realistic unit run-ins, AUROC): no tuning method beats keeping the model as trained; the dial with recalibration is the least harmful and beats the binary choice (STATUS.md E8).
10. E9 (exploratory, corrected tuning): shrinking w towards 1 makes tuning safe (within 0.002 AUROC of the trained model) but no method gains at realistic single-ICU run-ins; gains only after 1-2 years in the medical-ICU pairs.
11. E10 diagnostic: in the realistic sepsis pairs physiology shifted more than the testing pattern's meaning; the pattern's meaning shifted clearly in 7 of 26 pairs, mostly the 2012 mortality pairs. Literature: sepsis labels are built from testing (cultures, SOFA labs), which plausibly keeps the pattern's meaning stable (docs/GAP.md).
12. E11 (2012 mortality pairs, realistic run-ins): pre-specified mean -0.009, caused by one pH = 97 record; with impossible values removed the safe dial gains +0.0015 on average, up to +0.02-0.03 where the pattern's meaning shifted, and never harms elsewhere.
13. Need more hospital pairs to test claim 2. eICU (208 hospitals) needs credentialed access; Wesley wants open data without sign-in first (search running or done: out/lit/lit10_open_datasets.md).

## 3. Evidence map

| Claim or fact | Evidence | Files |
|---|---|---|
| Raw drop confounds fit at A with robustness | E1-E3 simulation with exact Bayes rules | STATUS.md (E3, full grid), analysis/e1-e3_*, docs/ARGUMENT.md |
| Lopsided trade-off, middle beats ends (simulation) | exact blend curves; E5 learned filter | figures/tradeoff_*.png, figures/filter_sigma1_gamma2.png, analysis/e5_summary.csv |
| Recalibration cannot replace the dial | E5 recalibration comparison | STATUS.md "Dial versus recalibration" |
| Best end depends on the target | Futoma (Memphis: remove) vs E6 (Emory: keep) | docs/GAP.md, STATUS.md E6 |
| Dial vs refit by run-in size | E6 run-in experiment | analysis/e6_runin.csv, figures/e6_two_hospitals.png |
| Cases a run-in yields | ICU volumes, silent deployments, updating literature | docs/GAP.md, out/lit/lit7_silent_trials.md, out/lit/lit8_runin_case_counts.md |
| Novelty and closest work | literature reports, papers read in full | docs/GAP.md, out/lit/lit1-6 |
| Datasets | PhysioNet 2019 (have); eICU and others | docs/DATA.md, out/lit/lit9_datasets.md, out/lit/lit10_open_datasets.md |

## 4. Files

| Area | Files |
|---|---|
| Overview and log | docs/MAP.md (this), STATUS.md, README.md |
| Design | SPEC.md (E1-E6 specifications), docs/ARGUMENT.md |
| Literature and gap | docs/GAP.md (current), docs/LITERATURE.md (superseded), out/lit/ (agent reports), tasks/lit*/ (briefs) |
| Simulation code | sim.py, bayes.py, models.py, run_grid.py, run_poe.py, latent_filter.py, analyze*.py, test_*.py |
| Real-data code | data/fetch_physionet2019.py, e6_cohort.py, e6_dial.py, figures/e6_figure.py (cache data/e6_cache.npz and raw data are gitignored; rerun the fetch script and e6_cohort.py) |
| Communication | docs/proposal_post_2026-10-19.md (group post), docs/meeting_2026-10-19/observation_shift_deck.html (deck) |
| Future work | docs/FUTURE.md |

## 5. Next steps

1. Review the open-dataset search (out/lit/lit10_open_datasets.md) and pick at least three open hospital or unit pairs.
2. Quick option already in hand: PhysioNet 2019 has ICU-unit columns (Unit1 medical, Unit2 surgical), giving hospital-by-unit sites within the open data.
3. Extend E6 to the new pairs: per target, the w curve (where is the optimum, how much does it beat the better end) and the run-in comparison (choose the end, choose w, refit).
4. Neural version: the latent-state model of E5 on real data (later).
5. Proposal deadline 25/26 Oct 2026.

## 6. Working rules

See CLAUDE.md: Claude specifies and reviews; GPT-6 Luna agents (Codex, xhigh) write code; quick runs by default; record every change in STATUS.md and push after each step.
