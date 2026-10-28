# Project map (read this first; updated 28 Oct 2026)

How the project got here, what it now claims, where each piece of evidence lives, and what comes next. STATUS.md holds the running log and full results; this file is the overview a new session or teammate needs to continue.

## 1. The question now (26 Oct 2026, proposal v3)

**Re-learning the testing policy without outcome labels.** A latent state-space model (discrete patient state; initial distribution and transition; three heads: value, testing, outcome) is trained at hospital A. At hospital B:
- **Label-free step (main contribution):** re-learn only the testing head from B's records (values and test orders), everything else frozen. Needs no outcomes, so B's historical records suffice before deployment.
- **Outcome ladder on top (secondary):** fitted on the run-in's outcome cases, ordered by cases needed: a weight on the testing evidence, a weight per test group, the outcome head, the full model (penalised towards the label-free fit).
- **Baselines:** keep and remove the testing evidence (Futoma), recalibration, GRU-D (unchanged, last layer, full fine-tune; its testing part cannot be separated).
- **Research question:** can the label-free step correct observation shift, and for which kinds of shift? Secondary: where it cannot, how many outcome cases does each rung need? If the final 5-page paper needs cuts, the label-free step is the focus.
- Key distinction (docs/ARGUMENT.md section 11): a shift is fixable without labels when B's testing depends differently on what the model can see; it needs outcomes when B's testing carries information the model never sees.

Proposal: `docs/proposal/overleaf/main.tex` (Overleaf source, ethdl2026 style), `docs/proposal/proposal_v3.pdf`, diff against the team's first version `docs/proposal/proposal_v3_diff.pdf`.

History of the question (kept for context):

**A trust dial for observation shift (19 Oct).** A model trained at hospital A learns A's testing policy (which tests are ordered, and when). At hospital B the policy differs. We give the model one weight w on the evidence from the testing pattern (w = 0 ignore it, w = 1 trust it as at A) and set w per hospital on the small labelled sample of a silent run-in.

Update (Wesley, 22 Oct): the dial becomes the bottom rung of a ladder of adaptations of the testing part (global dial, dial per test group, re-learn the testing part, full refit), and the question is which rung a run-in of a given size supports (docs/ARGUMENT.md section 10).

Scope decision (Wesley, 19 Oct; superseded on 22 and 26 Oct): the project studies the dial only. Adaptation without outcomes and the full adaptation ladder are future work (docs/FUTURE.md).

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
14. 22 Oct: teammate critique ("one dial is too simple") -> ladder of adaptations of the testing part; proposal moved to the course template (docs/proposal/overleaf/); state-space model kept, two-branch model dropped; E2 dataset to be determined.
15. E12/E12b (23 Oct, simulation): re-learning the testing head from B's records without outcomes (head_U) reaches near-optimal predictions from 200 unlabelled patients under all four testing shifts; the labelled head refit overfits small run-ins.
16. E13/E13b (23-24 Oct): over 1-104 weeks of deployment head_U is best from week 1 under every testing shift; a dial on top adds nothing (w stays about 0.95); full fine-tuning is worst at every week up to 2 years at any learning rate (lr 0.05 harms early, lr 0.001 learns nothing). No "staircase" of rungs.
17. E14 (25 Oct): a shift only outcomes reveal (B's doctors also test on a hidden risk factor): head_U neither helps nor harms, light labelled rungs do nothing, full fine-tuning recovers half the loss only after 2 years. The reverse (A tested on the factor, B does not) is harmless.
18. E14b (26 Oct): a dial allowed above 1 cannot fix that shift either, even with w chosen on the test cohort (about 3% of the loss).
19. E15 (25 Oct, real data, first run of the multivariate state-space model): stopped at the sanity gate; the model is 0.06-0.16 AUROC below E7's linear model on 5 of 6 sources, even without the value likelihood; cause: outcome read only from 32 discrete states, no static covariates (age etc.). Waiting for a model decision.
20. 26 Oct: proposal v3 restructured around the label-free step (section 1), handed to the team.

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
| Label-free testing-head refit fixes testing shifts (simulation) | E12/E12b, E13 | STATUS.md E12, E13; analysis/e12_*, analysis/e13_*; figures/e13_rungs_over_time.png |
| Fine-tuning does not pay off within realistic run-ins | E13, E13b (learning rates) | analysis/e13b_quick_runs.csv |
| Shifts only outcomes reveal cannot be fixed within 2 years | E14, E14b | analysis/e14_*, analysis/e14b_runs.csv, figures/e14_meaning_shift.png |
| State-space model not yet good enough on real data | E15 source gate, NLL-weight check | analysis/e15_sources.csv, out/e15/e15_train.log, out/e15/e15_lambda_check.log |
| Architectures, partial adaptation, novelty of the ladder | literature reports lit12-14 | docs/GAP.md, out/lit/lit12-14_*.md |

## 4. Files

| Area | Files |
|---|---|
| Overview and log | docs/MAP.md (this), STATUS.md, README.md |
| Design | SPEC.md (E1-E6 specifications), docs/ARGUMENT.md |
| Literature and gap | docs/GAP.md (current), docs/LITERATURE.md (superseded), out/lit/ (agent reports), tasks/lit*/ (briefs) |
| Simulation code | sim.py, bayes.py, models.py, run_grid.py, run_poe.py, latent_filter.py, analyze*.py, test_*.py |
| Real-data code | data/fetch_physionet2019.py, e6_cohort.py, e6_dial.py, figures/e6_figure.py (cache data/e6_cache.npz and raw data are gitignored; rerun the fetch script and e6_cohort.py) |
| Ladder experiments (simulation) | e12_unlabelled_head.py, e13_rungs_over_time.py, e13b_finetune_lr.py, sim_u.py (hidden-risk-factor simulator), e14_meaning_shift.py, e14b_dial_above1.py, figures/e13_figure.py, figures/e14_figure.py |
| Real pairs with the state-space model | e15_sequences.py (hourly cache data/e15_cache.npz, gitignored), latent_filter_mv.py (multivariate model), e15_real_pairs.py, figures/e15_figure.py |
| Proposal | docs/proposal/overleaf/ (Overleaf source and style files), proposal_v1_main.tex / proposal_v2_main.tex (earlier versions), proposal_v3.pdf, proposal_v3_diff.pdf |
| Communication | docs/proposal_post_2026-10-19.md (group post), docs/meeting_2026-10-19/observation_shift_deck.html (deck) |
| Future work | docs/FUTURE.md |

## 5. Where it stands (28 Oct 2026) and next steps

Current story (E12-E15, STATUS.md):
- In simulation, the label-free testing-head refit is the best adaptation from week 1 for every shift in how B's testing follows the visible patient state; outcomes add nothing on top (dial w about 1), and full fine-tuning is worse for up to 2 years.
- A shift where B's tests carry information the model never sees is not fixable by any light rung, nor by a dial above 1; only full fine-tuning helps, after about 2 years (half the loss). So the kind of shift, not the run-in size alone, decides what helps.
- The dial (global) has no simulated case where it helped after the label-free step; the per-test-group dial and the outcome-head rung are untested (need a multi-test simulator and a risk-shift simulation).
- On real data the multivariate state-space model is still too weak (E15 gate failed); the pair experiment waits for a model fix.

Next steps (28 Oct):
1. E15 model fix (needs Wesley's decision): outcome head reading the full filtered belief plus static covariates, small NLL weight, standardisation on the training split only; retrain, recheck the gate (within 0.05 AUROC of E7's linear model), then run the 8 pairs.
2. Simulations still open: a risk shift (same state, different outcome risk at B) where the outcome-head rung should win; a multi-test simulator for the per-group dial; fine-tuning started from the label-free fit and penalised towards it.
3. Architecture comparison with GRU-D (and possibly GRU, Neural CDE, Transformer), as the teammate asked.
4. E2 dataset still open (eICU needs credentialed access).

Earlier story (E6-E11, 20 Oct):
- Testing rates differ a lot between sites, but for sepsis the testing pattern's meaning largely transfers; physiology shifts more (E10). Likely reason: the Sepsis-3 label is built from testing (cultures, SOFA labs), see docs/GAP.md. For mortality the pattern's meaning does shift in several ICU pairs (E10).
- Naive tuning of the dial on a realistic run-in (log loss, few cases) makes ranking worse (E8). Safe tuning (separate recalibration, shrink w towards 1) does no harm (E9).
- On the 2012 mortality pairs the safe dial gains where the pattern shifted, up to +0.02-0.03 AUROC with longer run-ins, and is neutral elsewhere (E11, after removing a pH = 97 data error; the pre-specified result was -0.009 because of that record).
- Framing for the group: the dial is a very localized refit (one parameter, the trust in the testing pattern), which needs far fewer outcome cases than refitting a model. Answer to "a scalar does nothing in a deep net": the dial sits on the structure (the separate testing term of a deep state-space model), not on the neurons; E5 showed a learned model built this way follows the optimal trade-off.

Next steps as of 20 Oct (superseded by the list above):
1. Recalibration with outlier protection (clip logits or remove impossible values) as part of the method.
2. Simulation: map when the dial pays off by shift size and run-in size, and reproduce the E8 tuning failure.
3. Neural version (E5 latent-state model) on real data.
4. eICU (credentialed; 208 hospitals) for a clean multi-hospital confirmation, mortality outcome.
5. Proposal deadline 25/26 Oct 2026 (docs/proposal/, docs/proposal_post_2026-10-19.md; update with E8-E11).

## 6. Resuming on another machine or account

- Clone the repository (GitHub `wjremy/DL-Observation-Shift-Project`); everything except raw data and caches is in git.
- Data: `python data/fetch_physionet2019.py` and `python data/fetch_physionet2012.py` (both open, no account). Caches rebuild with `python e6_cohort.py`, `python e7_sites.py` and `python e15_sequences.py` (about 5 min); each later script (e8_runin.py ... e11_p12_runin.py, e10_diagnostic.py) reruns in minutes. Simulation A models are cached under results/ (gitignored); E13/E14 retrain them on first run (20-30 min each).
- Python 3.12 with numpy, pandas, matplotlib (torch only for the simulation models). LaTeX (MiKTeX) for docs/proposal.
- Code-writing agents: GPT-6 Luna through the user-level skill `codex-agents` (in `~/.claude/skills/codex-agents`, not in this repository); on another machine either install that skill or write code directly. Claude's own memory is per account; this file, STATUS.md, SPEC.md and CLAUDE.md hold everything needed.

## 7. Working rules

See CLAUDE.md: Claude specifies and reviews; GPT-6 Luna agents (Codex, xhigh) write code; quick runs by default; record every change in STATUS.md and push after each step.
