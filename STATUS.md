# Status

Update this file with every change: what started, what finished, what was decided. Newest first within each section.

## Running now

Nothing.

## Open

- Team: everyone applies for PhysioNet credentialing (CITI course plus approval, 1-2 weeks) for eICU-CRD; see docs/DATA.md.

- Review `analysis/contrasts.csv` (bootstrap intervals for architecture differences) and the figures.
- Possible quick follow-ups (minutes, not a grid): stronger informativeness (gamma 3-4) to enlarge the effects; a CDE with longer training to test whether its under-learning of the pattern is a prior or slow convergence.
- Real-data arm: PhysioNet Challenge 2019 sepsis data (open access, CC BY 4.0, 42 MB, two hospital systems A and B). Proposed: train the structured model at A, re-learn the testing head on B's unlabelled records, evaluate on B's labels. Downloaded 18 Oct (docs/DATA.md). Group meeting deck (19 Oct): https://claude.ai/artifact/CtdgYzJg2qEK2n7Y7pX6nE
- Literature search done (19 Oct): `docs/GAP.md`. Closest papers (Futoma 2021, Rockenschaub 2024, CLIO) read in full and summarised there.
- Earlier optional item: real-data arm (MIMIC-IV). Needs credentialed PhysioNet access, which a team member must apply for; on real data report mask gain on A against cost on B (no Bayes rules available).

## Results so far

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

- E5 (learned latent filter with tempered testing head) implemented by Luna (`tasks/e5_filter.md`, `out/e5_filter.md`), reviewed (numpy filter reproduces Bayes to 4e-9; torch matches numpy), run and analysed (18 Oct).
- E4 (product-of-experts discounting) implemented by Luna (`tasks/e4_poe.md`, `out/e4_poe.md`), reviewed, run and analysed (18 Oct).
- E3 (exact redundancy) implemented by Luna (`tasks/e3_recorded.md`, `out/e3_recorded.md`), reviewed, run and analysed (17 Oct).
- 17 Oct 2026: full grid finished (108 fits, 215 min). Bug fixed in `run_grid.py` `_bayes_arrays`: it unpacked the seed tuple in the wrong order, so Bayes predictions were computed on the training cohort's latent paths, not the test patients. All 27 `bayes.npz` files recomputed on the right cohort (y arrays now match the model shards); analysis rerun. Local folder renamed to `DL - Observation Shift Project`.
- 17 Oct 2026: repository pushed to https://github.com/wjremy/DL-Observation-Shift-Project (private).
- 17 Oct 2026: smoke run and all tests passed with real PyTorch (feature builders, forward passes, empty histories, one-step loss reduction, end-to-end smoke). Bayes filter matches Kalman to 1e-6; grid resolution 241 vs 481 changes Brier by 2e-13.
- 17 Oct 2026: code written by two GPT-6 Luna agents from SPEC.md (briefs in `tasks/`, reports in `out/`), reviewed by Claude.
- 17 Oct 2026: design fixed in SPEC.md, following docs/ARGUMENT.md.

## Decisions

- 19 Oct 2026 (Wesley): primary transfer = same ICU type in another hospital (most realistic deployment); primary metric AUROC; E7/E7b count as exploratory; E8 tests realistic run-in sizes with a design fixed before running (SPEC.md E8).

- 19 Oct 2026 (Wesley): the project studies the dial only; adaptation without outcomes is future work (docs/FUTURE.md). Use open data without sign-in for more hospital pairs first; eICU later. Project map in docs/MAP.md.

- 17 Oct 2026: simulation first, not MIMIC: exact Bayes benchmarks exist only in simulation, and MIMIC needs credentialed access.
- 17 Oct 2026: models are measured against the exact A_rule and V_rule, not ranked by B score (docs/ARGUMENT.md, step 6).
- 17 Oct 2026: CPU only (laptop has Intel UHD 620, no CUDA); 4 workers with one thread each.
