# The argument behind the study

This file records the reasoning that led to the current design, in the order it developed (discussion of Sarah Martinson's proposal "Deep Time-Series Models under Observation Shift", 17 Oct 2026). Read it before changing the design: each step answers an objection to the one before.

## 1. The original objection

Hospitals differ in observation policy. In hospital A, sicker patients are tested more often, so the observation pattern predicts sickness. In hospital B, everyone is tested equally, so it does not. The proposal trains on A, tests on B and asks which architectures are more robust.

Objection: a model that learns A's policy is just a good ML model on A, and it loses on B because the rule no longer holds. A model that does not learn it looks "robust" only because it failed to learn a real signal. A model trained only on A cannot know the relationship breaks in B. So a robustness ranking measures how well the policy was learned, not robustness.

This is correct for a naive design that ranks models by B score alone.

## 2. Redundancy: which of two routes a model takes

In the proposal's setup, the latent state and its link to the outcome are the same in A and B; only P(observed | state) changes. In A there are two routes to the prediction: infer the state from the measured values (works in A and B), or infer it from how often the patient is tested (works only in A). If both routes give the same loss, the data cannot choose between them, and the architecture's inductive bias does (underspecification, D'Amour et al. 2020; shortcut learning, Geirhos et al. 2020).

Refinement from the discussion: under equal loss, the shift experiment measures the architectures' priors for using the observation pattern. The prior, not robustness, is then the object of study, and the shift is only a probe of it.

## 3. Optimisation on A cannot justify ignoring the pattern

A model cannot infer from A's data that it should not use the pattern elsewhere. Any robustness to an unknown shift comes from prior knowledge built in by the modeller (inputs, architecture, regularisation), never from optimisation on the training distribution. That is true of all research on distribution shift. Whether an architecture's behaviour is a systematic prior or luck is testable: a prior replicates across seeds and data-generating settings.

## 4. Two different effects inside "observation shift"

- Informative observation: whether being measured says something about the state (dependence of P(observed) on the state).
- Sampling coverage: how dense and regular measurements are, independent of the state.

A policy change usually shifts both. They need separate experiments. Coverage is a clean architecture question, independent of any redundancy assumption.

## 5. Exact redundancy almost never holds

With noisy values and state-dependent testing, not being tested carries information about the state at times when there is no value. So the pattern almost always adds some information beyond the values, and a model fit well to A should use it to some degree. "Equal loss" is therefore not a usable assumption.

## 6. Resolution: measure against the exact Bayes rules

The problem is not that a model uses the pattern's extra information (justified), but that it may use the pattern instead of the values, beyond what the information warrants. In simulation the exact Bayes predictor is computable, which gives the benchmark:

- A_rule: the exact Bayes predictor under A. Applied unchanged to B, its loss increase is the justified transfer cost J_B: the price of learning A perfectly.
- V_rule: the exact Bayes predictor that treats observation as uninformative. It is optimal under B.
- A model whose B loss exceeds the A_rule's B loss (excess X_B > 0) relies on the pattern more than the data justify: a real, architecture-driven flaw. A model with lower B loss than the A_rule must be worse than optimal on A (source excess S > 0): under-learning.
- The Bayes mask gain G = R_A(V_rule) - R_A(A_rule) measures how redundant values and pattern are; the study varies it through the measurement noise sigma.

This turns objection 1 into the baseline every model is held to, and it needs no redundancy assumption. On real data (e.g. MIMIC-IV), where Bayes rules are unknown, the comparable report is the gain from adding masks on A against the cost on B.

## 7. Does redundancy save the robustness ranking? (E3)

Redundancy changes what a B-ranking measures; it does not make it a measure of robustness learned from A. If the pattern is exactly redundant, A does not decide how much a model uses it: many rules fit A equally well, and they differ only on histories that never occur under A. B shows how each architecture extrapolates there, i.e. its prior. E3 builds this case exactly (testing driven by recorded values only, missing at random): the justified cost is zero, yet models lose 0.006-0.030 on B, with a five-fold spread whose order differs from fit on A and between B_match and B_dense. So under redundancy, Sarah's design detects real shortcut use, larger than in the informative case. However, it ranks priors for a specific shift, not robustness. Two further limits remain. The redundancy needed fails in her motivating case (clinicians test because the patient looks sick, which is not in the record: E1, not E3). And on real data the same B drop means opposite things in the two worlds (justified cost in E1, shortcut in E3), and the B data cannot tell them apart. Only the Bayes benchmark, available in simulation, separates them.

## 8. The trade-off between A and B is lopsided (18 Oct 2026)

Blending the two exact rules, logit p_w = logit p_V + w (logit p_A - logit p_V), traces a curve from the perfect model of A (w = 1) to the model that ignores who is tested (w = 0). The curve is strongly bent. At sigma 1, gamma 2 and B_match, w = 0.5 costs 0.002 Brier at A and saves 0.009 at B. With A and B weighted equally, w around 0.4 beats both ends (sum 0.223 against 0.230 and 0.229). Figure: `figures/tradeoff_sigma1_gamma2_curve.png`.

Why, in general: Brier error = unavoidable error + mean squared distance to the ideal prediction. Each hospital has its own ideal, so the extra error at each grows with the square of the distance to its ideal. Halfway, each costs about a quarter, not a half. At A's optimum, A's error is flat (second order) while B's error changes at first order, so a small discount always pays as soon as any shift toward less selective testing is plausible.

The same holds for clinical decisions, not only for Brier. With threshold decisions (cost-weighted: a missed case costs 1 - t, a false alarm t), the middle of the curve again beats both ends at t = 0.1, 0.2 and 0.3. At t = 0.2, in missed-case equivalents per 1,000 patients, the total extra harm over A and B is about 10 at w = 0.4, against 27 at w = 1 and 23 at w = 0. The reason: discounting the pattern a little flips decisions only for patients near the threshold. At A those patients are near indifference, so little is lost; at B the same histories mean a clearly different risk, so clearly wrong decisions are fixed.

Requirement: the dial must scale only the evidence carried by being tested. A black-box network has no such lever, and product of experts (E4) supplies the wrong one: it scales everything the pattern-only model knows, which overlaps with what the values say. A model with an explicit testing head has the right lever by construction. E5 confirms this when the model is learned from A's data: tempering its testing evidence follows the exact curve to within about 0.001 Brier at B.

## 9. Practical consequence: separate patient physiology from hospital practice (18 Oct 2026)

Practical scenario: a sepsis or deterioration alert developed at hospital A and deployed at hospitals the developer does not know. The Epic Sepsis Model is the documented case: external validation showed much weaker performance than reported, and much of its signal came from clinician actions (cultures, antibiotics), i.e. from the clinicians' suspicion, which differs between hospitals (see `docs/LITERATURE.md`). Deployment itself also changes testing, because clinicians order tests after alerts, so the observation process shifts even within B.

What to do depends on what B can provide:

| Situation | Action |
|---|---|
| B's records without outcomes (the usual silent period before go-live) | Re-learn only the testing head at B by maximum likelihood on its unlabelled records; keep dynamics, value head and outcome head from A. No dial needed, and the direction of the shift (more or less selective) is learned too. |
| No data from B | Discount the testing evidence: choose w by average or worst-case error over a stated range of plausible testing practices, calibrated with variation within A (wards, shifts, weekdays, years) and scaled up by an explicit assumption, since differences between hospitals are probably larger than within one. |
| After go-live | Monitor and re-learn the testing head periodically, because the alert changes testing practice. |

Ordinary training on all of A does not set the dial. It fits the mix of A's groups and may learn group-specific rules, but it never sees a group with unseen practice. Holding out one group at a time mimics that.

Limit: re-learning without labels assumes physiology (dynamics and value head) transfers. If B's patients or equipment differ, the value head must change too, and that needs labels.

Proposed project question: can a model separate patient physiology from hospital testing practice, so that it can be adapted to a new hospital without outcome labels, and how much should it trust the testing pattern when no data from the new hospital exist?

Literature: anchor regression (a dial between best fit and invariance, chosen by expected shift strength), invariant prediction, distributionally robust optimisation, and the EHR literature on informative measurement processes. See `docs/LITERATURE.md`.

## 10. From one dial to a ladder (22 Oct 2026)

A teammate's objection: can one number capture observation shift? Not in general. A single dial scales all testing terms together, so it repairs a shift in how much the testing pattern should count overall (B tests everyone more, so every order means a bit less), but not a shift in which tests carry meaning. E10 found the second kind in several 2012 mortality pairs, and E6 shows a real example: troponin is ordered for 3% of Beth Israel patients but 32% at Emory, so "troponin ordered" loses meaning there while "lactate ordered" may keep it.

Resolution (Wesley, 22 Oct): a ladder of adaptations of the testing part only, from cheap to flexible: (1) one global dial; (2) one dial per test group; (3) re-learn the testing part; (4) full refit. Each rung repairs more kinds of shift but needs more outcome cases, and a silent run-in yields few (about 2-6 sepsis cases in 1-3 months for one ICU). The question becomes which rung a run-in of a given size supports, for which kind of shift. The dial stays as the low-data default, now justified by the comparison rather than assumed.

## Earlier Codex runs (17 Oct 2026)

Two studies were run earlier the same day in the Codex app; their reports are in `docs/prior_codex_runs/`. The policy-learning study fixed the physiology predictor for every model and learned only the observation correction. It showed that even the exact source rule loses under B (Brier 0.082 on A to 0.246 on B, against 0.163 without the cue) and that GRU, GRU-D and CDE recovered 97-98% of the source cue benefit. Because physiology was fixed, it could not test step 6 (using the pattern instead of the values); this study does.
