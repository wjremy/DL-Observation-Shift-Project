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

## Earlier Codex runs (17 Oct 2026)

Two studies were run earlier the same day in the Codex app; their reports are in `docs/prior_codex_runs/`. The policy-learning study fixed the physiology predictor for every model and learned only the observation correction. It showed that even the exact source rule loses under B (Brier 0.082 on A to 0.246 on B, against 0.163 without the cue) and that GRU, GRU-D and CDE recovered 97-98% of the source cue benefit. Because physiology was fixed, it could not test step 6 (using the pattern instead of the values); this study does.
