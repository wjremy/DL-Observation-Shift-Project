# What is known, and the gap (19 Oct 2026)

Six GPT-6 Luna agents searched the literature in parallel with live web search (briefs `tasks/lit/`, full reports with all references `out/lit/`). Claude reviewed the reports and checked the decisive sources directly: titles, and for the two closest papers the abstracts, of Zhou 2023, Stokes 2025, Rockenschaub 2024, Futoma 2021, Jeanselme 2022, Alaa 2017, Carmona & Nicholls 2020 and Yang 2022 were confirmed on arXiv, PMLR, Crossref or the publisher page. A search finds what exists; it cannot prove that nothing else exists.

## Our claims

- C1: the raw drop from A to B confounds fit on A with robustness; judge models against the drop a Bayes-optimal model of A would also have.
- C2: the trade-off is lopsided: a small discount of the testing evidence costs second order at A and gains first order at B; partial trust beats both ends.
- C3: a deployment-time dial weighting the testing-process evidence in a latent-state model, set per hospital.
- C4: recalibration and thresholds cannot repair observation shift because they keep the ranking; the dial does.
- C5: how much local data the dial needs, compared with recalibration and full fine-tuning.
- C6: re-learning only the testing component on unlabelled data from the new hospital.

## Verdict per claim

| Claim | Status | Closest work and what it leaves open |
|---|---|---|
| C1 | partly known | Taori et al. 2020 (effective robustness) and Miller et al. 2021 (accuracy on the line) already warn that out-of-distribution scores confound source accuracy, and adjust empirically with a trend across models. Nobody we found uses the Bayes-optimal source model's own drop as the justified cost, and no clinical benchmark applies this to a changed testing policy. |
| C2 | mechanism known, application open | The local asymmetry follows from risk geometry (anchor regression, Rothenhäusler et al. 2021; semi-modular inference, Carmona & Nicholls 2020; Frazier & Nott 2026). For missingness, **Rockenschaub et al. 2024** show that ignorable shifts leave the Bayes predictor unchanged (our E3), and that under non-ignorable shifts ignoring missingness helped most when it was highly informative. **Futoma et al. 2021** found that fitting with measurement indicators and dropping them at deployment transferred better across hospitals. Both compare all or nothing; partial trust, and why it beats both ends, is not shown. Condition to state: the gain at B is first order only if discounting moves predictions toward B's optimum (true for a shift toward less selective testing, not for every shift). |
| C3 | ingredients known, combination apparently open | Tempering one component's influence is established (power likelihoods; cut models, Plummer 2015; semi-modular inference with a tunable influence parameter chosen by predictive score, Carmona & Nicholls 2020). Models with an explicit testing or observation process exist (Alaa et al. 2017; Lange et al. 2015; Rubanova et al. 2019; DeepJoint, Jeanselme et al. 2022, which is more robust to a weekday/weekend change in clinical presence). None weights the testing evidence with a per-hospital dial. Wording to avoid: "neural networks cannot have such a lever"; correct: "standard architectures do not expose one". |
| C4 | known as such | Recalibration and thresholds preserve ranking and AUC (Janssen et al. 2008; Yang, Soltan & Clifton 2022 across four NHS trusts; Guo et al. 2017). Our contribution is not this fact but showing that the dial restores ranking where they cannot (E5: AUC 0.64 to 0.75 at B). |
| C5 | framework known, comparison open | The updating ladder and its sample-size logic are established (Steyerberg et al. 2004; Vergouwe et al. 2017; Riley et al. 2020); silent trials exist (Kwong et al. 2022). No study measures the data needed for an observation-evidence weight against recalibration and fine-tuning. |
| C6 | partly known | Adaptation to missingness shift with unlabelled target data exists: Zhou, Balakrishnan & Lipton 2023 (linear models, specific missingness assumptions) and Stokes et al. 2025 (preprint; MNAR, imputation and weighting, two-hospital EHR). Neither refits only an observation-process component while holding physiology fixed. |

## Gap statement (draft for the proposal)

Clinical prediction models exploit informative testing patterns, and these patterns change between hospitals. Prior work either evaluates such shifts with raw performance drops, which reward models that under-fit the source, or chooses between using and ignoring missingness altogether (Futoma et al. 2021; Rockenschaub et al. 2024). Methods that adapt to missingness shift adjust the whole predictor (Zhou et al. 2023; Stokes et al. 2025), and per-site recalibration cannot repair it because it keeps the ranking of patients. Separately, Bayesian statistics offers ways to temper the influence of one model component (Carmona & Nicholls 2020), and clinical latent-state models can represent the testing process explicitly (Alaa et al. 2017). We bring these together: a per-hospital weight on the evidence from the testing process, motivated by the lopsided trade-off between source and target error, evaluated against the drop that a Bayes-optimal source model would also have, and set with the limited local data of a silent run-in.

## Before claiming novelty, check

- An OpenReview submission titled "Policy-Aware Pretraining for EHR as Clinician-in-the-Loop Dynamical Systems" (https://openreview.net/pdf?id=BcOC4WCXmQ) may model clinician observation and actions with cross-hospital invariance. OpenReview blocked automated access (verification challenge), so it is unread. Open it in a browser and read the abstract.
- Read Rockenschaub et al. 2024 and Futoma et al. 2021 in full: they are the closest precedent for using versus ignoring informative missingness under shift.
