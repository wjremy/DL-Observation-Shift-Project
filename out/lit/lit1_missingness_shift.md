*Overlap is relative to C1, C3, and C6. I found no paper that fully matches that package.*

| Paper (authors, year, venue) | URL/DOI | What it does (1 sentence) | Claims touched | Overlap |
|---|---|---|---|---|
| Zhou, Balakrishnan & Lipton, 2023, AISTATS | [PMLR](https://proceedings.mlr.press/v206/zhou23b.html) | Formalizes domain adaptation when source and target differ in missingness; under specific missingness assumptions, derives a target adjustment for linear predictors using labeled source and unlabeled target data. | C1, C6 | partial |
| Stokes et al., 2025, arXiv preprint | [arXiv:2504.00322](https://arxiv.org/abs/2504.00322) | Reduces MNAR missingness-shift adaptation to imputation and domain adaptation, and illustrates it with EHR data from two hospitals. | C1, C6 | partial |
| Futoma et al., 2021, *Critical Care Explorations* | [DOI](https://doi.org/10.1097/CCE.0000000000000453) | Across three cohorts at two hospitals, measurement-recency indicators alone generalized poorly, while using them during fitting and omitting them at evaluation helped physiology-based models transfer. | C3 | partial |
| Lange et al., 2015, *Biometrics* | [DOI](https://doi.org/10.1111/biom.12252) | Jointly models latent disease states with state-dependent observation times using a hidden continuous-time Markov process and a Markov-modulated Poisson observation process. | C3 | partial |
| Sisk et al., 2021, *JAMIA* | [DOI](https://doi.org/10.1093/jamia/ocaa242) | Reviews 36 methods for informative presence and observation, including measurement indicators, HMMs, and joint models of outcomes and observation processes. | C3 | partial |
| Subbaswamy, Adams & Saria, 2021, AISTATS | [PMLR](https://proceedings.mlr.press/v130/subbaswamy21a.html) | Estimates worst-case model performance under user-specified shifts in conditional data-generating mechanisms, including clinical-practice shifts. | C1 | partial |
| Subbaswamy & Saria, 2020, arXiv preprint | [arXiv:2002.08948](https://arxiv.org/abs/2002.08948) | I-SPEC uses a learned causal graph to select predictions stable to specified mechanism shifts, including in a healthcare mortality task. | C1 | partial |
| Calderon et al., 2024, Findings of EMNLP | [ACL Anthology / DOI](https://doi.org/10.18653/v1/2024.findings-emnlp.7) | Proposes “Target Drop” alongside the usual source-based performance drop, showing in NLP that a large source drop can reflect a harder target domain. | C1 | partial |
| Wong et al., 2021, *JAMA Internal Medicine* | [DOI](https://doi.org/10.1001/jamainternmed.2021.2626) | Externally evaluates the widely deployed Epic Sepsis Model and reports weak discrimination and calibration in the evaluated hospital. | — | none-but-related |
| Kamran et al., 2024, *NEJM AI* | [DOI](https://doi.org/10.1056/AIoa2300032) | Evaluates Epic sepsis predictions relative to treatment and clinical recognition; AUROC falls from 0.62 to 0.47 when predictions after recognition are excluded. | — | none-but-related |

## Closest work

- **Zhou et al.** is closest to C1/C6: it studies missingness shift with unlabeled target data and compares source- and target-optimal predictors. Its main results concern linear models and specific missingness assumptions; it does not establish the proposed clinical “justified drop” metric or isolate a state-dependent measurement model.
- **Stokes et al.** is the closest clinical adaptation paper: it addresses MNAR missingness and uses two-hospital EHR data. It adapts through imputation and weighting, not by holding a physiology model fixed while refitting only an observation-process component; it is a preprint.
- **Lange et al.** is closest to C3’s model structure: latent disease state and observation intensity are modeled jointly. It is not a prediction-transfer method and does not introduce a per-site weight on the observation likelihood.

## Verdict per claim

- **C1 — partly known:** Target Drop and missingness-shift excess-risk analyses address shortcomings of raw source-to-target drops, but I found no paper using the specific drop of an A-optimal model as the justified-cost baseline.
- **C3 — partly known:** Explicit latent-state and observation-process models exist, as do methods that use measurement indicators. I found no verified work with the proposed per-site scalar weighting the observation-process log-likelihood.
- **C6 — partly known:** Zhou et al. and Stokes et al. adapt under missingness shift using target data, but I found no verified method that refits only the observation-process component on unlabeled target data while fixing transferable physiology.

## Search terms used

“domain adaptation under missingness shift”; “Domain Adaptation Under MNAR Missingness”; “informative observation” and “measurement indicator” with EHR and external validity; “joint disease observation process” and latent state; “shift-stable models” with Subbaswamy and Saria; Epic sepsis treatment timing with Kamran and Wong; “Source Drop” and “Target Drop.”