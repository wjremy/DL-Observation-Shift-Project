| Paper (authors, year, venue) | URL/DOI | What it does (1 sentence) | Claims touched | Overlap |
|---|---|---|---|---|
| *Latent ODEs for Irregularly-Sampled Time Series* (Rubanova et al., 2019, NeurIPS) | [Proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf) | Models observation-time probabilities with Poisson processes alongside latent ODE dynamics, but does not study hospital adaptation. | C3 | partial |
| *Learning from Clinical Judgments* (Alaa et al., 2017, ICML) | [Proceedings](https://proceedings.mlr.press/v70/alaa17a.html) | A latent semi-Markov clinical state modulates Hawkes-process monitoring intensity and physiological marks; evaluated at one medical center. | C3, C6 | partial |
| *A Joint Model for Multistate Disease Processes and Random Informative Observation Times* (Lange et al., 2015, *Biometrics*) | [DOI](https://doi.org/10.1111/biom.12252) | Jointly fits a latent disease CTMC and state-dependent Poisson observation process to EMR data for disease-rate estimation. | C3, C6 | partial |
| *Joint Modeling of Longitudinal EHR Data with Shared Random Effects…* (Yang et al., 2026, arXiv preprint) | [arXiv](https://arxiv.org/abs/2602.15374) | Separately estimates EHR visiting, biomarker-observation, and trajectory processes linked by latent frailty; it is not a cross-site prediction adaptation study. | C3, C6 | partial |
| *Joint Modeling and Analysis of Longitudinal Data with Informative Observation Times* (Liang et al., 2009, *Biometrics*) | [DOI](https://doi.org/10.1111/j.1541-0420.2008.01104.x) | Uses latent variables to link longitudinal outcomes and informative observation times for statistical estimation. | C3 | partial |
| *Analysis of Longitudinal Data with Irregular, Outcome-Dependent Follow-up* (Lin et al., 2004, *JRSS B*) | [DOI](https://doi.org/10.1111/j.1467-9868.2004.b5543.x) | Develops inverse visit-intensity weighting for marginal longitudinal analyses, rather than a latent-state prediction model. | C6 | none-but-related |
| *Longitudinal Data Subject to Irregular Observation* (Pullenayegum & Lim, 2016, *Statistical Methods in Medical Research*) | [DOI](https://doi.org/10.1177/0962280214536537) | Reviews visit-process methods and their assumptions; it does not propose site adaptation. | C6 | none-but-related |
| *Informative Presence and Observation in Routine Health Data* (Sisk et al., 2021, *JAMIA*) | [DOI](https://doi.org/10.1093/jamia/ocaa242) | Reviews clinical prediction methods for informative observation and notes that site-specific measurement patterns can limit transportability. | C3, C6 | none-but-related |
| *Modeling Missing Data in Clinical Time Series with RNNs* (Lipton et al., 2016, MLHC / JMLR W&C 56) | [Proceedings PDF](https://proceedings.mlr.press/v56/Lipton16.pdf) | Uses missingness indicators as predictive features and shows tests run can be informative, without modeling a separate process likelihood. | C3 | none-but-related |
| *Deep State Space Models for Computational Phenotyping* (Ghosh et al., 2016, IEEE ICHI) | [Paper page](https://d3ccyth396mz21.cloudfront.net/publications/deep-state-space-models-for-computational-phenotyping) | Applies deep state-space models to irregular, incomplete EHR data; its abstract does not describe a clinician-ordering likelihood. | C3 | none-but-related |
| *Deep Kalman Filters* (Krishnan et al., 2015, arXiv preprint) | [arXiv](https://arxiv.org/abs/1511.05121) | Presents a deep generative state-space model with an EHR counterfactual application, but no explicit testing-process model. | C3 | none-but-related |
| *Domain Adaptation under Missingness Shift* (Zhou et al., 2023, AISTATS) | [Proceedings](https://proceedings.mlr.press/v206/zhou23b.html) | Adapts from labeled source to unlabeled target under missingness shift, including an analytic correction for linear models under underreporting MCAR. | C6 | partial |
| *Domain Adaptation Under MNAR Missingness* (Stokes et al., 2025, arXiv preprint) | [arXiv](https://arxiv.org/abs/2504.00322) | Studies MNAR-shift adaptation via imputation, including an EHR regional-shift application, rather than refitting a latent observation-process component. | C6 | partial |

## Closest work

- **Alaa et al.** is closest for prediction: latent clinical states and clinician monitoring are modeled together. It leaves cross-hospital adaptation and a per-site observation-likelihood weight unstudied.
- **Lange et al.** gives the clearest clinical precedent for a state-dependent observation likelihood, but focuses on disease-rate estimation rather than prediction transfer.
- **Yang et al.** is a recent EHR joint model with distinct visit and measurement components. It estimates associations through shared frailty; it does not test label-free, cross-site refitting while freezing physiology.

## Verdict per claim

- **C3 — Apparently open in the verified papers found.** Explicit observation-process models are established, but I found no paper that tempers only that likelihood with a deployment-time, hospital-specific weight.
- **C6 — Partly known.** Unlabeled-target missingness adaptation exists, and joint models can estimate observation components separately; the specific procedure of refitting only that component at a new hospital while holding physiology fixed remains unshown in these papers.

**Unverified lead:** An [OpenReview search hit](https://openreview.net/pdf?id=BcOC4WCXmQ) titled *Policy-Aware Pretraining for EHR as Clinician-in-the-Loop Dynamical Systems* appeared to describe explicit observation–action modeling and cross-hospital invariance. OpenReview blocked access to its paper metadata, so I could not verify its status or count it in the verdict; check it before making a novelty claim.

## Search terms

“Latent ODE Poisson observation times”; “latent clinical state Hawkes monitoring EHR”; “joint informative observation process disease state CTMC”; “deep state-space EHR missingness”; “domain adaptation MNAR missingness unlabeled target.”