## Novelty verdicts

**1. Separate adaptation of the observation process — Partly anticipated.** Lasko, Strobl, and Stead propose a site-specific “Process Model” separate from a stable disease model [R1]. De Fauw et al. provide a concrete, non-EHR example: they retrained only the image-segmentation component for a new scanner while keeping the referral classifier fixed [R2]. In EHRs, Jeanselme et al. jointly model missingness and inter-observation times with survival, and test robustness under weekday/weekend shifts [R3]. These works anticipate process/disease separation, component-only updating, or modeling observation shift. **Not found:** a study that adapts only the testing-pattern component at a new hospital using a small sample of labeled target outcomes.

**2. A graded ladder compared by target sample size — Partly anticipated.** Nested updating strategies are established: Schnellinger et al. compare recalibration, revision, and no update in repeated clinical model updates [R5]. Meijerink et al. document partial network reuse and fine-tuning among AI model-updating methods [R6]. Mutnuri et al. compare transfer-learning strategies with target-data subsets from 1% to 75% in ICU prediction [R7]. These anticipate the *general* idea of comparing update complexity under data scarcity, but none compares the proposed global testing weight, test-group weights, testing-part refit, and full refit across target sample sizes. Phrase the contribution as the **observation-process-specific comparison**, not as the first graded model-updating ladder.

**3. Per-hospital choice of how much testing policy to trust — Partly anticipated; the exact dial appears unreported.** Site-specific process models are proposed in [R1], and the component/fine-tuning literature in [R6] makes partial adaptation a familiar idea. The fixed keep/drop precedents are already summarized in `GAP.md`. **Not found in targeted searches:** a method that estimates a continuous, hospital-specific trust weight on testing-pattern evidence from a small labeled run-in. The dial is the clearest novelty claim, but should be distinguished from general modular updating and from site-specific process modeling.

## Additional papers to cite or distinguish

Sisk et al.’s review is useful background on informative presence/observation, process features, and the need to validate or update models when measurement practice changes [R8]. Van Smeden et al. explain why missing-indicator methods depend on stability of the missingness mechanism across development and deployment [R9]. Subbaswamy and Saria give the broader causal framing for dataset shifts driven by clinician treatment or measurement patterns [R10].

These are targeted-search findings, not proof that no closer paper exists.

## References

[R1] Thomas A. Lasko, Eric V. Strobl, and William W. Stead. 2024. [“Why do probabilistic clinical models fail to transport between sites.”](https://doi.org/10.1038/s41746-024-01037-4) *npj Digital Medicine* 7, 53.

[R2] Jeffrey De Fauw, Joseph R. Ledsam, Bernardino Romera-Paredes, et al. 2018. [“Clinically applicable deep learning for diagnosis and referral in retinal disease.”](https://doi.org/10.1038/s41591-018-0107-6) *Nature Medicine* 24:1342–1350.

[R3] Vincent Jeanselme, Glen Martin, Matthew Sperrin, Niels Peek, Brian Tom, and Jessica Barrett. 2025. [“Prediction of Survival Outcomes under Clinical Presence Shift: A Joint Neural Network Architecture.”](https://arxiv.org/abs/2508.05472) arXiv:2508.05472 preprint; no later venue found.

[R4] Yichen Zhu, Jian Yuan, Bo Jiang, Tao Lin, Haiming Jin, Xinbing Wang, and Chenghu Zhou. 2023. [“Prediction with Incomplete Data under Agnostic Mask Distribution Shift.”](https://doi.org/10.24963/ijcai.2023/525) *Proceedings of IJCAI-23*, 4720–4728. StableMiss addresses mask-distribution shift in general incomplete-data prediction, not hospital-specific supervised adaptation of a testing component.

[R5] Erin M. Schnellinger, Wei Yang, and Stephen E. Kimmel. 2021. [“Comparison of dynamic updating strategies for clinical prediction models.”](https://doi.org/10.1186/s41512-021-00110-w) *Diagnostic and Prognostic Research* 5:20.

[R6] Lotta M. Meijerink, Zoë S. Dunias, Artuur M. Leeuwenberg, Anne A. H. de Hond, David A. Jenkins, Glen P. Martin, Matthew Sperrin, Niels Peek, René Spijker, Lotty Hooft, Karel G. M. Moons, Maarten van Smeden, and Ewoud Schuit. 2025. [“Updating methods for artificial intelligence-based clinical prediction models: a scoping review.”](https://doi.org/10.1016/j.jclinepi.2024.111636) *Journal of Clinical Epidemiology* 178:111636.

[R7] Maruthi Kumar Mutnuri, Henry Thomas Stelfox, Nils Daniel Forkert, and Joon Lee. 2024. [“Using Domain Adaptation and Inductive Transfer Learning to Improve Patient Outcome Prediction in the Intensive Care Unit: Retrospective Observational Study.”](https://doi.org/10.2196/52730) *Journal of Medical Internet Research* 26:e52730.

[R8] Rose Sisk, Lijing Lin, Matthew Sperrin, Jessica K. Barrett, Brian Tom, Karla Diaz-Ordaz, Niels Peek, and Glen P. Martin. 2021. [“Informative presence and observation in routine health data: A review of methodology for clinical risk prediction.”](https://doi.org/10.1093/jamia/ocaa242) *Journal of the American Medical Informatics Association* 28(1):155–166.

[R9] Maarten van Smeden, Rolf H. H. Groenwold, and Karel G. M. Moons. 2020. [“A cautionary note on the use of the missing indicator method for handling missing data in prediction research.”](https://doi.org/10.1016/j.jclinepi.2020.06.007) *Journal of Clinical Epidemiology* 125:188–190.

[R10] Adarsh Subbaswamy and Suchi Saria. 2020. [“From development to deployment: dataset shift, causality, and shift-stable models in health AI.”](https://doi.org/10.1093/biostatistics/kxz041) *Biostatistics* 21(2):345–352.