## Papers

| Paper (authors, year, venue) | URL/DOI | What it does (1 sentence) | Claims touched | Overlap |
|---|---|---|---|---|
| Steyerberg et al., 2004, *Statistics in Medicine* | [DOI](https://doi.org/10.1002/sim.1844) | Compares recalibration and more extensive revision across validation samples from 200 to 10,000 patients; recommends larger samples and shrinkage for structural revisions. | C5 | partial |
| Janssen et al., 2008, *Journal of Clinical Epidemiology* | [DOI](https://doi.org/10.1016/j.jclinepi.2007.04.018) | In one clinical example, simple recalibration improved calibration about as much as revision, while no updating method improved discrimination. | C4, C5 | partial |
| Vergouwe et al., 2017, *Statistics in Medicine* | [DOI](https://doi.org/10.1002/sim.7179) | Proposes increasing update complexity from intercept adjustment to slope adjustment and full coefficient revision as evidence allows, to limit overfitting. | C5 | partial |
| Riley et al., 2020, *BMJ* | [DOI](https://doi.org/10.1136/bmj.m441) | Gives tailored sample-size guidance and says intercept-only updating needs enough data to estimate the mean outcome risk precisely, while updating more parameters requires more data. | C5 | partial |
| Binuya et al., 2022, *BMC Medical Research Methodology* | [DOI](https://doi.org/10.1186/s12874-022-01801-8) | Systematically reviews clinical model evaluation and updating guidance, distinguishing discrimination from calibration and summarizing recalibration, revision, and extension. | C4, C5 | partial |
| Guo et al., 2017, ICML / PMLR | [Proceedings](https://proceedings.mlr.press/v70/guo17a.html) | Establishes temperature scaling as an effective one-parameter post-hoc calibration method for neural-network classifiers. | C4 | partial |
| Yang, Soltan & Clifton, 2022, *npj Digital Medicine* | [DOI](https://doi.org/10.1038/s41746-022-00614-9) | Compares site-specific threshold adjustment with transfer learning across four NHS trusts; threshold adjustment changed operating sensitivity but left the model’s AUROC unchanged. | C4, C5 | partial |
| Kwong et al., 2022, *Frontiers in Digital Health* | [DOI](https://doi.org/10.3389/fdgth.2022.929508) | Reports a prospective silent trial in which site drift was identified, the model refined, and performance checked in a second silent trial. | C5 | partial |
| Zhang, Chen & Bui, 2022, *Journal of Biomedical Informatics* | [DOI](https://doi.org/10.1016/j.jbi.2022.104168) | Uses unlabeled target EHR event sequences for adversarial adaptation of heart-failure prediction across institutions. | C5 | none-but-related |
| Valanarasu et al., 2023, MIDL / PMLR | [Proceedings](https://proceedings.mlr.press/v227/valanarasu24a.html) | Introduces zero-shot, single-image test-time adaptation for medical image segmentation, without target labels or test-time backpropagation. | C5 | none-but-related |

## Closest work

- **Yang et al.** is closest to C4: the study directly shows that site-specific threshold tuning can target sensitivity while preserving the original score ranking and AUROC. It also compares against transfer learning, which can change discrimination. It does not model a changed testing policy or selectively adjust evidence from the observation process.
- **Janssen et al.** directly compare recalibration and revision: with 752 labeled local patients for updating and 283 for evaluation, recalibration and revision improved calibration similarly, but did not improve discrimination in that example. It is one study, not a general sample-size rule.
- **Riley et al.** gives the clearest sample-size framework for C5. It supports the intuition that updating fewer parameters can require less data, but does not estimate the data needed for your observation-process weight or compare it directly with neural-network fine-tuning.

## Verdict per claim

- **C4 — Known for standard recalibration and thresholds.** Intercept changes, positive-slope logistic recalibration, and temperature scaling preserve score order; changing a decision threshold does not change AUROC. A feature-dependent model revision can reorder patients, so the claim applies to recalibration and thresholding, not every kind of update.
- **C5 — Partly known; the project-specific comparison appears open.** The updating ladder and method-specific sample-size principles are established, but there is no universal patient count, nor a direct comparison of local data needed for your one-parameter observation-evidence dial versus recalibration or full fine-tuning. Outcome-based recalibration, threshold selection, and dial selection need local outcomes; unsupervised adaptation papers use target inputs without labels but do not give the same comparison for this setting.

## Search terms used

“clinical prediction model updating recalibration sample size shrinkage”; “Janssen Vergouwe updating methods”; “site-specific threshold adjustment hospital transfer learning”; “silent trial clinical AI Kwong”; “temperature scaling calibration neural networks”; “unsupervised domain adaptation clinical event sequences”; “test-time adaptation medical imaging.”