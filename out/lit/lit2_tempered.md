*I checked `docs/ARGUMENT.md`. Its proposed dial interpolates between predictive log-odds using the **testing evidence**; that is more specific than simply raising a mask probability to a power.*

| Paper (authors, year, venue) | URL/DOI | What it does (1 sentence) | Claims touched | Overlap |
|---|---|---|---|---|
| Ibrahim et al., 2015, *Statistics in Medicine* | [DOI](https://doi.org/10.1002/sim.6728) | Reviews power priors, which discount a historical dataset’s likelihood when borrowing information into a current analysis. | C3 | none-but-related |
| Bhattacharya, Pati & Yang, 2019, *Annals of Statistics* | [DOI](https://doi.org/10.1214/18-AOS1712) | Studies fractional posteriors formed by raising the likelihood to a power, including contraction under model misspecification. | C3 | none-but-related |
| Grünwald & van Ommen, 2017, *Bayesian Analysis* | [DOI](https://doi.org/10.1214/17-BA1085) | Introduces SafeBayes, which learns a global likelihood learning rate to address misspecification. | C3 | none-but-related |
| Holmes & Walker, 2017, *Biometrika* | [DOI](https://doi.org/10.1093/biomet/asx010) | Gives a principled way to choose a power for a generally misspecified Bayesian likelihood. | C3 | none-but-related |
| Plummer, 2015, *Statistics and Computing* | [DOI](https://doi.org/10.1007/s11222-014-9503-z) | Describes cut models that block feedback between model components; this is a binary cut, not a continuous evidence weight. | C3 | partial |
| Jacob et al., 2017, arXiv preprint | [arXiv](https://arxiv.org/abs/1708.08719) | Examines when modular inference, which restricts information flow between components, can outperform full-model inference. | C2, C3 | partial |
| Carmona & Nicholls, 2020, AISTATS | [PMLR](https://proceedings.mlr.press/v108/carmona20a.html) | Develops semi-modular inference, with an influence parameter that interpolates between full and cut inference and can be chosen by predictive scoring. | C2, C3 | partial |
| Frazier & Nott, 2026, *Journal of the American Statistical Association* | [DOI](https://doi.org/10.1080/01621459.2025.2580690) | Formalizes a bias–variance trade-off between full and cut inference and develops a semi-modular posterior. | C2 | partial |
| Lin, Tarp & Evans, 2025, *Biometrics* | [DOI](https://doi.org/10.1093/biomtc/ujaf008) | Raises the observational-data likelihood to a tunable power and selects it by expected predictive log density when combining RCT and observational data. | C2, C3 | partial |
| Zhou, Balakrishnan & Lipton, 2023, AISTATS | [PMLR](https://proceedings.mlr.press/v206/zhou23b.html) | Studies adaptation under missingness shift, deriving adjustments for particular settings; it does not temper a Bayesian observation-process likelihood. | C3 | none-but-related |
| Sisk et al., 2021, *Journal of the American Medical Informatics Association* | [DOI](https://doi.org/10.1093/jamia/ocaa242) | Reviews clinical prediction methods that use informative presence and observation, including predictors, joint models, and latent structures. | C3 | none-but-related |
| Jeanselme et al., 2022, NeurIPS Workshop on Learning from Time Series for Health | [arXiv](https://arxiv.org/abs/2205.13481) | DeepJoint models clinical presence alongside survival and reports robustness to a weekday–weekend change in observation patterns. | C3 | partial |

## Closest work

- **Carmona & Nicholls (2020)** is the closest methodological precedent: a tunable influence parameter already regulates how much a module affects inference. It does not apply the parameter to clinical testing evidence or tune it per deployment site.
- **Frazier & Nott (2026)** make the general trade-off between full and cut inference precise. Their result is not about source-versus-target risk under observation-policy shift.
- **Jeanselme et al. (2022)** directly addresses clinical presence shift and models the observation process. It does not expose a site-specific dial for changing how strongly that process informs the prediction.

## Verdict per claim

- **C3 — partly known; the specific application appears open in this search.** Component influence weights and power likelihoods are established. I found no paper applying one specifically to the testing-process evidence in a clinical latent-state prediction model and setting it per hospital during run-in. The project’s log-odds interpolation corresponds most directly to tempering a testing **log Bayes factor**. A claim that standard neural networks cannot have any such lever would be too broad: they can be modified with weighted losses or heads, but do not expose this Bayesian evidence dial by default.
- **C2 — partly known; the shift-specific result appears open in this search.** Semi-modular inference already motivates intermediate trust and predictive tuning. The stated asymmetry follows locally when A’s risk is minimized at full trust and the shift makes reducing trust improve B’s risk: A’s increase is second order, while B’s decrease is first order. That directional condition matters; it is not guaranteed for every shift.

These are search findings, not proof that no exact precedent exists.

## Search terms used

“power likelihood missingness model,” “tempered likelihood observation process,” “semi-modular inference influence parameter,” “Bayesian fractional posterior,” “SafeBayes Grünwald,” “power prior,” “informative presence clinical risk prediction,” and “domain adaptation under missingness shift.”