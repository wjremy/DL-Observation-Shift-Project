## Literature map for C2

| Paper (authors, year, venue) | URL/DOI | What it does (1 sentence) | Claims touched | Overlap |
|---|---|---|---|---|
| Rothenhäusler et al., 2021, *JRSSB* | [DOI: 10.1111/rssb.12398](https://doi.org/10.1111/rssb.12398) | Anchor regression tunes between OLS and 2SLS, trading observational prediction against robustness to a specified class of linear shifts. | C2, general trade-off | partial |
| Mehrabi et al., 2021, *ICML* | [PMLR paper](https://proceedings.mlr.press/v139/mehrabi21a.html) | Derives Pareto trade-offs between standard and Wasserstein adversarial risk for several fixed model classes. | C2, DRO trade-off | partial |
| Choraria et al., 2023, *AAAI* | [DOI: 10.1609/aaai.v37i6.25875](https://doi.org/10.1609/aaai.v37i6.25875) | Relaxes IRM by enforcing invariance within partitions of environments when global invariance over-constrains prediction. | C2, partial invariance | partial |
| Arjovsky et al., 2019, arXiv preprint | [arXiv:1907.02893](https://arxiv.org/abs/1907.02893) | Introduces IRM, which seeks representations with a common optimal classifier across training environments. | C2, invariant prediction | none-but-related |
| Rockenschaub et al., 2024, arXiv preprint | [arXiv:2406.16484](https://arxiv.org/abs/2406.16484) | Studies informative missingness under shifts: Bayes predictions remain stable for ignorable shifts, while experiments find ignoring missingness can help when it is highly informative. | C2, missingness | partial |
| Zhou, Balakrishnan & Lipton, 2023, *AISTATS* | [PMLR paper](https://proceedings.mlr.press/v206/zhou23b.html) | Formalizes missingness shift and studies adapting with unlabeled target data, including cases where missingness indicators are unavailable. | C2, missingness shift | none-but-related |

## Closest work

**Anchor regression** is the closest general analogue: it explicitly tunes fit versus robustness and gives guarantees for a restricted shift family. Its linear structural setting does not analyze a weight on a clinical testing-process likelihood or the local cost and gain of reducing that weight.

**Rockenschaub et al.** are closest on missingness: they directly study predictors that use informative missingness under a change in the missingness mechanism. Their result is qualitative/empirical for non-ignorable shifts; it does not establish that partial trust beats both endpoints or derive the first-order/second-order asymmetry.

**Mehrabi et al.** provide formal robustness-versus-standard-risk frontiers, but for Wasserstein adversaries and specified model classes, not for a changing clinical observation policy.

## Verdict per claim

- **C2, general: partly known.** A lopsided local trade-off follows directly from risk geometry, though the direction of improvement is conditional. For Brier risk, if \(p_A\) is A’s Bayes predictor and \(p_\epsilon=p_A+\epsilon h\), source excess risk is exactly \(\epsilon^2 E_A[h^2]\). At B, the risk change is \(2\epsilon E_B[(p_A-p_B)h]+\epsilon^2 E_B[h^2]\), so it improves at first order when the chosen direction \(h\) points toward B’s predictor. Anchor regression and adversarial-risk work study broader trade-offs, but I found no paper stating this exact result for the project’s setup.
- **C2, missingness-specific: partly known qualitatively; the exact result appears open.** Existing work studies using versus disregarding missingness, but I found no verified analysis proving the local asymmetry or showing that *partial* trust in an informative testing pattern beats both full trust and ignoring it.

## Search terms used

“anchor regression” and shifted predictive performance; “IRM” and spurious correlations; “partial invariance” and OOD trade-off; standard versus adversarial risk Pareto trade-off; robust prediction under missingness shifts; domain adaptation under missingness shift; shrinkage toward invariant predictors and first-/second-order risk.