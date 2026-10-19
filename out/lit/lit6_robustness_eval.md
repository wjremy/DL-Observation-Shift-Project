## Literature map

| Paper (authors, year, venue) | URL/DOI | What it does (1 sentence) | Claims touched | Overlap |
|---|---|---|---|---|
| Taori et al., 2020, NeurIPS | [Proceedings paper](https://papers.nips.cc/paper_files/paper/2020/hash/d8330f857a17c53d217014ee776bfd50-Abstract.html) | Defines **effective robustness** as OOD accuracy above an empirical ID→OOD trend, and **relative robustness** as an intervention’s change in OOD accuracy; warns that OOD-only rankings confound source accuracy, and that effective robustness can rise while both accuracies fall. | C1 | partial |
| Miller et al., 2021, ICML | [PMLR paper](https://proceedings.mlr.press/v139/miller21b.html) | Finds a strong “accuracy on the line” relationship between ID and OOD performance across models and tasks, with some weaker-shift exceptions. | C1 | partial |
| Shi et al., 2023, NeurIPS | [Proceedings paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e9000ecb86d45c442a1d38fae68dd8fb-Abstract-Conference.html) | Extends effective robustness to models trained on different data by predicting OOD performance from accuracy on multiple ID test sets. | C1 | partial |
| Zhang et al., 2021, ACM CHIL | [DOI: 10.1145/3450439.3451878](https://doi.org/10.1145/3450439.3451878) | Benchmarks domain-generalization methods on multi-region eICU mortality data and synthetic shifts, including patient subsampling bias; its sequence model is a GRU, and its shifts do not change the observation policy. | C1 | partial |
| Spinnato & Landi, 2026, ICLR | [OpenReview paper](https://openreview.net/forum?id=qetBM8nLkf) | Benchmarks 12 classifiers, including GRU-D, Raindrop, and Neural CDE, across 34 irregular time-series datasets, including clinical data and partially observed series; it does not test train-hospital/test-hospital observation-policy shift. Its “shift” irregularity refers to temporal offsets among signals. | C1 | partial |
| Che et al., 2018, *Scientific Reports* | [DOI: 10.1038/s41598-018-24271-9](https://doi.org/10.1038/s41598-018-24271-9) | Introduces GRU-D, which uses masks and time since last observation, and shows that missingness can predict clinical outcomes; it does not benchmark cross-site observation-policy shift. | C1 | none-but-related |

## Closest work

- **Taori et al.** is the closest measurement proposal: it directly identifies source accuracy as a confounder and offers an empirical baseline. Its baseline is the fitted trend across a model set, not the target loss increase of the Bayes-optimal source predictor. Its “relative robustness” means an intervention’s OOD gain, not the A-to-B drop itself.
- **Miller et al. and Shi et al.** support adjusting OOD performance for ID performance; Shi et al. also address comparisons across training distributions. They still estimate expected OOD performance empirically rather than isolate the shift cost justified by the source optimum.
- **Zhang et al. and PYRREGULAR** provide relevant clinical and irregular-series benchmarks, but not the proposed Bayes-reference evaluation under a changed hospital testing policy.

## Verdict per claim

**C1 — Partly known; the exact proposal appears open in the papers located.** The literature clearly recognizes that raw OOD performance can conflate source fit with robustness and offers empirical ID-adjusted measures. I did not find a paper that defines justified shift cost as the Bayes-optimal source model’s A-to-B loss increase, then scores models against that cost. The clinical benchmarks located likewise do not test that comparison under observation-policy shift.

## Search terms used

“effective robustness” Taori; “accuracy on the line”; “relative robustness” and “absolute robustness” under distribution shift; Bayes-optimal/oracle baselines for shift benchmarks; “underfitting” robustness metric; clinical time-series domain generalization; “missingness shift” and “observation shift” benchmarks; GRU-D, mTAN, Raindrop, Neural CDE under shift.