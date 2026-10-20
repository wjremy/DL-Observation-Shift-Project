# Literature task: how large is observation shift between hospitals in practice?

You are a research assistant with live web search. Use primary sources; give a full reference (authors, year, title, venue, DOI or URL) for every claim; do not guess.

## Context

We study observation shift: the testing pattern (which labs/vitals are measured, how often) carries information about the outcome, and that relationship may change between hospitals. In our open-data experiments (PhysioNet/CinC 2019 sepsis: Beth Israel and Emory; PhysioNet 2012 ICU types), testing *rates* differed a lot, but the pattern's *predictive meaning* seemed to transfer, while physiology transferred worse. We want to know whether that is typical.

## Questions (essential)

1. Empirical studies that measure how much informative missingness or testing-pattern features (measurement indicators, counts, time since last measurement) help or hurt when a model is moved to another hospital or dataset. For each: data, outcome, sites, what was transferred, size of the effect. Include at least: Futoma et al. 2021 (Critical Care Explorations), Singh et al. 2021 (JMIR Medical Informatics, PhysioNet 2019 A to B with missingness indicators; report exactly what they found), Agniel, Kohane and Weber 2018 (BMJ, laboratory test timing), Sharafoddini et al. 2019 (missingness indicators, MIMIC), studies in eICU across hospitals, Moor et al. 2023 and other multi-dataset sepsis studies (any analysis of missingness features across sites), Rockenschaub et al. papers on missingness or cross-dataset generalisation (2023-2025), and any 2024-2026 work on "missingness shift", "observation process shift" or "measurement policy shift".
2. Is the sepsis label itself tied to testing? Describe how the PhysioNet 2019 Sepsis-3 label (and Sepsis-3 labels in MIMIC/eICU benchmarks) is derived (SOFA from labs, suspected infection from cultures and antibiotics), and any paper discussing that testing-defined labels make testing patterns predictive by construction (label leakage via testing).
3. Anything known about how the PhysioNet 2019 challenge data were preprocessed (hourly binning, which variables, any harmonisation between the hospitals) that would blur or align testing patterns between the two hospitals.

## Output

Compact report (max about 900 words plus tables): Table 1 (studies: data, sites, transfer, effect of testing-pattern features); short sections on question 2 and question 3; a paragraph answering "how much does observation shift typically matter between hospitals, and is sepsis special?"; references. Do not write files other than your answer.
