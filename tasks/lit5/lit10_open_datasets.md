# Research task: openly downloadable clinical time-series datasets from different hospitals or units (no sign-in)

You are a research assistant with live web search. Use primary sources (dataset pages, papers). Give a link for every claim and check the access terms on the dataset's own page. Do not guess; mark anything not found as "not found".

## Hard requirement

The data must be downloadable **without any account, sign-in, credentialing, training course, data-use agreement or application** (e.g. a direct HTTPS link, Zenodo, figshare, Dryad, GitHub, an open PhysioNet project, UCI). Kaggle, PhysioNet credentialed projects, Synapse and anything needing a login are **excluded**; list them separately in one line each only if they would otherwise be excellent.

## Context

We study observation shift: hospitals differ in which tests they order and how often. We train a model at hospital A that uses physiological values plus the testing pattern (which variables were measured, how often), and at hospital B we tune one weight w on how much to trust the testing pattern. We already use the open PhysioNet/CinC 2019 sepsis data (two hospitals, A and B; it also has ICU-unit columns Unit1/Unit2). We need more sites: different hospitals, or clearly different units/eras with different measurement practice, each with a usable outcome. Several separate datasets from different hospitals are fine (like Futoma et al. 2021, who used two hospitals' data separately), as long as they share enough variables to build one model (vitals and common labs such as lactate, WBC, creatinine, platelets, bilirubin, glucose, pH, etc.).

## Questions (essential)

1. Find openly downloadable datasets of hospital patients (ICU, ED or ward; adults preferred) with **time-stamped or hourly measurements where missingness reflects whether a test was done**, and an outcome (sepsis preferred; in-hospital mortality, deterioration, AKI, ICU transfer acceptable). For each: name, link, licence, country/hospital(s), number of patients and outcome cases, variables, time resolution, whether the site/unit is identifiable, download size, and exactly how to download (direct link?).
   Check at least: PhysioNet/CinC 2012 challenge (sets A, B, C; ICU types; open?), MIMIC-III and MIMIC-IV demo datasets (100 patients, open?), eICU demo (no hospital IDs, but a third hospital system), HiRID or AmsterdamUMCdb or SICdb open demo subsets if any, PhysioNet 2019 Unit1/Unit2 sizes, open Chinese or other national ICU/sepsis datasets on figshare/Zenodo/Dryad/Mendeley (e.g. from Scientific Data papers), COVID-19 hospital time-series datasets with labs from different hospitals (e.g. Wuhan Tongji/Yan et al. 2020, Brazilian Hospital Israelita/Fleury/Sirio-Libanes open data, Mexican or Spanish open COVID datasets), the open "Pediatric Intensive Care" or neonatal datasets, and any open multi-centre sepsis cohort.
2. For each candidate, say whether it shares enough variables with PhysioNet 2019 (hourly vitals and the common labs) to apply one model, and whether outcome counts are large enough (at least about 50 outcome cases).
3. Documented differences in measurement practice between the candidate sites (cite if any).

## Output

Compact report (max about 900 words plus tables):
- Table 1: open datasets meeting the hard requirement (columns as in question 1, plus overlap with PhysioNet 2019 and outcome count).
- One line each: excellent but excluded datasets (access reason).
- Recommendation: the best combination of open sites for 3 or more hospital/unit pairs within a week, with download sizes.
- References/links.

Do not download data, write code or modify files other than your answer.
