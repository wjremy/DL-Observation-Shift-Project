# Data

## PhysioNet/Computing in Cardiology Challenge 2019: early prediction of sepsis

- Source: https://physionet.org/content/challenge-2019/1.0.0/ (Reyna MA, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. Critical Care Medicine, 2020).
- Access: open access, Creative Commons Attribution 4.0. Cite the paper and PhysioNet when using it.
- Contents: one pipe-separated file (.psv) per ICU patient, one row per hour; vitals, labs and demographics, with unmeasured values left empty (NaN); `SepsisLabel` per hour. Two hospital systems: `training_setA` (20,336 patients) and `training_setB` (20,000 patients).
- Downloaded 18 Oct 2026: all 40,336 files, 323 MB uncompressed.
- Location: `data/raw_physionet2019/` (not in git; re-download with `python data/fetch_physionet2019.py`, which fetches the files one by one because PhysioNet offers no ZIP for this project and resumes if interrupted).
- Planned use (STATUS.md, ARGUMENT.md step 9): train at hospital system A, re-learn the testing head on B's records without labels, evaluate on B's labels; no exact reference model exists for real data, so models are compared with each other.

## More hospitals: dataset scan (19 Oct 2026; `out/lit/lit9_datasets.md`, Claude checked the eICU demo page and the closest preprint)

- **eICU-CRD v2.0** is the main candidate: 208 US hospitals, 200,859 ICU stays (2014-15), pseudonymous `hospitalid`, five-minute vitals and time-stamped labs; Sepsis-3 cohorts exist in YAIB and Moor et al. 2023 (both need customising for per-hospital splits). At least 10 hospitals have more than 1,600 eligible stays each (PLOS Digital Health hospital-transfer study). PhysioNet credentialed: CITI training plus approval, typically 1-2 weeks, and **every team member needs their own access**.
- The open **eICU demo** (ODbL, 2,500 stays from 20 hospitals) has hospital and unit IDs removed (checked on the PhysioNet page): usable only to prototype the pipeline.
- Same-hospital alternative: **MIMIC-III** CareVue (2001-08) versus MetaVision (2008-12), a documented change of recording system within Beth Israel (credentialed).
- Single-centre datasets (HiRID, AmsterdamUMCdb, SICdb, Zhejiang, PIC) add at most one site each; the hidden third PhysioNet 2019 hospital is not released.
- Closest prior work: Singh et al. 2021 (PhysioNet 2019 A to B with missingness indicators); Johnson et al. 2018 (eICU hospital-held-out mortality, no focus on measurement patterns); You et al., arXiv 2609.30781 (25 Sep 2026, missingness-aware conformal calibration across eICU hospitals, mortality; checked). None tunes a per-hospital trust weight.
- Caveat: documented eICU differences are partly recording and interface differences, not only ordering policy (as with HCO3 at Emory in E6).
- Plan: everyone applies for PhysioNet credentialing now; prototype on the eICU demo and PhysioNet 2019; then train on one large eICU hospital (or MIMIC) and find the best w at each of the larger eICU hospitals, giving a distribution of optimal dial settings.
