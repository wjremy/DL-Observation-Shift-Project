# Data

## PhysioNet/Computing in Cardiology Challenge 2019: early prediction of sepsis

- Source: https://physionet.org/content/challenge-2019/1.0.0/ (Reyna MA, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. Critical Care Medicine, 2020).
- Access: open access, Creative Commons Attribution 4.0. Cite the paper and PhysioNet when using it.
- Contents: one pipe-separated file (.psv) per ICU patient, one row per hour; vitals, labs and demographics, with unmeasured values left empty (NaN); `SepsisLabel` per hour. Two hospital systems: `training_setA` (20,336 patients) and `training_setB` (20,000 patients).
- Downloaded 18 Oct 2026: all 40,336 files, 323 MB uncompressed.
- Location: `data/raw_physionet2019/` (not in git; re-download with `python data/fetch_physionet2019.py`, which fetches the files one by one because PhysioNet offers no ZIP for this project and resumes if interrupted).
- Planned use (STATUS.md, ARGUMENT.md step 9): train at hospital system A, re-learn the testing head on B's records without labels, evaluate on B's labels; no exact reference model exists for real data, so models are compared with each other.
