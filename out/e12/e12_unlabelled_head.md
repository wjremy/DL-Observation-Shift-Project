Implemented [e12_unlabelled_head.py](C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/e12_unlabelled_head.py). It builds the four B policies, evaluates the cached E5 filter and oracle, runs the labelled and unlabelled head refits with deterministic pools and multiprocessing, and writes run, summary, and curve CSVs under `analysis/`.

No method departure from the spec. Cached `keep`, `drop`, and `oracle` results are repeated across size/draw rows to fill the run CSV schema; each is computed once per policy. Smoke mode writes its curve at `n_U=100`; full mode uses the specified `n_U=2,000`.

Checks passed: `py_compile`; NumPy predictions on a small B sample were within (0, 1); rank-based AUROC matched brute-force pairwise AUROC. I did not run refits.

Run from the project folder:

- Smoke: `python e12_unlabelled_head.py --smoke`
- Full: `python e12_unlabelled_head.py --workers 4`