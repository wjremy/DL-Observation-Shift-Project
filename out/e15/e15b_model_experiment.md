Implemented the three files:

- [latent_filter_mv.py](<C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/latent_filter_mv.py:84>) adds the multivariate NumPy and lazy PyTorch filters, training/refit methods, standardisation, and a callable Torch parity check.
- [e15_real_pairs.py](<C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/e15_real_pairs.py:529>) adds source training and caching, pair adaptations, metrics, summaries, and the spawn-safe worker pool.
- [e15_figure.py](<C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/figures/e15_figure.py:19>) plots AUROC changes with draw min–max ranges.

**Checks:** `py_compile` passed. The D=1 NumPy filter matched `latent_filter.numpy_forward` exactly across three dial weights; the `w=0` testing-head invariance and standardisation checks passed. Synthetic-cache loading, target splitting, run-in sampling, target scaling, and AUROC-difference summaries passed. I did not run model training or render the figure; Torch and matplotlib checks remain for an environment that has those packages.

**E7 comparison:** I used `analysis/e7_sites.csv`, column `source_holdout_auroc_w1`, for the combined linear model with physiology and pattern features. E7 has no whole-hospital source rows for P19 A-all/B-all, so those two use the combined `local_auroc` from `analysis/e10_sites.csv`, recorded with that provenance. In smoke mode, the source-AUROC gate logs warnings but continues so the deliberately short run can exercise pair handling; full mode stops after a gate warning. Smoke trains only the two source models needed for its two pairs.

**Commands** (PowerShell):

```powershell
$py = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'

# Build the sequence cache if needed
& $py e15_sequences.py

# Smoke, source training only, full run, figure, and Torch parity check
& $py e15_real_pairs.py --smoke
& $py e15_real_pairs.py --train-only
& $py e15_real_pairs.py --workers 4 --draws 3 --lambda-nll 1.0
& $py figures/e15_figure.py
& $py -c "import latent_filter_mv; assert latent_filter_mv.check_torch_matches_numpy()"
```