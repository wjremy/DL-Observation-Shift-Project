Built [e13_rungs_over_time.py](<C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/e13_rungs_over_time.py:1>) and [e13_figure.py](<C:/Users/wesre/Documents/Claude/DL - Observation Shift Project/figures/e13_figure.py:1>). The runner implements the cached A model, nested deployment streams, adaptation methods, spawn-safe jobs, and run/summary CSVs. The figure script creates the four-panel PNG and PDF.

No substantive departure from the SPEC. `keep` and `oracle` are computed once per policy and repeated across weeks as static reference rows for the weekly summary and plot.

Checks passed: both scripts compiled; the threshold-patched outcome rate was **1.94%** on 20,000 simulated patients; Bayes predictions averaged **2.18%** on 1,500; weekly stream sizes were **25, 50, 100, 200, 325, 650, 1,300, 2,600**. Torch and matplotlib were not loaded. I did not run model training or render the figure.

Commands:

```powershell
python e13_rungs_over_time.py --smoke
python e13_rungs_over_time.py --train-only
python e13_rungs_over_time.py --workers 4 --lr 0.05 --steps 160
python figures\e13_figure.py
```