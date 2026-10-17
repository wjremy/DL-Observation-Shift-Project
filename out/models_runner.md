Written [models.py](C:/Users/wesre/Documents/Claude/obs-shift-sim/models.py), [run_grid.py](C:/Users/wesre/Documents/Claude/obs-shift-sim/run_grid.py), and [test_models.py](C:/Users/wesre/Documents/Claude/obs-shift-sim/test_models.py).

**Checks:** All three files passed `py_compile`. `test_models.py` passed its NumPy feature checks, including LOCF, GRU-D delta, CDE interpolation, cumulative counts, and an unobserved row (feature shape `[2, 6, 3]`).

**Not run:** PyTorch forward/training checks and `run_grid --smoke`; the supplied Python has no PyTorch. Those checks are included in `test_models.py` for a PyTorch-enabled environment.

**Deviations:** None identified. **Open concern:** PyTorch execution and the end-to-end smoke run still need verification with real PyTorch.