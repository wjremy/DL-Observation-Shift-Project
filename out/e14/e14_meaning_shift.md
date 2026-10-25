Created [sim_u.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/sim_u.py), [e14_meaning_shift.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/e14_meaning_shift.py), and [figures/e14_figure.py](/C:/Users/wesre/Documents/Claude/DL%20-%20Observation%20Shift%20Project/figures/e14_figure.py). No existing files were modified.

**Checks:** all three files passed `py_compile`. NumPy checks found an outcome threshold of **2.59309** and a **2.11%** outcome rate on 20,000 patients. Policy testing rates ranged from **29.87% to 29.97%**. With beta 0, the inferred trait probability stayed at 0.3 within **1.0e-15**; with delta 0, oracle predictions matched the existing Bayes grid to **2.8e-17**. Under gain’s B policy, mean inferred trait probability was **0.8745** for u=1 and **0.0679** for u=0; mean oracle prediction was **1.92%**, versus a sampled outcome rate of **2.00%**.

No design departures. The mixture filter omits Gaussian constants common to both trait branches; they cancel when computing the trait posterior. I did not run training or generate the figure because this runtime lacks torch and matplotlib.

From the project folder, the commands are:

```powershell
$py = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py e14_meaning_shift.py --smoke --workers 4
& $py e14_meaning_shift.py --train-only --workers 4
& $py e14_meaning_shift.py --workers 4 --draws 2
& $py figures/e14_figure.py
```