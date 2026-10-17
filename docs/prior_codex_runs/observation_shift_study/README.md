# Synthetic observation-shift study

This package reproduces a controlled simulation of binary risk prediction from 49 half-hourly measurement opportunities. Patient trajectories and labels are generated independently of the observation policy. The main experiment has four clinical-like measurement channels. The diagnostic experiment adds an independent noise-valued channel whose observation mask carries information, while the four physiological masks remain fixed across diagnostic policies.

Three sequence models are included: a GRU, GRU-D, and a neural CDE. Models can receive observation masks and elapsed times alongside measurements. No model receives future measurements. Normalization is fit on observed training entries only and then reused for validation and evaluation. An observation-only logistic baseline uses masks, elapsed gaps, and the time grid, never measured values or latent states.

Training policies use density `rho=.25` with informative observation dependence `gamma=2` (source) or independent observation `gamma=0` (reference). Evaluation crosses densities `{.1,.25,.5}` with dependence `{2,1,0,-2}`. The `gamma=-2` condition is a prespecified stress test. The final matrix has five training seeds, 2 experiments, 2 dynamics regimes, and 2 training policies. The baseline has a separate 40-shard launcher. Evaluation reuses the same test patients and pairs observation draws across policies with seeds `61000` through `61009`.

## Environment

Use Python 3.10 or newer. From this folder, create an isolated environment and install the dependencies at the minimum supported versions listed in `requirements.txt`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate` instead. Each launcher invokes the same Python interpreter running the launcher and limits math-library threads to one per worker.

The recorded environment is Python 3.12.10, CPU-only PyTorch 2.14.1, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6, and Matplotlib 3.11.2. See `environment.json` for the full record. In the neural-model training history, `train_loss` is the final minibatch loss from that epoch; use validation BCE to assess convergence and select checkpoints.

## Run

Run the six-model tuning grid, then the final neural-model matrix:

```powershell
python execute.py tune --workers 3
python execute.py final --workers 3
```

The tuning stage chooses each model configuration from source-policy validation BCE only. The final stage runs the neural models and their no-observation-feature controls. Shards save checkpoints, predictions, metadata, logs, and execution manifests under `tuning/`, `results/`, and `execution/`. Rerunning `execute.py` starts those jobs again; the post-fit checkpoint sensitivity audit supports resuming from completed audit notes.

The completed final matrix has 200 shards: 120 primary neural-model shards, 40 no-observation-feature controls, and 40 observation-only baseline shards. The primary neural models use the source-validation-selected configuration for each experiment and model; the final CDE runs use two integration substeps.

Run the observation-only baseline as a separate job:

```powershell
python baseline_execute.py
```

It runs 40 shards with two workers and saves predictions, coefficient arrays, metadata, logs, and a completion manifest under `results/observation_only_batch/` and `results/{experiment}/{regime}/observation_only/`.

The cohort audit can be regenerated with:

```powershell
python baseline.py --audit --output results
```

After all final shards complete, produce the analysis tables and figures with:

```powershell
python summarize_results.py --input results --output analysis
```

To wait for the neural-model matrix to finish, run the checkpoint sensitivity audit, summarize results, and build the archive in one resumable pipeline, run `python finish_pipeline.py`. It records progress under `execution/finish/` and invokes `package_results.py` only after the final summary passes its integrity checks. To package manually instead, run `python package_results.py` after final audit and analysis files exist.

The summary includes Brier score, log loss, AUROC, calibration error, and paired source-trained versus reference-trained contrasts. It averages repeated mask draws within each patient before uncertainty estimates.

## Tests and saved outputs

Run the integration tests with `python -m unittest test_runner test_analysis`. The simulator and baseline focused tests use pytest: `python -m pytest test_data.py test_baseline.py test_models.py`.

Prediction archives store `y`, `prob`, and `patient_id` arrays in `[condition, repeat, patient]` order. Adjacent metadata records the condition table, seeds, patient ID ranges, training configuration, feature source, and model selection details. `audit_data.py` independently checks completed baseline shards and paired diagnostic masks/values. These synthetic results describe this data-generating process and do not establish performance or transportability in real hospitals.

## Final results and limits

The full analysis completed all 200 fits and 24,576,000 held-out predictions. At the matched target policy (`rho=.25`, `gamma=0`), the primary contrast is the source-trained model's Brier score minus that of a separately reference-trained (`gamma=0`) model. Positive values mean the source-trained fit has higher target Brier loss. Estimates below are mean differences with unadjusted paired-bootstrap 95% intervals over five training seeds and 1,024 shared test patients; intervals are descriptive and not multiplicity-adjusted.

| Architecture | Main, slow | Main, fast | Diagnostic, slow | Diagnostic, fast |
|---|---:|---:|---:|---:|
| GRU | 0.0263 (0.0144, 0.0410) | 0.0217 (0.0129, 0.0314) | 0.0015 (-0.0013, 0.0044) | -0.0002 (-0.0030, 0.0025) |
| GRU-D | 0.0233 (0.0136, 0.0335) | 0.0175 (0.0098, 0.0256) | 0.0013 (-0.0014, 0.0040) | 0.0005 (-0.0015, 0.0029) |
| Neural CDE | 0.0252 (0.0175, 0.0336) | 0.0310 (0.0228, 0.0393) | 0.0016 (-0.0016, 0.0048) | 0.0009 (-0.0011, 0.0029) |

In the main experiment, the source-trained core models' Brier scores ranged from 0.194–0.207 at the source policy to 0.216–0.237 when evaluated at the matched-density `gamma=0` target. Separately reference-trained fits scored 0.193–0.206 at that target. In the diagnostic experiment, where the four physiological observation masks are held fixed and only the noise-channel mask changes, all core-model transfer intervals include zero. The observation-only baseline's diagnostic AUROC fell from 0.737/0.722 at `gamma=2` to 0.501/0.501 at `gamma=0` in slow/fast regimes, confirming that this baseline learned the mask cue.

For fast dynamics in the main experiment, the GRU-D transfer gap was 0.0175 Brier (0.0098, 0.0256), versus 0.0310 (0.0228, 0.0393) for the CDE. The CDE-minus-GRU-D gap difference was 0.0135 (0.0067, 0.0199), so GRU-D had the smaller mean transfer penalty in this condition. This contrast does not establish a universal architecture ranking: the corresponding slow-regime interval includes zero, as do all diagnostic architecture-gap intervals. See [`analysis/contrasts.csv`](analysis/contrasts.csv), [`analysis/metrics.csv`](analysis/metrics.csv), and [`analysis/results_summary.json`](analysis/results_summary.json) for the complete estimates.

Each fit saw one training observation policy; source and reference fits were trained separately. In the main experiment, observation patterns are predictive by construction, so source-policy learning may capture useful risk information as well as policy-specific structure. There was no general source-performance/transfer tradeoff: CDE source-policy Brier was higher than GRU and GRU-D in both regimes (0.1968 vs 0.1937/0.1938 slow; 0.2069 vs 0.2011/0.2016 fast), yet in fast dynamics its transfer penalty was also larger than GRU-D's. A low transfer gap alone cannot distinguish learned transferable structure from underlearning useful policy-specific signal, while a larger gap can reflect source-policy information that no longer transfers. The reference fit is a finite-sample target-trained comparator, not a Bayes-optimal benchmark. No hospital identity or hospital-level covariates enter this simulation, so it cannot estimate hospital-specific density interactions or support hospital-level inference. The equal-budget hyperparameter search uses source-policy validation only, the five seeds vary training cohorts as well as initialization, and the intervals are descriptive across multiple contrasts. Convergence was generally adequate for full-input fits (118/120 stopped before the epoch cap), but 10/40 partial ablations reached the cap and five had best validation loss at epoch 36 or later; these secondary comparisons carry greater budget uncertainty and showed mixed source-versus-target effects. CDE cohort metrics met the predeclared solver-sensitivity thresholds across all 60 checkpoints, although one validation prediction shifted by 0.0307 when increasing from two to four RK4 substeps. These results describe this synthetic setup and do not prove that robust behavior is impossible or that observed transfer gaps are caused solely by robustness versus underlearning.

## Evidence and rationale

The supplied proposal cites six useful foundations. They motivate different parts of the design; together they do not settle the controlled state-dependent policy comparison tested here:

- [TIME](https://proceedings.mlr.press/v306/qiao26g.html) provides pattern-level forecasting benchmarks. It does not directly test robustness to missing-not-at-random policy shifts.
- The [tree-model comparison](https://doi.org/10.1016/j.asoc.2026.114983) by Pablo Reina-Jiménez, María Martínez-Ballesteros, and José C. Riquelme reports data-dependent model rankings. It does not establish performance under a controlled observation-policy shift.
- [GRU-D](https://www.nature.com/articles/s41598-018-24271-9) uses masks and elapsed times as predictive inputs. That supports modeling observation history, without implying universal robustness to distribution shift.
- [Neural CDEs](https://arxiv.org/abs/2005.08926) model irregularly sampled time series. Their formulation does not guarantee invariance to a changed observation policy.
- The [MIMIC-IV resource paper](https://www.nature.com/articles/s41597-022-01899-x) documents a clinical dataset. It is a dataset reference, not a controlled architecture comparison.
- [S4](https://arxiv.org/abs/2111.00396) includes sampling-rate transfer experiments. Sampling-rate robustness has therefore been studied; this work does not claim to be the first such study.

Two further papers motivate explicit observation-policy interventions. [Rockenschaub et al.](https://arxiv.org/abs/2406.16484) show that a Bayes predictor can change under non-ignorable shift and discuss cases where disregarding informative missingness improves transfer; this study therefore makes no novelty claim for that idea. [Jeanselme et al.](https://arxiv.org/abs/2508.05472) report that jointly modeling observation and outcome processes can improve transportability. Together, these sources support studying policy change without assuming that predictive systems must always be brittle.

The design rationale is an inference from this literature, not a claim of an exhaustive review or literature-first contribution: controlled data provide a way to separate observation-density changes, state-dependent observation changes, and a diagnostic mask-only cue. Hospital identity and observation density need not combine multiplicatively; hospital-specific nonlinear response curves are possible. This simulation intervenes on density and observation policy, but does not estimate hospital interactions.
