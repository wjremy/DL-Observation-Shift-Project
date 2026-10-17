# Observation-cue learning mechanism study

This controlled follow-up asks whether a model that learns a predictive observation pattern can lose performance when that pattern changes. It isolates one cue-learning mechanism; it is not a clinical forecasting benchmark and does not explain the earlier end-to-end architecture comparison.

## Design

Each patient has a balanced, static latent health state `H ∈ {-1,+1}` and binary outcome `Y=(H+1)/2`. Seventeen dense physiological measurements span 0–24 hours. Their class means are `±0.35`; noise follows a known Gaussian AR(1) covariance with marginal standard deviation 1 and correlation 0.7. This physiology is informative but imperfect. Its exact Bayes logit, `b(X)=2 μ 1ᵀΣ⁻¹X`, is fixed and shared by every model and policy.

At each time, a zero-valued cue may be observed. Under source policy A, observation probability is 0.35 for `H=+1` and 0.15 for `H=-1`; under target policy B, it is 0.25 independently of health. Both policies have marginal expected observation density 0.25. Models cannot use `H`, `Y`, or physiology in their cue branch: they receive only the observation mask, elapsed gaps, and time. Their prediction is `sigmoid(b(X) + λ c(M))`.

The learned correction is trained either on the genuine source masks (`true`) or on source mask rows permuted across patients (`permuted`). Permutation preserves the collection of trajectories and recomputes gaps, while breaking mask-health dependence in expectation; chance association can remain in a finite sample. True and permuted fits use the same generated cohort at each paired seed, changing only mask assignment. Each fit uses one training assignment. At evaluation, the frozen correction is applied to paired A and B masks for the same fixed test patients and mask redraws.

The three architectures are GRU, GRU-D-style hidden-decay GRU, and Neural CDE. They differ only in how they map the observation history to `c(M)`; none learns or changes the physiological predictor. Because cue values are always zero, this GRU-D branch does not exercise input-value decay. A count-logistic correction is a positive control because the source-optimal cue rule is linear in the number observed.

## Interventions and interpretation

At inference, `λ ∈ {0, 0.5, 1}` scales the fitted correction. This is a cue-use intervention, not a change to what was learned. For Brier risk `R`, the paired quantities are:

- `I = R_A(0) − R_A(1)`, the benefit from using the learned correction under A;
- `C = R_B(1) − R_B(0)`, its cost under B;
- `D = R_B(1) − R_A(1)`, the total A-to-B loss change.

Because the same base predictions and outcomes are used at `λ=0`, `D = I + C` exactly for paired mean losses. The analysis reports the components separately. This helps distinguish a learned source benefit from harm after transfer; predictive source performance alone does not distinguish useful learning from robustness.

The exact source Bayes cue correction is `k log(.35/.15) + (17−k) log(.65/.85)`, where `k` is the observed count. The target Bayes cue correction is zero. The analysis also applies the exact source rule unchanged to target masks. This transferred-rule control quantifies the consequence of perfect source-policy learning under B; it is deliberately different from B's own Bayes predictor.

This is a descriptive mechanism experiment. It does not establish a universal architecture ranking, demonstrate that robustness is generally caused by underlearning, or explain the previous full-model results. A model may transfer well or poorly depending on its learned cue correction and the changed policy.

## Final run and analysis

The final design uses 2,048 training patients, 512 validation patients, 4,096 shared test patients, five training seeds, and eight paired observation redraws. It comprises 30 neural fits and 10 count-logistic positive-control fits. A source-only pilot with 2,048/512 patients found validation BCE of 0.281 for GRU, 0.284 for GRU-D, and 0.290 for CDE at epoch 80; the CDE was still improving. The final cap was set to 120 epochs with patience 12 before final test scoring. Validation uses the corresponding training assignment; target test results did not select checkpoints or settings. Initial zero correction remained eligible. The compact pilot record is in [pilot_validation_summary.json](pilot_validation_summary.json).

From this directory, install the listed minimum dependencies in an environment with Python 3.10 or later, then run:

```powershell
python -m pip install -r requirements.txt
python benchmark_cpu.py --output cpu_benchmark.json
python run_tests.py
python run_mechanism.py --output results/final --epochs 120 --patience 12 --workers 3
python audit_mechanism.py --results results/final --output audits/cde_mechanism_audit.json
python analyze_mechanism.py --results results/final --output analysis
```

The runner writes one directory per architecture, training assignment, and seed, with metadata, initial and selected checkpoints, and paired predictions. It records code/configuration hashes and checks that runner, data, and model source files did not change during a fit. The analysis validates the 40-shard matrix and writes CSV summaries, JSON results, and plots. The CDE audit compares two and four integration substeps and selectively refines when its configured threshold is exceeded.

The recorded host is an Intel i5-8365U with four physical cores, eight logical processors, and Intel UHD 620 graphics. PyTorch was CPU-only; no CUDA or modern accelerator was validated. All 40 fits completed in 213.95 seconds (3 minutes 34 seconds) using three concurrent workers with one thread per fit. GRU ran eagerly, while GRU-D and CDE used TorchScript for measured speed in the installed runtime. TorchScript is deprecated in this PyTorch version, so its use is an environment-specific optimization. See [PROTOCOL.md](PROTOCOL.md), [hardware.json](hardware.json), [environment.json](environment.json), and [cpu_benchmark.json](cpu_benchmark.json) for details.

## Findings

The randomized cue-learning intervention produced a clear source benefit and target cost in this fixed-physiology simulator. Genuine-source-trained neural correction branches recovered about 97–98% of the source Bayes cue benefit. When evaluated on B, those learned source corrections increased Brier loss by about 0.081–0.085 over the shared cue-free baseline. Models trained on patient-permuted masks learned no comparable source benefit; their estimated target cost stayed near zero. The permuted arm intentionally removes the cue-health association during training, so it is a learnability intervention. It does not show that an unchanged A-trained robust model failed to learn the cue.

At full cue use (`λ=1`), the paired mean test results were:

| True-source-trained branch | Source Brier | Target Brier | Source benefit I | Target cost C | Source oracle benefit recovered |
|---|---:|---:|---:|---:|---:|
| GRU | 0.08421 | 0.24662 | 0.07869 | 0.08372 | 97.8% |
| GRU-D-style | 0.08446 | 0.24799 | 0.07844 | 0.08508 | 97.4% |
| Neural CDE | 0.08443 | 0.24345 | 0.07847 | 0.08054 | 97.5% |
| Count-logistic control | 0.08253 | 0.25047 | 0.08037 | 0.08757 | 99.8% |

All branches share a cue-free Brier score of 0.16290 at `λ=0`. For the permuted-training controls, I was approximately −0.0005 to −0.0017 and C was approximately 0.00004 to 0.00012. The analysis reports paired 95% seed-and-patient bootstrap intervals and gate curves in [analysis](analysis/).

The analytic oracle gives a useful mechanism check independent of neural fit quality. The exact source cue rule reaches Brier 0.08241 under A, but applying that same perfect source rule unchanged under B gives 0.24630, compared with the cue-free Brier 0.16290. This is source benefit `I=0.08050` and transfer cost `C=0.08339`: the specified policy shift can create substantial harm even when the source cue rule is learned perfectly. Across all fitted arms, the numerical residual in `D = I + C` was `1.11×10⁻¹⁶`.

A post-hoc probability-scale accounting separates this common transferred-rule harm from each fitted branch's deviation. For target Bayes probability `q`, transferred source-rule probability `pS`, and fitted target probability `p`, conditional excess risk is `Q=mean((p−q)²)=H+A+E`, where `H=mean((pS−q)²)` is common source-rule harm, `A=2mean((pS−q)(p−pS))` measures whether the fitted deviation attenuates or reinforces that harm, and `E=mean((p−pS)²)` is rule-estimation error. In the true-source-trained neural arms:

| Branch | Q target excess | H common harm | A alignment | E rule error | Source probability MSE vs oracle |
|---|---:|---:|---:|---:|---:|
| GRU | 0.08391 | 0.08372 | −0.00249 | 0.00268 | 0.00159 |
| GRU-D-style | 0.08527 | 0.08372 | −0.00145 | 0.00300 | 0.00182 |
| Neural CDE | 0.08114 | 0.08372 | −0.00599 | 0.00341 | 0.00201 |

CDE deviations attenuate more of the transferred-rule harm than GRU or GRU-D deviations, while also having somewhat more target rule-estimation error; the stronger attenuation dominates in this decomposition. Its source probability error is slightly higher, while source-benefit differences were small, with paired 95% intervals spanning zero. This accounting is descriptive algebra, not a causal mediation test or a general claim that underlearning creates robustness. Paired architecture contrasts and bootstrap intervals for each component are in [analysis/mechanism_results.json](analysis/mechanism_results.json).

The learned-effect contrasts support this mechanism within the simulator. They do not explain the earlier end-to-end architecture results, establish a universal robustness-versus-learning trade-off, or show that one cue architecture is superior. CDE training also approached the fixed optimization budget: four of five true-source CDE fits selected epoch 117 or later and reached the 120-epoch cap; one permuted-source CDE fit selected the cap as well. Thus the 97–98% recovery is strong but not proof of complete neural convergence. No additional fits were run after final analysis.

The final runner, data, model, analysis, and numerical audit checks passed: 40/40 fits, 10/10 CDE audit checkpoints, and 18 focused tests (12 function checks and six analysis unit tests). The CDE audit triggered no selective eight-substep refinements. See [analysis/mechanism_results.json](analysis/mechanism_results.json), [audits/cde_mechanism_audit.json](audits/cde_mechanism_audit.json), and [test_results.json](test_results.json) for machine-readable results.
