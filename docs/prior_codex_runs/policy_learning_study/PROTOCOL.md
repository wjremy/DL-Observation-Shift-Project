# Isolating learned observation-cue dependence

Protocol fixed before final test inspection. This follow-up addresses whether learning a genuinely predictive source observation cue increases loss after that cue becomes independent of health. It is a mechanism experiment, not a new hospital benchmark or a retrospective explanation of the previous study.

## Remove physiology learning as a competing explanation

Each patient has a static binary latent health state H, with equally likely classes. The outcome is that state. Dense physiological measurements have class means +/-0.35 and known Gaussian AR(1) noise (marginal SD 1, correlation .7), at 17 times from 0 to 24 hours. This is a deliberately simple classification task, not future clinical forecasting.

The exact physiology-only Bayes logit is b(X)=2*mu*1' Sigma^-1 X. Every architecture receives the SAME fixed b(X) in its final prediction. No architecture trains, changes, or replaces the physiological predictor. Its only trainable component is a cue correction c(M), giving sigmoid(b(X)+lambda*c(M)). The correction receives the observation mask, its causal gaps and time; it cannot access physiology, latent health or labels at inference.

## Source and target policies

Source A observes a zero-valued cue at each time with probability .35 for H=1 and .15 for H=-1. Target B observes it with probability .25 independently of H. Both have expected marginal density .25. The exact SAME patients, physiology, labels and base predictions are reused in A and B, with paired uniforms for the masks. Realized finite-sample densities need not be equal. The zero-valued cue conveys information exclusively through its observation pattern.

Conditional on H, physiology and cue masks are independent. Therefore the exact source cue logit is k*log(.35/.15)+(17-k)*log(.65/.85), where k is the number of observations. The exact target cue logit is zero. These oracles establish incremental cue information and quantify missed source information directly, rather than treating source accuracy as proof of policy learning.

## Two distinct interventions

1. **Cue learnability during training:** train each correction branch on genuine source masks or patient-permuted source masks. Permutation changes only mask assignment, preserves the collection of observed mask trajectories exactly, and recomputes gaps. It breaks cue-health dependence in expectation; finite samples may retain chance association. Physiology and outcomes are never permuted. Validation uses the corresponding training assignment, without target-based checkpoint selection.
2. **Cue use at inference:** evaluate each frozen fitted correction at lambda=0,.5,1. This controls use of the learned correction; it is not itself an intervention on learning.

GRU, a GRU-D-style hidden-decay cue branch, and Neural CDE use the same available observation history. Here GRU-D input-value decay has no effect because cue values are zero; conclusions concern these isolated cue extractors, not all defining strengths of full GRU-D models. Neural CDE uses causal piecewise-linear controls and two RK4 substeps. A fitted count-logistic correction and analytic oracles are positive controls.

## Scope and computation

Final target: 2,048 training, 512 validation and 4,096 fixed test patients; five paired training seeds and eight paired observation redraws. Thirty neural fits (three architectures, two training assignments, five seeds), plus ten cheap count-logistic fits. Initial zero correction is an eligible validation checkpoint. A separate seed-90 pilot checks implementation, source learning and practical runtime; test performance must not select settings. Final settings and any pilot-driven adjustments are recorded before final evaluation.

Benchmark CPU thread counts, execution modes and batch sizes before choosing the run configuration. The verified machine has an i5-8365U, four physical cores/eight logical processors, and Intel UHD 620 graphics. Installed PyTorch is CPU-only. Prefer measured throughput and bounded parallelism over using every logical thread indiscriminately. Preserve the earlier completed study unchanged.

## Outcomes and checks

For Brier loss R, report source benefit I=R_A(lambda0)-R_A(lambda1), target cost C=R_B(lambda1)-R_B(lambda0), and total shift loss D=R_B(lambda1)-R_A(lambda1). Since the lambda0 predictions and outcomes are identical in A/B, **D=I+C exactly** for paired empirical means. Check this identity. It separates the lost source advantage from additional target harm.

Report true-versus-permuted training differences in I and C; source excess Brier relative to the source Bayes oracle; cue-logit error; and gate trade-off curves. Pair training seeds, patients and observation redraws. Descriptive 95% bootstrap intervals resample training replications and one shared patient sample, retaining all redraws within patients. No multiplicity-adjusted or universal architecture claim is planned.

An analytic positive control applies the exact source cue rule unchanged to B's masks. This is deliberately NOT B's Bayes predictor, which ignores the cue. Its target cost shows what happens after perfect recovery of A's policy rule. Compare neural cue functions to that same source rule on both A and B mask inputs; distinguish rule-estimation error from the fact that the source-optimal rule becomes inappropriate in B. This control was added before inspection of final test outcomes; it requires no model fitting or target-based selection.

Tests cover the analytic likelihood, incremental oracle benefit, population cue independence in B, exact physiological invariance, permutation preservation, model gradients, zero correction, implementation equivalence, numerical refinement and the loss decomposition. Evaluate final CDE checkpoint sensitivity to four substeps; disclose individual as well as average changes.

### Pilot and final execution addendum

Before inspecting any final test predictions, the seed-90 pilot with 512 training patients and 40 epochs showed all neural branches still improving. A second source-only pilot with the final 2,048/512 training/validation sizes and 80 epochs reached source-validation BCE about .28 for GRU/GRU-D and .29 for CDE, versus about .28 for the count control; the CDE was still improving at the cap. Therefore the final maximum is **120 epochs, patience 12**, with the same fixed learning rate .003 and batch size 512 for every neural branch. This replaces the initial 40-epoch budget above. No target test results selected this change. Epoch zero remains eligible. Late stopping and remaining source-oracle gaps will be reported rather than assumed absent.

The measured 36-configuration CPU benchmark completed in 11.49 seconds. GRU uses eager execution; GRU-D-style and CDE branches use the faster measured TorchScript execution in this installed runtime. All parallel fits use one compute thread each with three worker processes, leaving one physical core for system activity. TorchScript is deprecated in this PyTorch version, so this choice is an environment-specific speed optimization rather than a future-proof recommendation. Script/eager predictions were checked for equivalence. The pilot-stage test suite had 16 checks. The final suite has 18 passing checks: 12 function tests and six analysis unit tests; no target test outcomes were used to tune the models.

## Interpretation

The design makes physiological learning identical by construction. A difference between the paired training assignments is attributable to the cue-learning intervention within this specified experiment. It does not establish why the old end-to-end architectures differed, nor that greater cue learning must cause monotonically greater harm for every architecture. Underlearning and miscalibration remain measurable properties of the correction branch. A null architecture difference is valid. Novelty or a general critique of existing literature is not presumed.
