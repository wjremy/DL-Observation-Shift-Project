# Controlled observation-shift study: protocol

Protocol established 17 October 2026 before examining model test results. This is an executed synthetic study, not a clinical validation. The source proposal is Sarah Martinson, *Deep Time-Series Models under Observation Shift* (three pages, supplied by the user).

## Questions and hypotheses

1. Do GRU, GRU-D and Neural CDE differ in prediction under density changes, changes in state-dependent observation policy at matched marginal density, and their combination?
2. Does the answer differ between slow and fast underlying dynamics?
3. Does a model's transferred prediction error exceed that of the same architecture trained directly under the target policy?
4. Can a policy-sensitive noise-valued cue channel induce transfer failures while all physiological observations remain identical?

No architecture ordering or accuracy/robustness tradeoff is presumed. Equivalent behavior, within uncertainty, is a valid outcome. Failing to reject a difference does not establish equivalence.

## Data-generating process

Each patient has a stationary Gaussian OU fluctuation (SD 0.8) plus an independent persistent baseline (SD 0.7). Four measurement channels have fixed loadings [1, 0.8, -1, 0.6] and independent Gaussian measurement noise SD 0.5. The input window is 0--24 hours on a half-hour grid; the binary outcome is whether latent severity at 30 hours exceeds zero. The dynamics regimes have OU timescales 6 hours (slow) and 1 hour (fast). Dynamics do not change between training and testing within a regime. Observation policy never uses future labels or states.

All input measurements, observation opportunities and labels are simulated. This grid-based missingness design is an approximation to continuous irregular sampling; the CDE processes an interpolated control path, not an autonomous physical model.

## Observation interventions

Per-variable observation probability is sigmoid(a + gamma * standardized current latent severity). The intercept is calibrated using Gaussian quadrature against the known stationary latent marginal to give desired expected observation rate rho. This is MNAR with respect to the measured inputs when gamma is nonzero; gamma=0 is MCAR. Average-density matching does not guarantee equal information content, value distributions, or per-patient measurement counts.

Source: rho=0.25, gamma=2. Test grid: rho in {0.10, 0.25, 0.50} and gamma in {2, 1, 0, -2}. Negative gamma reverses the cue and is a deliberately extreme stress test, not the primary deployment scenario. The primary transfer comparison is gamma=2 to gamma=0 at rho=0.25.

For each latent test patient, reuse the same potential measurements and outcome across policies. Use at least 10 observation draws; pair uniform randomness across policies, models and training objectives. Repeated observations of one patient are not independent patients.

## Diagnostic arm

Add a fifth input channel of iid standard Gaussian noise, independent of latent physiology and future outcome. Only this channel's observation pattern depends on severity and changes with the intervention. The four physiological channels always use identical MCAR rho=0.25 masks for a given patient and observation draw. Thus the shift leaves physiological values and observations identical. Removing the informative cue can nevertheless remove useful information; a target-trained comparator is needed to interpret transfer loss.

## Models and controls

Main models: standard GRU with observation features; GRU-D with learned input/hidden decay; Neural CDE with causal forward-filled values and piecewise linear controls, integrated using RK4. Supply the same raw observed information, encoded appropriately, to all three. Normalize using source-training observed entries only. Use no future-point spline interpolation, full-trajectory normalization, hospital identifier, or future observation features.

Train each architecture under source gamma=2 and, separately, target-reference gamma=0 at rho=0.25. The latter is a finite-sample empirical reference, not an optimal predictor. Each model selects a checkpoint using validation from its own training policy; test results must not choose hyperparameters or checkpoints.

Additional controls: observation-only predictor to establish cue informativeness; GRU and CDE without explicit observation features, labeled as partial ablations because timing can leak through forward-filled values. Removing GRU-D's masks and gaps changes its defining architecture and is not treated as the same model.

Use equal numbers of tuning configurations per architecture. Report parameter counts and actual time; do not claim exact architecture-only identification from unequal optimization/capacity. Review numerical CDE integration convergence on pilot and final checkpoints. Correct implementation failures and disclose any change prompted by diagnostics.

## Training and computational scope

At least five final training seeds in every primary arm and both dynamics regimes, plus at least 10 paired observation draws of an independent test cohort. A pilot determines practical sample sizes and epoch budgets on local CPU before final test inspection. Record those choices in an execution addendum. Small models and early stopping keep the study feasible. Training checkpoints, configurations, predictions and seed identities must be retained for reproduction.

## Outcomes and analysis

Primary outcome: Brier score (lower is better). Secondary: log loss, AUROC, 10-bin ECE, calibration intercept/slope, and reliability plots. Report absolute target error, not only changes in error.

Primary transfer contrast for each architecture a is G_a = Brier_target(source-trained a) - Brier_target(target-trained a), at rho=0.25,gamma=0. Architecture contrasts compare these G_a values. Additional contrasts compare the source-trained model across paired test policies and regimes. These are experiment-specific differences, not universal rankings. Report separate slow/fast regime estimates.

Uncertainty must account for both repeated training and patient clustering: pair seeds and test draws; resample whole patients, carrying all observation versions together. Report between-training-seed variation separately or use a documented hierarchical bootstrap over training seeds and patient clusters. The five training seeds remain five training replications, not five times the test cohort size. Use at least 1000 bootstrap draws for primary Brier contrasts. Primary comparisons are predeclared; secondary grid findings are exploratory and multiplicity must be acknowledged.

## Interpretation boundaries

Clinical hospital differences, treatment feedback, changing outcomes, censoring, missing labels, and unrecorded real-world physiology are outside this simulation. Sensitivity to state-dependent observation is not automatically pathological: missingness can contain legitimate information, and the optimal observed-data predictor may change with policy. The diagnostic and target-trained controls help distinguish unavailable information from avoidable transfer error, without proving biological understanding.

References: Che et al. (2018), https://www.nature.com/articles/s41598-018-24271-9 ; Kidger et al. (2020), https://arxiv.org/abs/2005.08926 ; Gu et al. (2022), https://arxiv.org/abs/2111.00396 ; Rockenschaub et al. (2024), https://arxiv.org/abs/2406.16484 ; Jeanselme et al. (2025), https://arxiv.org/abs/2508.05472 .
