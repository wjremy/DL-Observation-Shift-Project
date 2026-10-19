# Observation-shift simulation study: specification (Claude, 17 Oct 2026)

## Questions

Q1 (justified cost). How much does the exact A-optimal predictor lose when applied unchanged under policy B? This is the loss no model trained only on A can avoid without giving up A accuracy.
Q2 (excess reliance). Do end-to-end architectures, trained on raw values + masks under A, lose MORE than that justified cost on B (they rely on the observation pattern instead of the values, beyond what the information warrants), or less (they under-learn the mask)? Measured against the exact Bayes rules.
Q3 (redundancy). How do Q1 and Q2 change as the values become more or less informative (noise sigma), i.e. as the mask's extra information over the values (Bayes mask gain) shrinks or grows?
Q4 (sampling coverage). With UNINFORMATIVE observation in both hospitals, how much does each architecture lose beyond Bayes when only the density/regularity of sampling changes?

## Data-generating process (module `sim.py`)

- Time grid t = 0..T-1, T = 48. Latent state: stationary Gaussian AR(1), Z_0 ~ N(0,1), Z_t = rho Z_{t-1} + sqrt(1-rho^2) eps_t, rho = 0.95.
- Outcome: Y = 1{Z_{T-1+h} > c}, horizon h = 6, threshold c = 1.0 (prevalence about 0.16). Z beyond T-1 continues the same AR(1).
- Observation indicator O_t ~ Bernoulli(p_t). Value X_t = Z_t + sigma * eta_t (eta standard normal), available only when O_t = 1.
- Policies (dataclass `Policy`, each with a name):
  - `informative(gamma, rate)`: p_t = sigmoid(alpha + gamma Z_t), with alpha solved numerically so that E[p_t] = rate under Z ~ N(0,1).
  - `uniform(rate)`: p_t = rate, independent of Z.
  - `bursty(rate, stay)`: O_t a two-state Markov chain independent of Z, P(O_t = O_{t-1}) = stay, stationary marginal = rate (derive the switching probabilities; for rate 0.5 both stay-probabilities equal `stay`). O_0 drawn from the stationary marginal.
- `simulate_latent(n, seed) -> dict(z: float32[n, T], z_future: float32[n], y: int8[n])`
- `observe(latent, policy, sigma, seed) -> dict(x: float32[n, T] with NaN where unobserved, m: int8[n, T])`
  The same latent cohort can be observed under several policies (paired test sets). Different seeds for latent and observation draws; all seeds explicit.

## Exact Bayes predictors (module `bayes.py`, numpy only)

Grid forward filter on z in [-5.5, 5.5] with G = 241 points (check that G = 481 changes Brier by < 1e-4). Transition kernel from the AR(1), rows renormalised. Per step likelihood given the ASSUMED observation model:
- O_t = 1: p_assumed(O=1 | z) * Normal(x_t; z, sigma^2)
- O_t = 0: 1 - p_assumed(O=1 | z)
Uniform/bursty policies are uninformative about z, so their likelihood factor reduces to Normal(x; z, sigma^2) when observed and 1 when not ("values-only rule"; it is exact Bayes for any Z-independent policy, including bursty).
Final prediction: P(Y=1 | history) = sum_z post_T(z) * [1 - Phi((c - rho^h z) / sqrt(1 - rho^(2h)))].
Function: `bayes_predict(x, m, sigma, assumed_policy) -> float64[n]`, vectorised over patients, work in log space or renormalise each step.
Rules used in analysis:
- `A_rule`: assumed = training policy A. Applied to A test data = Bayes optimum under A. Applied to B data = transferred source rule.
- `V_rule`: assumed uninformative (values-only). Exact Bayes under any Z-independent B; on A data it is the best rule that ignores what the pattern says about z.
Tests (pytest-free plain asserts in `test_sim_bayes.py`, runnable with numpy only):
1. With an uninformative policy and every step observed, the grid filter matches an exact Kalman filter (posterior mean and variance at T-1, and P(Y=1)) to < 1e-3.
2. With an uninformative policy and random missingness, same Kalman check (Kalman skips the update when unobserved).
3. Calibration: on 20000 simulated patients under an informative policy, A_rule predictions binned in deciles match observed Y frequencies within Monte Carlo error; and A_rule Brier < V_rule Brier on A data.
4. alpha solver gives the requested marginal rate (Monte Carlo check).
5. Bursty chain has the requested marginal rate and stay probability.

## Models (module `models.py`, PyTorch CPU)

All end-to-end on raw observed data. Output: one logit for Y at the end of the window. Hidden size 64 unless stated.
Per-step features for recurrent models: x_locf (last observed value carried forward; 0 before the first observation), m_t, delta_t (steps since last observation, divided by T; GRU-D definition from Che et al. 2018).
- `gru`: nn.GRU on [x_locf, m, delta], linear head on final hidden state.
- `grud`: GRU-D (Che et al. 2018): input decay of x toward the training-set empirical mean of observed values, hidden-state decay from delta, mask as input. Implemented as an explicit loop over T with a GRUCell.
- `cde`: Neural CDE (Kidger et al. 2020), own implementation, no torchcde. Control path channels: [t / T, x linearly interpolated between observations (forward-filled after the last and back-filled before the first observation; 0 if never observed), cumulative observation count / T]. Initial hidden h_0 = linear map of the path at t = 0. Vector field: MLP hidden -> 128 -> 128 -> hidden * channels with tanh output. Integrate with RK4-free Euler over the 47 path increments plus 2 substeps per increment (h += f(h) dX/substeps). Head on final hidden.
- `transformer`: tokens only at observed steps: input projection of [x_t] plus sinusoidal embedding of t (d = 64), learned CLS token, 2 encoder layers, 4 heads, padding mask for absent steps. Patients with zero observations: CLS only. Head on CLS output.
Training: Adam lr 1e-3, weight decay 0, batch 256, BCE loss, max 120 epochs, early stopping on validation BCE (validation drawn under the TRAINING policy), patience 12, keep best checkpoint. Inputs computed once as tensors.

## Experiments (runner `run_grid.py`)

Cohort sizes: train 8000, validation 2000, test 8000 latent patients; one test latent cohort per seed, observed under every test policy below with a common observation seed (paired across policies).
Seeds: 0, 1, 2 (each seed redraws train, val, test and model init).

E1 informative (Q1-Q3). Training policy A = informative(gamma, rate 0.3). Grid: sigma in {0.25, 0.5, 1.0, 2.0} x gamma in {1.0, 2.0}. Test policies for every fitted model: A itself, B_match = uniform(0.3), B_dense = uniform(0.6).
E2 coverage (Q4). sigma = 0.5. Training policy A = uniform(0.5). Test policies: A, B_sparse = uniform(0.15), B_bursty = bursty(0.5, stay 0.9).
Fits: E1 8 configs x 4 archs x 3 seeds = 96; E2 1 x 4 x 3 = 12. Total 108.
Execution: multiprocessing pool with `--workers` (default 4, the physical core count), torch.set_num_threads(1) in each worker, one fit per task, each fit writes its own shard directory `results/<exp>/<config>/<arch>/seed<k>/` with `preds.npz` (keys: one float32[n_test] array of predicted probabilities per test policy name, plus `y`), `meta.json` (config, arch, seed, best epoch, epochs run, train/val BCE history, wall time). Skip shards that already exist (resume). Also write `bayes.npz` once per config and seed with A_rule and V_rule predictions on every test policy (and `y`), computed in the main process before or after the fits.
CLI: `python run_grid.py --exp E1|E2|all --workers 4 --out results [--smoke]`. `--smoke`: n_train 800, n_val 200, n_test 1000, 5 epochs, one config per experiment, all 4 archs, seed 0, into `results_smoke/`.

## Analysis (`analyze.py`, numpy + pandas; `plot.py`, matplotlib)

Risk R = Brier score (also report BCE). For each config, arch, seed, test policy B, with A the training policy:
- Bayes mask gain on A: G = R_A(V_rule) - R_A(A_rule)  (redundancy measure; small G = near-redundant).
- Justified transfer cost: J_B = R_B(A_rule) - R_B(V_rule)  (perfect A learning, applied to B, versus B's Bayes optimum).
- Model source excess: S = R_A(model) - R_A(A_rule).
- Model target excess over the transferred rule: X_B = R_B(model) - R_B(A_rule).
- Model excess over B's optimum: R_B(model) - R_B(V_rule).
- Reliance probe on A test data: regress logit(p_model) on [1, logit(p_V), logit(p_A) - logit(p_V)] by least squares, giving b_values and b_mask. Exact Bayes has b_values = 1, b_mask = 1 by construction. b_mask > 1 with b_values < 1 = substitution (pattern used instead of values). Also the same probe with R^2.
- Mask gain captured: (R_A(V_rule) - R_A(model)) / G.
Aggregate: mean over seeds with seed-level min/max; paired differences between architectures with a patient-level bootstrap (1000 resamples, paired across arch and policy) for X_B and S.
E2: excess over Bayes (= V_rule there) on A and on each B, per arch.
Outputs: `analysis/e1_summary.csv`, `analysis/e2_summary.csv`, `analysis/contrasts.csv`, `analysis/summary.json` with the headline numbers. `plot.py`: (1) per sigma, X_B vs S per arch with the A_rule point at (0, 0) and J_B annotated; (2) b_mask and b_values vs sigma per arch; (3) E2 excess bars.

## E3: exact redundancy, testing driven by recorded values only (Claude, 17 Oct 2026)

Question: when the observation pattern is exactly redundant given the recorded values, so that the Bayes rule on A ignores it and the justified transfer cost is zero, do architectures still lose on B? Any loss on B is then pure shortcut use, decided by the architecture's prior off A's support.

Policy `recorded(gamma, rate, sigma)` in `sim.py` (name `recorded_g{gamma}_r{rate}_s{sigma}`, fields gamma, alpha, rate). Sequential over t: x_last = most recent recorded noisy value at s < t (0 if none yet); m_t ~ Bernoulli(sigmoid(alpha + gamma * x_last)); x_t = z_t + sigma * eps_t if m_t. Noise is drawn as a full (n, T) matrix before the masks. alpha is calibrated by bisection so that the mean of m over (patients, t) is `rate` on a fixed calibration cohort (n 4000, latent seed 990001, observation seed 990002, common random numbers so the bisection is monotone).
Why it is exactly redundant: P(m_t | z, past values) = P(m_t | past recorded values), so the policy factor cancels from the likelihood (missing at random). The exact Bayes rule on any such policy is the values-only rule. `bayes._is_informative` already treats non-`informative` policies as values-only, so A_rule = V_rule identically and G = J_B = 0.
Setting: sigma 1.0, gamma 2.0, train A = recorded(2.0, 0.3, 1.0); test A, B_match = uniform(0.3), B_dense = uniform(0.6). Seed 0 only, standard cohort sizes and training. 4 fits; about 10 minutes on 4 workers.
Metrics: S = R_A(model) - R_A(V_rule); X_B = R_B(model) - R_B(V_rule) (Brier, also BCE). Shortcut probe on each test policy: regress logit(p_model) - logit(p_V) on [1, n_obs / T, gap since last observation / T] (least squares, report coefficients and R^2); Bayes gives all zeros. Patient-level paired bootstrap (1000) for X_B per arch.

## E4: can a network be taught to discount the testing pattern? (Claude, 18 Oct 2026)

Motivation: blending the exact rules, logit p_w = logit p_V + w (logit p_A - logit p_V), traces a very lopsided trade-off. Near w = 1, error at A is flat (first-order optimum) while error at B falls linearly. So a small discount costs almost nothing at A and saves much at B. The fitted networks lie far above this curve. Question: can a standard shortcut-debiasing method bring a network onto it?
Method: product of experts (Clark et al. 2019; He et al. 2019). (1) Bias model b: GRU (hidden 64) on [0, m, delta] (the value channel zeroed), i.e. sees only who was tested and when, trained on A as usual. (2) Main model f: GRU with the standard features [x_locf, m, delta], trained on A with logit = f(x, m) + b(m), b frozen (its logits precomputed), same loss, optimiser, early stopping (validation loss also on f + b). (3) Deployment family: logit = f + w * b, w in {0, 0.1, ..., 1.0}. w = 1 is the model fitted to A; w = 0 removes the testing-pattern signal as far as the method can.
Setting: E1 config sigma 1, gamma 2, seed 0 (same cohorts and seeds as E1 via run_grid helpers), test policies A, B_match, B_dense. Compare: the family's (Brier A, Brier B) points against the exact-rule blend curve and the plain E1 GRU (results/E1/sigma1_gamma2/gru/seed0).
Output: results/E4/sigma1_gamma2/seed0/{bias,poe_gru}/ with logits.npz (one float32[n_test] logit array per test policy, plus y) and meta.json.

## E5: a learned model with a separate testing head, tempered at deployment (Claude, 18 Oct 2026)

Motivation: E4 showed that a black-box network has no lever that turns down only the evidence carried by being tested. The exact rule has one, because the testing pattern enters the filter only through the factor P(m_t | Z_t). E5 asks whether a model that learns this structure from A's data, rather than being given it, gives a tempering curve that follows the exact blend curve.
Model (`latent_filter.py`, PyTorch): a discrete latent-state filter with K = 32 states, all parameters learned by gradient descent on A's training data:
- transition matrix: row-softmax of a learned K x K logit matrix (initialised near-diagonal: +4 on the diagonal, +2 on the neighbours);
- initial distribution: softmax of learned logits;
- value head: Gaussian emission, learned mean per state (initialised evenly from -2.5 to 2.5) and one shared learned log-sd;
- testing head: P(m_t = 1 | state k) = sigmoid(c_k), learned;
- outcome head: P(Y = 1 | state at T-1 = k) = sigmoid(u_k), learned; prediction = filtered belief at T-1 times sigmoid(u).
Forward filter in log space over t = 0..T-1: predict with the transition matrix; update with log N(x_t; mu_k, s) when m_t = 1 and with w * [m_t log sigmoid(c_k) + (1 - m_t) log(1 - sigmoid(c_k))] for every t. Training uses w = 1. Loss = BCE(y) + (1/T) * mean negative log marginal likelihood of (x, m), the latter from the same forward pass (log-sum-exp normalisers), so that the heads are proper likelihoods and not free features. Adam lr 1e-2, batch 512, up to 200 epochs, early stopping on validation loss (patience 15), torch.set_num_threads(4).
Deployment family: the same trained model with w in {0, 0.05, ..., 1}.
Setting: E1 config sigma 1, gamma 2, seed 0 (same cohorts as E1 via run_grid helpers); test policies A, B_match, B_dense.
Compare: (Brier A, Brier B) of the tempered family against the exact blend curve and the plain E1 GRU. Also report the learned testing-head curve sigmoid(c_k) against the true policy sigmoid(alpha + gamma * mu_k) as a sanity check.
Expectation, stated in advance: the simulator is itself a latent-state model of this kind, so success here is a proof of principle, not evidence for real data.

## E6: the dial on real data from two hospitals (Claude, 19 Oct 2026)

Question: on real data, does partial trust in the testing pattern (0 < w < 1) transfer better from hospital A to hospital B than full trust (w = 1) or ignoring it (w = 0, Futoma et al. 2021's "fit with indicators, deploy without")? And how many patients from B (a silent run-in) are needed to choose w, compared with recalibration and refitting?
Data: PhysioNet/CinC 2019 (docs/DATA.md). Hospital A = training_setA (Beth Israel, the hospital of MIMIC and of Futoma et al.); hospital B = training_setB (Emory).
Cohort: patients with at least 24 hourly rows (ICULOS 1..24 present); exclude patients with SepsisLabel = 1 in any of hours 1-24. Prediction time: end of hour 24. Outcome: SepsisLabel = 1 in any of hours 25-72 (patients leaving earlier count as 0).
Features from hours 1-24, following Futoma: physiology = last observed value of each vital and lab (HR ... Platelets; not Unit1, Unit2) plus its square, Age, Gender, HospAdmTime; never-measured values filled with hospital A's median; all standardised with A's training statistics. Testing pattern = for each vital and lab: measured in hours 17-24 (0/1), measured at all in 1-24 (0/1), log(1 + number of measurements in 1-24).
Model: L2-penalised logistic regression on all features, fit on 80% of A (rest = A test); penalty chosen by 5-fold cross-validation on A's training part. Also physiology-only and pattern-only models for reference.
Dial: logit_w = b0 + physiology terms + w * (pattern terms), w in 0, 0.1, ..., 1.5, applied to the A-fitted coefficients without refitting.
Evaluation: AUROC, Brier and log loss on A test and on B (B test set below), with patient-level bootstrap intervals (1000) for the w curve on B.
Run-in experiment: split B once into a run-in pool and a fixed B test set of 8,000 patients. For n_run in 100, 250, 500, 1000, 2500, 5000 (and the full pool), draw 50 random run-in samples; on each, choose: (a) w by log loss on the run-in, (b) intercept and slope recalibration of the w = 1 model, (c) w plus intercept, (d) a full refit on the run-in only (same penalty grid, CV on the run-in), (e) Futoma's w = 0 and the unadapted w = 1 as fixed references. Report mean and 10-90% range of B-test AUROC and log loss per method and n_run, plus the number of sepsis cases per run-in.
Descriptive: per variable, the measurement rate in A and in B, and the AUROC of the pattern-only model within each hospital (how informative testing is in each).

## E7: where does the best dial sit across many site pairs? (Claude, 19 Oct 2026)

Question: across real source-target pairs, is the best trust weight w at an end (0 or 1) or in between, and does a dial tuned on target data beat choosing the better end on the same data? Open data only (no sign-in).

Sites:
- **P19 (sepsis).** PhysioNet 2019, sites = hospital x unit: A-MICU, A-SICU, B-MICU, B-SICU (Unit1 = 1 medical, Unit2 = 1 surgical, taken at the first row with either set; patients with neither unit excluded). Cohort, features and outcome exactly as E6 (24-hour landmark; outcome sepsis in hours 25-72; physiology last values and squares plus Age, Gender, HospAdmTime; pattern: measured in hours 17-24, ever measured, log(1 + total count)). 12 directed pairs.
- **P12 (mortality).** PhysioNet/CinC 2012 sets a, b, c (12,000 ICU stays, BIDMC, open, ODC-By; outcome In-hospital_death from Outcomes-*.txt), sites = ICUType 1 CCU, 2 cardiac surgery recovery, 3 medical, 4 surgical. Window: first 48 hours. Variables: all time-varying parameters except RecordID, Age, Gender, Height, ICUType, Weight (static); MechVent and Urine count as measurements. Physiology: last value of each variable in hours 0-48 and its square, plus Age, Gender, Height, Weight (static, -1 means missing). Pattern: measured in hours 40-48, ever measured in 0-48, log(1 + total count). 12 directed pairs.

Per source site s: split 80/20 (seed fixed); medians, scaling and the L2 logistic penalty (5-fold CV on the E6 grid) from the 80%. Report held-out 20% AUROC and log loss. Dial as E6: logit_w = b0 + physiology + w * pattern, one fitted model, no refit.

Per target t (all of t's patients, using s's medians and scaling):
- w curve, w = 0 to 1.5 in steps of 0.05: log loss, Brier, AUROC.
- Tuned comparison: 100 random 50/50 splits of t into tune and evaluation halves. On the tune half choose (i) w from the grid by log loss and (ii) the better end, w in {0, 1}, by log loss. Evaluate both on the evaluation half. Report the mean and 10-90% range of the difference (dial minus end) in log loss and AUROC, the share of splits where the chosen w is strictly inside (0.05 to 0.95), and the mean chosen w.
- Also report n and outcome cases per site.

Outputs: `analysis/e7_pairs.csv` (one row per dataset, source, target), `analysis/e7_curves.csv`, `analysis/e7_sites.csv`, `analysis/e7_summary.json`, figure script `figures/e7_figure.py` (small multiples: target log loss relative to w = 1 against w, one panel per pair, best w marked; Claude renders it).

## E7b: dial above 1 versus recalibration (Claude, 19 Oct 2026)

E7 found optima above 1 (up to the grid edge 1.5) with the largest log-loss gains, while AUROC fell, which suggests part of the gain is calibration (scaling the pattern term also stretches the predictions). E7b: same pairs and source fits as E7; extended grid w = 0 to 3; tuned comparison on 100 tune/evaluation splits between choosing the end (0 or 1), the dial on 0-1 and the dial on 0-3, each with and without intercept-and-slope recalibration fitted on the tune half (w chosen jointly with its recalibration). The dial counts as a real improvement over the binary choice only if `dial_recal` beats `end_recal`. Brief: tasks/e7b/e7b_recal.md.
