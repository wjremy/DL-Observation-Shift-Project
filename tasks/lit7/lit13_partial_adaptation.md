# Literature task: adapting only part of a trained model with few labels

You are a research assistant with live web search.

## Project context (read before searching)

ETH Zurich Deep Learning course project. Clinical prediction models trained at hospital A learn A's testing policy: which labs and vitals are measured, and when (the "testing pattern" or "observation process"). At hospital B the policy differs (observation shift), so testing-pattern features can change meaning. Our model has a physiology part (measured values) and a separate testing part (measurement indicators, counts, time since last measurement; in the neural version, a latent-state model with a separate head for the observation process).

Proposed method, a ladder of adaptations of the testing part only, tuned on the few labelled patients of a silent run-in at hospital B (about 2-6 sepsis cases in 1-3 months for one ICU):
1. one global dial w that scales the whole testing part (w=0 ignore, w=1 trust as at A);
2. one dial per group of tests (e.g. routine labs, cultures, blood gases);
3. re-learn the testing part entirely;
4. full refit of the model.
Higher rungs fix more kinds of shift but need more outcome cases. Research question: which rung is best for a given run-in size and kind of shift.

Read docs/GAP.md in the working folder first: it records what we already know (Futoma et al. 2021 "fit with, deploy without"; Rockenschaub et al. 2024 missingness shift; model updating: Steyerberg 2004, Vergouwe 2017; Riley 2024; silent run-ins). Do not repeat that material; add to it.

Rules: live web search; primary sources; full reference (authors, year, title, venue, DOI or URL) for every claim; never guess a reference or a number; say "not found" when you could not find something. Do not write any files; your final answer is the report.


## Questions (essential)

1. Methods that adapt only part of a trained neural network to a new domain with few labelled samples, and evidence on which part to adapt for which kind of shift: surgical fine-tuning (Lee et al. 2023, ICLR: which layer depends on the shift type), last-layer retraining / Deep Feature Reweighting (Kirichenko et al. 2023), linear probing then fine-tuning (Kumar et al. 2022), adapters / LoRA / BitFit, test-time adaptation for time series (2023-2026). For each: how many parameters adapted, how many labelled samples needed, any result on the trade-off between flexibility and sample size.
2. Clinical prediction specifically: transfer learning or fine-tuning of deep EHR / ICU models to a new hospital with few labels (e.g. Desautels et al. 2017 sepsis transfer; Gupta et al. 2018; eICU / MIMIC / HiRID fine-tuning studies; 2022-2026 work on few-shot or sample-efficient hospital adaptation). Report sample sizes at which fine-tuning starts to help.
3. Hierarchies of updating methods chosen by sample size: the clinical model-updating ladder (intercept, recalibration, revision, refit), the closed testing procedure (Vergouwe et al. 2017), Bayesian or shrinkage updating (e.g. Su et al., Debray et al.), and any rule for how many parameters can be updated per outcome event (events per parameter for updating). Give concrete numbers where stated.
4. Any work that adapts or reweights measurement-indicator / missingness features separately from value features when moving to a new site (per-feature or grouped), even in linear models.

## Output

Compact report (max about 1,000 words plus tables). Table 1: method, year, venue, part adapted, parameters, labelled samples used, key finding. Table 2: clinical transfer studies (data, source->target, model, labelled target samples, gain). Section on events-per-parameter rules for updating. Then 3-5 sentences: which 4-6 references a 2-page proposal should cite to motivate a ladder of adaptations chosen by run-in size, and why. References at the end.
