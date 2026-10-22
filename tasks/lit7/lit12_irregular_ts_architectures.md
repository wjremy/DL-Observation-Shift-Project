# Literature task: deep architectures for irregular clinical time series and how they use the observation process

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

1. Deep architectures for irregularly sampled, partially observed clinical time series, and how each treats the observation pattern: GRU-D (Che et al. 2018), ODE-RNN / Latent ODE (Rubanova et al. 2019), Neural CDE (Kidger et al. 2020), mTAN (Shukla and Marlin 2021), SeFT (Horn et al. 2020), Raindrop (Zhang et al. 2022), and important 2023-2026 successors (e.g. ContiFormer, state-space / Mamba models for irregular series, Warpformer, others you find). For each: does it consume masks / time gaps explicitly, model the observation times as a point process, or treat them as nuisance?
2. Deep models that model the observation process as a separate component (joint models of values and measurement times, e.g. neural point-process or "informative observation" latent models; Alaa et al.; Lasko; Hawkes-process EHR models; any 2022-2026 work). Which ones separate a physiology part from a testing part so that the testing part could be adapted on its own?
3. Multi-site deep-learning benchmarks for ICU prediction that report cross-hospital transfer of deep models: YAIB (van de Water et al. 2024), Moor et al. 2023 (multi-dataset sepsis), HiRID/AUMCdb/eICU/MIMIC transfer studies, eICU per-hospital studies. Report: architectures compared, size of the transfer drop, whether missingness/testing features were analysed, compute used (GPU hours or model size) if stated.
4. Any study comparing these architectures under a change in the observation process (missingness shift, sampling-rate shift) specifically.

## Output

Compact report (max about 1,000 words plus tables). Table 1: architecture, year, venue, how it uses the observation pattern, has separable testing component (yes/no/partly). Table 2: multi-site studies (data, sites, models, transfer drop, missingness analysed?, compute). Then 3-5 sentences: which 4-6 references a 2-page proposal should cite for this area, and why. References at the end.
