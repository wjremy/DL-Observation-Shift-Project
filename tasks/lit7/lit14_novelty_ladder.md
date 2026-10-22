# Literature task: novelty check for a ladder of observation-process adaptations

You are a skeptical reviewer with live web search. Your job is to find prior work that would make our idea not novel.

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


## Claims to attack

1. Nobody has adapted the observation-process (testing-pattern) part of a clinical model separately from the physiology part when moving between hospitals, using a small labelled target sample.
2. Nobody has compared a graded set of such adaptations (one global weight, per-test-group weights, re-learning the observation part, full refit) as a function of target sample size.
3. Nobody has proposed choosing how much of the learned testing policy to trust per hospital, rather than a fixed rule (keep or remove indicators).

## Where to search (essential)

Missingness shift / observation-process shift / measurement policy shift (2018-2026), informative missingness under distribution shift, "fit with, deploy without" follow-ups, domain adaptation for EHR with missing data, invariant or causal approaches to missingness indicators (e.g. clinician-behaviour or treatment-policy shift), feature-group or block-wise model updating, partial pooling across hospitals (hierarchical models with hospital-specific missingness effects), power-prior or tempered-likelihood weighting of one model component. Include ML venues (NeurIPS, ICML, ICLR, MLHC, CHIL, AISTATS) and clinical/statistics venues (Stat Med, JAMIA, npj Digit Med, Lancet Digit Health).

## Output

For each claim: verdict (novel / partly anticipated / anticipated), the closest 1-4 papers with full references and one line each on what they did and how it differs from our claim. Then a list of any other papers we must cite or distinguish ourselves from. Max about 900 words. References at the end.
