You are a literature researcher. Use live web search (Google Scholar, arXiv, PubMed, proceedings, publisher pages). Work in the current folder only to read docs/ARGUMENT.md (our reasoning) if useful; write no files.

PROJECT CONTEXT. A student deep-learning project on observation shift in clinical time series. At hospital A, sick patients are tested more often, so the testing pattern (who is measured when; the missingness mask) predicts the outcome. At hospital B the testing policy differs (e.g. independent of sickness). A model trained at A and deployed at B suffers. Our claims, which we need to position against the literature:
C1 Measuring robustness by the raw performance drop from A to B conflates fit on A with robustness; a model that fails to learn A's informative testing pattern looks robust. Robustness should be judged against the drop a Bayes-optimal model of A would also have (the justified cost).
C2 The trade-off is lopsided: slightly discounting the evidence carried by the testing pattern costs almost nothing at A (second order, A is at its optimum) but gains at first order at B; partial trust beats both full trust and ignoring the pattern.
C3 A deployment-time dial: a weight w multiplying the log-likelihood of the observation (testing) process inside a probabilistic latent-state model (hidden sickness state; separate heads for values, testing, outcome), set per hospital, e.g. during a silent run-in period. Ordinary neural networks (GRU, GRU-D, Transformer, Neural CDE) have no such lever.
C4 Per-site recalibration or thresholds cannot fix observation shift, because they do not change the ranking of patients (AUC); the dial does.
C5 How much local data (patients, outcomes) is needed to set the dial, compared with full fine-tuning or standard recalibration.
C6 Re-learning only the observation-process component on unlabelled data from the new hospital (physiology assumed to transfer), without outcomes.

RULES. Only cite papers you actually found and opened (title page or abstract at least); give the URL or DOI for every entry. Never invent a reference; if unsure, say "not verified". Prefer peer-reviewed work and well-known preprints; note year and venue. Be precise about what each paper does and does NOT do relative to C1-C6. Stop when the search space is covered; quality over quantity (typically 6-15 papers).

OUTPUT (your final answer, compact markdown, under 900 words):
1. Table: | Paper (authors, year, venue) | URL/DOI | What it does (1 sentence) | Claims touched | Overlap: full / partial / none-but-related |
2. "Closest work": the 1-3 papers nearest to our claims, and precisely what they leave open.
3. "Verdict per claim" for the claims in your focus: known / partly known / apparently open, one line of reasoning each.
4. Search terms you used.

YOUR FOCUS:

Theory of trading in-distribution fit against robustness. Find work on: anchor regression (Rothenhausler et al. 2021), invariant prediction and IRM, distributionally robust optimization, interpolating between using and ignoring spurious or unstable features ("partially invariant", "trade-off between in-distribution and out-of-distribution accuracy"), shrinkage toward an invariant predictor, and any result that a small move away from the source optimum is second-order costly but first-order beneficial under shift. Also any such analysis specific to missingness indicators. Claims in focus: C2 (is the lopsided trade-off known in general, and for missingness?).
