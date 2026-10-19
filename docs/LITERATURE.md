# Literature map (Claude, 18 Oct 2026)

Entries were written from memory. The verified literature search of 19 Oct 2026 supersedes this map where they differ: see `docs/GAP.md` (claims, verdicts, gap statement) and `out/lit/` (six agent reports with DOIs or URLs). Each entry states what it contributes to this project.

## Informative measurement in health records (the problem exists)

- Agniel D, Kohane IS, Weber GM (2018). Biases in electronic health record data due to processes within the healthcare system. BMJ. The timing of lab orders predicted survival better than many lab values: the measurement process carries outcome information.
- Wong A, et al. (2021). External validation of a widely implemented proprietary sepsis prediction model in hospitalized patients. JAMA Internal Medicine. Epic Sepsis Model at Michigan Medicine: AUC about 0.63 against the developer's 0.76-0.83.
- Kamran F, et al. (2024). Evaluation of sepsis prediction models before onset of treatment. NEJM AI. Restricted to the time before clinicians acted (cultures, antibiotics), the Epic model's discrimination dropped to near chance: its signal came largely from clinicians' suspicion.
- Sperrin M, et al. (2020). Missing data should be handled differently for prediction than for description or causal explanation. Journal of Clinical Epidemiology. Informative missingness can help prediction, but only while the missingness mechanism stays the same.
- Groenwold RHH (2020). Informative missingness in electronic health record systems: the curse of knowing. Diagnostic and Prognostic Research. The same point from the clinical-epidemiology side.

## Architectures for irregular, informatively sampled series (Sarah's proposal)

- Che Z, et al. (2018). Recurrent neural networks for multivariate time series with missing values. Scientific Reports. GRU-D, which uses the mask and time gaps as features.
- Kidger P, et al. (2020). Neural controlled differential equations for irregular time series. NeurIPS.
- Rubanova Y, Chen RTQ, Duvenaud D (2019). Latent ODEs for irregularly-sampled time series. NeurIPS. Includes a Poisson model of observation times: a learned testing process, close to the E5 idea.

## Shortcuts and underspecification (why the B ranking measures priors)

- Geirhos R, et al. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence.
- D'Amour A, et al. (2022). Underspecification presents challenges for credibility in modern machine learning. JMLR (arXiv 2020).
- Clark C, Yatskar M, Zettlemoyer L (2019). Don't take the easy way out: ensemble based methods for avoiding known dataset biases. EMNLP. Product of experts, used in E4.
- He H, Zha S, Wang H (2019). Unlearn dataset bias in natural language inference by fitting the residual. Workshop paper. Product of experts, used in E4.

## Choosing how much to trust an unstable signal (the dial)

- Rothenhaeusler D, Meinshausen N, Buehlmann P, Peters J (2021). Anchor regression: heterogeneous data meet causality. JRSS-B. A dial between best fit and invariance, chosen by the expected shift strength; the linear analogue of our grey curve.
- Peters J, Buehlmann P, Meinshausen N (2016). Causal inference by using invariant prediction. JRSS-B.
- Arjovsky M, et al. (2019). Invariant risk minimization. arXiv.
- Duchi JC, Namkoong H (2021). Learning models with uniform performance via distributionally robust optimization. Annals of Statistics. Worst-case choice over a set of plausible shifts.
- Subbaswamy A, Saria S (2020). From development to deployment: dataset shift, causality, and shift-stable models in health AI. Biostatistics. Shifts in clinical practice (for example, ordering patterns) as a named source of failure.

## Evaluating with clinical decisions (beyond Brier)

- Vickers AJ, Elkin EB (2006). Decision curve analysis. Medical Decision Making. Net benefit at a treatment threshold; the basis of the cost-weighted check in ARGUMENT.md step 8.
- Proper scoring rules as mixtures of threshold losses: Schervish MJ (1989), Annals of Statistics; Hernandez-Orallo J, Flach P, Ferri C (2012), JMLR. Brier weights all thresholds equally, which is why the Brier result carries over to decisions.

## Data

- PhysioNet/Computing in Cardiology Challenge 2019, early prediction of sepsis (Reyna MA, et al., 2020, Critical Care Medicine). Open access, CC BY 4.0, ZIP 42 MB (checked on the PhysioNet page, 18 Oct 2026): hourly vitals and labs with unmeasured values left empty, sepsis labels, and two hospital systems (training sets A and B). A natural real-data test of training at A and adapting at B.
- MIMIC-IV and eICU-CRD (eICU covers about 200 US hospitals): credentialed PhysioNet access needed.
