# Future directions (not in the current project)

Decision (Wesley, 19 Oct 2026): the project studies the dial only. The items below are recorded so they are not lost.

## Adaptation without outcomes, and the adaptation ladder

The latent-state model has a separate testing component. The testing pattern is recorded for every patient, so that component can be re-learned on hospital B's records **without outcome labels** (claim C6 in docs/GAP.md).

Advantages over the dial:
- no labels needed, so it can use every run-in patient from day one;
- it adapts per variable instead of with one global weight. Example from E6: Emory almost never records HCO3. A Beth Israel model reads "HCO3 missing" as meaningful, while a testing component re-learned at Emory would learn that its absence carries no information there;
- it can move in any direction, not only scale trust in A's policy.

Theoretical disadvantages:
- **Not identified without outcomes.** A change in testing at B can come from a different policy or from sicker or different patients (case mix). Without outcomes the model must assume the first; if the second is true, it adapts in the wrong direction without noticing.
- **Wrong objective.** It maximises the fit of B's testing pattern, not the prediction of B's outcomes; under misspecification a better fit of the pattern can predict worse. The dial is chosen on prediction loss and includes w = 1, so with enough data it cannot do much worse than not adapting.
- **The link from tests to outcome is what matters, and only outcomes show it.** Clinicians test on information not in the data. Unlabelled data show how often B tests, not whether a test at B still signals suspicion as strongly as at A.

Proposed ladder, by number of outcome cases at B (cut-offs from E6, one linear model and one hospital pair; neural models would shift the refit band later):

| Outcome cases at B | Best option |
|---|---|
| 0 | re-learn the testing component without labels (or keep w at most 1) |
| fewer than about 5 | label-free re-learning with a cautious dial on top |
| about 5-20 | dial (one parameter, checked on outcomes) |
| about 50 or more | full refit, or a refit shrunk towards A's model |

Natural combination: re-learn the testing component on all of B's records without labels, then set the dial on the labelled run-in cases as a safety check.

## Other directions

- A refit shrunk towards A's model (standard in clinical model updating) as a further comparison.
- Choosing w without labels from how badly A's testing model fits B's tests (a heuristic; untested).
- eICU-CRD (208 hospitals; credentialed PhysioNet access) to estimate the distribution of the best w over many target hospitals (docs/DATA.md).
