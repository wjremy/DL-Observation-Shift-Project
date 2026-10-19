# Proposal idea: a trust dial for observation shift

(Group-chat post, Wesley, 19 Oct 2026. Sources: docs/GAP.md, STATUS.md E6.)

**1. Problem.** Models are usually trained on data from a single hospital. Hospitals differ in when and how often they run tests: their observation policy. A model trained at hospital A learns A's policy, for example that a lactate order means the doctor suspects sepsis. When it is deployed at hospital B, which tests differently, that signal changes meaning and performance drops. The change in policy is called observation shift.

**2. Current literature.**
- a) The binary case: Futoma et al. (2021) trained with measurement indicators and dropped them at deployment ("fit with, deploy without"). This transferred best for models trained at Beth Israel and moved to a Memphis hospital, but not the other way round. Rockenschaub et al. (2024) found that ignoring informative missingness costs performance at the training hospital but is more robust elsewhere, especially when missingness is highly informative. Both treat the learned policy as all or nothing, and both end in a general rule rather than a choice per hospital.
- b) The domain adaptation case (model updating): re-estimating the model on outcome data from the new hospital. The options range from adjusting the baseline risk, to recalibrating, to refitting all parameters.

**3. Gap.**
- a) Binary case:
  - The learned policy is useful. The testing pattern alone predicts sepsis at both hospitals (AUROC 0.73 at A, 0.69 at B). Removing it throws away real information.
  - The trade-off is lopsided. Turning the policy down slightly from what is optimal at A costs almost nothing at A, because the model sits at A's optimum, where the error curve is flat. At B the model is not at its optimum, so the same small step helps directly. So partial trust can beat both ends.
  - The best setting depends on the target hospital. From the same source hospital (Beth Israel), dropping the policy was best for Memphis, but keeping it is best for Emory (our data). No fixed on/off rule is right for both.
- b) Domain adaptation case: refitting needs many outcome cases. Even validating a model properly needs 100+ events (Riley et al. 2024), and with few events simple updates beat full refits (Steyerberg et al. 2004; Vergouwe et al. 2017). Sepsis is rare: about 2% of our ICU patients develop it within two days. A dial has one parameter, so it needs far fewer cases.

**4. Solution context.** Before going live, models are often run silently in the background to see how they would have performed on real patients. Documented sepsis run-ins: Duke Sepsis Watch about 3 months; UCSD COMPOSER about 46 days; Epic's sepsis model at Tufts 77 days. No guideline sets the length.

**5. Solution.**
- a) Give the model a dial w for how much it trusts the learned observation policy. In a linear model, scale the policy coefficients by w. In neural sequence models, use a state-space model with separate parts for physiology, testing and outcome, and scale the testing part.
- b) Train on A, then tune w on a run-in subset of B.
- c) Compare with the on/off cases from earlier work (binary case).
- d) Compare with full refitting and recalibration (domain adaptation case).
- e) Compare all of them as the run-in grows.

**Not novel**
1. Tempering one part of a model with a tunable weight.
2. Domain adaptation: recalibrating or refitting on the new hospital.
3. Removing the policy entirely (Futoma: w = 0).

**Novel**
1. Keep or remove the learned policy depends on the target hospital: from the same source, the best choice flips between hospitals. So it should be set per hospital from run-in data, not fixed as a general rule (as in Futoma).
2. Whether the best setting lies *between* keep and remove, which a binary choice cannot reach. Theory and simulation predict it; we test it on real hospitals.
3. How many run-in cases each option needs (binary choice, dial, full refit), compared with how many cases a sepsis run-in actually yields.

**Preliminary results**
- Binary case: in simulation an in-between setting beats both on and off. On the one real pair we have (Beth Israel to Emory) "keep" was best, the opposite of Futoma's Memphis result from the same source hospital; the dial, set on the run-in, landed on par with it and avoided the wrong end, which cost 0.05 AUROC. More hospital pairs are needed to see how often the middle wins.
- Domain adaptation case: a one-parameter update beats a full refit up to about 20 sepsis cases; refitting wins from about 50. A typical ICU admits about 600-1,300 patients a year, so a 1-3 month run-in yields only about 2-6 sepsis cases. For ICU sepsis models a one-parameter update is the better choice, and neural networks need even more data to refit.
