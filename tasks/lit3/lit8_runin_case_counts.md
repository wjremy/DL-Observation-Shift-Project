# Literature task: how many outcome cases does a realistic silent run-in yield, and how many does model updating need?

You are a research assistant with live web search. Work from primary sources (papers, official statistics); give a full reference (authors, year, title, venue, DOI or URL) for every number. Do not guess numbers; if a figure is not found, say so.

## Context (do not search for this; it is our own work)

We study a clinical sepsis prediction model trained at hospital A and deployed at hospital B. Before going live, hospitals often run models silently for weeks to months. We compare two ways to use those silent-run data from B: (i) set one scalar parameter (a "dial" weighting part of the model), or (ii) refit or update the whole model. In our own experiment (ICU patients with a stay of at least 24 h, prediction at hour 24 of sepsis in the next 48 h, about 2% outcome rate), a full refit of a ~100-feature logistic regression beat the one-parameter dial only once the run-in had about 2,500 eligible patients (about 50 sepsis cases); below about 1,000 patients (about 20 cases) the dial was better. We want to know, from the literature, how many cases a realistic run-in yields in different settings. Report what the evidence shows, whichever way it points.

## Questions (essential)

1. ICU volume: typical number of ICU beds per ICU and ICU admissions per year per unit and per hospital (US and Europe; give medians or ranges, e.g. from national registries, SCCM, ICNARC, NICE, ANZICS, Dutch NICE registry, Leapfrog). Typical ICU length of stay and the share of ICU stays lasting at least 24 h.
2. Sepsis incidence in ICU patients: share of ICU admissions that develop sepsis (Sepsis-3) after ICU admission, i.e. ICU-acquired sepsis, versus sepsis present on admission. The PhysioNet/Computing in Cardiology Challenge 2019 cohort rates (sets A and B) and any reported admission counts and time spans of those two hospitals' data.
3. Reported case counts in real silent or shadow deployments of sepsis or deterioration models: for each, setting (single ICU, ED, whole hospital, health system), duration, number of patients or encounters, number of outcome cases. Include those we already know (Duke Sepsis Watch, UCSD COMPOSER, Epic ESM at Tufts, Penn, a pediatric ward study) only if you can add the case counts; prioritise ICU-level deployments and smaller or non-academic hospitals.
4. Sample-size requirements for model updating at a new site: minimum events for (a) recalibration-in-the-large / intercept, (b) logistic recalibration (intercept and slope), (c) re-estimating all coefficients, ideally from Riley et al. (sample size for external validation and for development, pmsampsize), Vergouwe et al. 2017, Steyerberg et al. 2004, Van Calster, Collins et al. ("at least 100 events" / 200 events for validation), and any study comparing updating methods by target sample size. Also any study of how much local data fine-tuning a neural network (e.g. a deep sepsis model) needs at a new hospital, with the case counts used.

## Optional

5. Any paper that discusses the practical time to accumulate enough local events for recalibration or refitting (e.g. "it would take N months to collect 100 events").

## Output

Write a compact report (max about 900 words plus tables):
- Table 1: ICU volume and sepsis incidence figures (number, setting, source).
- Table 2: silent deployments with setting, duration, patients, cases (source).
- Table 3: events needed per updating method (source, recommendation, assumptions).
- A short paragraph: given these numbers, roughly how many sepsis cases would a 1-3 month run-in yield for (a) one ICU, (b) a whole hospital's ICUs, (c) a health system; show the arithmetic and label it as derived.
- References list.

Do not write or modify any files other than your answer. No code.
