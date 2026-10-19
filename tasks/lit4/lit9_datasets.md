# Research task: datasets with several hospitals for testing a per-hospital "trust dial" on testing patterns

You are a research assistant with live web search. Use primary sources (dataset papers, PhysioNet or repository pages, benchmark papers). Give a link for every claim. Do not guess; mark anything not found as "not found".

## Context

We train a clinical prediction model (main task: sepsis onset in ICU; other ICU outcomes such as mortality, acute kidney injury or circulatory failure are acceptable) at hospital A and deploy it at hospital B. The model uses both physiological values and the testing pattern (which labs/vitals were measured, and how often). A scalar w in [0, 1] (or above) scales how much the model trusts the testing pattern; we tune w on a small "run-in" sample from B. So far we have one real pair (PhysioNet/CinC 2019: Beth Israel to Emory), where the best w sat at 1. We need many source-target hospital pairs, ideally with clearly different testing practices, to see how often the best w lies strictly between 0 and 1.

We are students in an ETH Zurich Deep Learning course project (deadline for the proposal 25 Oct 2026; project runs about 8 weeks), on CPU laptops.

## Questions (essential)

1. List candidate datasets with **more than one hospital or care site that can be separated** (hospital ID, site, unit type, or time periods with a documented change of practice or EHR system). For each give:
   - name, link, number of hospitals/sites and patients (ICU stays), years, country;
   - whether a hospital/site identifier is available to users;
   - hourly or time-stamped vitals and labs (so a testing pattern can be built);
   - sepsis labels or a published sepsis definition implementation for it (e.g. Sepsis-3 code in a benchmark), and other outcome labels;
   - access: open, credentialed (e.g. PhysioNet credentialing with CITI training), or application to a committee; typical time to get access; licence restrictions relevant to students.
   Candidates to check at least: eICU Collaborative Research Database (and eICU-CRD v2.0 hospital IDs and per-hospital sizes), MIMIC-III (CareVue vs MetaVision periods) and MIMIC-IV, HiRID, AmsterdamUMCdb, SICdb (Salzburg), the PhysioNet/CinC 2019 challenge (is any data of the hidden third hospital released?), INSPIRE, the Paediatric Intensive Care database (PIC), Zhejiang or other Chinese ICU databases, MIMIC-IV-ED, and any multi-centre sepsis datasets released after 2022.
2. Harmonised multi-dataset benchmarks that already build sepsis cohorts across several of these databases, e.g. "Yet Another ICU Benchmark" (YAIB, van de Water et al.), ricu (R package), Moor et al. 2023 multi-site sepsis study, BlendedICU, and whether their code gives hourly features plus missingness indicators per hospital. State whether they work at the hospital level within eICU.
3. For eICU specifically: how many hospitals have at least about 500 or 1,000 ICU stays; whether published work documents that lab-ordering or charting practice differs strongly between eICU hospitals (cite); whether vitals are hourly and labs time-stamped; sepsis label options.
4. Any published study that already trains on one hospital and evaluates across many hospitals of eICU (or similar) **with attention to missingness or testing patterns**, so we know the closest prior work.

## Optional

5. Fully open (no credentialing) multi-hospital datasets we could start with this week.

## Output

A compact report (max about 900 words plus tables):
- Table 1: datasets (columns as in question 1).
- Table 2: harmonised benchmarks and what they give.
- Short eICU section (question 3).
- Closest prior work (question 4), one line each.
- A recommendation: which dataset(s) let us get the most hospital pairs fastest, given access delays and an 8-week project.
- References list.

Do not write or modify any files other than your answer. No code.
