You are a literature researcher. Use live web search. Write no files.
CONTEXT: We propose setting one parameter of a sepsis prediction model (how much it trusts the testing pattern) at a new hospital during a silent run-in period, and we measure how many local patients and sepsis cases are needed. We need real-world anchors.
QUESTIONS:
1. Silent (shadow) deployments of sepsis or deterioration models in hospitals: which ones are documented (e.g. Epic Sepsis Model local validations, Duke Sepsis Watch, UCSD COMPOSER, Johns Hopkins TREWS, Kaiser AAM, Michigan, others), how long did the silent period last, and how many patients, encounters and sepsis cases did it accrue?
2. Guidance on silent trials and local validation of clinical AI (e.g. Kwong et al. 2022 "The silent trial", Wiens et al. 2019 "Do no harm", FDA or NICE guidance, TRIPOD-AI, DECIDE-AI): recommended durations or sample sizes, if any.
3. Typical sepsis incidence in ICU and ward cohorts used in those deployments, so we can convert weeks of run-in into numbers of cases.
4. Any study measuring how much local data is needed to adapt or recalibrate a sepsis model at a new site (learning curves of local updating).
RULES: cite only sources you opened (abstract at least), with URL or DOI; never invent numbers; mark anything uncertain.
OUTPUT (compact markdown, under 800 words): a table | deployment or guideline | URL/DOI | silent period length | patients/encounters | sepsis cases or rate | note |; then 3-5 bullets of the most useful anchors for converting run-in time into patients and cases; then the answer to question 4.
