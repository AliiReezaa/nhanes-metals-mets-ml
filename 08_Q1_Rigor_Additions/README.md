# Q1 rigor additions: temporal validation, real WQS mixture analysis, PR curve, SHAP waterfalls, importance stability, correlation heatmap, TRIPOD+AI/PROBAST-AI checklists

Built on top of the same faithfully-replicated CatBoost model used in
`07_Novelty_Additions` (test AUROC reproduces the manuscript's 0.8567
exactly). Where `07_Novelty_Additions` asked "what new findings can
strengthen the story," this folder answers a different question you asked
afterward: "what is still genuinely *missing* on the AI side, independent of
any new finding" -- the reporting-rigor and validation gaps a picky Q1
reviewer would flag. Same house style throughout: `pubstyle.py`
(Okabe-Ito/cividis, Liberation Sans, no baked-in titles, 600 dpi PNG + vector
PDF for every figure), human-readable table columns, journal-formatted
p-values.

**This is exploratory/discussion material for your professor, same as
before -- not a finished manuscript section.**

## 1. Temporal (forward-in-time) validation (`code/06_temporal_validation.py`)

Every result anywhere in this submission -- the manuscript's own AUROC
0.8567 and everything in `07_Novelty_Additions` -- comes from one random
SEQN split that pools all six NHANES cycles (2007-2018) before splitting.
That never tests whether the model generalizes *forward in time*, only
whether it generalizes to a random resample of the same pooled distribution
it trained on. The Methods draft itself already flags this precise gap
(Methods, Section 10: "No external or temporal validation was performed,
though the six-cycle design would permit holding out the most recent
cycle") -- this addition is a direct answer to that self-identified
limitation.

Same predictors, same preprocessing, same tuned hyperparameters; only the
partition changes, from random-by-SEQN to a hard cutoff: train on
2007_2008-2013_2014 (n=8,102), test on 2015_2016+2017_2018 (n=3,042),
never seen in any form during training.

**Result: a modest, honest drop.** AUROC falls from 0.857 (random split) to
0.840 (temporal holdout) -- a real but small degradation, and calibration is
if anything *better* under the temporal split (slope 0.987 vs. 1.064,
observed:expected 0.986 vs. 0.970). The model generalizes forward in time
reasonably well; it is not simply memorizing the pooled cross-sectional
distribution.

Outputs: `figures/fig_temporal_validation_roc.png/.pdf`,
`tables/table_temporal_validation.csv`.

## 2. A real WQS mixture analysis (`code/11_wqs_mixture.py`)

`07_Novelty_Additions/code/05_mixture_score.py` was explicitly flagged there
as "the lightweight analogue" of a real mixture-index method -- a single
unconstrained logistic regression collapsed into one weighted sum, not
actual Weighted Quantile Sum (WQS) regression. This replaces it with the
standard Carrico et al. (2015) WQS estimator used throughout
environmental-epidemiology metals-mixture papers: quartile-scored metals,
component weights estimated by constrained (simplex, non-negative-direction)
optimization over 200 bootstrap resamples of a 40% training partition, and
the resulting index tested in a single adjusted logistic regression fit only
on the held-out 60% validation partition -- so the final p-value is not
inflated by the same data used to estimate the weights.

**Result: a real, validated, and honestly surprising signal.** The WQS
index is significantly associated with MetS in the validation partition
(beta=0.160, P=0.011; OR=1.61 comparing the highest vs. lowest
quartile-burden category) -- a materially stronger and more defensible
result than the informal proxy's in-sample p=0.019. The headline is *which*
metal drives it: **selenium carries by far the largest component weight
(0.51 of 1.0)**, more than double manganese (0.19) and cadmium (0.16), with
lead and mercury contributing almost nothing (0.06 and 0.08, both with CIs
touching zero). This is worth flagging rather than smoothing over: selenium
is conventionally framed as protective/antioxidant, but its dominant weight
here is in the *harmful-direction* index, consistent with a growing
literature that high (not just low) selenium status has been positively
associated with diabetes/MetS-type outcomes in some populations -- this is
a specific, testable claim for the MR arm, not a contradiction to explain
away.

Outputs: `figures/fig_wqs_component_weights.png/.pdf`,
`figures/fig_wqs_dose_response.png/.pdf`,
`tables/table_wqs_component_weights.csv`,
`tables/table_wqs_validation_test.csv`.

## 3. Precision-recall curve (`code/07_pr_curve.py`)

AUPRC (0.8114 for CatBoost) was already in `model_comparison.csv`, but there
was no PR curve figure to go with it -- only the ROC curve (Figure 2). Added
as the standard companion; reproduces the exact reported AUPRC.

Outputs: `figures/fig_precision_recall_curve.png/.pdf`,
`tables/table_precision_recall_summary.csv`.

## 4. Individual-level SHAP waterfall plots (`code/08_shap_waterfall.py`)

Every SHAP figure anywhere in this submission (Figure 3, Figure 5, and
everything in `07_Novelty_Additions`) is population-level. This shows the
most-confidently-correct true positive (P(MetS)=0.969, actual=1) and true
negative (P(MetS)=0.004, actual=0) in the test set, with the exact
feature-by-feature contribution that produced each individual prediction --
the standard way to make an interpretability story concrete for a reader who
is not going to parse a beeswarm plot unaided. Both are dominated by BMI and
age, consistent with the population-level ranking.

Outputs: `figures/fig_shap_waterfall_true_positive.png/.pdf`,
`figures/fig_shap_waterfall_true_negative.png/.pdf`,
`tables/table_shap_waterfall_examples.csv`.

## 5. Bootstrap stability of feature importance (`code/09_shap_stability.py`)

Every importance ranking in the manuscript (`permutation_importance.csv`,
`shap_importance.csv`, and the new stability analysis here) was previously a
single point estimate with no uncertainty -- no way to tell whether feature
8 is really different from feature 11. This bootstraps the test set
(2,000 resamples, no refitting -- the fixed model's SHAP matrix is
resampled row-wise, the standard nonparametric approach for this question)
and reports both a 95% CI per feature and how often each feature lands in
the top 10 across resamples.

**Result: the ranking is more stable than it looks, with one clean
boundary.** The top 9 predictors (BMI, age, vigorous recreational activity,
resting pulse, and five demographic/lifestyle variables) land in the top 10
in 99.9-100% of resamples -- genuinely stable. Blood cadmium is the last
reliably-stable entrant. Below that, blood lead, selenium, sex, mercury and
income-to-poverty ratio are essentially tied and interchangeable for the
10th spot (0-0.1% inclusion each) -- meaning any claim about which single
predictor is "10th most important" is noise, not signal, and should not be
made.

Outputs: `figures/fig_shap_importance_stability.png/.pdf`,
`tables/table_shap_importance_stability.csv`.

## 6. Full predictor correlation heatmap (`code/10_correlation_heatmap.py`)

`collinearity_flags.csv` only ever showed the one pair that got excluded
(BMXHIP vs. BMXBMI, r=0.94) -- there was no figure showing the correlation
structure of the 33 predictors that were actually kept. This is the full
Spearman correlation matrix, hierarchically clustered for readability (one
constant column, `RIDRETH1_nan`, was dropped as undefined-correlation since
no participant in the training partition has missing race/ethnicity).

**Result: reassuring.** Only two pairs reach |rho| >= 0.5 among the 33
retained predictors (ever-smoked/smoking-intensity at 0.557, an expected
structural relationship since one gates the other; smoking-intensity/blood
cadmium at 0.556, the expected smoking-cadmium pathway) -- nothing else
approaches even moderate collinearity. The predictor-selection screen
described in Methods Section 4 held up.

Outputs: `figures/fig_predictor_correlation_heatmap.png/.pdf`,
`tables/table_predictor_correlation_matrix.csv`,
`tables/table_predictor_correlation_flags.csv`.

## 7. TRIPOD+AI checklist and PROBAST-AI risk-of-bias summary (`code/12_reporting_checklists.py`)

The Methods draft already states "Reporting follows TRIPOD+AI [3] and
STROBE [4]" (Methods, Section 1) -- but nowhere in any of the three
manuscript PDFs is there an itemized checklist backing that claim up (a
direct text search for "TRIPOD"/"PROBAST" across all three finds only the
one citation). Q1 clinical-ML journals routinely ask for the completed
checklist as a supplementary file, so this substantiates a claim that is
already being made rather than introducing a new one. Built by actually
reading Methods_AI.pdf and Results_AI.pdf section-by-section and citing
exactly where, if anywhere, each item is addressed -- not inferred from the
README.

**Honest tally across the 27 items (52 sub-items):** 23 satisfied, 11
partial, 6 not applicable (mostly subjective-assessment items that don't
apply to lab-based/survey data), and 12 not addressed *at this draft
stage specifically because the sections they belong to don't exist yet* --
Title, Abstract, Introduction and Discussion are genuinely not written in
this Methods+Results-only package, and items 18/19 (funding, conflicts of
interest, protocol, registration, patient/public involvement) have no
statement anywhere yet. Those are not failures of what exists; they are
exactly the sections you told me to set aside ("forget about the
manuscripts... this is all only as drafts"). Within what Methods and
Results actually cover, TRIPOD+AI compliance is strong -- the missing-data
handling, leakage prevention, calibration/decision-curve reporting, and
sensitivity-analysis sections in particular are more thorough than most
published papers in this space.

**PROBAST-AI domain summary:** Low risk of bias in Participants and
Predictors (well-defined cohort, pre-specified predictors, no data-driven
selection). "Some concerns" in Outcome -- not because the outcome is poorly
measured, but because the manuscript's own sensitivity analysis shows the
outcome-definition choice (harmonized vs. lab-only) moves discrimination by
0.029-0.034 AUROC units, the largest swing of any sensitivity arm, meaning a
different defensible definition would have produced a different headline
number (disclosed, not hidden -- but real). "Some concerns" in Analysis,
per the single-random-split issue that item 1 above (temporal validation)
now partially -- not fully -- addresses.

Outputs: `tables/table_tripod_ai_checklist.csv`,
`tables/table_probast_ai_risk_of_bias.csv`.

## Bottom line

Ranked by how much this changes what you can honestly claim: (1) the real
WQS analysis (section 2) is the strongest new result in this batch -- a
validated, out-of-sample p=0.011 with a genuinely interesting and
literature-consistent headline (selenium dominates the mixture weight,
not the usual suspects lead/cadmium). (2) Temporal validation (section 1)
is the most important *rigor* addition, because it closes a gap the authors
already flagged in their own Methods section, and the answer is reassuring
(a small, not alarming, AUROC drop). (3) The TRIPOD+AI/PROBAST-AI checklists
(section 7) cost nothing scientifically but are the cheapest credibility
signal in the whole package -- they turn an unverifiable blanket claim
("reporting follows TRIPOD+AI") into an auditable one. (4) The remaining
four (PR curve, SHAP waterfalls, importance stability, correlation heatmap)
are genuine gaps closed with no surprises in the results -- exactly what you
want from polish items: cheap, expected, and one fewer thing for a reviewer
to ask for.
