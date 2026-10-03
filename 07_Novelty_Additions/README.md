# Novelty additions: dose-response confirmation, metal interactions, component-level modeling, effect modification, mixture score

Built on top of the existing SUBMISSION pipeline (same predictors, same
train/test split by SEQN, same tuned CatBoost hyperparameters from
`best_params.json`). The refit CatBoost model reproduced the manuscript's
reported test AUROC exactly (0.8567), confirming the replication is
faithful before any new analysis was run on top of it.

**This is exploratory work for discussion with your professor, not a
finished manuscript section.** Some of what follows complicates the
existing SHAP-based story rather than confirming it -- that is reported
honestly below rather than smoothed over.

**Figure/table style note:** every figure below is built with your own
`pubstyle.py` house style (the same module used elsewhere in `06_Code/`):
Okabe-Ito colorblind-safe palette for line/scatter plots, the cividis
colormap for heatmaps, Liberation Sans (metrically identical to Arial),
600 dpi raster + vector PDF for every figure (`figures/*.png` and the
matching `figures/*.pdf`), panel labels (A/B/C) on multi-panel figures,
and no titles baked into the images -- per that module's own convention,
the caption belongs in the surrounding text (i.e., the descriptions
below), not burned into the picture. All p-values are given in
journal-standard formatting (3 decimals, or scientific notation below
0.001) and every CSV table uses human-readable column headers.

## 1. Dose-response confirmation (`code/01_dose_response.py`)

Fit a logistic-GAM (natural cubic spline, df=4) for lead, cadmium and
selenium, adjusted for age/BMI/sex/income-to-poverty ratio, on the full
analytic cohort, and tested spline-vs-linear with a likelihood-ratio test.

**Honest result: none of the three reached statistical significance for
non-linearity** (lead p=0.21, cadmium p=0.89, selenium p=0.39; a
full-32-predictor-adjustment sensitivity check for lead alone gave a
consistent p=0.22, at the cost of dropping to n=1,743 via complete-case
deletion). The segmented (two-piece) model for lead identified a
"breakpoint" at 2.80 µg/dL, but its bootstrap 95% CI (0.50-2.80 µg/dL)
spans almost the entire search range -- the location is not well
identified. The plotted spline curve for lead shows an overall *decline*
across most of its range that flattens at the tail, not the clear
rise-then-fall inversion the SHAP dependence plot visually suggests.

**What this means:** the SHAP-observed shapes are real properties of the
fitted CatBoost model (that part is not in question), but they are not
independently confirmed as statistically distinguishable from a linear
(or, for lead, monotonic) association once tested in an adjusted
parametric model. This is a legitimate, reportable finding in its own
right -- it's evidence that a flexible tree ensemble can produce a
visually compelling non-monotonic SHAP curve that a formal test does not
support at this sample size, which is exactly the kind of caution a Q1
reviewer would want to see rather than have to raise themselves.

Outputs: `figures/fig_dose_response_confirmation.png/.pdf` (3-panel: A lead,
B cadmium, C selenium), `figures/fig_lead_breakpoint.png/.pdf`,
`tables/table_dose_response_lrt.csv`, `tables/table_lead_breakpoint.csv`.

## 2. Metal-metal SHAP interactions (`code/02_shap_interactions.py`)

Computed SHAP interaction values (TreeExplainer) for all 5x5 metal pairs
on the full held-out test partition (n=2,229).

**Selenium x cadmium**: a real, clearly significant signal. Mean
|interaction| = 0.0072 (4th-strongest of the 10 metal pairs), and the
interaction term becomes more negative as cadmium rises (Spearman
rho=-0.33, p=6x10^-57) -- directionally consistent with the known
toxicological antagonism between selenium and cadmium.

**Selenium x lead**: a clear null. Weakest of the 10 pairs (mean
|interaction| = 0.0030), and no significant association with lead level
(rho=0.013, p=0.54). No support here for a selenium-lead antagonism.

**Incidental finding, not hypothesized in advance**: lead x manganese
(0.0118) and mercury x manganese (0.0103) were the two strongest
metal-metal interaction pairs in the model, stronger than either
selenium pairing. Worth noting but not chased further here -- flagged as
an observation for follow-up, not a tested hypothesis.

Outputs: `figures/fig_metal_interaction_heatmap.png/.pdf`,
`figures/fig_selenium_interaction_dependence.png/.pdf` (2-panel: A selenium
x cadmium, B selenium x lead), `tables/table_metal_interaction_ranking.csv`,
`tables/table_selenium_antagonism_test.csv`.

## 3. Component-level modeling (`code/03_component_level.py`)

Refit the same predictor set against each of the 5 individual MetS
components separately (5 independent CatBoost models), to see which
physiological pathway each metal's signal concentrates in.

| Metal | Concentrates in | SHAP value | Caveat |
|---|---|---|---|
| Selenium | Triglycerides | 0.091 (vs. 0.015-0.035 elsewhere) | Consistent with selenoprotein/lipid biology |
| Cadmium | Fasting glucose | 0.059 (vs. 0.025-0.049 elsewhere) | Consistent with cadmium's known beta-cell toxicity |
| Lead | HDL cholesterol | 0.078 (vs. 0.021-0.045 elsewhere) | -- |
| Mercury | Waist circumference | 0.100 (vs. 0.021-0.053 elsewhere) | **Read cautiously** -- see below |
| Manganese | Blood pressure | 0.040 (fairly flat overall, 0.017-0.040) | Weakest signal of the five, consistent with earlier permutation-importance finding |

**Important caveat on the waist-circumference component model**: it
reached AUROC=0.97, which is not a genuine metals-driven result -- it's
driven by BMI-waist collinearity (BMI is a direct predictor of a
component defined by a correlated anthropometric measurement). Mercury's
apparent concentration there should be read with that in mind, not as a
clean adiposity-specific mercury effect; residual confounding by fish
consumption (a source of both mercury exposure and dietary pattern) is a
more likely explanation than a first-pass reading would suggest.

This is a single-model-per-component comparison with no bootstrap CI
across components -- directionally informative, not formally tested for
significance.

Outputs: `figures/fig_metal_by_component_heatmap.png/.pdf`,
`tables/table_component_level_auc.csv`,
`tables/table_metal_by_component_shap.csv`.

## 4. ML-to-MR hypothesis dossier (`tables/table_ml_to_mr_hypothesis_dossier.csv`)

Translates each finding above into a specific, appropriately-hedged
testable hypothesis for the Mendelian randomization arm, explicitly
framing this ML analysis as hypothesis generation rather than as an
endpoint in itself. Each row states the strength of evidence honestly
(several are "weak on its own" or "moderate, single-model" rather than
"confirmed") so the MR side knows which hypotheses are well-supported
enough to prioritize and which are exploratory leads.

## 5. Metal x demographic effect modification (`code/04_effect_modification.py`)

Reused the already-fitted CatBoost model's SHAP interaction tensor (no
refitting) to test all 15 metal x {age, sex, BMI} interaction terms on
the held-out test partition, with a Bonferroni correction for the 15
comparisons (alpha=0.0033).

**Headline finding: blood manganese x age is the single strongest and
most significant interaction found anywhere in this whole novelty
package** (Spearman rho=0.376, p=1.2x10^-75 -- far beyond Bonferroni).
This matters because manganese showed almost no *marginal* signal
anywhere else in the original analysis (flat SHAP dependence curve,
permutation importance indistinguishable from noise). Manganese is not
"uninformative" -- its association with MetS risk appears to be strongly
age-dependent rather than diffuse, which the original main-effects-only
analysis had no way of surfacing.

9 of the 15 interaction terms survived Bonferroni correction: manganese
x age, mercury x BMI, cadmium x age, manganese x BMI, selenium x BMI,
cadmium x sex, lead x sex, mercury x sex, manganese x sex. Lead x age
(the single largest RAW interaction magnitude) and selenium x sex did
NOT survive correction (p=0.025 and p=0.020 respectively) -- flagged
explicitly so the strongest-looking bar in the heatmap isn't mistaken for
the most statistically defensible one.

Outputs: `figures/fig_metal_demographic_interaction.png/.pdf`,
`figures/fig_top_effect_modification.png/.pdf` (highlights manganese x age,
the most statistically significant hit -- not lead x age, which has the
larger raw magnitude but does not survive correction),
`tables/table_metal_demographic_interaction.csv`.

## 6. Joint 5-metal mixture dose-response (`code/05_mixture_score.py`)

Addition 1 found no individual metal's dose-response curve was
significantly non-linear. This asks the complementary question: does the
**joint** 5-metal burden, combined into a single weighted mixture index
(logistic-regression weights on the 5 standardized metals, the
lightweight analogue of a weighted-quantile-sum index), behave
differently than any one metal alone?

**Result: yes, modestly.** The mixture score's linear term is not
significant on its own (p=0.151), but the spline-vs-linear LRT for the
combined mixture IS significant (chi2=9.94, df=3, **p=0.019**, n=6,317
metals-complete-case). The shape is a shallow U/J: predicted MetS
probability declines to a minimum at a slightly-below-average combined
metal burden, then rises again at both the low and high extremes,
consistent with the notion that it's the *joint* pattern across metals,
not any single one, that carries a detectable non-linear signal. CIs
widen substantially at both tails (fewer participants there), so this
should be read as suggestive rather than conclusive -- and p=0.019 is a
single a priori test, not corrected against the 15 tests in section 5.

Outputs: `figures/fig_mixture_score_dose_response.png/.pdf`,
`tables/table_mixture_score_test.csv`.

## Bottom line

Ranked by how defensible the evidence actually is: (1) the manganese x
age interaction (section 5) is the strongest finding in the entire
package -- essentially unimpeachable statistically, and it overturns the
original "manganese contributes nothing" reading by showing the null was
only true on average, not conditionally. (2) The selenium-cadmium
antagonism interaction (section 2) is the second-strongest -- large,
significant, and biologically motivated in advance rather than found by
trawling. (3) The joint metal-mixture nonlinearity (section 6) is real
but modest (p=0.019, single test, wide tails) -- a legitimate lead, not
a headline. (4) The component-level breakdown (section 3) is directionally
interesting but single-model and not formally tested. (5) The dose-response
confirmation attempt (section 1) is the most *scientifically important*
result precisely because it complicated the SHAP story instead of
confirming it -- reporting that honestly is worth more to the credibility
of the AI arm than a rubber-stamp would have been.
