# Exploratory additions: dose-response, metal interactions, component-level models, effect modification

These analyses sit on top of the main pipeline. They use the same cohort
(10,879 adults), the same predictors and the same leading model (CatBoost, with the hyperparameters chosen in
`06_Code/06_cv_tuning.py`). They are exploratory: none was prespecified, and
none is reported in the manuscript Results. Treat every p-value here as
hypothesis-generating.

Run them after the main pipeline (the launcher does this):

    python 07_Novelty_Additions/code/01_dose_response.py      # and 02, 03, 04

Tables and figures are written to `outputs/07_Novelty_Additions/` and copied
into `tables/` and `figures/` here by `06_Code/26_build_submission.py`.

The SHAP summaries use the model fitted on the training partition and the
held-out test participants. The regression models use every participant in
the cohort with the relevant metals observed (test participants included),
because they estimate associations rather than predictive performance. All
regression models below adjust for the same set: age, BMI, sex,
income-to-poverty ratio, race/ethnicity, education, smoking (ever and
current intensity), any alcohol in the past 12 months, and survey cycle.
Metals enter on the log scale. They are fitted on participants with the
metal(s) observed, never on imputed values.

## 1. Dose-response (`code/01_dose_response.py`)

A logistic model with a natural cubic spline (4 df) for each metal was
compared with the same model using a linear term, by likelihood-ratio test
(`tables/table_dose_response_lrt.csv`, `figures/fig_dose_response_confirmation.png`).

| Metal | n | LRT chi-square (3 df) | p |
|---|---|---|---|
| Lead | 9,301 | 10.19 | 0.017 |
| Cadmium | 9,301 | 12.51 | 0.006 |
| Mercury | 9,301 | 5.43 | 0.143 |
| Selenium | 5,349 | 2.77 | 0.428 |
| Manganese | 5,349 | 3.17 | 0.366 |

Lead and cadmium depart from linearity on the log scale (both remain below
0.05 after Benjamini-Hochberg correction across the five tests: q = 0.043 and
0.030). A segmented model for lead places a breakpoint at 2.39 µg/dL, but its
bootstrap 95% interval (0.74 to 2.75 µg/dL, from only 80 bootstrap
replicates, so the interval itself is rough) spans most of the searched range,
so the location is poorly identified (`tables/table_lead_breakpoint.csv`,
`figures/fig_lead_breakpoint.png`). Read the curves in the figure for the
shape; the test only says that a straight line on the log scale does not fit
lead and cadmium.

## 2. Metal-metal interactions (`code/02_shap_interactions.py`)

Two kinds of evidence are kept separate.

* Descriptive: mean absolute SHAP interaction values from CatBoost for each
  of the 10 metal pairs, computed only on test participants who had both
  metals measured (`tables/table_metal_interaction_ranking.csv`,
  `figures/fig_metal_interaction_heatmap.png`).
* Inferential: for each pair, a logistic model with both log metals, their
  product and the adjustment set, tested by likelihood ratio, with
  Benjamini-Hochberg control across the 10 pairs.

No metal-metal interaction survived FDR control (smallest q = 0.218). The
two pairs with nominal p < 0.05 were lead × selenium (p = 0.027) and
mercury × selenium (p = 0.045). The selenium × cadmium antagonism that the
earlier draft reported is not supported (p = 0.151, q = 0.303;
`tables/table_selenium_antagonism_test.csv`). The earlier version tested
Spearman correlations between SHAP interaction values and metal levels,
including imputed values; that approach was replaced because it measures
the fitted model's internal structure, not an association in the data.

## 3. Component-level models (`code/03_component_level.py`)

The leading model was refitted once for each of the five MetS components,
with the same predictors and hyperparameters (no SMOTE, because component
prevalence ranges from 38% to 57%).

| Component | Test AUROC | Test prevalence |
|---|---|---|
| Waist circumference | 0.968 | 57.4% |
| Blood pressure | 0.835 | 46.0% |
| Fasting glucose | 0.770 | 55.3% |
| Triglycerides | 0.749 | 37.6% |
| HDL cholesterol | 0.716 | 42.3% |

The waist model's 0.968 comes from BMI, which tracks waist circumference
almost one to one; it says nothing about metals. Mean |SHAP| by metal and
component is in `tables/table_metal_by_component_shap.csv` and
`figures/fig_metal_by_component_heatmap.png`. The largest values are lead for
HDL (0.090), selenium for triglycerides (0.076), mercury for waist (0.069),
cadmium for glucose (0.061) and manganese for blood pressure (0.052). These
are single-model attributions with no uncertainty interval, so they suggest
where to look in the MR phase rather than show where an effect lies.

## 4. Effect modification by age, sex and BMI (`code/04_effect_modification.py`)

For each of the 15 metal × modifier pairs, a logistic model with the log
metal, the modifier, their product and the adjustment set was tested by
likelihood ratio, with Benjamini-Hochberg control across the 15 tests
(`tables/table_metal_demographic_interaction.csv`,
`figures/fig_metal_demographic_interaction.png`). Six survived at q < 0.05:

| Interaction | Beta (log metal × modifier) | q |
|---|---|---|
| Lead × BMI | 0.039 | < 0.001 |
| Lead × age | −0.013 | < 0.001 |
| Mercury × BMI | 0.024 | < 0.001 |
| Lead × sex (female) | 0.234 | 0.010 |
| Cadmium × sex (female) | 0.174 | 0.021 |
| Manganese × BMI | −0.039 | 0.049 |

Manganese × BMI only just clears the threshold. The pattern centres on lead: its association with MetS changes with
BMI, age and sex. `figures/fig_top_effect_modification.png` shows the
strongest one. The manganese × age finding that the earlier draft called its
headline result does not hold up once it is tested in a regression model
(q = 0.888); it came from correlating SHAP interaction values on imputed
rows.

## What changed from the earlier draft

The earlier README described results from a cohort that included pregnant
and non-fasting participants, used first-reading blood pressure, mis-coded
some refusal codes and skip patterns, adjusted only for age, BMI, sex and
income, and tested interactions with SHAP correlations. Every number above
comes from the corrected pipeline. The earlier "mixture score" script
(05_mixture_score.py) and the ML-to-MR hypothesis dossier were removed; the
proper mixture analysis is the weighted quantile sum in
`08_Q1_Rigor_Additions`.
