# Rigor additions: temporal validation, mixture analysis, PR curve, SHAP checks, correlations, reporting checklists

These scripts check the main analysis rather than add new findings. They use
the same cohort (10,879 adults), split, predictors and leading model
(CatBoost with the hyperparameters from `06_Code/06_cv_tuning.py`). The
temporal validation is reported in the manuscript (Results section 5,
Table S10); the rest are supporting material.

Run them after the main pipeline (the launcher does this). Tables and figures
are written to `outputs/08_Q1_Rigor_Additions/` and copied into `tables/` and
`figures/` here by `06_Code/26_build_submission.py`.

## 1. Temporal validation (`code/06_temporal_validation.py`)

The main result comes from one random split of the pooled cycles, which tests
generalization to a random subsample of the same population, not forward in
time. Here CatBoost was re-tuned by 10-fold cross-validation on the 2007-2014
cycles only (n = 7,786), refitted there, and tested on 2015-2018 (n = 3,093),
which it never saw (`tables/table_temporal_validation.csv`,
`figures/fig_temporal_validation_roc.png`).

| Design | Test n | AUROC (95% CI) | Calibration slope | Calibration-in-the-large | O:E |
|---|---|---|---|---|---|
| Random split (primary) | 2,176 | 0.853 (0.837 to 0.869) | 1.06 | −0.051 | 0.983 |
| Temporal holdout | 3,093 | 0.833 (0.819 to 0.846) | 1.06 | −0.008 | 0.997 |

Discrimination drops by about 0.02 forward in time and calibration holds.
The earlier version reused hyperparameters tuned on a training set that
contained most of the 2015-2018 participants; it now re-tunes on 2007-2014
alone. The calibration intercept is now calibration-in-the-large (slope fixed
at 1), the same quantity as in the main calibration table.

## 2. Weighted quantile sum mixture (`code/11_wqs_mixture.py`)

A WQS regression treats the five metals as one mixture. Weights were
estimated by 200 bootstraps on a 40% training split and the index was tested
on the remaining 60% (n = 3,209 with all five metals), adjusted for age, BMI,
sex, income, race/ethnicity, education, smoking, alcohol and cycle.

The mixture index was not associated with MetS: OR 0.994 per quartile
(p = 0.936; `tables/table_wqs_validation_test.csv`). Selenium took the largest
weight (0.46, 95% interval 0.19 to 1.00), but with a null index the weights
describe the fit rather than a mixture effect
(`tables/table_wqs_component_weights.csv`, `figures/fig_wqs_component_weights.png`,
`figures/fig_wqs_dose_response.png`). The positive-direction constraint of WQS
also means it cannot detect a mixture that lowers risk.

## 3. Precision-recall curve (`code/07_pr_curve.py`)

AUPRC 0.823 for CatBoost against a no-skill baseline equal to the test
prevalence, 0.467 (`tables/table_precision_recall_summary.csv`,
`figures/fig_precision_recall_curve.png`). With near-balanced classes the
PR curve adds little beyond the ROC curve; it is kept because TRIPOD+AI
asks for it.

## 4. SHAP waterfalls (`code/08_shap_waterfall.py`)

One correctly classified participant with MetS (predicted probability 0.95)
and one without (0.003). In both, BMI makes the largest single contribution
(`tables/table_shap_waterfall_examples.csv`,
`figures/fig_shap_waterfall_true_positive.png`,
`figures/fig_shap_waterfall_true_negative.png`).

## 5. SHAP importance stability (`code/09_shap_stability.py`)

Mean |SHAP| recomputed on 2,000 bootstrap resamples of the test set with the
model held fixed (`tables/table_shap_importance_stability.csv`,
`figures/fig_shap_importance_stability.png`). Nine predictors are in the top
ten in every resample; sex and smoking intensity share the tenth place. No
metal enters the top ten in any resample. This measures resampling of the
test rows only; it does not refit the model, so it understates the
instability shown across models in Figure 4B of the manuscript.

## 6. Predictor correlations (`code/10_correlation_heatmap.py`)

Spearman correlations among the predictors on the training partition (after imputation)
(`tables/table_predictor_correlation_matrix.csv`,
`figures/fig_predictor_correlation_heatmap.png`). Three pairs exceed
|rho| = 0.5 (`tables/table_predictor_correlation_flags.csv`): any alcohol and
drinks per drinking day (0.81, by construction, since non-drinkers have 0
drinks), smoking intensity and blood cadmium (0.55), and ever smoked and
smoking intensity (0.55). Because smoking is the main source of cadmium,
part of any cadmium association can reflect smoking; both smoking items are
in the model and in every adjustment set for that reason.

## 7. TRIPOD+AI and PROBAST-AI (`code/12_reporting_checklists.py`)

An item-by-item TRIPOD+AI checklist for the Methods and Results drafts
(`tables/table_tripod_ai_checklist.csv`) and a domain-level PROBAST-AI
risk-of-bias summary (`tables/table_probast_ai_risk_of_bias.csv`). Numbers in
both are read from the pipeline outputs when the script runs. Items that
belong to sections not yet written (title page, abstract, introduction,
discussion, funding) are marked as not addressed at this draft stage.
