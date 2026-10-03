"""
Single combined supplementary file for the machine-learning component, covering
both the Methods and the Results.

Journals want one supplement per manuscript, not one per section, and the
numbering here is continuous so that Table S1-S7 and Figure S1-S5 can be cited
from either section without collision:

  Table S1  predictor dictionary (Methods 4)
  Table S2  metabolic syndrome component definitions (Methods 3)
  Table S3  hyperparameter grids and selected configurations (Methods 7)
  Table S4  threshold-level classification metrics, 11 models (Results 2)
  Table S5  predictor importance, SHAP and permutation (Results 3)
  Table S6  subgroup discrimination and calibration (Results 5)
  Table S7  sensitivity analyses (Results 5)
  Figure S1 analysis pipeline (Methods 6)
  Figure S2 AUROC forest and paired-bootstrap differences (Results 2)
  Figure S3 calibration curves (Results 2)
  Figure S4 decision-curve analysis (Results 2)
  Figure S5 subgroup discrimination (Results 5)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docx_style as ds
from references import cite

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs" / "metrics"
FIGP = ROOT / "outputs" / "figures_pub"
OUT = ROOT / "manuscript"

N = json.loads((MET / "publication_numbers.json").read_text())
tune = pd.read_csv(MET / "tuning_summary.csv")
cmp_df = pd.read_csv(MET / "model_comparison.csv")
sg = pd.read_csv(MET / "subgroup_auc.csv")
shap_all = pd.read_csv(MET / "shap_importance.csv")
perm = pd.read_csv(MET / "permutation_importance.csv")
s_diq = pd.read_csv(MET / "sensitivity_diq010.csv")
s_met = pd.read_csv(MET / "sensitivity_metals_scope.csv")
s_smo = pd.read_csv(MET / "sensitivity_smote_vs_classweight.csv")
s_out = pd.read_csv(MET / "sensitivity_outcome_definition.csv")

n_pred = N["n_predictors"]
TOP = N["top_model"]
f = N["flow"]

shap33 = shap_all[~shap_all["feature_code"].str.startswith("missingindicator")].copy()
shap33["rank"] = shap33["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
perm["rank"] = perm["perm_importance_mean"].rank(ascending=False, method="min").astype(int)

MINUS = "−"


def num(v, dp=3):
    return f"{v:.{dp}f}".replace("-", MINUS)


def ci(lo, hi):
    return f"{num(lo)}–{num(hi)}"


PREDICTORS = [
    ("Blood metals", [
        ("LBXBPB", "Blood lead concentration", "Continuous, µg/dL", "1"),
        ("LBXBCD", "Blood cadmium concentration", "Continuous, µg/L", "1"),
        ("LBXTHG", "Blood total mercury concentration", "Continuous, µg/L", "1"),
        ("LBXBSE", "Blood selenium concentration", "Continuous, µg/L", "1"),
        ("LBXBMN", "Blood manganese concentration", "Continuous, µg/L", "1"),
    ]),
    ("Demographic and socioeconomic", [
        ("RIDAGEYR", "Age at screening", "Continuous, years", "1"),
        ("RIAGENDR", "Sex", "Binary (female = 1)", "1"),
        ("RIDRETH1", "Race/Hispanic origin", "One-hot; Mexican American = reference", "5"),
        ("DMDEDUC2", "Highest education level completed (adults)",
         "Ordinal 1–5 (<9th grade → college graduate)", "1"),
        ("DMDMARTL", "Marital status", "One-hot; married = reference", "6"),
        ("DMDHHSIZ", "Total number of people in the household", "Continuous, count", "1"),
        ("INDFMPIR", "Ratio of family income to the federal poverty threshold",
         "Continuous, 0–5", "1"),
        ("RIDEXMON", "Six-month period in which the examination took place",
         "Binary (May–October = 1)", "1"),
    ]),
    ("Anthropometric", [
        ("BMXBMI", "Body mass index from measured height and weight", "Continuous, kg/m²", "1"),
    ]),
    ("Lifestyle", [
        ("SMQ020", "Smoked at least 100 cigarettes in life", "Binary (yes = 1)", "1"),
        ("SMQ040", "Current cigarette smoking frequency",
         "Ordinal, recoded 0 = not at all, 1 = some days, 2 = every day", "1"),
        ("ALQ101 / ALQ111", "Alcohol gatekeeper item, harmonized across cycles: \"had at least 12 "
                            "alcoholic drinks in any one year\" (2007–2016) and \"ever had a drink "
                            "of any kind of alcohol\" (2017–2018)", "Binary (ever = 1)", "1"),
        ("ALQ130", "Average number of alcoholic drinks per day on drinking days, past 12 months",
         "Continuous, drinks/day", "1"),
        ("DBQ700", "Self-rated overall diet quality", "Ordinal 1–5 (excellent → poor)", "1"),
        ("PAQ605", "Vigorous-intensity activity as part of work", "Binary (yes = 1)", "1"),
        ("PAQ650", "Vigorous-intensity recreational activity", "Binary (yes = 1)", "1"),
        ("PAQ665", "Moderate-intensity recreational activity", "Binary (yes = 1)", "1"),
        ("PAD645", "Minutes spent walking or bicycling for transportation on a typical day "
                   "(gated on PAQ635, \"walk or use a bicycle for transportation\")",
         "Continuous, minutes/day", "1"),
        ("BPXPLS", "Resting pulse, 60-second count", "Continuous, beats/min", "1"),
    ]),
]

METS_CRITERIA = [
    ["Abdominal obesity", "Waist circumference ≥ 102 cm (men) or ≥ 88 cm (women)",
     "BMXWAIST", "Not applicable"],
    ["Elevated triglycerides", "Triglycerides ≥ 150 mg/dL", "LBXTR",
     "BPQ100D (currently taking prescribed medicine for cholesterol)"],
    ["Reduced HDL cholesterol", "HDL-C < 40 mg/dL (men) or < 50 mg/dL (women)", "LBDHDD",
     "BPQ100D (currently taking prescribed medicine for cholesterol)"],
    ["Elevated blood pressure", "Systolic ≥ 130 mmHg or diastolic ≥ 85 mmHg",
     "BPXSY1, BPXDI1", "BPQ050A (currently taking prescribed medicine for hypertension)"],
    ["Elevated fasting glucose", "Fasting plasma glucose ≥ 100 mg/dL", "LBXGLU",
     "DIQ050 (currently taking insulin); DIQ070 (taking pills to lower blood sugar)"],
]

GRIDS = {
    "Logistic Regression": "C ∈ {0.01, 0.1, 1, 10}; penalty = L2",
    "Decision Tree": "max_depth ∈ {3, 5, 8, None}; min_samples_leaf ∈ {1, 5, 20}",
    "Random Forest": "n_estimators ∈ {200, 500}; max_depth ∈ {None, 8, 15}; min_samples_leaf ∈ {1, 5}",
    "Gradient Boosting": "n_estimators ∈ {100, 200}; learning_rate ∈ {0.01, 0.1}; max_depth ∈ {2, 3}",
    "LightGBM": "n_estimators ∈ {200, 500}; learning_rate ∈ {0.01, 0.05, 0.1}; num_leaves ∈ {15, 31}",
    "XGBoost": "n_estimators ∈ {200, 500}; learning_rate ∈ {0.01, 0.05, 0.1}; max_depth ∈ {3, 5}",
    "CatBoost": "iterations ∈ {200, 500}; learning_rate ∈ {0.01, 0.05, 0.1}; depth ∈ {4, 6}",
    "SVM": "C ∈ {0.1, 1, 10}; kernel ∈ {RBF, linear}",
    "Naive Bayes": "none (no hyperparameters searched)",
    "MLP": "hidden_layer_sizes ∈ {(64), (64, 32)}; alpha ∈ {0.0001, 0.001}",
    "KNN": "n_neighbors ∈ {5, 15, 25}",
}

SHORT = {"Self-rated diet quality (1=excellent...5=poor)": "Self-rated diet quality",
         "Current smoking intensity (0=none, 1=some days, 2=every day)": "Current smoking intensity"}


def main():
    doc = ds.new_document(line_spacing="1.5")
    ds.title(doc, "Supplementary Material")
    ds.subtitle(doc, "Machine-learning analysis of blood metals and metabolic syndrome "
                     "in NHANES 2007–2018")

    # ================= Table S1 =================
    ds.caption(doc, "Table S1.",
               "Predictor dictionary. Each NHANES item code is given with a plain description of "
               "what the item records, its coding in this analysis, and the number of model columns "
               f"it contributes after encoding (total {n_pred}). Categorical variables with more "
               "than two levels were one-hot encoded with the stated reference category omitted.",
               above=True)
    rows = []
    for domain, items in PREDICTORS:
        first = True
        for code, desc, coding, ncol in items:
            rows.append([domain if first else "", code, desc, coding, ncol])
            first = False
    ds.add_table(doc, ["Domain", "NHANES code", "What the item records", "Coding", "Cols"],
                 rows, col_widths_in=[1.05, 0.88, 2.55, 1.52, 0.40],
                 align_right_from=4, font_size=ds.Pt(8))
    ds.table_note(doc,
                  "Survey modules used, all merged on the respondent sequence number (SEQN): "
                  "demographics (DEMO), body measures (BMX), blood pressure (BPX), triglycerides "
                  "(TRIGLY), HDL cholesterol (HDL), plasma fasting glucose (GLU), blood lead–"
                  "cadmium–mercury–selenium–manganese (PBCD), alcohol use (ALQ), diet behavior "
                  "(DBQ), smoking (SMQ), diabetes (DIQ), physical activity (PAQ), and blood "
                  "pressure and cholesterol (BPQ). Hip circumference (BMXHIP) and physician-"
                  "diagnosed diabetes (DIQ010) were screened but not retained in the primary "
                  "predictor set.")
    ds.page_break(doc)

    # ================= Table S2 =================
    ds.caption(doc, "Table S2.",
               "Metabolic syndrome component definitions. A component is scored positive when "
               "either the measured criterion or the drug-treatment criterion is met, following the "
               f"harmonized joint interim statement {cite('alberti')} and the earlier AHA/NHLBI "
               f"thresholds {cite('grundy')}. Participants meeting at least three of five "
               "components were classified as cases.", above=True)
    ds.add_table(doc, ["Component", "Measured criterion", "NHANES variable",
                       "Drug-treatment criterion"],
                 METS_CRITERIA, col_widths_in=[1.12, 1.92, 0.95, 2.41],
                 align_right_from=99, font_size=ds.Pt(8))
    ds.table_note(doc,
                  "HDL-C, high-density lipoprotein cholesterol. NHANES carries only one "
                  "cholesterol-medication item in the cycles used here and does not distinguish "
                  "HDL-specific from triglyceride-specific treatment, so BPQ100D is applied to both "
                  "lipid components. Component measurements and medication items were used solely "
                  "to construct the outcome and were excluded from the predictor set.")

    # ================= Table S3 =================
    rows3 = []
    for _, t in tune.sort_values("cv_auc", ascending=False).iterrows():
        bp = json.loads(t["best_params"]) if isinstance(t["best_params"], str) else {}

        def _fmt(v):
            return "(" + ", ".join(str(x) for x in v) + ")" if isinstance(v, list) else str(v)
        sel = ", ".join(f"{k.replace('model__', '')} = {_fmt(v)}" for k, v in bp.items()) or "—"
        rows3.append([t["model"], GRIDS.get(t["model"], "—"), sel, f"{t['cv_auc']:.4f}"])
    ds.caption(doc, "Table S3.",
               "Hyperparameter search space, selected configuration and cross-validated "
               "discrimination for each algorithm. Configurations were chosen by randomized search "
               "(capped at 12 configurations; smaller grids searched exhaustively) with 10-fold "
               "stratified cross-validation on the training partition, optimizing AUROC. *These "
               "values are tuning-selection statistics and are optimistically biased; "
               "test-partition performance is reported in the main text.*", above=True)
    ds.add_table(doc, ["Algorithm", "Search space", "Selected configuration", "CV AUROC"],
                 rows3, col_widths_in=[1.12, 2.05, 2.05, 0.68],
                 align_right_from=3, font_size=ds.Pt(7.5))
    ds.page_break(doc)

    # ================= Table S4 =================
    rows4 = []
    for _, r in cmp_df.iterrows():
        rows4.append([r["Model"], f"{r['Cutoff_tuned_cutoff']:.3f}",
                      f"{r['Accuracy_tuned_cutoff'] * 100:.1f}",
                      f"{r['Sensitivity_tuned_cutoff'] * 100:.1f}",
                      f"{r['Specificity_tuned_cutoff'] * 100:.1f}",
                      f"{r['PPV_tuned_cutoff'] * 100:.1f}", f"{r['NPV_tuned_cutoff'] * 100:.1f}",
                      f"{r['F1_tuned_cutoff']:.3f}", f"{r['Kappa_tuned_cutoff']:.3f}",
                      f"{r['Accuracy_at_0.5'] * 100:.1f}",
                      f"{r['Sensitivity_at_0.5'] * 100:.1f}",
                      f"{r['Specificity_at_0.5'] * 100:.1f}"])
    ds.caption(doc, "Table S4.",
               "Classification performance on the held-out test partition at the decision threshold "
               "selected by maximizing the Youden index on out-of-fold training predictions, and at "
               "the conventional 0.5 cut-point for comparison. All values except the threshold, F1 "
               "and κ are percentages.", above=True)
    ds.add_table(doc,
                 ["Model", "Thresh.", "Acc.", "Sens.", "Spec.", "PPV", "NPV", "F1", "κ",
                  "Acc.₀.₅", "Sens.₀.₅", "Spec.₀.₅"],
                 rows4,
                 col_widths_in=[1.05, 0.50, 0.44, 0.46, 0.46, 0.44, 0.44, 0.44, 0.44,
                                0.50, 0.52, 0.52],
                 align_right_from=1, font_size=ds.Pt(7))
    ds.table_note(doc,
                  "Acc., accuracy; Sens., sensitivity; Spec., specificity; PPV, positive predictive "
                  "value; NPV, negative predictive value; κ, Cohen's kappa. Subscript 0.5 denotes "
                  "the conventional cut-point. Thresholds were selected without reference to the "
                  "test partition.")
    ds.page_break(doc)

    # ================= Table S5 =================
    merged = shap33.merge(
        perm[["feature_code", "perm_importance_mean", "perm_importance_std", "rank"]],
        on="feature_code", suffixes=("_shap", "_perm"))
    rows5, n_resolved = [], 0
    for _, r in merged.nsmallest(15, "rank_shap").iterrows():
        pv, sd = r["perm_importance_mean"], r["perm_importance_std"]
        resolved = abs(pv) > sd
        n_resolved += int(resolved)
        rows5.append([SHORT.get(r["clinical_name"], r["clinical_name"]),
                      f"{r['mean_abs_shap']:.4f}", f"{int(r['rank_shap'])}",
                      f"{num(pv, 5)} ± {sd:.5f}", f"{int(r['rank_perm'])}",
                      "yes" if resolved else "no"])
    ds.caption(doc, "Table S5.",
               f"The fifteen highest-ranked predictors by mean absolute SHAP value, with their "
               f"permutation-importance values and ranks for comparison. Both methods are applied "
               f"to the {TOP} model on the held-out test partition and both rank over the same "
               f"{N['n_predictors_ranked']} predictors.", above=True)
    ds.add_table(doc,
                 ["Predictor", "Mean |SHAP|", "SHAP rank", "Permutation Δ AUROC (mean ± SD)",
                  "Perm. rank", "Above noise"],
                 rows5, col_widths_in=[1.95, 0.80, 0.72, 1.62, 0.72, 0.72],
                 align_right_from=1, font_size=ds.Pt(8))
    ds.table_note(doc,
                  "Permutation Δ AUROC is the mean decrease in test-set AUROC over ten permutations "
                  "of the predictor, with the standard deviation across those permutations. *Above "
                  "noise* is a descriptive heuristic, not a hypothesis test: it flags predictors "
                  "whose mean exceeds one permutation standard deviation, a roughly one-sigma bar "
                  "reflecting permutation variability only and not sampling variability of the test "
                  f"partition. Only {n_resolved} of these fifteen clear it, so ranks below the "
                  "leading few should not be read as a resolved ordering. A negative value means "
                  "permuting the predictor left discrimination unchanged or marginally improved it. "
                  "Two of the 33 columns (the missing-category indicators for race/ethnicity and "
                  "marital status) are constant or near-constant and receive exactly zero importance "
                  "under both methods; under SHAP they fall to the bottom (joint rank 32), but under "
                  "permutation importance an exact zero ranks above the seven predictors with "
                  "negative importance, placing them at joint rank 25 of 33 and pushing genuinely "
                  "uninformative predictors such as mercury below them.")
    ds.page_break(doc)

    # ================= Table S6 =================
    rows6 = []
    for _, r in sg.iterrows():
        rows6.append([r["subgroup"], f"{int(r['n']):,}", f"{r['prevalence_pct']:.1f}",
                      f"{r['auc']:.3f} ({ci(r['ci_low'], r['ci_high'])})",
                      f"{r['calibration_slope']:.2f} "
                      f"({r['slope_ci_low']:.2f}–{r['slope_ci_high']:.2f})",
                      f"{r['observed_expected_ratio']:.3f} "
                      f"({r['oe_ci_low']:.3f}–{r['oe_ci_high']:.3f})"])
    ds.caption(doc, "Table S6.",
               f"Discrimination and calibration of the {TOP} model within subgroups of the held-out "
               f"test partition. Calibration is reported alongside discrimination because a model "
               f"can rank participants poorly within a stratum while still stating their absolute "
               f"risk correctly, and the two have different consequences in use.", above=True)
    ds.add_table(doc, ["Subgroup", "n", "Prev. (%)", "AUROC (95% CI)",
                       "Cal. slope (95% CI)", "O:E ratio (95% CI)"],
                 rows6, col_widths_in=[1.30, 0.50, 0.62, 1.42, 1.38, 1.38],
                 align_right_from=1, font_size=ds.Pt(8))
    ds.table_note(doc,
                  "Cal. slope, calibration slope, describing dispersion of the predictions "
                  "(1.00 = perfect); O:E, ratio of observed to expected outcome frequency, "
                  "describing calibration in the large (1.00 = perfect). All intervals come from "
                  "the same 1,000 percentile bootstrap resamples, with the calibration model "
                  "refitted in each. Slope and O:E answer different questions and are not "
                  "interchangeable. Age groups are tertiles of the test-partition age distribution. "
                  "Subgroups are nested within the overall sample, so their intervals are not "
                  "independent of the overall row and should not be used to test against it.")

    # ================= Table S7 =================
    rows7 = []
    for _, r in s_diq.iterrows():
        rows7.append(["Add diagnosed diabetes", r["model"], f"{r['cv_auc_primary']:.4f}",
                      f"{r['cv_auc_plus_diq010']:.4f}", f"{r['delta']:+.4f}"])
    for _, r in s_smo.iterrows():
        cw = r["cv_auc_class_weight"]
        rows7.append(["Class weighting instead of SMOTENC", r["model"],
                      f"{r['cv_auc_smote']:.4f}",
                      "not applicable" if pd.isna(cw) else f"{cw:.4f}",
                      "—" if pd.isna(r["delta"]) else f"{r['delta']:+.4f}"])
    for _, r in s_met.iterrows():
        rows7.append(["Lead only, other four metals dropped", r["model"],
                      f"{r['cv_auc_all_5_metals']:.4f}", f"{r['cv_auc_lead_only']:.4f}",
                      f"{r['delta']:+.4f}"])
    for _, r in s_out.iterrows():
        rows7.append(["Measured-value-only outcome definition", r["model"],
                      f"{r['cv_auc_harmonized_label']:.4f}", f"{r['cv_auc_labonly_label']:.4f}",
                      f"{r['delta'] * -1:+.4f}"])
    ds.caption(doc, "Table S7.",
               "Prespecified sensitivity analyses. Each entry is the mean AUROC from 10-fold "
               "stratified cross-validation on the training partition, using the three leading "
               "algorithms at their default hyperparameter configurations so that the comparison "
               "between arms is not confounded by re-tuning. *Absolute values are consequently "
               "lower than the tuned values in Table S3 and are not comparable to them; the "
               "difference between arms is the quantity of interest.*", above=True)
    ds.add_table(doc, ["Analysis", "Model", "Primary", "Alternative", "Δ AUROC"],
                 rows7, col_widths_in=[2.30, 1.25, 0.80, 1.05, 0.80],
                 align_right_from=2, font_size=ds.Pt(8.5))
    ds.table_note(doc,
                  "Δ AUROC is the alternative specification minus the primary, so a positive value "
                  "indicates the alternative discriminated better. For the outcome-definition arm "
                  "the primary specification is the harmonized definition used throughout, and the "
                  "negative values indicate that the measured-value-only definition is harder to "
                  "predict.")
    ds.page_break(doc)

    # ================= Figures =================
    ds.figure(doc, FIGP / "figM1_analysis_pipeline.png", width_in=6.3)
    ds.caption(doc, "Figure S1.",
               "Analysis pipeline. Stages 1–4 are performed once on the cohort. Every step in "
               "stage 5 is refitted independently inside each training fold during cross-validation, "
               "and inside the training partition alone for the final models, so that no quantity "
               "estimated from held-out data can influence a model later scored on it.")

    ds.figure(doc, FIGP / "fig3_auc_forest.png", width_in=6.3)
    ds.caption(doc, "Figure S2.",
               f"Model discrimination and pairwise comparison against {TOP}. **(A)** AUROC with "
               f"bootstrap 95% confidence intervals. **(B)** Difference in AUROC from {TOP} from a "
               f"paired bootstrap in which the same resampled participants were scored under both "
               f"models; intervals crossing zero indicate models that cannot be distinguished from "
               f"{TOP}. Using unpaired intervals from panel A for these comparisons would overstate "
               f"the uncertainty, because all models are scored on the same participants.")
    ds.page_break(doc)

    ds.figure(doc, FIGP / "fig4_calibration.png", width_in=6.3)
    ds.caption(doc, "Figure S3.",
               "Calibration on the held-out test partition. **(A)** Calibration curves for the four "
               "leading models using ten quantile-based bins, with calibration slopes in the "
               "legend; the diagonal denotes perfect calibration. **(B)** Distribution of predicted "
               f"probabilities from {TOP} by true outcome status.")

    ds.figure(doc, FIGP / "fig5_decision_curve.png", width_in=4.5)
    ds.caption(doc, "Figure S4.",
               "Decision-curve analysis on the held-out test partition. Net benefit is plotted "
               "against the risk threshold at which intervention would be considered warranted, for "
               "the four leading models and for the treat-all and treat-none reference strategies.")
    ds.page_break(doc)

    ds.figure(doc, FIGP / "fig9_subgroup_auc.png", width_in=5.2)
    ds.caption(doc, "Figure S5.",
               f"Discrimination of the {TOP} model within subgroups of the held-out test partition, "
               f"with bootstrap 95% confidence intervals from 1,000 resamples. The dashed vertical "
               f"line marks the overall test-set AUROC and subgroup sizes are given at the right. "
               f"Age groups are tertiles of the test-partition age distribution. Because every "
               f"subgroup is nested within the overall sample, these intervals are not independent "
               f"of the overall estimate and overlap with it should not be read as a significance "
               f"test. Subgroup calibration is reported in Table S6.")

    path = OUT / "Manuscript_Supplement_AI.docx"
    # Convert break-only paragraphs into page_break_before on the following
    # block, so a section that happens to end on a page boundary cannot leave
    # a blank page behind it.
    n_breaks = ds.normalize_page_breaks(doc)
    doc.save(path)
    print(f"Saved: {path}")
    print(f"paragraphs={len(doc.paragraphs)} tables={len(doc.tables)}")


if __name__ == "__main__":
    main()
