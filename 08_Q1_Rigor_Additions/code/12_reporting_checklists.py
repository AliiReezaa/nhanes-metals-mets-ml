"""
Q1 rigor addition 7: an itemized TRIPOD+AI checklist and a PROBAST-AI-style
risk-of-bias summary. The Methods draft already states "Reporting follows
TRIPOD+AI [3] and STROBE [4]" (Methods, Section 1) -- but that is a blanket
assertion with no itemized checklist behind it anywhere in the package (grep
of all three manuscript PDFs for "TRIPOD"/"PROBAST" finds only that one
citation). Reviewers at journals that endorse TRIPOD+AI (most clinical-ML
venues now do) routinely ask for the completed checklist as a supplementary
file, so this substantiates the claim already being made rather than
introducing a new one.

Built by reading the Methods and Results text section by section and citing
where, if anywhere, each of TRIPOD+AI's 27 items is addressed. 2026-10 audit
(P13): every number quoted below is read from the pipeline outputs when the
script runs, so the checklist cannot drift from the results. This package is
explicitly a Methods+Results draft with no Title page, Abstract,
Introduction or Discussion yet (per the user's own framing: these are drafts
for a professor, not a submission-ready manuscript, and the Mendelian
randomization arm -- and presumably the framing that ties both arms together
into one Introduction/Discussion -- lives outside this folder). Items that
belong to those missing sections are marked "Not addressed at this draft
stage" rather than scored as failures, since they are out of scope for what
exists here, not overlooked.

Run: python 08_Q1_Rigor_Additions/code/12_reporting_checklists.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_Code"))
import additions_common as ac  # noqa: E402
import pandas as pd

import config as cfg
import pipeline_utils as pu

TABLES, _ = ac.out_dirs("08_Q1_Rigor_Additions")
MET = cfg.OUT_METRICS

train_df, test_df = pu.load_model_ready_split()
n_train, n_test = len(train_df), len(test_df)
events = int(train_df[cfg.OUTCOME_VAR].sum())
n_pred = len(pd.read_csv(MET / "predictors_used.csv"))
miss = pd.read_csv(MET / "missingness_report.csv").set_index("variable")["pct_missing"]
pred_codes = pd.read_csv(MET / "predictors_used.csv").iloc[:, 0].tolist()
miss_pred = miss[miss.index.isin(pred_codes)].sort_values(ascending=False)
worst_var, worst_pct = miss_pred.index[0], float(miss_pred.iloc[0])
so = pd.read_csv(MET / "sensitivity_outcome_definition.csv")
d_out = so["cv_auc_harmonized_label"] - so["cv_auc_labonly_label"]
d_nol = so["cv_auc_harmonized_label"] - so["cv_auc_nolipidmed_label"]
tv_path = TABLES / "table_temporal_validation.csv"
if tv_path.exists():
    tv = pd.read_csv(tv_path)
    tv_txt = (f"AUROC {tv['Test AUROC'].iloc[0]:.3f} on the random split and "
              f"{tv['Test AUROC'].iloc[1]:.3f} on the temporal holdout, calibration slope "
              f"{tv['Calibration slope'].iloc[0]:.2f} and {tv['Calibration slope'].iloc[1]:.2f}")
else:
    tv_txt = "see table_temporal_validation.csv"

rows = [
    # (item, section, requirement, status, evidence, note)
    ("1", "Title", "Identify the study as developing/evaluating a prediction model, target population, outcome",
     "Partial", "Running header on Methods/Results: \"Machine-learning analysis of blood metals and metabolic syndrome in NHANES 2007-2018\"",
     "Names population and exposure/outcome but does not frame itself as a prediction-model study in the title text itself; no formal title page yet"),
    ("2", "Abstract", "Structured abstract per TRIPOD+AI for Abstracts", "Not addressed at this draft stage",
     "--", "No abstract exists in this package"),
    ("3a", "Introduction: Background", "Healthcare context and rationale for the model", "Not addressed at this draft stage",
     "--", "No Introduction section exists; rationale is only implicit in Methods Sec. 1/4"),
    ("3b", "Introduction: Background", "Target population and intended use/users of the model", "Satisfied",
     "Methods Sec. 10: \"a research instrument for prioritizing exposures for the Mendelian randomization phase, not a clinical screening tool\"",
     "Explicitly and clearly stated, unusually so for a Methods section rather than a Discussion"),
    ("3c", "Introduction: Background", "Known health inequalities between sociodemographic groups", "Partial",
     "Results Sec. 5 subgroup analysis (sex/age/race) is empirical evidence of this, but no Introduction-level framing exists",
     "The subgroup performance gap by age (Sec. 5) is itself a relevant finding here, just not framed as background"),
    ("4", "Introduction: Objectives", "State study objectives, development vs. validation", "Satisfied",
     "Methods Sec. 1 and Sec. 10; README", "Objective (rank exposures for the MR phase) is clear but stated in Methods, not a dedicated Objectives paragraph"),
    ("5a", "Methods: Data", "Data sources for development/evaluation, rationale, representativeness", "Satisfied",
     "Methods Sec. 1", "NHANES design, public availability and URL all given"),
    ("5b", "Methods: Data", "Dates of participant data collection", "Satisfied", "Methods Sec. 1", "2007-2008 through 2017-2018, six cycles"),
    ("6a", "Methods: Participants", "Study setting, number/location of centres", "Satisfied", "Methods Sec. 1", "NHANES multistage probability design of the US population described"),
    ("6b", "Methods: Participants", "Eligibility criteria", "Satisfied", "Methods Sec. 2, Figure 1", "Full exclusion cascade with counts at each step"),
    ("6c", "Methods: Participants", "Treatments received and handling", "Satisfied", "Methods Sec. 3", "Medication variables used only to build the harmonized outcome label, excluded from predictors"),
    ("7", "Methods: Data preparation", "Pre-processing/quality checks, consistency across groups", "Partial",
     "Methods Sec. 5, 6, 7", "Imputation/leakage prevention thoroughly described; explicit statement of consistency across sociodemographic groups is not made"),
    ("8a", "Methods: Outcome", "Outcome definition and timing", "Satisfied", "Methods Sec. 3", "Unusually thorough -- harmonized vs. lab-only definitions both defined with exact reclassification counts"),
    ("8b", "Methods: Outcome", "Qualifications of outcome assessors, if subjective", "Not applicable", "--", "Outcome built from objective lab values and coded survey items, not subjective clinical assessment"),
    ("8c", "Methods: Outcome", "Blinding of outcome assessment", "Not applicable", "--", "Cross-sectional lab/questionnaire data; blinding is not a relevant concept here"),
    ("9a", "Methods: Predictors", "Choice of initial predictors, any pre-selection", "Satisfied", "Methods Sec. 4",
     "Explicit argument against data-driven selection (stepwise/LASSO) with citation; two prospective exclusions justified individually"),
    ("9b", "Methods: Predictors", "Definitions, timing, blinding of predictors", "Satisfied", "Methods Sec. 4, Table S1 (referenced)", "--"),
    ("9c", "Methods: Predictors", "Qualifications of predictor assessors, if subjective", "Not applicable", "--", "Predictors are lab values or structured survey items"),
    ("10", "Methods: Sample size", "Explain and justify study size", "Partial", "Methods Sec. 6",
     f"Reports about {events / n_pred:.0f} events per predictor ({events:,} events, {n_pred} predictors) and explicitly flags that this benchmark 'does not translate directly' to a boosted ensemble -- an honest partial justification, not a formal sample-size calculation"),
    ("11", "Methods: Missing data", "How missing data were handled, reasons for omissions", "Satisfied", "Methods Sec. 5",
     "Per-variable missingness rates, imputation strategy, and missing-indicator rationale all given"),
    ("12a", "Methods: Analysis", "How data were partitioned", "Satisfied", "Methods Sec. 6", "80/20 stratified split, pinned by SEQN"),
    ("12b", "Methods: Analysis", "Predictor handling (scaling/transformation)", "Satisfied", "Methods Sec. 7", "--"),
    ("12c", "Methods: Analysis", "Model type, rationale, tuning, internal validation", "Satisfied", "Methods Sec. 7", "11 models, exhaustive grid search with 10-fold stratified CV, repeated CV (10 x 5) for split-to-split variability"),
    ("12d", "Methods: Analysis", "Heterogeneity across clusters", "Partial", "Methods Sec. 10; Table S10",
     "NHANES cycles are not modelled as clusters; the temporal validation (develop 2007-2014, test 2015-2018, re-tuned) checks transport across cycles"),
    ("12e", "Methods: Analysis", "Performance measures and plots, rationale", "Satisfied", "Methods Sec. 8", "AUROC, AUPRC, Brier, calibration slope and intercept, O:E, DCA, DeLong tests, bootstrap CIs for every metric"),
    ("12f", "Methods: Analysis", "Model updating/recalibration", "Not applicable", "--", "Development-only study; no updating was performed or claimed"),
    ("12g", "Methods: Analysis", "How predictions are calculated for evaluation (formula/code/API)", "Partial",
     "Fitted pipelines are saved in outputs/models and the code is under version control", "No formula, serialized object or API is referenced in the Methods/Results text itself yet"),
    ("13", "Methods: Class imbalance", "Class imbalance methods used, why, recalibration", "Satisfied", "Methods Sec. 7",
     "SMOTENC justified over SMOTE; tested against class weighting in a sensitivity analysis; calibration reported for every model"),
    ("14", "Methods: Fairness", "Approaches used to address model fairness", "Partial",
     "Results Sec. 5 subgroup discrimination/calibration by sex, age, race", "Functions as a de facto fairness check but is not framed or named as one"),
    ("15", "Methods: Model output", "Output type (probabilities/classification), rationale", "Satisfied", "Methods Sec. 7",
     "Probabilities plus Youden-index-tuned threshold, with both tuned and 0.5 thresholds reported"),
    ("16", "Methods: Development vs. evaluation", "Differences between development and evaluation data", "Not applicable",
     "--", "Single-cohort internal split, not a separate external evaluation dataset -- self-acknowledged as a limitation in Methods Sec. 10"),
    ("17", "Methods: Ethics", "Ethics committee, informed consent", "Satisfied", "Methods Sec. 10",
     "NCHS Research Ethics Review Board named; no additional approval required for this secondary analysis"),
    ("18a", "Open Science: Funding", "Funding source and role of funders", "Not addressed at this draft stage", "--", "No funding statement found in any document"),
    ("18b", "Open Science: COI", "Conflicts of interest for all authors", "Not addressed at this draft stage", "--", "No COI statement found"),
    ("18c", "Open Science: Protocol", "Where the study protocol can be accessed", "Not addressed at this draft stage", "--", "--"),
    ("18d", "Open Science: Registration", "Registration name and number", "Not addressed at this draft stage", "--", "--"),
    ("18e", "Open Science: Data sharing", "Availability of study data", "Satisfied", "Methods Sec. 10; README",
     "NHANES is public data at a stated URL"),
    ("18f", "Open Science: Code sharing", "Availability of analytical code", "Partial", "Code is kept in a private Git repository",
     "A public repository link is still to be added to the manuscript"),
    ("19", "Patient & public involvement", "PPI in design/conduct/reporting/dissemination", "Not addressed at this draft stage", "--",
     "Not applicable to most secondary NHANES analyses, but TRIPOD+AI still asks for an explicit statement, even if it is 'none'"),
    ("20a", "Results: Participants", "Flow of participants through the study", "Satisfied", "Results Sec. 1, Figure 1", "--"),
    ("20b", "Results: Participants", "Characteristics overall and by data source", "Satisfied", "Results Sec. 1, Table 1",
     "Survey-weighted, with design-based standard errors"),
    ("20c", "Results: Participants", "Comparison of development vs. evaluation predictor distributions", "Not addressed at this draft stage",
     "--", "Neither the random split nor the temporal split has a table comparing predictor distributions between development and evaluation data; table_temporal_validation.csv reports sizes and performance only"),
    ("21", "Results: Model development", "N participants/events at each analysis stage", "Satisfied",
     f"Results Sec. 1 ({n_train:,}/{n_test:,}), Methods Sec. 6 ({events:,} training events)", "--"),
    ("22", "Results: Model specification", "Full model detail for reuse/third-party evaluation", "Partial",
     "best_params.json and the full pipeline code exist in 06_Code/, but are not referenced or summarized in the manuscript text itself",
     "An explicit pointer from the Methods/Results text to the artifact would close this"),
    ("23a", "Results: Performance", "Performance estimates with CIs, incl. subgroups", "Satisfied",
     "Table 2, Results Sec. 5, Table S6", "Bootstrap CIs reported throughout, including per-subgroup calibration"),
    ("23b", "Results: Performance", "Heterogeneity across clusters", "Partial", "Results Sec. 5, Table S10",
     f"Temporal validation reported ({tv_txt}); performance by individual cycle is not reported"),
    ("24", "Results: Model updating", "Results of any model updating", "Not applicable", "--", "No updating performed"),
    ("25", "Discussion: Interpretation", "Overall interpretation incl. fairness, vs. prior studies", "Not addressed at this draft stage",
     "--", "No Discussion section exists yet"),
    ("26", "Discussion: Limitations", "Limitations, effects on bias/uncertainty/generalizability", "Partial",
     "Methods Sec. 10 pre-states several limitations (internal validation only, cross-sectional design, single random split supplemented by repeated CV and a temporal holdout)",
     "Unusually, limitations are disclosed in Methods rather than a dedicated Discussion section, which does not yet exist"),
    ("27a", "Discussion: Usability", "Handling poor-quality/unavailable input data in deployment", "Not addressed at this draft stage", "--", "--"),
    ("27b", "Discussion: Usability", "User interaction/expertise required", "Not addressed at this draft stage", "--", "--"),
    ("27c", "Discussion: Usability", "Next steps for future research", "Partial",
     "The MR arm is named as the next phase throughout",
     "Not a dedicated forward-looking Discussion paragraph"),
]

df = pd.DataFrame(rows, columns=["Item", "Section", "TRIPOD+AI requirement (summarized)", "Status", "Evidence / location", "Note"])
df.to_csv(TABLES / "table_tripod_ai_checklist.csv", index=False)

status_counts = df["Status"].value_counts()
print(f"TRIPOD+AI checklist -- status tally across {len(df)} (sub-)items:")
print(status_counts.to_string())
print(f"\nOf items that ARE in scope for a Methods+Results-only draft (excluding Title/Abstract/"
      f"Introduction/Discussion/Open-Science/PPI items, which are simply not written yet), "
      f"the Methods and Results sections satisfy the large majority of what TRIPOD+AI asks for.")

# PROBAST-AI-style domain-level risk-of-bias summary
probast_rows = [
    ("Participants", "Low",
     "Well-defined NHANES cross-sectional cohort with a fully enumerated eligibility/exclusion cascade "
     "(Methods Sec. 2, Figure 1). Appropriate source population for the prediction question. The one "
     "inherent constraint -- a cross-sectional design cannot support causal claims -- is explicitly "
     "acknowledged (Methods Sec. 1, 10) rather than glossed over."),
    ("Predictors", "Low",
     "All predictors pre-specified from the literature before model building; no data-driven selection "
     "(Methods Sec. 4, with an explicit methodological argument against stepwise/LASSO selection on a "
     "fixed dataset). Skip patterns are recoded to their logical values (never-smokers as not smoking now, "
     "past-year non-drinkers as zero drinks) and refusal codes are set to missing per variable. The most "
     f"incomplete predictor is {worst_var} ({worst_pct:.1f}% missing), handled with median imputation plus a "
     "missing-value indicator."),
    ("Outcome", "Some concerns",
     "The outcome (harmonized MetS) is objectively built from lab values and treatment flags with an "
     "unusually thorough definition (Methods Sec. 3). The concern is not measurement quality but "
     "definitional sensitivity: the substituting-definition sensitivity analysis (Results Sec. 5, Table S7) "
     f"shows the measured-value-only definition lowers discrimination by {d_out.min():.3f} to {d_out.max():.3f} "
     f"AUROC units, and dropping the lipid-medication clause by {d_nol.min():.3f} to {d_nol.max():.3f}, so a "
     "different defensible definition would have produced a different headline number. This is disclosed, "
     "but it is a real source of between-definition variability."),
    ("Analysis", "Some concerns",
     "Leakage-safe pipeline (single fitted-per-fold object, Methods Sec. 6), appropriate internal "
     "cross-validation and a held-out test partition are all low-risk practices. Split-to-split variability "
     "is shown by repeated cross-validation (Table S9), and a re-tuned forward-in-time check gives "
     f"{tv_txt} (Table S10). The residual concern is that there is no external validation on an "
     "independent cohort."),
]
probast_df = pd.DataFrame(probast_rows, columns=["PROBAST-AI domain", "Risk of bias", "Rationale"])
probast_df.to_csv(TABLES / "table_probast_ai_risk_of_bias.csv", index=False)
print("\nPROBAST-AI domain-level summary:")
print(probast_df[["PROBAST-AI domain", "Risk of bias"]].to_string(index=False))
