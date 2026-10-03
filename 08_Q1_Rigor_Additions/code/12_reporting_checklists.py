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

Built by reading the actual Methods_AI.pdf and Results_AI.pdf text
section-by-section (not inferred from the README) and citing exactly where,
if anywhere, each of TRIPOD+AI's 27 items is addressed. This package is
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

TABLES, _ = ac.out_dirs("08_Q1_Rigor_Additions")

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
     "Reports ~125 events/predictor and explicitly flags that this benchmark 'does not translate directly' to a boosted ensemble -- an honest partial justification, not a formal sample-size calculation"),
    ("11", "Methods: Missing data", "How missing data were handled, reasons for omissions", "Satisfied", "Methods Sec. 5",
     "Per-variable missingness rates, imputation strategy, and missing-indicator rationale all given"),
    ("12a", "Methods: Analysis", "How data were partitioned", "Satisfied", "Methods Sec. 6", "80/20 stratified split, pinned by SEQN"),
    ("12b", "Methods: Analysis", "Predictor handling (scaling/transformation)", "Satisfied", "Methods Sec. 7", "--"),
    ("12c", "Methods: Analysis", "Model type, rationale, tuning, internal validation", "Satisfied", "Methods Sec. 7", "11 models, randomized search, 10-fold stratified CV, all justified"),
    ("12d", "Methods: Analysis", "Heterogeneity across clusters", "Not addressed at this draft stage", "--",
     "NHANES cycles are a natural clustering structure not analyzed as such in the original pipeline -- 08_Q1_Rigor_Additions/code/06_temporal_validation.py now partially fills this gap"),
    ("12e", "Methods: Analysis", "Performance measures and plots, rationale", "Satisfied", "Methods Sec. 8", "AUROC, AUPRC, Brier, calibration slope/O:E, DCA -- all justified with citations"),
    ("12f", "Methods: Analysis", "Model updating/recalibration", "Not applicable", "--", "Development-only study; no updating was performed or claimed"),
    ("12g", "Methods: Analysis", "How predictions are calculated for evaluation (formula/code/API)", "Partial",
     "README states code will be deposited publicly on publication", "No formula, serialized object, or API is referenced in the Methods/Results text itself yet"),
    ("13", "Methods: Class imbalance", "Class imbalance methods used, why, recalibration", "Satisfied", "Methods Sec. 7",
     "SMOTENC justified over SMOTE; tested against class-weighting in a prespecified sensitivity analysis; calibration reported for every model"),
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
    ("18f", "Open Science: Code sharing", "Availability of analytical code", "Partial", "README: \"code, derived metrics and model artifacts will be deposited publicly on publication\"",
     "Stated intent, not yet a live repository link"),
    ("19", "Patient & public involvement", "PPI in design/conduct/reporting/dissemination", "Not addressed at this draft stage", "--",
     "Not applicable to most secondary NHANES analyses, but TRIPOD+AI still asks for an explicit statement, even if it is 'none'"),
    ("20a", "Results: Participants", "Flow of participants through the study", "Satisfied", "Results Sec. 1, Figure 1", "--"),
    ("20b", "Results: Participants", "Characteristics overall and by data source", "Satisfied", "Results Sec. 1, Table 1",
     "Survey-weighted, with design-based standard errors"),
    ("20c", "Results: Participants", "Comparison of development vs. evaluation predictor distributions", "Partial",
     "08_Q1_Rigor_Additions/tables/table_temporal_validation.csv now reports this across the temporal train/test split", "Not present for the original random split, since there is no separate external evaluation dataset"),
    ("21", "Results: Model development", "N participants/events at each analysis stage", "Satisfied",
     "Results Sec. 1 (8,915/2,229), Methods Sec. 6 (4,130 events)", "--"),
    ("22", "Results: Model specification", "Full model detail for reuse/third-party evaluation", "Partial",
     "best_params.json and the full pipeline code exist in 06_Code/, but are not referenced or summarized in the manuscript text itself",
     "An explicit pointer from the Methods/Results text to the artifact would close this"),
    ("23a", "Results: Performance", "Performance estimates with CIs, incl. subgroups", "Satisfied",
     "Table 2, Results Sec. 5, Table S6", "Bootstrap CIs reported throughout, including per-subgroup calibration"),
    ("23b", "Results: Performance", "Heterogeneity across clusters", "Not addressed at this draft stage", "--",
     "Same gap as item 12d; NHANES-cycle heterogeneity is not analyzed as a clustering structure in the original results"),
    ("24", "Results: Model updating", "Results of any model updating", "Not applicable", "--", "No updating performed"),
    ("25", "Discussion: Interpretation", "Overall interpretation incl. fairness, vs. prior studies", "Not addressed at this draft stage",
     "--", "No Discussion section exists yet"),
    ("26", "Discussion: Limitations", "Limitations, effects on bias/uncertainty/generalizability", "Partial",
     "Methods Sec. 10 pre-states several limitations (internal-validation-only, single seed, no temporal validation)",
     "Unusually, limitations are disclosed in Methods rather than a dedicated Discussion section, which does not yet exist"),
    ("27a", "Discussion: Usability", "Handling poor-quality/unavailable input data in deployment", "Not addressed at this draft stage", "--", "--"),
    ("27b", "Discussion: Usability", "User interaction/expertise required", "Not addressed at this draft stage", "--", "--"),
    ("27c", "Discussion: Usability", "Next steps for future research", "Partial",
     "Methods Sec. 10 gestures at this (\"would permit holding out the most recent cycle\") and the MR arm is named as the next phase throughout",
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
     "fixed dataset). Consistently defined and measured across all participants. One moderate concern: "
     "current-smoking-intensity has 55.3% missingness from a genuine skip pattern, handled with imputation "
     "plus a missingness indicator and reported as such rather than hidden."),
    ("Outcome", "Some concerns",
     "The outcome (harmonized MetS) is objectively built from lab values and treatment flags with an "
     "unusually thorough definition (Methods Sec. 3). The concern is not measurement quality but "
     "definitional sensitivity: the substituting-definition sensitivity analysis (Results Sec. 5, Table S7) "
     "shows the outcome-definition choice moves discrimination by 0.029-0.034 AUROC units -- the largest "
     "swing of the four sensitivity arms -- meaning a materially different, defensible outcome definition "
     "would have produced a different headline number. This is disclosed, not hidden, which is itself "
     "good practice, but it is a real source of between-definition variability."),
    ("Analysis", "Some concerns",
     "Leakage-safe pipeline (single fitted-per-fold object, Methods Sec. 6), appropriate internal "
     "cross-validation and a held-out test partition are all low-risk practices. The residual concern, "
     "explicitly self-flagged in Methods Sec. 10, is that this is internal validation on a SINGLE random "
     "split under ONE fixed seed, with no external or temporal validation -- split-to-split variability is "
     "not captured by within-partition bootstrapping. 08_Q1_Rigor_Additions/code/06_temporal_validation.py "
     "now provides a genuine forward-in-time check (AUROC 0.857 to 0.840 from random-split to temporal-"
     "holdout, well-calibrated in both), which moves this domain from what would otherwise be a clean "
     "'some concerns' toward the low end of it -- but a single temporal split is still not the same as "
     "true external validation on an independent cohort."),
]
probast_df = pd.DataFrame(probast_rows, columns=["PROBAST-AI domain", "Risk of bias", "Rationale"])
probast_df.to_csv(TABLES / "table_probast_ai_risk_of_bias.csv", index=False)
print("\nPROBAST-AI domain-level summary:")
print(probast_df[["PROBAST-AI domain", "Risk of bias"]].to_string(index=False))
