"""
Shared configuration for the metabolic syndrome (MetS) + blood metals pipeline.
Every numbered script imports from here. Change paths/variable lists in ONE place.

REBUILD NOTE (this version): this is a corrected rebuild of the original pipeline.
Every change below is tied to a specific, verified problem found in an independent
review of the original code and the actual NHANES documentation/data. Each fix is
flagged with a "FIX:" comment at the point it's made so the audit trail survives
in the code itself, not just in a handoff document.
"""
from pathlib import Path

# ----------------------------------------------------------------------
# Paths (relative to project root; works on any machine as long as the
# folder structure below is preserved)
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "NHANES_data"
DATA_PROC_DIR = PROJECT_ROOT / "data_processed"
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_FIGURES = OUT_DIR / "figures"
OUT_MODELS = OUT_DIR / "models"
OUT_METRICS = OUT_DIR / "metrics"

for d in [DATA_PROC_DIR, OUT_FIGURES, OUT_MODELS, OUT_METRICS]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# NHANES cycles present in NHANES_data/
# ----------------------------------------------------------------------
CYCLES = [
    {"letter": "E", "years": "2007_2008"},
    {"letter": "F", "years": "2009_2010"},
    {"letter": "G", "years": "2011_2012"},
    {"letter": "H", "years": "2013_2014"},
    {"letter": "I", "years": "2015_2016"},
    {"letter": "J", "years": "2017_2018"},
]

# NHANES module (file) prefixes to load per cycle. All are confirmed present
# in NHANES_data/NHANES_<years>/<PREFIX>_<letter>.xpt
# FIX: "BPQ" (Blood Pressure & Cholesterol Questionnaire) added. The original
# pipeline never downloaded this module, which made it structurally impossible
# to apply the medication-treatment clause of the AHA/NHLBI MetS definition
# (see METS below). Verified present for all 6 cycles with BPQ050A (antihypertensive
# medication) and BPQ100D (cholesterol medication) in every cycle before adding this.
MODULES = ["DEMO", "BMX", "BPX", "TRIGLY", "HDL", "GLU", "PBCD", "ALQ", "DBQ", "SMQ", "DIQ", "PAQ", "BPQ"]

# DEMO defines the sampled-person universe; every other module is left-joined onto it.
ANCHOR_MODULE = "DEMO"

# ----------------------------------------------------------------------
# Population scope
# ----------------------------------------------------------------------
MIN_AGE = 20  # adults only

# ----------------------------------------------------------------------
# MetS-defining component variables (used ONLY to build the label,
# never as predictors)
# ----------------------------------------------------------------------
METS_COMPONENT_VARS = [
    "BMXWAIST",  # waist circumference (BMX)
    "BPXSY1", "BPXDI1",  # systolic / diastolic BP (BPX)
    "LBDHDD",  # HDL cholesterol (HDL)
    "LBXTR",  # triglycerides (TRIGLY)
    "LBXGLU",  # fasting glucose (GLU)
]

# FIX: medication/treatment variables used to complete the AHA/NHLBI "harmonized"
# 2009 definition (Alberti et al., Circulation 2009;120:1640-1645, building on
# Grundy et al., Circulation 2005;112:2735-2752). Each of the BP, HDL,
# triglyceride, and glucose components must be scored positive if EITHER the lab/
# exam value crosses the threshold OR the person is on drug treatment for that
# condition. The original pipeline used lab/exam values only, which
# systematically undercounts MetS+ among people whose conditions are
# pharmacologically controlled. Variable names verified directly against the
# NHANES BPQ/DIQ codebooks for 2007-2008 and 2017-2018 (identical wording and
# variable names across all 6 cycles):
#   BPQ050A - "Now taking prescribed medicine for HBP"            -> BP component
#   BPQ100D - "Now taking prescribed medicine [for cholesterol]"  -> HDL & TG components
#             (NHANES does not separately ask about HDL-specific vs. TG-specific
#             drug treatment in any cycle used here; BPQ100D is the only
#             cholesterol-medication-use item available, so -- consistent with
#             the harmonized definition's OR logic and with standard practice in
#             the NHANES MetS literature -- it is applied to BOTH the HDL and
#             triglyceride components. This is a documented modeling choice, not
#             an oversight; it is stated explicitly in the manuscript methods.)
#   DIQ050  - "Taking insulin now"                                -> glucose component
#   DIQ070  - "Taking diabetic pills to lower blood sugar"        -> glucose component
# These variables are treated exactly like METS_COMPONENT_VARS: used only to
# build the label, excluded entirely from the predictor set (see PREDICTOR_VARS
# and the leakage-safe assembly logic in 03_feature_engineering.py).
MEDICATION_VARS_FOR_LABEL = ["BPQ050A", "BPQ100D", "DIQ050", "DIQ070"]

# AHA/NHLBI (2005) adult MetS thresholds -- CORRECTED triglyceride cutoff (150, not 130)
METS_THRESHOLDS = {
    "waist_men": 102,       # cm, RIAGENDR == 1
    "waist_women": 88,      # cm, RIAGENDR == 2
    "triglycerides": 150,   # mg/dL
    "hdl_men": 40,           # mg/dL, below this is positive
    "hdl_women": 50,         # mg/dL, below this is positive
    "sbp": 130,              # mmHg
    "dbp": 85,               # mmHg
    "glucose": 100,          # mg/dL
}
METS_POSITIVE_COUNT = 3  # >= this many components positive => MetS = 1

# ----------------------------------------------------------------------
# NHANES sentinel / refusal codes to recode to missing BEFORE modeling.
# Only applied to the questionnaire variables listed (lab/exam variables
# use different, mostly-unambiguous missing conventions and are handled
# separately in 02_clean_and_outcome.py).
# ----------------------------------------------------------------------
SENTINEL_RECODE_VARS = [
    "SMQ020", "SMQ040", "ALQ111", "ALQ101", "ALQ130", "DBQ700",
    "PAQ605", "PAQ650", "PAQ665", "PAQ635", "DIQ010",
    "DMDMARTL", "DMDEDUC2",  # confirmed present in data: DMDMARTL has 77 (Refused);
                              # DMDEDUC2 has 7 (Refused) and 9 (Don't know)
    # FIX: medication variables also carry 7 (Refused) / 9 (Don't know) NHANES
    # sentinel codes and must be cleaned the same way before they're used to
    # build the outcome label.
    "BPQ050A", "BPQ100D", "DIQ050", "DIQ070",
]
SENTINEL_CODES = {7, 9, 77, 99, 777, 999, 7777, 9999}

# ----------------------------------------------------------------------
# FIX: alcohol-ever variable harmonization.
# The original pipeline used ALQ111 ("Ever had a drink of any kind of
# alcohol") as a predictor. Verified directly against nhanes_merged_raw.csv:
# ALQ111 has ZERO non-null values in 2007-2008, 2009-2010, 2011-2012,
# 2013-2014, and 2015-2016 -- it was only added to the NHANES alcohol
# questionnaire in 2017-2018. Its ~86% "missingness" in the original pipeline
# was therefore not random: it was structural absence from 5 of 6 pooled
# cycles, harmonized by mode-imputing a near-constant value into 83% of the
# cohort and reporting whatever fell out of that as "signal." The equivalent
# gatekeeper question in the five earlier cycles was ALQ101 ("Had at least 12
# alcohol drinks/1 yr"), confirmed present in exactly those five cycles and
# absent from 2017-2018 (a clean 1-for-1 questionnaire-redesign swap, verified
# by non-null counts per cycle). The two questions are not word-for-word
# identical, but both function as the survey's alcohol-use gatekeeper item in
# their respective cycles and are harmonized here into a single "ever drank
# alcohol" indicator, which is standard practice for pooling redesigned NHANES
# questionnaire items across cycles (e.g. NHANES's own guidance on combining
# ALQ101/ALQ111 for trend analyses).
ALQ_EVER_SOURCES = {"ALQ111": ["2017_2018"], "ALQ101": ["2007_2008", "2009_2010", "2011_2012", "2013_2014", "2015_2016"]}
ALQ_EVER_VAR = "ALQ_EVER"

# FIX: PAD645 skip-pattern recode. Verified against the NHANES 2007-2008 PAQ
# codebook: PAD645 ("minutes walking/bicycling for transportation") is only
# asked of respondents who answered "Yes" to PAQ635 ("Walk or bicycle [for
# transportation]"). A "No" on PAQ635 routes directly past PAD645, leaving it
# blank -- this blank means zero minutes of walk/bike transportation, not
# genuinely unknown. The original pipeline treated all PAD645 blanks
# identically (median-imputed), conflating "confirmed zero" with "truly
# unknown." Rows with PAQ635 == 2 (No) get PAD645 recoded to 0 before any
# imputation; rows with PAQ635 missing or == 1 (Yes) but PAD645 still blank
# are left as genuine missing data for imputation.
PAD645_GATE_VAR = "PAQ635"
PAD645_VAR = "PAD645"

# ----------------------------------------------------------------------
# Predictor set (Step 4). Deliberately EXCLUDES METS_COMPONENT_VARS and
# MEDICATION_VARS_FOR_LABEL.
# ----------------------------------------------------------------------
PREDICTOR_VARS = {
    "metals": ["LBXBPB", "LBXBCD", "LBXTHG", "LBXBSE", "LBXBMN"],
    "demographic": ["RIDAGEYR", "RIAGENDR", "RIDRETH1", "DMDEDUC2", "DMDMARTL",
                     "DMDHHSIZ", "INDFMPIR", "RIDEXMON"],
    # BMXHIP dropped: r=0.9367 with BMXBMI and 81.97% missing, both computed
    # on the training partition only, from real data, by the dedicated
    # candidate-variable screen in 04_split.py (see its FIX note -- this
    # number used to be a hardcoded, never-actually-reproduced comment here;
    # it is now written to outputs/metrics/collinearity_flags.csv every run).
    # Redundant signal, heavy imputation burden -- excluded on both grounds.
    "anthropometric": ["BMXBMI"],
    # NOTE: DBQ010 (infant tap-water question, mean respondent age ~2.5) and
    # PAD590 (youth sedentary-time item) were dropped after 02_clean_and_outcome.py
    # showed them 100% missing in the adults-only cohort -- both are pediatric/youth
    # NHANES items, not applicable to this population.
    # FIX: ALQ111 replaced by the harmonized ALQ_EVER (see ALQ_EVER_SOURCES above).
    "lifestyle": ["SMQ020", "SMQ040", "ALQ_EVER", "ALQ130", "DBQ700",
                   "PAQ605", "PAQ650", "PAD645", "PAQ665", "BPXPLS"],
}
# DIQ010 (diagnosed diabetes) is intentionally NOT in the primary predictor set --
# it's added only in the Step 13 sensitivity analysis (10_sensitivity.py).
DIQ_SENSITIVITY_VAR = "DIQ010"

# ----------------------------------------------------------------------
# Encoding plan (Step 4). Every variable in PREDICTOR_VARS must appear in
# exactly one of these buckets -- 03_feature_engineering.py asserts this.
# Codes verified against the NHANES codebook (and against actual unique
# values observed in nhanes_with_outcome.csv) before writing this, not
# assumed from memory alone.
# ----------------------------------------------------------------------

# 1=Yes, 2=No in the raw NHANES coding -> recoded to 1/0 (1=Yes)
# FIX: ALQ111 -> ALQ_EVER (already built as clean 1/0 upstream in
# 02_clean_and_outcome.py, so it is carried through as continuous-style 0/1
# here rather than re-mapped from a raw 1/2 code -- see CONTINUOUS_AS_IS).
BINARY_YESNO_1_2 = ["SMQ020", "PAQ605", "PAQ650", "PAQ665"]

# Ordinal, already in a sensible increasing direction as coded -- kept as-is
# DBQ700: 1=excellent ... 5=poor (higher = worse diet)
# DMDEDUC2: 1=<9th grade ... 5=college grad+ (higher = more education)
ORDINAL_AS_IS = ["DBQ700", "DMDEDUC2"]

# Ordinal but raw NHANES coding runs backwards for our purposes (1=Every day
# is the *most* intense smoking, 3=Not at all is the *least*) -- explicit
# remap so higher value = higher exposure/risk, consistent with everything else
ORDINAL_REMAP = {
    "SMQ040": {1: 2, 2: 1, 3: 0},  # 0=not at all, 1=some days, 2=every day
}

# True unordered categories -> one-hot encoded
# RIDRETH1: 1=Mexican American,2=Other Hispanic,3=Non-Hispanic White,4=Non-Hispanic Black,5=Other
# DMDMARTL: 1=Married,2=Widowed,3=Divorced,4=Separated,5=Never married,6=Living w/ partner
NOMINAL_ONEHOT = ["RIDRETH1", "DMDMARTL"]

# 2-level variables recoded to a single named binary column instead of a raw 1/2 code
# RIAGENDR: 1=Male, 2=Female -> sex_female (1 if Female)
# RIDEXMON: 1=Nov-Apr, 2=May-Oct -> exam_summer (1 if May-Oct)
BINARY_RENAME = {
    "RIAGENDR": {"new_name": "sex_female", "positive_raw_value": 2},
    "RIDEXMON": {"new_name": "exam_summer", "positive_raw_value": 2},
}

# Left as numeric, imputed + scaled later inside the modeling pipeline (Step 6) --
# NOT touched in 03_feature_engineering.py beyond being carried through as-is.
# FIX: ALQ_EVER added here -- it is built as a clean 0/1 column directly in
# 02_clean_and_outcome.py (harmonized from ALQ101/ALQ111), so there is no raw
# 1/2 NHANES code left to remap; treating it as a pass-through 0/1 numeric
# column is equivalent to BINARY_YESNO_1_2 treatment downstream (imputed via
# the continuous pipeline's median imputer, which for a clean 0/1 column with
# low missingness is effectively the same as mode imputation) while keeping
# the encode() function's bucket logic simple and auditable.
CONTINUOUS_AS_IS = [
    "RIDAGEYR", "INDFMPIR", "DMDHHSIZ", "BMXBMI",
    "ALQ130", "PAD645", "BPXPLS", "ALQ_EVER",
    "LBXBPB", "LBXBCD", "LBXTHG", "LBXBSE", "LBXBMN",
]

ID_VAR = "SEQN"
OUTCOME_VAR = "mets_status"
CYCLE_VAR = "cycle"

# Survey design variables -- descriptive/Table-1 use only, never fed to sklearn.
SURVEY_WEIGHT_MEC = "WTMEC2YR"
SURVEY_WEIGHT_FASTING = "WTSAF2YR"  # correct weight for triglycerides/glucose-derived stats
SURVEY_PSU = "SDMVPSU"
SURVEY_STRATA = "SDMVSTRA"

# FIX: cycle-specific correct weight variable for the 5 blood metals. Verified
# directly against each cycle's NHANES lab documentation (PBCD_<letter>.htm,
# "Eligible Sample" and "Analytic Notes" sections), not assumed to be
# WTMEC2YR throughout:
#   2007-2008 (E): lead/cadmium/mercury only (selenium & manganese NOT measured
#                  at all in this cycle -- confirmed by inspecting PBCD_E.xpt's
#                  actual columns, which contain no LBXBSE/LBXBMN fields).
#                  "Exam sample weights should be used" -> WTMEC2YR.
#   2009-2010 (F): same as 2007-2008 -- no selenium/manganese in this cycle.
#                  WTMEC2YR.
#   2011-2012 (G): all 5 metals, full MEC-examined sample (no subsample
#                  language in the documentation). WTMEC2YR.
#   2013-2014 (H): "a one-half sample from participants aged 12 years and
#                  older were eligible" for the WHOLE metals panel (all 5
#                  metals share the same half-subsample this cycle). Docs:
#                  "The appropriate sample weights are provided in the
#                  variable WTSH2YR in this data file ... and should be used."
#   2015-2016 (I): same one-half-subsample design as 2013-2014. WTSH2YR.
#   2017-2018 (J): "all examined participants" (no subsample language) for
#                  cadmium/manganese/mercury/selenium; lead has a narrower age
#                  floor (6+ vs 1+) but that's irrelevant to this adults-only
#                  cohort. No special weight named -> WTMEC2YR.
# WTSH2YR is present in nhanes_merged_raw.csv already (it's part of the PBCD_H/
# PBCD_I modules and was pulled in automatically by 01_merge_data.py's
# whole-module merge) -- confirmed non-null exactly in the 2013-2014/2015-2016
# rows and nowhere else, consistent with the documentation above.
METALS_WEIGHT_VAR = "WTSH2YR"
METALS_WEIGHT_BY_CYCLE = {
    "2007_2008": "WTMEC2YR",
    "2009_2010": "WTMEC2YR",
    "2011_2012": "WTMEC2YR",
    "2013_2014": "WTSH2YR",
    "2015_2016": "WTSH2YR",
    "2017_2018": "WTMEC2YR",
}
# Cycles in which selenium/manganese were not measured at all (structural
# absence, not missingness) -- used to scope any selenium/manganese-specific
# descriptive statistic or sensitivity analysis to the cycles where the assay
# actually exists, and to caption figures/tables accordingly.
SE_MN_NOT_MEASURED_CYCLES = ["2007_2008", "2009_2010"]

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# Clinical / human-readable labels. Used only for display (plot axes, SHAP
# labels, Table 1) -- every script keeps working on the raw NHANES codes
# internally; this is purely a rendering layer, applied at the last moment
# via label_feature() below.
# ----------------------------------------------------------------------
CLINICAL_NAMES = {
    # metals
    "LBXBPB": "Blood lead (µg/dL)",
    "LBXBCD": "Blood cadmium (µg/L)",
    "LBXTHG": "Blood mercury, total (µg/L)",
    "LBXBSE": "Blood selenium (µg/L)",
    "LBXBMN": "Blood manganese (µg/L)",
    # demographic
    "RIDAGEYR": "Age (years)",
    "sex_female": "Sex: Female",
    "RIDRETH1": "Race/ethnicity",
    "DMDEDUC2": "Education level",
    "DMDMARTL": "Marital status",
    "DMDHHSIZ": "Household size",
    "INDFMPIR": "Income-to-poverty ratio",
    "exam_summer": "Exam month: May-Oct (vs. Nov-Apr)",
    # anthropometric
    "BMXBMI": "Body mass index (kg/m²)",
    # lifestyle
    "SMQ020": "Smoked ≥100 cigarettes in life",
    "SMQ040": "Current smoking intensity (0=none, 1=some days, 2=every day)",
    "ALQ_EVER": "Ever consumed alcohol (harmonized ALQ101/ALQ111)",
    "ALQ130": "Avg. alcoholic drinks/day (past 12 mo)",
    "DBQ700": "Self-rated diet quality (1=excellent...5=poor)",
    "PAQ605": "Vigorous work activity",
    "PAQ650": "Vigorous recreational activity",
    "PAQ665": "Moderate recreational activity",
    "PAD645": "Minutes walking/biking for transportation",
    "BPXPLS": "Resting pulse (60 sec)",
    # sensitivity-only
    "DIQ010": "Diagnosed diabetes",
    "diq010_yes": "Diagnosed diabetes: Yes",
    "diq010_borderline": "Diagnosed diabetes: Borderline",
    # MetS components (for Table 1 / context, never predictors)
    "BMXWAIST": "Waist circumference (cm)",
    "BPXSY1": "Systolic blood pressure (mmHg)",
    "BPXDI1": "Diastolic blood pressure (mmHg)",
    "LBDHDD": "HDL cholesterol (mg/dL)",
    "LBXTR": "Triglycerides (mg/dL)",
    "LBXGLU": "Fasting glucose (mg/dL)",
    "RIAGENDR": "Sex",
    # medication variables (label-building only, for Table 1 / audit, never predictors)
    "BPQ050A": "On antihypertensive medication",
    "BPQ100D": "On cholesterol-lowering medication",
    "DIQ050": "On insulin",
    "DIQ070": "On oral diabetes medication",
}

# Category value labels for the one-hot-encoded nominal variables, used to
# turn e.g. "RIDRETH1_2.0" into "Race/ethnicity: Other Hispanic".
CATEGORY_LABELS = {
    "RIDRETH1": {
        1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 5: "Other/Multi-Racial",
    },
    "DMDMARTL": {
        1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated",
        5: "Never married", 6: "Living with partner",
    },
}


def label_feature(col_name: str) -> str:
    """Best-effort clinical label for any column that can appear in the
    final feature matrix: a plain predictor (CLINICAL_NAMES), a one-hot
    dummy of a nominal variable (base name + coded category, via
    CATEGORY_LABELS), a missing-value indicator column (sklearn
    MissingIndicator output, prefixed "missingindicator_"), or its "_nan"
    missing-indicator variant from the pre-split one-hot step. Falls back to
    the raw column name if nothing matches, so this never raises."""
    # FIX: handle sklearn's MissingIndicator-style output name so the new
    # "was this value actually observed?" flags (see pipeline_utils.py) get a
    # readable label in SHAP plots / importance tables instead of a raw code.
    if col_name.startswith("missingindicator_"):
        base = col_name[len("missingindicator_"):]
        base_label = CLINICAL_NAMES.get(base, base)
        return f"{base_label}: was measured/observed"

    if col_name in CLINICAL_NAMES:
        return CLINICAL_NAMES[col_name]

    for base, levels in CATEGORY_LABELS.items():
        prefix = f"{base}_"
        if col_name.startswith(prefix):
            suffix = col_name[len(prefix):]
            base_label = CLINICAL_NAMES.get(base, base)
            if suffix == "nan":
                return f"{base_label}: missing"
            try:
                code = float(suffix)
                level_label = levels.get(int(code), suffix)
            except ValueError:
                level_label = suffix
            return f"{base_label}: {level_label}"

    return col_name


def label_features(col_names) -> list:
    return [label_feature(c) for c in col_names]
