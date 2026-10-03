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
    "SBP_MEAN", "DBP_MEAN",  # mean of all valid BP readings (built in 02 from BPX)
    "LBDHDD",  # HDL cholesterol (HDL)
    "LBXTR",  # triglycerides (TRIGLY)
    "LBXGLU",  # fasting glucose (GLU)
]

# FIX (2026-10 audit, C16): blood pressure is the mean of all valid readings,
# not the first reading only. NHANES takes up to four auscultatory readings;
# averaging them is the standard way to reduce white-coat and measurement
# noise. A diastolic reading of 0 mmHg (sounds heard to zero) is not a valid
# value and is treated as missing. The first-reading version is kept for a
# sensitivity arm (mets_status_bpfirst).
BP_SYS_READINGS = ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]
BP_DIA_READINGS = ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]

# FIX (2026-10 audit, C5): fasting check. Triglycerides and glucose are only
# valid after an 8- to 24-hour fast. NCHS assigns a fasting-subsample weight
# (WTSAF2YR) of zero to every participant who did not meet that criterion, in
# every cycle; that is the only fasting indicator available for 2015-2018
# (the separate FASTQX file is not part of this download). Participants with
# WTSAF2YR == 0 are excluded.
FASTING_WEIGHT_VAR = "WTSAF2YR"

# FIX (2026-10 audit, C13): pregnant participants are excluded (RIDEXPRG == 1,
# "Yes, positive lab pregnancy test or self-reported pregnant"); pregnancy
# changes waist, blood pressure, lipids and glucose.
PREGNANCY_VAR = "RIDEXPRG"

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
# FIX (2026-10 audit, C1/C14): refusal and don't-know codes are recoded PER
# VARIABLE, using the code width each NHANES codebook actually uses. The old
# version applied one shared set {7, 9, 77, 99, 777, 999, 7777, 9999} to every
# listed variable, which silently erased genuine answers of 7 and 9 drinks per
# day in ALQ130 (184 and 44 adults), and it left PAD645's 9999 ("Don't know")
# in the data as 9,999 minutes because PAD645 was not listed at all.
_YES_NO = (7, 9)        # 1-digit items: 7 = Refused, 9 = Don't know
_TWO_DIGIT = (77, 99)
_THREE_DIGIT = (777, 999)
_FOUR_DIGIT = (7777, 9999)
SENTINEL_MAP = {
    "SMQ020": _YES_NO, "SMQ040": _YES_NO,
    "ALQ101": _YES_NO, "ALQ110": _YES_NO, "ALQ111": _YES_NO,
    "ALQ120Q": _THREE_DIGIT, "ALQ121": _TWO_DIGIT, "ALQ130": _THREE_DIGIT,
    "DBQ700": _YES_NO,
    "PAQ605": _YES_NO, "PAQ650": _YES_NO, "PAQ665": _YES_NO, "PAQ635": _YES_NO,
    "PAD645": _FOUR_DIGIT,
    "DIQ010": _YES_NO,
    "DMDMARTL": _TWO_DIGIT, "DMDEDUC2": _YES_NO,
    # medication items used to build the outcome label
    "BPQ050A": _YES_NO, "BPQ100D": _YES_NO, "DIQ050": _YES_NO, "DIQ070": _YES_NO,
}
SENTINEL_RECODE_VARS = list(SENTINEL_MAP)  # kept for backward compatibility

# ----------------------------------------------------------------------
# FIX (2026-10 audit, C15): alcohol variables.
# ALQ111 ("ever had a drink") exists only in 2017-2018, and the earlier
# version combined it with ALQ101 ("at least 12 drinks in any one year",
# 2007-2016) into one "ever drank" variable. Those are different questions
# (26.9% "no" in 2007-2016 vs 9.0% in 2017-2018), so the combined variable
# measured different things in different cycles. It is replaced by a
# construct that every cycle measures the same way: ANY ALCOHOL IN THE PAST
# 12 MONTHS.
#   2007-2016: ALQ120Q (how often drank in the past 12 months; 0 = never).
#              People who never had 12 drinks in their life (ALQ101 = 2 and
#              ALQ110 = 2) skip ALQ120Q and are coded as non-drinkers.
#   2017-2018: ALQ121 (past-12-month frequency; 0 = never), and ALQ111 = 2
#              (never had a drink) skips ALQ121 and is coded as non-drinker.
# FIX (C2): ALQ130 (average drinks on a drinking day) is only asked of
# past-year drinkers; for everyone else it is a structural zero, not an
# unknown value, and is set to 0 before imputation.
ALQ_CURRENT_VAR = "ALQ_PAST12"

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
    # FIX (C15): ALQ_PAST12 (any alcohol in the past 12 months) replaces the
    # earlier ALQ_EVER, which mixed two different questions across cycles.
    "lifestyle": ["SMQ020", "SMQ040", "ALQ_PAST12", "ALQ130", "DBQ700",
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
BINARY_YESNO_1_2 = ["SMQ020", "PAQ605", "PAQ650", "PAQ665"]

# FIX (C19): binary variables already built as clean 0/1 in
# 02_clean_and_outcome.py. They go through the categorical (mode-impute,
# unscaled) path, not the continuous one, so SMOTENC never creates
# fractional values for them.
BINARY_01_AS_IS = ["ALQ_PAST12"]

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
CONTINUOUS_AS_IS = [
    "RIDAGEYR", "INDFMPIR", "DMDHHSIZ", "BMXBMI",
    "ALQ130", "PAD645", "BPXPLS",
    "LBXBPB", "LBXBCD", "LBXTHG", "LBXBSE", "LBXBMN",
]

# FIX (C18): selenium and manganese were not assayed in 2007-2010, so their
# missingness is identical and is a period marker, not informative
# non-response. The two flags are merged into one column with an honest name.
SE_MN_FLAG_NAME = "missingindicator_LBXBSE"

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
    "ALQ_PAST12": "Drank alcohol in past 12 months",
    "ALQ130": "Avg. drinks per drinking day (past 12 mo; 0 = non-drinker)",
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
    "SBP_MEAN": "Systolic blood pressure, mean of readings (mmHg)",
    "DBP_MEAN": "Diastolic blood pressure, mean of readings (mmHg)",
    "BPXSY1": "Systolic blood pressure, first reading (mmHg)",
    "BPXDI1": "Diastolic blood pressure, first reading (mmHg)",
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
    # FIX (2026-10 audit, C4): sklearn's MissingIndicator outputs 1 when the
    # value is MISSING (and was therefore imputed). The earlier label said
    # "was measured/observed", which inverted the meaning of every SHAP colour
    # for these flags.
    if col_name == SE_MN_FLAG_NAME:
        return "Selenium & manganese not measured (2007-2010 cycles)"
    if col_name.startswith("missingindicator_"):
        base = col_name[len("missingindicator_"):]
        base_label = CLINICAL_NAMES.get(base, base)
        return f"{base_label}: missing (imputed)"

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
