"""
Step 2 -- Data Cleaning
Step 3 -- Outcome Definition (MetS, AHA/NHLBI harmonized adult criteria)

Reads data_processed/nhanes_merged_raw.csv (from 01_merge_data.py) and:
  1. Filters to adults (RIDAGEYR >= config.MIN_AGE)
  2. Recodes NHANES refusal/don't-know sentinel codes to missing in the
     questionnaire variables listed in config.SENTINEL_RECODE_VARS
  3. Excludes pregnant participants (2026-10 audit, C13)
  4. Builds ALQ_PAST12 (any alcohol in the past 12 months, same construct in
     every cycle), sets ALQ130 = 0 for non-drinkers, sets SMQ040 = 'not at
     all' for never-smokers, and recodes PAD645 to 0 minutes wherever the
     PAQ635 skip gate says the respondent doesn't walk/bike for transportation
  5. Builds mean systolic/diastolic blood pressure from all valid readings
  5. Reports missingness (saved to outputs/metrics/missingness_report.csv)
  6. Drops rows missing any of the 5 MetS-defining lab/exam components (label
     can't be computed otherwise), rows that did not meet the NCHS fasting
     criterion (2026-10 audit, C5), or rows missing ALL 5 metals (no exposure
     signal at all). Medication variables are NOT part of this drop -- a missing
     medication-use item is treated as "not on treatment" (see build_mets_status
     docstring), consistent with NHANES's own skip-pattern logic, so it never
     removes anyone from the cohort.
  7. FIX: builds the 5 MetS component flags + mets_status using the AHA/NHLBI
     HARMONIZED definition (Alberti et al. 2009): each component is positive if
     the lab/exam threshold is crossed OR the person is on documented drug
     treatment for that condition. The original pipeline used lab/exam values
     only. Both versions are computed and saved side by side
     (mets_status_labonly vs mets_status) with an explicit comparison table so
     the size of this fix's effect is auditable, not just asserted.
  8. Writes a STROBE-style exclusion flow table

Output: data_processed/nhanes_with_outcome.csv
        outputs/metrics/exclusion_flow.csv
        outputs/metrics/missingness_report.csv
        outputs/metrics/outcome_definition_comparison.csv

Run:
    python 06_Code/02_clean_and_outcome.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def record(flow, label, df):
    flow.append({"step": label, "n_remaining": len(df)})
    return df


def recode_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """FIX (2026-10 audit, C1/C14): per-variable refusal / don't-know codes
    (config.SENTINEL_MAP), so a genuine answer that happens to equal a code
    used elsewhere (7 or 9 drinks per day) is never erased."""
    df = df.copy()
    for col, codes in cfg.SENTINEL_MAP.items():
        if col not in df.columns:
            continue
        hit = df[col].isin(codes)
        if hit.any():
            print(f"  {col}: recoding {int(hit.sum())} values coded {codes} to NaN")
            df.loc[hit, col] = np.nan
    return df


def build_alcohol_vars(df: pd.DataFrame) -> pd.DataFrame:
    """FIX (2026-10 audit, C2/C15): ALQ_PAST12 = any alcohol in the past 12
    months, built the same way in every cycle (see config.ALQ_CURRENT_VAR),
    and ALQ130 = 0 for everyone who did not drink in the past 12 months
    (they were never asked ALQ130, so their blank is a structural zero)."""
    df = df.copy()
    get = lambda c: df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
    q120, q121 = get("ALQ120Q"), get("ALQ121")
    q101, q110, q111 = get("ALQ101"), get("ALQ110"), get("ALQ111")

    drinker = (q120 > 0) | (q121 > 0)
    non_drinker = (q120 == 0) | (q121 == 0) | (q111 == 2) | ((q101 == 2) & (q110 == 2))
    past12 = pd.Series(np.nan, index=df.index)
    past12[non_drinker] = 0
    past12[drinker] = 1
    df[cfg.ALQ_CURRENT_VAR] = past12
    print(f"  ALQ_PAST12 built: {int((past12 == 1).sum())} drinkers, "
          f"{int((past12 == 0).sum())} non-drinkers, {int(past12.isna().sum())} unknown")

    before = int(df["ALQ130"].isna().sum())
    df.loc[past12 == 0, "ALQ130"] = 0
    print(f"  ALQ130: set to 0 for {before - int(df['ALQ130'].isna().sum())} past-year non-drinkers")
    return df


def recode_smq040_skip_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """FIX (2026-10 audit, C3): SMQ040 (do you now smoke?) is only asked of
    people who smoked at least 100 cigarettes (SMQ020 = 1). Never-smokers
    (SMQ020 = 2) do not smoke now: raw code 3 ('Not at all'). Before this fix
    their blank was left to mode imputation, which only gave the right answer
    because the mode happened to be 'not at all'."""
    df = df.copy()
    n = int(((df["SMQ020"] == 2) & df["SMQ040"].isna()).sum())
    df.loc[df["SMQ020"] == 2, "SMQ040"] = 3
    print(f"  SMQ040: set to 'not at all' for {n} never-smokers")
    return df


def recode_pad645_skip_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """FIX: PAD645 is only asked of respondents who said "Yes" to PAQ635 (walk
    or bicycle for transportation). A "No" (code 2) on PAQ635 skips PAD645
    entirely, leaving it blank -- that blank means confirmed zero minutes, not
    unknown. Recode those rows to 0 before PAD645 is treated as a genuine
    missing-data problem for imputation."""
    df = df.copy()
    gate = cfg.PAD645_GATE_VAR
    var = cfg.PAD645_VAR
    if gate not in df.columns or var not in df.columns:
        print(f"  [NOTE] {gate} or {var} not found -- skipping PAD645 skip-pattern recode")
        return df
    n_recoded = (df[gate] == 2).sum()
    df.loc[df[gate] == 2, var] = 0
    print(f"  PAD645: recoded {n_recoded} rows to 0 minutes based on PAQ635=='No' skip pattern")
    return df


def build_mean_bp(df: pd.DataFrame) -> pd.DataFrame:
    """FIX (2026-10 audit, C16): mean of all valid readings. Diastolic 0 is
    not a valid reading and is treated as missing."""
    df = df.copy()
    sys_cols = [c for c in cfg.BP_SYS_READINGS if c in df.columns]
    dia = df[[c for c in cfg.BP_DIA_READINGS if c in df.columns]].copy()
    n_zero = int((dia == 0).sum().sum())
    dia = dia.mask(dia == 0)
    df["SBP_MEAN"] = df[sys_cols].mean(axis=1)
    df["DBP_MEAN"] = dia.mean(axis=1)
    df["BP_N_READINGS"] = df[sys_cols].notna().sum(axis=1)
    print(f"  BP: mean of up to {len(sys_cols)} readings; {n_zero} diastolic readings of 0 set to missing")
    return df


def _med_flag(df: pd.DataFrame, col: str) -> pd.Series:
    """Missing-safe 'on treatment' flag: NaN (item not asked, per NHANES skip
    logic -- e.g. BPQ050A is only asked of people who said they'd been told
    they have high blood pressure) is treated as 'not on treatment', which is
    the standard, defensible reading of these particular skip patterns (you
    cannot be on a medication for a condition you were never told you have).
    This is stated as an explicit modeling assumption, not silently assumed."""
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return (df[col] == 1).fillna(False)


def build_mets_status(df: pd.DataFrame) -> pd.DataFrame:
    """Builds both the original lab/exam-only MetS label (mets_status_labonly,
    reproducing the pipeline's original -- and incomplete -- definition) and
    the corrected, AHA/NHLBI-harmonized label (mets_status) that also counts a
    component positive when the person is on documented drug treatment for it.
    See config.MEDICATION_VARS_FOR_LABEL for the exact variables and citations."""
    df = df.copy()
    t = cfg.METS_THRESHOLDS
    male = df["RIAGENDR"] == 1
    female = df["RIAGENDR"] == 2

    # --- lab/exam-only components (original definition) ---
    df["mets_waist"] = (
        (male & (df["BMXWAIST"] >= t["waist_men"])) |
        (female & (df["BMXWAIST"] >= t["waist_women"]))
    ).astype(int)  # waist has no medication-treatment clause in the harmonized definition

    trigly_lab = df["LBXTR"] >= t["triglycerides"]
    hdl_lab = (male & (df["LBDHDD"] < t["hdl_men"])) | (female & (df["LBDHDD"] < t["hdl_women"]))
    bp_lab = (df["SBP_MEAN"] >= t["sbp"]) | (df["DBP_MEAN"] >= t["dbp"])
    # first-reading-only version, kept for the sensitivity arm (decision C16)
    dia1 = df["BPXDI1"].mask(df["BPXDI1"] == 0)
    bp_first_lab = (df["BPXSY1"] >= t["sbp"]) | (dia1 >= t["dbp"])
    glu_lab = df["LBXGLU"] >= t["glucose"]

    df["mets_trigly_labonly"] = trigly_lab.astype(int)
    df["mets_hdl_labonly"] = hdl_lab.astype(int)
    df["mets_bp_labonly"] = bp_lab.astype(int)
    df["mets_glu_labonly"] = glu_lab.astype(int)

    # --- FIX: medication-adjusted components (harmonized definition) ---
    on_bp_med = _med_flag(df, "BPQ050A")
    on_lipid_med = _med_flag(df, "BPQ100D")  # applied to both HDL & TG, see config.py note
    on_insulin = _med_flag(df, "DIQ050")
    on_glu_pills = _med_flag(df, "DIQ070")
    on_glu_med = on_insulin | on_glu_pills

    df["mets_trigly"] = (trigly_lab | on_lipid_med).astype(int)
    df["mets_hdl"] = (hdl_lab | on_lipid_med).astype(int)
    df["mets_bp"] = (bp_lab | on_bp_med).astype(int)
    df["mets_glu"] = (glu_lab | on_glu_med).astype(int)

    df["on_bp_med"] = on_bp_med.astype(int)
    df["on_lipid_med"] = on_lipid_med.astype(int)
    df["on_glu_med"] = on_glu_med.astype(int)

    df["mets_score_labonly"] = df[["mets_waist", "mets_trigly_labonly", "mets_hdl_labonly",
                                    "mets_bp_labonly", "mets_glu_labonly"]].sum(axis=1)
    df["mets_status_labonly"] = (df["mets_score_labonly"] >= cfg.METS_POSITIVE_COUNT).astype(int)

    df["mets_score"] = df[["mets_waist", "mets_trigly", "mets_hdl", "mets_bp", "mets_glu"]].sum(axis=1)
    df[cfg.OUTCOME_VAR] = (df["mets_score"] >= cfg.METS_POSITIVE_COUNT).astype(int)

    # Sensitivity arms for the two outcome-definition decisions (A8):
    # (a) cholesterol-medication clause NOT applied to TG/HDL (BP and glucose
    #     treatment clauses kept), because NHANES cannot tell statins from
    #     fibrates or niacin;
    # (b) harmonized definition but with the first BP reading only.
    score_nolipid = (df["mets_waist"] + trigly_lab.astype(int) + hdl_lab.astype(int)
                     + df["mets_bp"] + df["mets_glu"])
    df["mets_status_nolipidmed"] = (score_nolipid >= cfg.METS_POSITIVE_COUNT).astype(int)
    bp_first = (bp_first_lab.fillna(False) | on_bp_med).astype(int)
    score_bpfirst = (df["mets_waist"] + df["mets_trigly"] + df["mets_hdl"] + bp_first + df["mets_glu"])
    df["mets_status_bpfirst"] = (score_bpfirst >= cfg.METS_POSITIVE_COUNT).astype(int)

    return df


def report_outcome_definition_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Quantifies how much the drug-treatment clause, and each sensitivity
    variant of the definition, changes the outcome label."""
    n = len(df)
    n_pos_old = int(df["mets_status_labonly"].sum())
    n_pos_new = int(df[cfg.OUTCOME_VAR].sum())
    reclassified = df["mets_status_labonly"] != df[cfg.OUTCOME_VAR]
    n_reclass_to_positive = int(((df["mets_status_labonly"] == 0) & (df[cfg.OUTCOME_VAR] == 1)).sum())
    n_reclass_to_negative = int(((df["mets_status_labonly"] == 1) & (df[cfg.OUTCOME_VAR] == 0)).sum())

    n_nolip = int(df["mets_status_nolipidmed"].sum())
    n_bpf = int(df["mets_status_bpfirst"].sum())
    rows = [
        {"definition": "lab_only (measured values only)", "n": n, "n_positive": n_pos_old,
         "prevalence_pct": round(100 * n_pos_old / n, 2)},
        {"definition": "harmonized (measured value or drug treatment; primary)", "n": n,
         "n_positive": n_pos_new, "prevalence_pct": round(100 * n_pos_new / n, 2)},
        {"definition": "harmonized without the lipid-medication clause (sensitivity)", "n": n,
         "n_positive": n_nolip, "prevalence_pct": round(100 * n_nolip / n, 2)},
        {"definition": "harmonized with the first blood-pressure reading only (sensitivity)", "n": n,
         "n_positive": n_bpf, "prevalence_pct": round(100 * n_bpf / n, 2)},
    ]
    summary = pd.DataFrame(rows)
    summary["n_reclassified_negative_to_positive"] = [np.nan, n_reclass_to_positive, np.nan, np.nan]
    summary["n_reclassified_positive_to_negative"] = [np.nan, n_reclass_to_negative, np.nan, np.nan]
    summary["n_on_any_relevant_medication"] = [
        np.nan, int((df["on_bp_med"] | df["on_lipid_med"] | df["on_glu_med"]).sum()), np.nan, np.nan]

    out_path = cfg.OUT_METRICS / "outcome_definition_comparison.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nOutcome-definition comparison (saved to {out_path}):")
    print(summary.to_string(index=False))
    print(f"Total reclassified by the medication fix: {int(reclassified.sum())} of {n} "
          f"({100*reclassified.sum()/n:.2f}%) -- "
          f"{n_reclass_to_positive} moved MetS-negative -> MetS-positive, "
          f"{n_reclass_to_negative} moved MetS-positive -> MetS-negative (expected: the harmonized "
          f"definition only ever adds positives, so this should be 0 unless a data anomaly exists).")
    return summary


def main():
    in_path = cfg.DATA_PROC_DIR / "nhanes_merged_raw.csv"
    df = pd.read_csv(in_path, low_memory=False)
    print(f"Loaded {in_path.name}: {df.shape[0]} rows, {df.shape[1]} cols")

    flow = []
    flow.append({"step": "0. Pooled sample (all cycles, all ages)", "n_remaining": len(df)})

    # 1. Age filter
    df = df[df["RIDAGEYR"] >= cfg.MIN_AGE].copy()
    flow.append({"step": f"1. Age >= {cfg.MIN_AGE}", "n_remaining": len(df)})
    print(f"After age filter (>= {cfg.MIN_AGE}): {len(df)} rows")

    # 1b. FIX (C13): exclude pregnant participants
    preg = df[cfg.PREGNANCY_VAR] == 1
    df = df[~preg].copy()
    flow.append({"step": "2. Not pregnant", "n_remaining": len(df)})
    print(f"After excluding {int(preg.sum())} pregnant participants: {len(df)} rows")

    # 2. Sentinel recoding (per-variable codes)
    print("\nRecoding refusal/don't-know codes to missing:")
    df = recode_sentinels(df)

    # 3. Skip patterns and derived variables
    print("\nAlcohol variables:")
    df = build_alcohol_vars(df)
    print("\nSmoking skip pattern:")
    df = recode_smq040_skip_pattern(df)
    print("\nApplying PAD645 skip-pattern recode:")
    df = recode_pad645_skip_pattern(df)
    print("\nBlood pressure:")
    df = build_mean_bp(df)

    # 4. Missingness report (on the age-filtered cohort, before dropping anything).
    # FIX (found during the "check every step" precision review): this report
    # used to ALSO be the file that 14_generate_diagrams.py and the proposal
    # text cite for "the model-ready analytic dataset's worst-missing
    # predictor" -- but it is computed here on the age-filtered, PRE-exclusion
    # cohort (n=34,770), before the MetS-completeness and metals-availability
    # filters run below. That is a real, if modest, population mismatch: for
    # example SMQ040 (the actual worst-missing predictor) is 56.03% missing
    # here vs. 55.28% missing in the true final n=11,144 analytic cohort.
    # This age-filtered report is still useful in its own right (it is what
    # motivates/explains the exclusion filters below, which by definition
    # haven't run yet at this point in the script), so it is kept -- but
    # renamed and clearly labeled, and a SECOND report computed on the true
    # final cohort (after both exclusion filters + outcome labeling) is added
    # further down and saved as missingness_report.csv, which is the name
    # everything downstream already expects and correctly describes what
    # they cite it as: missingness within the actual model-ready dataset.
    cols_of_interest = (
        cfg.METS_COMPONENT_VARS
        + cfg.MEDICATION_VARS_FOR_LABEL
        + cfg.PREDICTOR_VARS["metals"]
        + cfg.PREDICTOR_VARS["demographic"]
        + cfg.PREDICTOR_VARS["anthropometric"]
        + cfg.PREDICTOR_VARS["lifestyle"]
    )
    cols_present = [c for c in cols_of_interest if c in df.columns]
    missing_report_age_filtered = (
        df[cols_present].isna().mean().mul(100).round(2)
        .sort_values(ascending=False)
        .rename("pct_missing").reset_index().rename(columns={"index": "variable"})
    )
    missing_report_age_filtered.to_csv(cfg.OUT_METRICS / "missingness_report_age_filtered_precohort.csv", index=False)
    print("\nTop 20 variables by % missing (age >= 20 cohort, BEFORE the MetS-completeness")
    print("and metals-availability exclusion filters below -- NOT the final analytic cohort):")
    print(missing_report_age_filtered.head(20).to_string(index=False))

    # 5a. Drop rows missing any of the 5 MetS-defining LAB components
    #     (medication variables are NOT part of this filter -- see _med_flag)
    df = df.dropna(subset=cfg.METS_COMPONENT_VARS)
    flow.append({"step": "3. Complete data on all 5 MetS lab/exam components", "n_remaining": len(df)})
    print(f"\nAfter requiring complete MetS lab components: {len(df)} rows")

    # 5b. FIX (C5): met the NCHS fasting criterion (8 to <24 h). NCHS gives a
    #     zero fasting-subsample weight to everyone who did not; values that
    #     the SAS transport reader returns as ~1e-79 are true zeros.
    fasted = df[cfg.FASTING_WEIGHT_VAR].fillna(0) > 0
    n_not_fasted = int((~fasted).sum())
    df = df[fasted].copy()
    flow.append({"step": "4. Met fasting criterion (8 to 24 h; non-zero fasting weight)", "n_remaining": len(df)})
    print(f"After excluding {n_not_fasted} participants who did not meet the fasting criterion: {len(df)} rows")

    # 5c. Drop rows missing ALL 5 metals (no exposure signal at all)
    metal_cols = cfg.PREDICTOR_VARS["metals"]
    n_metals_present = df[metal_cols].notna().sum(axis=1)
    df = df[n_metals_present > 0].copy()
    flow.append({"step": "5. At least 1 of 5 metals measured", "n_remaining": len(df)})
    print(f"After requiring >=1 metal measured: {len(df)} rows")

    # 6. Build outcome (both label-only and harmonized versions)
    df = build_mets_status(df)
    prevalence = df[cfg.OUTCOME_VAR].mean() * 100
    print(f"\nFinal analytic cohort: {len(df)} rows")
    print(f"MetS prevalence, harmonized definition (unweighted): {prevalence:.2f}%")
    print(f"MetS cases: {df[cfg.OUTCOME_VAR].sum()}  /  Non-cases: {(df[cfg.OUTCOME_VAR] == 0).sum()}")

    report_outcome_definition_impact(df)

    # Missingness report on the TRUE final analytic cohort (post both exclusion
    # filters, post outcome labeling) -- this is the file 14_generate_diagrams.py
    # and the proposal text actually mean when they say "missingness in the
    # model-ready dataset" / "worst-missing predictor", so it is saved under
    # the name they already read: missingness_report.csv. See the FIX note
    # above cols_of_interest for why this is a separate computation from the
    # age-filtered, pre-exclusion report saved just above.
    cols_present_final = [c for c in cols_of_interest if c in df.columns]
    missing_report_final = (
        df[cols_present_final].isna().mean().mul(100).round(2)
        .sort_values(ascending=False)
        .rename("pct_missing").reset_index().rename(columns={"index": "variable"})
    )
    missing_report_final.to_csv(cfg.OUT_METRICS / "missingness_report.csv", index=False)
    print(f"\nTop 20 variables by % missing (TRUE final analytic cohort, n={len(df)}):")
    print(missing_report_final.head(20).to_string(index=False))

    flow.append({"step": "6. Final analytic cohort (mets_status computed)", "n_remaining": len(df)})
    flow_df = pd.DataFrame(flow)
    flow_df["n_excluded_this_step"] = flow_df["n_remaining"].shift(1) - flow_df["n_remaining"]
    flow_df.loc[0, "n_excluded_this_step"] = 0
    flow_df.to_csv(cfg.OUT_METRICS / "exclusion_flow.csv", index=False)
    print("\nExclusion flow:")
    print(flow_df.to_string(index=False))

    out_path = cfg.DATA_PROC_DIR / "nhanes_with_outcome.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({df.shape[0]} rows, {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
