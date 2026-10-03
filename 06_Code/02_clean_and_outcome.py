"""
Step 2 -- Data Cleaning
Step 3 -- Outcome Definition (MetS, AHA/NHLBI harmonized adult criteria)

Reads data_processed/nhanes_merged_raw.csv (from 01_merge_data.py) and:
  1. Filters to adults (RIDAGEYR >= config.MIN_AGE)
  2. Recodes NHANES refusal/don't-know sentinel codes to missing in the
     questionnaire variables listed in config.SENTINEL_RECODE_VARS
  3. FIX: builds the harmonized ALQ_EVER alcohol variable from ALQ101 (2007-2016)
     / ALQ111 (2017-2018) -- see config.py ALQ_EVER_SOURCES for the full
     rationale. Structural, not random, missingness; verified against real
     per-cycle non-null counts before this fix was written.
  4. FIX: recodes PAD645 to 0 minutes wherever the PAQ635 skip gate says the
     respondent doesn't walk/bike for transportation (see config.py).
  5. Reports missingness (saved to outputs/metrics/missingness_report.csv)
  6. Drops rows missing any of the 5 MetS-defining lab/exam components (label
     can't be computed otherwise), or missing ALL 5 metals (no exposure signal
     at all). Medication variables are NOT part of this drop -- a missing
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
    python scripts/02_clean_and_outcome.py
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
    df = df.copy()
    for col in cfg.SENTINEL_RECODE_VARS:
        if col not in df.columns:
            continue
        n_flagged = df[col].isin(cfg.SENTINEL_CODES).sum()
        if n_flagged:
            print(f"  {col}: recoding {n_flagged} sentinel values to NaN")
            df.loc[df[col].isin(cfg.SENTINEL_CODES), col] = np.nan
    return df


def build_alq_ever(df: pd.DataFrame) -> pd.DataFrame:
    """FIX: harmonize the alcohol-ever gatekeeper question across the NHANES
    2013-2014 questionnaire redesign. ALQ111 exists only in 2017-2018; ALQ101
    exists only in 2007-2016. Take whichever is non-null for a given row (by
    construction, at most one of the two is ever populated for any single
    respondent, since they come from mutually exclusive cycles) and recode
    from the raw NHANES 1=Yes/2=No coding to a clean 1/0."""
    df = df.copy()
    has_111 = "ALQ111" in df.columns
    has_101 = "ALQ101" in df.columns
    if not (has_111 or has_101):
        raise SystemExit("Neither ALQ111 nor ALQ101 present -- cannot build ALQ_EVER.")
    if has_111 and has_101:
        raw = df["ALQ111"].where(df["ALQ111"].notna(), df["ALQ101"])
        overlap = (df["ALQ111"].notna() & df["ALQ101"].notna()).sum()
        if overlap:
            print(f"  [WARNING] {overlap} rows have BOTH ALQ111 and ALQ101 non-null "
                  f"(expected 0 given the cycles are mutually exclusive) -- ALQ111 takes priority")
    elif has_111:
        raw = df["ALQ111"]
    else:
        raw = df["ALQ101"]
    df[cfg.ALQ_EVER_VAR] = raw.map({1: 1, 2: 0})
    n_by_source = {
        "from_ALQ111_2017_2018": int((df["ALQ111"].notna()).sum()) if has_111 else 0,
        "from_ALQ101_2007_2016": int((df["ALQ101"].notna()).sum()) if has_101 else 0,
    }
    print(f"  ALQ_EVER built: {n_by_source}, "
          f"{df[cfg.ALQ_EVER_VAR].notna().sum()} non-null of {len(df)} total")
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
    bp_lab = (df["BPXSY1"] >= t["sbp"]) | (df["BPXDI1"] >= t["dbp"])
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

    return df


def report_outcome_definition_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Quantifies exactly how much the medication-adjusted fix changes the
    outcome label, instead of just asserting that it matters."""
    n = len(df)
    n_pos_old = int(df["mets_status_labonly"].sum())
    n_pos_new = int(df[cfg.OUTCOME_VAR].sum())
    reclassified = df["mets_status_labonly"] != df[cfg.OUTCOME_VAR]
    n_reclass_to_positive = int(((df["mets_status_labonly"] == 0) & (df[cfg.OUTCOME_VAR] == 1)).sum())
    n_reclass_to_negative = int(((df["mets_status_labonly"] == 1) & (df[cfg.OUTCOME_VAR] == 0)).sum())

    rows = [
        {"definition": "lab_only (original pipeline)", "n": n, "n_positive": n_pos_old,
         "prevalence_pct": round(100 * n_pos_old / n, 2)},
        {"definition": "harmonized (this fix: + drug-treatment clause)", "n": n, "n_positive": n_pos_new,
         "prevalence_pct": round(100 * n_pos_new / n, 2)},
    ]
    summary = pd.DataFrame(rows)
    summary["n_reclassified_negative_to_positive"] = [np.nan, n_reclass_to_positive]
    summary["n_reclassified_positive_to_negative"] = [np.nan, n_reclass_to_negative]
    summary["n_on_any_relevant_medication"] = [np.nan, int((df["on_bp_med"] | df["on_lipid_med"] | df["on_glu_med"]).sum())]

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

    # 2. Sentinel recoding
    print("\nRecoding sentinel/refusal codes to missing:")
    df = recode_sentinels(df)

    # 3. FIX: harmonized alcohol variable + PAD645 skip-pattern recode
    print("\nBuilding harmonized ALQ_EVER variable:")
    df = build_alq_ever(df)
    print("\nApplying PAD645 skip-pattern recode:")
    df = recode_pad645_skip_pattern(df)

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
    flow.append({"step": "2. Complete data on all 5 MetS lab/exam components", "n_remaining": len(df)})
    print(f"\nAfter requiring complete MetS lab components: {len(df)} rows")

    # 5b. Drop rows missing ALL 5 metals (no exposure signal at all)
    metal_cols = cfg.PREDICTOR_VARS["metals"]
    n_metals_present = df[metal_cols].notna().sum(axis=1)
    df = df[n_metals_present > 0].copy()
    flow.append({"step": "3. At least 1 of 5 metals measured", "n_remaining": len(df)})
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

    flow.append({"step": "4. Final analytic cohort (mets_status computed)", "n_remaining": len(df)})
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
