"""
Step 14 -- Reproducibility and Reporting

Produces two things every write-up of this analysis needs:
  1. A consolidated STROBE-style exclusion flow (pulls together the per-cycle
     counts from 01_merge_data.py and the exclusion steps from
     02_clean_and_outcome.py into one table)
  2. A survey-weighted Table 1 (baseline characteristics by MetS status)

REBUILD NOTE / FIX (two changes from the original 11_reporting.py):

1. Real design-based (Taylor-linearization, "ultimate cluster") standard
   errors, not the renormalized-weight DescrStatsW approximation the
   original script used and explicitly flagged as a stopgap. NHANES's own
   analytic guidelines specify the "ultimate cluster" simplification for a
   stratified, multistage design with (by construction, in the public-use
   masked variance files) exactly two sampled PSUs per pseudo-stratum:
   treat each pseudo-stratum's PSUs as if drawn with replacement, and
   estimate the variance of any weighted total/mean/proportion from the
   between-PSU variance of PSU-level totals of the (possibly linearized)
   statistic of interest, within stratum, summed across strata (see Wolter,
   "Introduction to Variance Estimation," 2nd ed., Ch. 2; CDC/NCHS NHANES
   Analytic and Reporting Guidelines). This is implemented from scratch in
   pure numpy/pandas below (ultimate_cluster_domain_mean) rather than via a
   third-party survey package, specifically so every arithmetic step is
   inspectable and citable in a methods section, and because the two
   Python survey packages evaluated (`samplics`, effectively archived; a
   newer `svy` package, not yet independently vetted for this project)
   both add a dependency whose internals would otherwise need to be taken
   on faith for a Q1 submission's stats-reviewer.

   The function also handles DOMAIN (subpopulation) estimation correctly:
   Table 1 splits the cohort by mets_status, and a group-conditional mean
   is a domain mean, not a simple-random-sample mean of a pre-filtered
   table. The linearized variable is computed over the FULL sample (every
   stratum/PSU, not just domain members) with domain non-members
   contributing exactly zero to the linearized sum -- this is what lets a
   stratum's between-PSU variance correctly reflect a PSU that happens to
   contain zero domain members, which a naive "filter to the domain rows,
   then treat what's left as a fresh stratified sample" approach would
   silently drop.

2. Correct per-cycle metals survey weight. The original script used
   WTMEC2YR uniformly for every continuous Table-1 row, including the five
   blood metals. Per config.METALS_WEIGHT_BY_CYCLE (built from each
   cycle's own PBCD_<letter> "Analytic Notes" documentation -- see
   config.py for the cycle-by-cycle citations), 2013-2014 and 2015-2016
   used a one-half subsample for the metals panel and must be weighted
   with WTSH2YR, not WTMEC2YR; the other four cycles correctly use
   WTMEC2YR. Using WTMEC2YR throughout would silently misrepresent the
   selection probability of every metals observation from the two
   half-subsample cycles.

REMAINING CAVEAT (read before citing these SEs anywhere): the ultimate-
cluster estimator below is the standard, NHANES-recommended approximation
for a two-PSU-per-stratum masked public-use design, and is a materially
more accurate design-based SE than the original script's renormalized-
weight approximation (which ignored PSU/stratum clustering entirely). It
still does NOT replicate every nuance of what a dedicated package (R
`survey`, Python `samplics`/`svy`) would do for a Rao-Scott-corrected
categorical association test -- the categorical p-values below remain an
UNWEIGHTED chi-square on a weight-rescaled contingency table (same
approach as the original script) and should be read as descriptive, not
design-based inferential, tests; the continuous-row p-values (design-based
Wald tests built directly on the ultimate-cluster SEs of each domain mean)
ARE now properly design-based. If categorical Table-1 p-values need to be
design-corrected for the final submission, redo those specific tests with
R's svychisq() or samplics before submission.

Output: outputs/metrics/strobe_flow.csv
        outputs/metrics/table1_by_mets_status.csv

Run:
    python scripts/11_reporting.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

TABLE1_CONTINUOUS = ["RIDAGEYR", "BMXBMI", "LBXBPB", "LBXBCD", "LBXTHG", "LBXBSE", "LBXBMN", "INDFMPIR"]
TABLE1_CATEGORICAL = ["RIAGENDR", "RIDRETH1", "DMDEDUC2", "SMQ020"]

# Value labels for Table 1's categorical rows. RIDRETH1/DMDMARTL reuse
# config.CATEGORY_LABELS; the rest are Table-1-specific (not otherwise
# one-hot encoded with a stored label) so they're defined here.
TABLE1_LEVEL_LABELS = {
    "RIAGENDR": {1: "Male", 2: "Female"},
    "SMQ020": {1: "Yes", 2: "No"},
    "DMDEDUC2": {1: "<9th grade", 2: "9-11th grade", 3: "HS grad/GED",
                 4: "Some college/AA degree", 5: "College grad or above"},
    **cfg.CATEGORY_LABELS,
}

# components fitting for context (not predictors) -- need the fasting subsample weight
FASTING_DERIVED = {"LBXTR", "LBXGLU"}
METALS_VARS = set(cfg.PREDICTOR_VARS["metals"])


def table1_level_label(var, level):
    base = cfg.label_feature(var)
    levels = TABLE1_LEVEL_LABELS.get(var, {})
    try:
        level_label = levels.get(int(float(level)), str(level))
    except (ValueError, TypeError):
        level_label = str(level)
    return f"{base}: {level_label}"


def build_strobe_flow():
    merged_path = cfg.DATA_PROC_DIR / "nhanes_merged_raw.csv"
    exclusion_path = cfg.OUT_METRICS / "exclusion_flow.csv"
    if not exclusion_path.exists():
        raise SystemExit(f"{exclusion_path} not found -- run 02_clean_and_outcome.py first.")

    rows = []
    if merged_path.exists():
        merged = pd.read_csv(merged_path, usecols=[cfg.CYCLE_VAR], low_memory=False)
        per_cycle = merged.groupby(cfg.CYCLE_VAR).size()
        for cyc, n in per_cycle.items():
            rows.append({"stage": f"Cycle {cyc} (raw, all ages)", "n": n})
        rows.append({"stage": "TOTAL pooled (all cycles, all ages)", "n": len(merged)})

    exclusion = pd.read_csv(exclusion_path)
    for _, r in exclusion.iterrows():
        rows.append({"stage": r["step"], "n": r["n_remaining"]})

    flow_df = pd.DataFrame(rows)
    flow_df.to_csv(cfg.OUT_METRICS / "strobe_flow.csv", index=False)
    print("STROBE flow:")
    print(flow_df.to_string(index=False))
    return flow_df


# ----------------------------------------------------------------------
# FIX: real Taylor-linearization ("ultimate cluster") design-based SE,
# replacing the original script's renormalized-weight DescrStatsW
# approximation. See module docstring for the full derivation/citations.
# ----------------------------------------------------------------------
def ultimate_cluster_domain_mean(df_full: pd.DataFrame, y, weight_col: str,
                                  domain_mask: pd.Series, strata_col=cfg.SURVEY_STRATA,
                                  psu_col=cfg.SURVEY_PSU):
    """Design-based weighted mean of y within the subpopulation (domain)
    identified by domain_mask, with a Taylor-linearized ultimate-cluster
    standard error that accounts for NHANES's stratified, two-PSU-per-
    stratum multistage sampling design.

    y may be a column name (str) in df_full, or any pd.Series aligned to
    df_full's index (e.g. a one-off 0/1 level indicator built by the
    caller for a categorical proportion -- see build_table1) -- passing a
    Series directly avoids mutating/fragmenting the caller's dataframe.

    domain_mask should already encode the subpopulation of interest (e.g.
    mets_status == 1) AND, for a continuous variable, that variable's own
    eligibility (var.notna()) -- see callers below; y's own notna is also
    enforced here as a final guard. Rows outside the domain contribute
    exactly 0 to the linearized variable, which is what makes this a
    correct domain (subpopulation) variance rather than a simple-random-
    sample variance computed on a pre-filtered table (see module docstring).

    Returns (mean, se, n_domain). If fewer than 2 non-singleton PSUs
    contribute anywhere, se is np.nan (cannot estimate a design-based
    variance from a single PSU).
    """
    d = df_full[[weight_col, strata_col, psu_col]].copy()
    d["y"] = y.reindex(df_full.index) if isinstance(y, pd.Series) else df_full[y]
    d["in_domain"] = (domain_mask.reindex(df_full.index, fill_value=False)
                       & d[weight_col].notna() & d["y"].notna())

    w = d["weight"] = d[weight_col]
    x_hat = w[d["in_domain"]].sum()
    n_domain = int(d["in_domain"].sum())
    if x_hat == 0 or n_domain == 0:
        return np.nan, np.nan, 0

    y_hat = (w[d["in_domain"]] * d.loc[d["in_domain"], "y"]).sum()
    r_hat = y_hat / x_hat

    # Linearized variable: u_i = w_i * (y_i - R_hat) for domain members, 0 otherwise.
    u = pd.Series(0.0, index=d.index)
    u.loc[d["in_domain"]] = w[d["in_domain"]] * (d.loc[d["in_domain"], "y"] - r_hat)

    tmp = pd.DataFrame({strata_col: df_full[strata_col], psu_col: df_full[psu_col], "u": u})
    psu_totals = tmp.groupby([strata_col, psu_col])["u"].sum()

    var_total = 0.0
    any_usable_stratum = False
    for _, psu_vals in psu_totals.groupby(level=0):
        n_h = len(psu_vals)
        if n_h < 2:
            # Singleton-PSU stratum (can happen after domain restriction if,
            # e.g., collapsed strata leave one PSU with zero eligible rows
            # some cycle) -- NHANES guidance is to collapse with an adjacent
            # stratum; lacking that mapping here, we skip this stratum's
            # contribution rather than fabricate a variance from n=1. This
            # is conservative (slightly understates SE) but avoids a
            # divide-by-zero / undefined n_h/(n_h-1) term.
            continue
        any_usable_stratum = True
        mean_psu = psu_vals.mean()
        var_total += (n_h / (n_h - 1)) * ((psu_vals - mean_psu) ** 2).sum()

    if not any_usable_stratum:
        return r_hat, np.nan, n_domain

    se = np.sqrt(var_total) / x_hat
    return r_hat, se, n_domain


def design_based_wald_test(m0, se0, n0, m1, se1, n1):
    """Wald test for a difference in two independent design-based means/
    proportions, using the ultimate-cluster SEs above (asymptotically
    normal, which is the standard reference distribution used for Taylor-
    linearized survey estimates -- see Wolter, Ch. 2; unlike the original
    script's Welch-Satterthwaite t-approximation, which was borrowed from
    the simple-random-sample two-sample t-test and does not have a natural
    "df" concept for a design-based variance estimate in the first place)."""
    if np.isnan(se0) or np.isnan(se1):
        return np.nan, np.nan
    se_diff = np.sqrt(se0 ** 2 + se1 ** 2)
    if se_diff == 0:
        return np.nan, np.nan
    z_stat = (m1 - m0) / se_diff
    p_val = 2 * (1 - norm.cdf(abs(z_stat)))
    return z_stat, p_val


def build_metals_weight_column(df: pd.DataFrame) -> pd.Series:
    """FIX: per-cycle correct metals survey weight (WTMEC2YR for four
    cycles, WTSH2YR for the two half-subsample cycles 2013-2014/2015-2016)
    -- see config.METALS_WEIGHT_BY_CYCLE and its cycle-by-cycle citations.
    Returns a single Series with, for each row, the value of whichever
    weight column is correct for that row's own cycle."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for cyc, weight_col in cfg.METALS_WEIGHT_BY_CYCLE.items():
        mask = df[cfg.CYCLE_VAR] == cyc
        if weight_col in df.columns:
            out.loc[mask] = df.loc[mask, weight_col]
    return out


def weight_col_for_var(var: str) -> str:
    if var in FASTING_DERIVED:
        return cfg.SURVEY_WEIGHT_FASTING
    if var in METALS_VARS:
        return "metals_weight"  # constructed column, see build_metals_weight_column
    return cfg.SURVEY_WEIGHT_MEC


def build_table1():
    df = pd.read_csv(cfg.DATA_PROC_DIR / "nhanes_with_outcome.csv", low_memory=False)
    print(f"\nLoaded {len(df)} rows for Table 1 (full analytic cohort, not split by train/test -- "
          f"Table 1 describes the whole analytic sample, this is standard practice and not a leakage concern "
          f"since it's descriptive, not used to fit or select models)")

    df["metals_weight"] = build_metals_weight_column(df)
    n_se_mn_eligible = (~df[cfg.CYCLE_VAR].isin(cfg.SE_MN_NOT_MEASURED_CYCLES)).sum()
    print(f"Selenium/manganese not measured at all in cycles {cfg.SE_MN_NOT_MEASURED_CYCLES} "
          f"(structural absence, not missingness) -- their Table-1 rows are domain-restricted "
          f"to the {n_se_mn_eligible} rows from cycles where the assay exists, same as every "
          f"other row's own notna() eligibility.")

    rows = []
    domain0 = df[cfg.OUTCOME_VAR] == 0
    domain1 = df[cfg.OUTCOME_VAR] == 1

    for var in TABLE1_CONTINUOUS:
        if var not in df.columns:
            continue
        weight_col = weight_col_for_var(var)
        row = {"variable": var, "clinical_name": cfg.label_feature(var), "type": "continuous",
               "weight_used": weight_col}

        m0, se0, n0 = ultimate_cluster_domain_mean(df, df[var], weight_col, domain0)
        m1, se1, n1 = ultimate_cluster_domain_mean(df, df[var], weight_col, domain1)
        row.update({
            "no_mets_mean": round(m0, 3) if not np.isnan(m0) else np.nan,
            "no_mets_se": round(se0, 4) if not np.isnan(se0) else np.nan, "no_mets_n": n0,
            "mets_mean": round(m1, 3) if not np.isnan(m1) else np.nan,
            "mets_se": round(se1, 4) if not np.isnan(se1) else np.nan, "mets_n": n1,
        })
        if not (np.isnan(m0) or np.isnan(m1)):
            _, p_val = design_based_wald_test(m0, se0, n0, m1, se1, n1)
            row["p_value"] = round(p_val, 4) if not np.isnan(p_val) else np.nan
        rows.append(row)

    for var in TABLE1_CATEGORICAL:
        if var not in df.columns:
            continue
        weight_col = cfg.SURVEY_WEIGHT_MEC
        sub = df[[var, cfg.OUTCOME_VAR, weight_col]].dropna()
        weighted_ct = sub.groupby([var, cfg.OUTCOME_VAR])[weight_col].sum().unstack(fill_value=0)
        weighted_pct = weighted_ct / weighted_ct.sum(axis=0) * 100

        # Design-based SE per (level, domain) cell: the standard error of the
        # weighted PROPORTION with this level, WITHIN each mets_status
        # domain -- i.e. y is a 0/1 "is this row == level" indicator and the
        # domain is (mets_status group) AND (var observed), so
        # ultimate_cluster_domain_mean's r_hat reproduces weighted_pct/100
        # exactly and its se is the correct design-based SE of that
        # proportion (not the degenerate "mean of a within-domain constant"
        # that a domain of level-matching rows alone would give).
        var_notna = df[var].notna()
        for level in weighted_ct.index:
            level_indicator = (df[var] == level).astype(float)
            m0, se0, n0 = ultimate_cluster_domain_mean(df, level_indicator, weight_col, domain0 & var_notna)
            m1, se1, n1 = ultimate_cluster_domain_mean(df, level_indicator, weight_col, domain1 & var_notna)
            rows.append({
                "variable": f"{var} = {level}", "clinical_name": table1_level_label(var, level), "type": "categorical",
                "weight_used": weight_col,
                "no_mets_mean": round(weighted_pct.loc[level, 0], 2) if 0 in weighted_pct.columns else np.nan,
                "no_mets_se": round(se0 * 100, 4) if not np.isnan(se0) else np.nan,
                "mets_mean": round(weighted_pct.loc[level, 1], 2) if 1 in weighted_pct.columns else np.nan,
                "mets_se": round(se1 * 100, 4) if not np.isnan(se1) else np.nan,
            })

        # Same-scale (see caveat) rescale, so its total matches the actual sample
        # size before testing, or chi2_contingency sees an inflated "N" in
        # the hundreds of millions and returns p~0 regardless of the real signal.
        # NOTE (caveat, unchanged from original): this p-value is an
        # UNWEIGHTED-N chi-square on weight-rescaled cell counts -- it is
        # NOT a Rao-Scott design-corrected test. Treat as descriptive.
        scaled_ct = weighted_ct / weighted_ct.values.sum() * len(sub)
        try:
            _, p_val, _, _ = chi2_contingency(scaled_ct.values)
        except ValueError:
            p_val = np.nan
        for r in rows[-len(weighted_ct.index):]:
            r["p_value"] = round(p_val, 4) if not np.isnan(p_val) else np.nan
            r["p_value_note"] = "descriptive chi-square, NOT design/Rao-Scott corrected"

    table1 = pd.DataFrame(rows)
    out_path = cfg.OUT_METRICS / "table1_by_mets_status.csv"
    table1.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(table1.to_string(index=False))
    return table1


def main():
    build_strobe_flow()
    build_table1()
    print("\n" + "=" * 70)
    print("Continuous-row SEs/p-values above are design-based (Taylor-linearization,")
    print("ultimate-cluster method) accounting for SDMVPSU/SDMVSTRA and, for the 5")
    print("blood metals, the correct per-cycle weight (WTMEC2YR or WTSH2YR -- see")
    print("config.METALS_WEIGHT_BY_CYCLE). Categorical omnibus p-values remain an")
    print("UNWEIGHTED chi-square on weight-rescaled counts (descriptive, not")
    print("Rao-Scott design-corrected) -- redo those specific tests in R's survey")
    print("package (svychisq) or Python samplics before final submission if a fully")
    print("design-corrected categorical test is required by the target journal.")
    print("=" * 70)


if __name__ == "__main__":
    main()
