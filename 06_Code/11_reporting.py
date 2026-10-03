"""
Step 14 -- Reproducibility and Reporting

Produces two things every write-up of this analysis needs:
  1. A consolidated STROBE-style exclusion flow (pulls together the per-cycle
     counts from 01_merge_data.py and the exclusion steps from
     02_clean_and_outcome.py into one table)
  2. A survey-weighted Table 1 (baseline characteristics by MetS status)

Design-based estimation (see the block comment above build_table1 for the
2026-10 changes): Taylor-linearized ultimate-cluster standard errors
(Wolter, "Introduction to Variance Estimation", 2nd ed., Ch. 2; NCHS NHANES
analytic guidelines), implemented in numpy/pandas so every step is
inspectable. Every row uses the fasting-subsample weight (WTSAF2YR) divided
by the number of pooled cycles, because the analytic cohort is the fasting
subsample. Caveat: in 2013-2016 the metals panel was measured in a separate
one-half subsample, and NCHS provides no weight for the overlap of the two
subsamples, so the metal rows for those cycles are approximate.

Output: outputs/metrics/strobe_flow.csv
        outputs/metrics/table1_by_mets_status.csv

Run:
    python 06_Code/11_reporting.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
# Design-based estimation (Taylor linearization, "ultimate cluster").
# 2026-10 audit (C6, C7, C20):
#   * every row uses the fasting-subsample weight WTSAF2YR divided by the
#     number of pooled cycles (the analytic cohort is, by construction, the
#     fasting subsample);
#   * metals are summarised as weighted GEOMETRIC means (they are strongly
#     right-skewed), with tests on the log scale;
#   * the MetS vs no-MetS difference is linearized directly, so the
#     covariance between the two domains (they share PSUs) is included, and
#     tests use t / F reference distributions with design degrees of freedom
#     = number of PSUs - number of strata;
#   * categorical rows use the adjusted Wald test (Korn & Graubard 1990)
#     instead of an unweighted chi-square on rescaled counts.
# Domain estimation: rows outside a domain contribute 0 to the linearized
# variable. Because every PSU of the design still contains cohort members,
# computing this on the cohort file gives the same variance as computing it
# on the full NHANES file with zero-weighted non-members (checked in
# check_all_psus_present below).
# ----------------------------------------------------------------------
def _psu_totals(df, u_cols):
    """Ultimate-cluster covariance of the column totals in u_cols."""
    g = df.groupby([cfg.SURVEY_STRATA, cfg.SURVEY_PSU])[u_cols].sum()
    k = len(u_cols)
    V = np.zeros((k, k))
    for _, block in g.groupby(level=0):
        n_h = len(block)
        if n_h < 2:
            continue  # no singleton strata occur in this cohort (checked below)
        dev = block.values - block.values.mean(axis=0)
        V += n_h / (n_h - 1) * dev.T @ dev
    return V


def design_df(df):
    n_psu = df.groupby([cfg.SURVEY_STRATA, cfg.SURVEY_PSU]).ngroups
    n_strata = df[cfg.SURVEY_STRATA].nunique()
    return n_psu - n_strata


def check_all_psus_present(df, full_path):
    if not full_path.exists():
        return
    full = pd.read_csv(full_path, usecols=[cfg.CYCLE_VAR, cfg.SURVEY_STRATA, cfg.SURVEY_PSU], low_memory=False)
    full_psus = full.groupby([cfg.SURVEY_STRATA, cfg.SURVEY_PSU]).ngroups
    cohort_psus = df.groupby([cfg.SURVEY_STRATA, cfg.SURVEY_PSU]).ngroups
    print(f"PSUs in the full NHANES file: {full_psus}; PSUs with cohort members: {cohort_psus}"
          + ("  (all present: cohort-file variance equals full-file domain variance)"
             if full_psus == cohort_psus else "  [NOTE] some PSUs have no cohort members"))


def domain_mean(df, y, w, mask):
    """Weighted domain mean, its linearized variable, and its SE."""
    m = mask & y.notna() & w.notna()
    x = w[m].sum()
    r = (w[m] * y[m]).sum() / x
    u = pd.Series(0.0, index=df.index)
    u[m] = w[m] * (y[m] - r) / x
    tmp = df[[cfg.SURVEY_STRATA, cfg.SURVEY_PSU]].copy()
    tmp["u"] = u
    se = float(np.sqrt(_psu_totals(tmp, ["u"])[0, 0]))
    return float(r), se, int(m.sum()), u


def domain_difference(df, y, w, d0, d1, ddf):
    """Mean(d1) - mean(d0) with a linearized SE that includes the covariance
    between the two domains; two-sided t test with ddf design df."""
    m0, se0, n0, u0 = domain_mean(df, y, w, d0)
    m1, se1, n1, u1 = domain_mean(df, y, w, d1)
    tmp = df[[cfg.SURVEY_STRATA, cfg.SURVEY_PSU]].copy()
    tmp["u"] = u1 - u0
    se_d = float(np.sqrt(_psu_totals(tmp, ["u"])[0, 0]))
    t = (m1 - m0) / se_d if se_d > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), ddf)) if se_d > 0 else np.nan
    return (m0, se0, n0), (m1, se1, n1), p


def adjusted_wald_categorical(df, var, w, d0, d1, ddf):
    """Design-based adjusted Wald F test that the level distribution of var
    is the same in d0 and d1 (Korn & Graubard 1990)."""
    ok = df[var].notna()
    levels = sorted(df.loc[ok, var].unique())[:-1]  # k-1 free proportions
    tmp = df[[cfg.SURVEY_STRATA, cfg.SURVEY_PSU]].copy()
    diffs, cols = [], []
    for lv in levels:
        ind = (df[var] == lv).astype(float).where(ok)
        m0, _, _, u0 = domain_mean(df, ind, w, d0 & ok)
        m1, _, _, u1 = domain_mean(df, ind, w, d1 & ok)
        c = f"u_{lv}"
        tmp[c] = u1 - u0
        cols.append(c)
        diffs.append(m1 - m0)
    d = np.array(diffs)
    V = _psu_totals(tmp, cols)
    k = len(d)
    W = float(d @ np.linalg.pinv(V) @ d)
    F = (ddf - k + 1) / (ddf * k) * W
    return float(stats.f.sf(F, k, ddf - k + 1))


def build_table1():
    df = pd.read_csv(cfg.DATA_PROC_DIR / "nhanes_with_outcome.csv", low_memory=False)
    print(f"\nLoaded {len(df)} rows for Table 1 (whole analytic cohort; descriptive only).")
    check_all_psus_present(df, cfg.DATA_PROC_DIR / "nhanes_merged_raw.csv")

    n_cycles = df[cfg.CYCLE_VAR].nunique()
    w = df[cfg.SURVEY_WEIGHT_FASTING] / n_cycles
    weight_note = f"{cfg.SURVEY_WEIGHT_FASTING}/{n_cycles}"
    ddf = design_df(df)
    print(f"Weight: {weight_note}; design degrees of freedom = {ddf}")

    rows = []
    d0 = df[cfg.OUTCOME_VAR] == 0
    d1 = df[cfg.OUTCOME_VAR] == 1

    for var in TABLE1_CONTINUOUS:
        if var not in df.columns:
            continue
        is_metal = var in METALS_VARS
        y = np.log(df[var].where(df[var] > 0)) if is_metal else df[var]
        # 2026-10 review: degrees of freedom from the strata and PSUs that hold
        # data on this variable (selenium and manganese exist only in 2011-2018,
        # so their df is smaller than the whole-design 94).
        vdf = design_df(df[y.notna()])
        (m0, se0, n0), (m1, se1, n1), p = domain_difference(df, y, w, d0, d1, vdf)
        row = {"variable": var, "clinical_name": cfg.label_feature(var), "type": "continuous",
               "summary": "geometric mean" if is_metal else "mean", "weight_used": weight_note}
        if is_metal:
            tq = stats.t.ppf(0.975, vdf)
            row.update({
                "no_mets_mean": round(float(np.exp(m0)), 3), "no_mets_se": round(float(np.exp(m0) * se0), 4),
                "no_mets_ci_low": round(float(np.exp(m0 - tq * se0)), 3),
                "no_mets_ci_high": round(float(np.exp(m0 + tq * se0)), 3), "no_mets_n": n0,
                "mets_mean": round(float(np.exp(m1)), 3), "mets_se": round(float(np.exp(m1) * se1), 4),
                "mets_ci_low": round(float(np.exp(m1 - tq * se1)), 3),
                "mets_ci_high": round(float(np.exp(m1 + tq * se1)), 3), "mets_n": n1,
            })
        else:
            row.update({"no_mets_mean": round(m0, 3), "no_mets_se": round(se0, 4), "no_mets_n": n0,
                        "mets_mean": round(m1, 3), "mets_se": round(se1, 4), "mets_n": n1})
        row["p_value"] = float(f"{p:.3g}")
        row["design_df"] = vdf
        row["p_value_note"] = ("design-based t test on the log scale" if is_metal
                               else "design-based t test") + f", df = {vdf}"
        rows.append(row)

    for var in TABLE1_CATEGORICAL:
        if var not in df.columns:
            continue
        ok = df[var].notna()
        p = adjusted_wald_categorical(df, var, w, d0, d1, ddf)
        for level in sorted(df.loc[ok, var].unique()):
            ind = (df[var] == level).astype(float).where(ok)
            (m0, se0, _), (m1, se1, _), _ = domain_difference(df, ind, w, d0 & ok, d1 & ok, ddf)
            rows.append({
                "variable": f"{var} = {level}", "clinical_name": table1_level_label(var, level),
                "type": "categorical", "summary": "weighted %", "weight_used": weight_note,
                "no_mets_mean": round(100 * m0, 2), "no_mets_se": round(100 * se0, 4),
                "mets_mean": round(100 * m1, 2), "mets_se": round(100 * se1, 4),
                "p_value": float(f"{p:.3g}"),
                "p_value_note": f"design-based adjusted Wald F test, df = {ddf}",
            })

    table1 = pd.DataFrame(rows)
    out_path = cfg.OUT_METRICS / "table1_by_mets_status.csv"
    table1.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(table1.to_string(index=False))
    return table1


def main():
    build_strobe_flow()
    build_table1()
    print("\nTable 1: fasting-subsample weights / number of cycles; design-based (Taylor")
    print("linearization, ultimate cluster) SEs and tests with design degrees of freedom.")


if __name__ == "__main__":
    main()
