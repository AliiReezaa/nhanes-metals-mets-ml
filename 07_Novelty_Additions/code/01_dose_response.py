"""
Addition 1: confirm the non-monotonic dose-response shapes seen in the
CatBoost SHAP dependence plot (Figure 5) with an independent, non-ML
model: a logistic GAM (natural cubic spline per metal) adjusted for age,
BMI, sex and income-to-poverty ratio, fit on the full analytic cohort
(n=11,144; more power than the n=2,229 test-only SHAP plot). For each
metal:
  1. Likelihood-ratio test: spline term vs. linear term (is the curve
     really non-linear, or is that shape sampling noise in the SHAP plot?)
  2. Predicted-probability curve with 95% CI across the metal's range,
     holding covariates at their means -- the classic "confirmatory"
     companion to a SHAP dependence plot.
For lead specifically (the metal whose SHAP curve inverts direction, the
most consequential and most surprising pattern): a two-piece (hinge)
segmented logistic model, breakpoint chosen by profile-likelihood grid
search, with a bootstrap CI on the breakpoint location.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubstyle as ps

ps.apply_style()

RNG = np.random.default_rng(42)


def _fmt_p(p: float) -> str:
    """Journal-standard p-value formatting: 3 decimals down to 0.001, then
    scientific notation, never a bare 0.0."""
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"

CLINICAL = {
    "LBXBPB": ("Blood lead", "µg/dL"),
    "LBXBCD": ("Blood cadmium", "µg/L"),
    "LBXBSE": ("Blood selenium", "µg/L"),
}
ADJUST = "RIDAGEYR + BMXBMI + sex_female + INDFMPIR"


def load_full_cohort():
    df = pd.read_csv("SUBMISSION/05_Data_Processed/model_ready.csv", low_memory=False)
    return df


def fit_linear_and_spline(df, metal, outcome="mets_status", df_spline=4):
    cols_needed = [metal, "RIDAGEYR", "BMXBMI", "sex_female", "INDFMPIR", outcome]
    sub = df[cols_needed].dropna().copy()
    n = len(sub)

    # Linear model
    y, X_lin = patsy.dmatrices(f"{outcome} ~ {metal} + {ADJUST}", sub, return_type="dataframe")
    lin_model = sm.GLM(y, X_lin, family=sm.families.Binomial()).fit()

    # Spline model (natural cubic spline, df_spline degrees of freedom).
    # constraints="center" avoids rank-deficiency: an uncentered cr() basis
    # already spans the constant function, which is collinear with the
    # formula's own intercept term otherwise.
    formula_spline = f'{outcome} ~ cr({metal}, df={df_spline}, constraints="center") + {ADJUST}'
    y2, X_spl = patsy.dmatrices(formula_spline, sub, return_type="dataframe")
    spl_model = sm.GLM(y2, X_spl, family=sm.families.Binomial()).fit()

    # Likelihood-ratio test: spline vs linear (nested models, df difference)
    lr_stat = 2 * (spl_model.llf - lin_model.llf)
    df_diff = spl_model.df_model - lin_model.df_model
    from scipy import stats as sps
    p_value = sps.chi2.sf(lr_stat, df_diff)

    return {
        "metal": metal, "n": n,
        "lin_model": lin_model, "spl_model": spl_model,
        "lr_stat": lr_stat, "lr_df": df_diff, "lr_p": p_value,
        "sub": sub, "design_info": X_spl.design_info,
    }


def predicted_curve(res, metal, n_points=60):
    sub = res["sub"]
    spl_model = res["spl_model"]
    grid = np.linspace(sub[metal].quantile(0.01), sub[metal].quantile(0.99), n_points)
    means = sub[["RIDAGEYR", "BMXBMI", "sex_female", "INDFMPIR"]].mean()
    new_df = pd.DataFrame({
        metal: grid,
        "RIDAGEYR": means["RIDAGEYR"], "BMXBMI": means["BMXBMI"],
        "sex_female": means["sex_female"], "INDFMPIR": means["INDFMPIR"],
    })
    # Build design matrix using the same spline basis (patsy handles this via design_info)
    X_new = patsy.build_design_matrices([res["design_info"]], new_df)[0]
    pred = spl_model.get_prediction(X_new)
    summary = pred.summary_frame(alpha=0.05)
    return grid, summary["mean"].values, summary["mean_ci_lower"].values, summary["mean_ci_upper"].values


N_BOOT = 80


def segmented_breakpoint_search(df, metal="LBXBPB", outcome="mets_status",
                                  candidate_range=None, n_candidates=25):
    cols_needed = [metal, "RIDAGEYR", "BMXBMI", "sex_female", "INDFMPIR", outcome]
    sub = df[cols_needed].dropna().copy()
    if candidate_range is None:
        lo, hi = sub[metal].quantile(0.10), sub[metal].quantile(0.90)
    else:
        lo, hi = candidate_range
    candidates = np.linspace(lo, hi, n_candidates)

    def deviance_for_breakpoint(bp, data):
        d = data.copy()
        d["hinge"] = np.clip(d[metal] - bp, 0, None)
        y, X = patsy.dmatrices(
            f"{outcome} ~ {metal} + hinge + {ADJUST}", d, return_type="dataframe"
        )
        m = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        return m.deviance, m, X.design_info

    best_bp, best_dev, best_model, best_design_info = None, np.inf, None, None
    for bp in candidates:
        dev, m, di = deviance_for_breakpoint(bp, sub)
        if dev < best_dev:
            best_dev, best_bp, best_model, best_design_info = dev, bp, m, di

    # Bootstrap CI on the breakpoint (resample rows, re-run the same grid search)
    boot_bps = []
    for b in range(N_BOOT):
        idx = RNG.integers(0, len(sub), len(sub))
        boot_sub = sub.iloc[idx]
        b_best_bp, b_best_dev = None, np.inf
        for bp in candidates:
            try:
                dev, _, _ = deviance_for_breakpoint(bp, boot_sub)
            except Exception:
                continue
            if dev < b_best_dev:
                b_best_dev, b_best_bp = dev, bp
        if b_best_bp is not None:
            boot_bps.append(b_best_bp)
    boot_bps = np.array(boot_bps)
    ci_low, ci_high = np.percentile(boot_bps, [2.5, 97.5])

    return {
        "breakpoint": best_bp, "ci_low": ci_low, "ci_high": ci_high,
        "n_boot": len(boot_bps), "model": best_model, "sub": sub,
        "design_info": best_design_info,
    }


def main():
    df = load_full_cohort()
    results = {}
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(ps.W_DOUBLE, ps.W_DOUBLE / 3.1))
    panel_letters = ["A", "B", "C"]
    for ax, metal, letter in zip(axes, ["LBXBPB", "LBXBCD", "LBXBSE"], panel_letters):
        res = fit_linear_and_spline(df, metal)
        results[metal] = res
        grid, mean, lo, hi = predicted_curve(res, metal)
        name, unit = CLINICAL[metal]
        ax.plot(grid, mean, color=ps.OKABE_BLUE, lw=1.7, zorder=3)
        ax.fill_between(grid, lo, hi, color=ps.OKABE_BLUE, alpha=0.18, lw=0, zorder=2)
        ax.axhline(df["mets_status"].mean(), color="#999999", ls="--", lw=0.8, zorder=1)
        ax.set_xlabel(f"{name} ({unit})")
        ax.set_ylabel("Predicted P(MetS)" if metal == "LBXBPB" else "")
        sig = "p < 0.001" if res["lr_p"] < 0.001 else f"p = {res['lr_p']:.3f}"
        ax.text(0.04, 0.96, f"n = {res['n']:,}\nspline vs. linear: {sig}",
                 transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                 color="#333333")
        ps.panel_label(ax, letter, dx=-0.16, dy=1.12)
        rows.append({
            "Metal": name, "N": res["n"],
            "LRT chi-square": round(res["lr_stat"], 2), "df": int(res["lr_df"]),
            "P value": _fmt_p(res["lr_p"]),
            "Nonlinear (P<0.05)": "Yes" if res["lr_p"] < 0.05 else "No",
        })
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.subplots_adjust(wspace=0.32)
    ps.save(fig, "fig_dose_response_confirmation")
    print("Saved fig_dose_response_confirmation.png/.pdf")

    lrt_table = pd.DataFrame(rows)
    lrt_table.to_csv("table_dose_response_lrt.csv", index=False)
    print(lrt_table.to_string(index=False))

    # Lead breakpoint search
    print("\nSearching for lead's inversion breakpoint (segmented logistic model)...")
    seg = segmented_breakpoint_search(df, metal="LBXBPB")
    print(f"Lead breakpoint: {seg['breakpoint']:.3f} ug/dL "
          f"(bootstrap 95% CI: {seg['ci_low']:.3f}-{seg['ci_high']:.3f}, "
          f"{seg['n_boot']}/{N_BOOT} bootstrap reps converged)")
    pd.DataFrame([{
        "Metal": "Blood lead",
        "Breakpoint (µg/dL)": round(seg["breakpoint"], 2),
        "95% CI low": round(seg["ci_low"], 2), "95% CI high": round(seg["ci_high"], 2),
        "Bootstrap reps": f"{seg['n_boot']}/{N_BOOT}",
        "Method": "segmented logistic GLM, profile-likelihood breakpoint search, "
                  "adjusted for age/BMI/sex/PIR",
    }]).to_csv("table_lead_breakpoint.csv", index=False)

    # Plot the segmented fit for lead specifically
    sub = seg["sub"]
    bp = seg["breakpoint"]
    grid = np.linspace(sub["LBXBPB"].quantile(0.01), sub["LBXBPB"].quantile(0.99), 100)
    means = sub[["RIDAGEYR", "BMXBMI", "sex_female", "INDFMPIR"]].mean()
    new_df = pd.DataFrame({
        "LBXBPB": grid, "hinge": np.clip(grid - bp, 0, None),
        "RIDAGEYR": means["RIDAGEYR"], "BMXBMI": means["BMXBMI"],
        "sex_female": means["sex_female"], "INDFMPIR": means["INDFMPIR"],
    })
    X_new = patsy.build_design_matrices([seg["design_info"]], new_df)[0]
    pred = seg["model"].get_prediction(X_new).summary_frame(alpha=0.05)
    fig2, ax2 = plt.subplots(figsize=(ps.W_SINGLE * 1.55, ps.W_SINGLE * 1.2))
    ax2.plot(grid, pred["mean"], color=ps.OKABE_VERMIL, lw=1.7, zorder=3)
    ax2.fill_between(grid, pred["mean_ci_lower"], pred["mean_ci_upper"],
                       color=ps.OKABE_VERMIL, alpha=0.18, lw=0, zorder=2)
    ax2.axvline(bp, color="#333333", ls=":", lw=1.2, zorder=1)
    ax2.text(bp, ax2.get_ylim()[1] * 0.97, f"  breakpoint = {bp:.2f} µg/dL",
              fontsize=8, va="top", color="#333333")
    ax2.text(0.97, 0.06, f"bootstrap 95% CI: {seg['ci_low']:.2f}–{seg['ci_high']:.2f} µg/dL\n"
                          "(spans most of the search range -- not localized)",
              transform=ax2.transAxes, fontsize=7.5, va="bottom", ha="right", color="#333333")
    ax2.set_xlabel("Blood lead (µg/dL)")
    ax2.set_ylabel("Predicted P(MetS)")
    fig2.tight_layout()
    ps.save(fig2, "fig_lead_breakpoint")
    print("Saved fig_lead_breakpoint.png/.pdf")


if __name__ == "__main__":
    main()
