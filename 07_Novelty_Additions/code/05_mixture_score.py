"""
Extra novelty mining #2: individual metals showed no significant
non-linear dose-response (Addition 1). Does the JOINT 5-metal mixture
behave differently? Build a simple, standard mixture index (weighted sum
via a logistic-regression fit on the 5 metals alone, the closest
lightweight analogue to a weighted-quantile-sum index) and test its
dose-response against MetS with the same spline-vs-linear LRT design used
in Addition 1, on the full analytic cohort, adjusted for age/BMI/sex/PIR.
"""
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubstyle as ps

ps.apply_style()


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


METALS = ["LBXBPB", "LBXBCD", "LBXTHG", "LBXBSE", "LBXBMN"]
ADJUST = "RIDAGEYR + BMXBMI + sex_female + INDFMPIR"


def main():
    df = pd.read_csv("SUBMISSION/05_Data_Processed/model_ready.csv", low_memory=False)
    # Metals-only complete cases (selenium/manganese restrict to their assayed cycles)
    sub = df[METALS + ["mets_status"]].dropna().copy()
    print(f"Metals-complete-case n = {len(sub)}")

    Xm = StandardScaler().fit_transform(sub[METALS])
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xm, sub["mets_status"])
    weights = dict(zip(METALS, lr.coef_[0]))
    print("Metals-only logistic-regression weights (standardized):")
    for m, w in weights.items():
        print(f"  {m}: {w:+.4f}")

    # Weighted mixture index = weighted sum of standardized metals
    sub["mixture_score"] = Xm @ lr.coef_[0]

    # Merge back covariates for the full adjusted model
    cov = df.loc[sub.index, ["RIDAGEYR", "BMXBMI", "sex_female", "INDFMPIR", "mets_status"]]
    sub2 = pd.concat([sub[["mixture_score"]], cov], axis=1)

    y, X_lin = patsy.dmatrices(f"mets_status ~ mixture_score + {ADJUST}", sub2, return_type="dataframe")
    lin_model = sm.GLM(y, X_lin, family=sm.families.Binomial()).fit()

    y2, X_spl = patsy.dmatrices(
        f'mets_status ~ cr(mixture_score, df=4, constraints="center") + {ADJUST}', sub2, return_type="dataframe"
    )
    spl_model = sm.GLM(y2, X_spl, family=sm.families.Binomial()).fit()

    lr_stat = 2 * (spl_model.llf - lin_model.llf)
    df_diff = spl_model.df_model - lin_model.df_model
    p_value = sps.chi2.sf(lr_stat, df_diff)
    print(f"\nJoint 5-metal mixture score: spline-vs-linear LRT chi2={lr_stat:.3f}, "
          f"df={df_diff}, p={p_value:.5f}")

    # Also test: is the LINEAR mixture-score term itself significant (joint metals effect)?
    lin_pval_mixture = lin_model.pvalues.get("mixture_score", np.nan)
    print(f"Linear mixture-score coefficient: {lin_model.params['mixture_score']:.4f} "
          f"(p={lin_pval_mixture:.2e})")

    pd.DataFrame([{
        "N": len(sub2),
        "Linear mixture coefficient": round(lin_model.params["mixture_score"], 4),
        "Linear mixture P value": _fmt_p(lin_pval_mixture),
        "Spline vs. linear LRT chi-square": round(lr_stat, 3),
        "df": int(df_diff),
        "Spline vs. linear P value": _fmt_p(p_value),
        "Weight: lead": round(weights["LBXBPB"], 4),
        "Weight: cadmium": round(weights["LBXBCD"], 4),
        "Weight: mercury": round(weights["LBXTHG"], 4),
        "Weight: selenium": round(weights["LBXBSE"], 4),
        "Weight: manganese": round(weights["LBXBMN"], 4),
    }]).to_csv("table_mixture_score_test.csv", index=False)

    # Plot predicted probability across the mixture-score range
    grid = np.linspace(sub2["mixture_score"].quantile(0.01), sub2["mixture_score"].quantile(0.99), 60)
    means = sub2[["RIDAGEYR", "BMXBMI", "sex_female", "INDFMPIR"]].mean()
    new_df = pd.DataFrame({
        "mixture_score": grid, "RIDAGEYR": means["RIDAGEYR"], "BMXBMI": means["BMXBMI"],
        "sex_female": means["sex_female"], "INDFMPIR": means["INDFMPIR"],
    })
    X_new = patsy.build_design_matrices([X_spl.design_info], new_df)[0]
    pred = spl_model.get_prediction(X_new).summary_frame(alpha=0.05)

    fig, ax = plt.subplots(figsize=(ps.W_SINGLE * 1.55, ps.W_SINGLE * 1.2))
    ax.plot(grid, pred["mean"], color=ps.OKABE_GREEN, lw=1.7, zorder=3)
    ax.fill_between(grid, pred["mean_ci_lower"], pred["mean_ci_upper"],
                      color=ps.OKABE_GREEN, alpha=0.18, lw=0, zorder=2)
    ax.axhline(sub2["mets_status"].mean(), color="#999999", ls="--", lw=0.8, zorder=1)
    ax.text(0.04, 0.96,
             f"n = {len(sub2):,}\nlinear term: P = {_fmt_p(lin_pval_mixture)}\n"
             f"spline vs. linear: P = {_fmt_p(p_value)}",
             transform=ax.transAxes, fontsize=7.5, va="top", ha="left", color="#333333")
    ax.set_xlabel("Joint 5-metal mixture score (weighted sum, standardized)")
    ax.set_ylabel("Predicted P(MetS)")
    fig.tight_layout()
    ps.save(fig, "fig_mixture_score_dose_response")
    print("Saved fig_mixture_score_dose_response.png/.pdf")


if __name__ == "__main__":
    main()
