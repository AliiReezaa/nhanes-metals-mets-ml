"""
Addition 2: metal-metal interactions, including the toxic-essential
antagonism hypothesis (selenium offsetting cadmium or lead).

2026-10 audit (A5). The earlier version reported Spearman p-values computed
from correlations between SHAP interaction values and metal levels, on all
test rows. Two problems: (1) SHAP values are deterministic outputs of one
fitted model, so those p-values ignore model uncertainty and are not valid
population inference; (2) selenium and manganese were not measured in
2007-2010, so for about 40% of rows their values were a single imputed
median, which manufactured strong-looking correlations. This version:
  * keeps the SHAP interaction summary as a DESCRIPTION of the fitted model,
    computed only on test rows where both metals were actually measured, and
    with no p-values;
  * tests each metal-metal interaction formally with an explicit logistic
    regression on the whole analytic cohort (observed values only, metals on
    the log scale, full adjustment set), likelihood-ratio test for the
    product term, Benjamini-Hochberg false-discovery-rate control across the
    10 pairs.
The heatmap and the table use the same scale (the full pairwise interaction,
i.e. both symmetric halves of the SHAP interaction matrix).

Run (after the main pipeline): python 07_Novelty_Additions/code/02_shap_interactions.py
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_Code"))
import additions_common as ac  # noqa: E402

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pubstyle as ps

ps.apply_style()
TABLES, FIGURES = ac.out_dirs("07_Novelty_Additions")
METALS = ac.METAL_NAMES


def interaction_lrt(df, m1, m2):
    """Likelihood-ratio test of log(m1) x log(m2) in an adjusted logistic model."""
    sub = df[[m1, m2, "mets_status"] + ac.ADJUST_COLUMNS].dropna().copy()
    sub = sub[(sub[m1] > 0) & (sub[m2] > 0)]
    sub["a"], sub["b"] = np.log(sub[m1]), np.log(sub[m2])
    f0 = f"mets_status ~ a + b + {ac.ADJUST_FORMULA}"
    y, X0 = patsy.dmatrices(f0, sub, return_type="dataframe")
    _, X1 = patsy.dmatrices(f0 + " + a:b", sub, return_type="dataframe")
    m0 = sm.GLM(y, X0, family=sm.families.Binomial()).fit()
    mfull = sm.GLM(y, X1, family=sm.families.Binomial()).fit()
    lr = 2 * (mfull.llf - m0.llf)
    return len(sub), float(mfull.params["a:b"]), float(stats.chi2.sf(lr, 1))


def main():
    st = ac.primary_state()
    Xt = st["Xt_test_df"]
    raw = st["test_df"]
    iv = st["explainer"].shap_interaction_values(Xt)
    iv = np.asarray(iv[1] if isinstance(iv, list) else iv)
    names = list(Xt.columns)
    idx = {m: names.index(m) for m in METALS}
    cohort = ac.analysis_frame()

    rows = []
    metal_cols = list(METALS)
    sub = np.zeros((5, 5))
    for i, j in combinations(range(5), 2):
        m1, m2 = metal_cols[i], metal_cols[j]
        observed = raw[m1].notna().values & raw[m2].notna().values
        pair = iv[observed, idx[m1], idx[m2]] * 2  # full pairwise interaction
        sub[i, j] = sub[j, i] = np.mean(np.abs(pair))
        n_lrt, beta, p = interaction_lrt(cohort, m1, m2)
        rows.append({"Metal 1": METALS[m1], "Metal 2": METALS[m2],
                     "Test rows with both measured": int(observed.sum()),
                     "Mean |SHAP interaction| (descriptive)": round(float(sub[i, j]), 5),
                     "Regression n": n_lrt, "Interaction beta (log x log)": round(beta, 4),
                     "LRT p": p})
    table = pd.DataFrame(rows)
    table["FDR-adjusted p"] = ac.bh_fdr(table["LRT p"])
    table["LRT p"] = table["LRT p"].map(ac.fmt_p)
    table["FDR-adjusted p"] = table["FDR-adjusted p"].map(ac.fmt_p)
    table = table.sort_values("Mean |SHAP interaction| (descriptive)", ascending=False)
    table.to_csv(TABLES / "table_metal_interaction_ranking.csv", index=False)
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(ps.W_SINGLE * 1.5, ps.W_SINGLE * 1.4))
    labels = [METALS[m] for m in metal_cols]
    disp = sub.copy()
    np.fill_diagonal(disp, np.nan)
    im = ax.imshow(disp, cmap="cividis")
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    for i in range(5):
        for j in range(5):
            if i != j:
                rgba = im.cmap(im.norm(disp[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, f"{disp[i, j]:.4f}", ha="center", va="center", fontsize=7.5,
                        color="white" if lum < 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean |SHAP interaction| (rows with both metals measured)", fontsize=7.8)
    cbar.ax.tick_params(labelsize=7.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    ps.save(fig, FIGURES / "fig_metal_interaction_heatmap")

    # selenium x cadmium and selenium x lead, observed selenium only
    se_obs = raw["LBXBSE"].notna().values
    fig2, axes = plt.subplots(1, 2, figsize=(ps.W_DOUBLE * 0.95, ps.W_DOUBLE * 0.42))
    summary = []
    for ax, other, label, letter in zip(axes, ["LBXBCD", "LBXBPB"], ["cadmium", "lead"], "AB"):
        inter = iv[se_obs, idx["LBXBSE"], idx[other]] * 2
        se_raw = raw.loc[se_obs, "LBXBSE"].values
        other_raw = raw.loc[se_obs, other].values
        rho = spearmanr(other_raw, inter)[0]
        sc = ax.scatter(se_raw, inter, c=np.log(other_raw), cmap=ps.shap_cmap(), s=10, alpha=0.75,
                        linewidths=0, rasterized=True)
        ax.axhline(0, color="#999999", lw=0.8, zorder=1)
        ax.set_xlabel("Blood selenium (µg/L)")
        ax.set_ylabel(f"SHAP interaction: selenium × {label}", labelpad=4)
        ax.text(0.04, 0.96, f"descriptive ρ = {rho:.3f}\n(n = {se_obs.sum():,} with selenium measured)",
                transform=ax.transAxes, fontsize=7.2, va="top", ha="left", color="#333333")
        cbar = fig2.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(f"log {label}", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ps.panel_label(ax, letter, dx=-0.24, dy=1.12)
        reg = table[(table["Metal 1"].isin([METALS["LBXBSE"], METALS[other]]))
                    & (table["Metal 2"].isin([METALS["LBXBSE"], METALS[other]]))].iloc[0]
        summary.append({"Pair": f"Selenium × {label}", "Test rows with selenium measured": int(se_obs.sum()),
                        "Mean |SHAP interaction| (descriptive)": round(float(np.mean(np.abs(inter))), 5),
                        "Spearman rho, SHAP interaction vs. metal (descriptive)": round(float(rho), 4),
                        "Regression interaction beta": reg["Interaction beta (log x log)"],
                        "Regression LRT p": reg["LRT p"], "FDR-adjusted p": reg["FDR-adjusted p"]})
    fig2.subplots_adjust(left=0.09, right=0.97, top=0.96, bottom=0.15, wspace=0.65)
    ps.save(fig2, FIGURES / "fig_selenium_interaction_dependence")
    pd.DataFrame(summary).to_csv(TABLES / "table_selenium_antagonism_test.csv", index=False)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
