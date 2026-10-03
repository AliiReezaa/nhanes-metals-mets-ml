"""
Addition 4: does each metal's association with MetS depend on age, sex or BMI
(effect modification)?

2026-10 audit (A5). The earlier version used Spearman correlations between
SHAP interaction values and the modifier as "tests", on all test rows. Those
p-values are not valid inference (SHAP values are deterministic outputs of
one fitted model), and for selenium and manganese about 40% of the rows held
a single imputed value, which created most of the headline "manganese x age"
signal. This version tests each of the 15 metal x modifier interactions with
an explicit logistic regression on the whole analytic cohort (observed
values only; metal on the log scale; full adjustment set), using a
likelihood-ratio test for the product term and Benjamini-Hochberg
false-discovery-rate control across the 15 tests. The SHAP interaction
magnitude is kept as a description of the fitted model, on rows where the
metal was measured, without p-values.

Run (after the main pipeline): python 07_Novelty_Additions/code/04_effect_modification.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_Code"))
import additions_common as ac  # noqa: E402

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pubstyle as ps

ps.apply_style()
TABLES, FIGURES = ac.out_dirs("07_Novelty_Additions")
METALS = ac.METAL_NAMES
MODIFIERS = {"RIDAGEYR": "Age", "sex_female": "Sex (female)", "BMXBMI": "BMI"}


def interaction_lrt(df, metal, mod):
    sub = df[[metal, "mets_status"] + ac.ADJUST_COLUMNS].dropna().copy()
    sub = sub[sub[metal] > 0]
    sub["lm"] = np.log(sub[metal])
    sub["mod"] = sub[mod]
    f0 = f"mets_status ~ lm + {ac.ADJUST_FORMULA}"
    y, X0 = patsy.dmatrices(f0, sub, return_type="dataframe")
    _, X1 = patsy.dmatrices(f0 + " + lm:mod", sub, return_type="dataframe")
    m0 = sm.GLM(y, X0, family=sm.families.Binomial()).fit()
    m1 = sm.GLM(y, X1, family=sm.families.Binomial()).fit()
    return len(sub), float(m1.params["lm:mod"]), float(stats.chi2.sf(2 * (m1.llf - m0.llf), 1))


def main():
    st = ac.primary_state()
    Xt, raw = st["Xt_test_df"], st["test_df"]
    iv = st["explainer"].shap_interaction_values(Xt)
    iv = np.asarray(iv[1] if isinstance(iv, list) else iv)
    names = list(Xt.columns)
    cohort = ac.analysis_frame()

    rows = []
    for m, mlab in METALS.items():
        obs = raw[m].notna().values
        for mod, modlab in MODIFIERS.items():
            inter = iv[obs, names.index(m), names.index(mod)] * 2
            n, beta, p = interaction_lrt(cohort, m, mod)
            rows.append({"Metal": mlab, "Modifier": modlab,
                         "Mean |SHAP interaction| (descriptive)": round(float(np.mean(np.abs(inter))), 5),
                         "Regression n": n, "Interaction beta (log metal x modifier)": round(beta, 4),
                         "p": p})
    t = pd.DataFrame(rows)
    t["FDR-adjusted p"] = ac.bh_fdr(t["p"])
    t["Significant after FDR (q < 0.05)"] = np.where(t["FDR-adjusted p"] < 0.05, "Yes", "No")
    best = t.sort_values("p").iloc[0]
    out = t.copy()
    out["LRT p"] = out["p"].map(ac.fmt_p)
    out["FDR-adjusted p"] = out["FDR-adjusted p"].map(ac.fmt_p)
    out = out.drop(columns="p").sort_values(["Metal", "Modifier"])
    out.to_csv(TABLES / "table_metal_demographic_interaction.csv", index=False)
    print(out.to_string(index=False))

    # heatmap: -log10(FDR p) with the interaction beta written in each cell
    t["neglog"] = -np.log10(t["FDR-adjusted p"].clip(lower=1e-12))
    mat = t.pivot(index="Metal", columns="Modifier", values="neglog").loc[list(METALS.values()),
                                                                          list(MODIFIERS.values())]
    betas = t.pivot(index="Metal", columns="Modifier",
                    values="Interaction beta (log metal x modifier)").loc[mat.index, mat.columns]
    fig, ax = plt.subplots(figsize=(ps.W_SINGLE * 1.55, ps.W_SINGLE * 1.4))
    im = ax.imshow(mat.values, cmap="cividis", vmin=0)
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            rgba = im.cmap(im.norm(mat.values[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, f"β {betas.values[i, j]:+.3f}", ha="center", va="center", fontsize=7.2,
                    color="white" if lum < 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("−log10 FDR-adjusted p (1.3 = q of 0.05)", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    ps.save(fig, FIGURES / "fig_metal_demographic_interaction")

    # descriptive SHAP view of the pair with the smallest regression p-value
    m_key = [k for k, v in METALS.items() if v == best["Metal"]][0]
    mod_key = [k for k, v in MODIFIERS.items() if v == best["Modifier"]][0]
    obs = raw[m_key].notna().values
    inter = iv[obs, names.index(m_key), names.index(mod_key)] * 2
    fig2, ax2 = plt.subplots(figsize=(ps.W_SINGLE * 1.55, ps.W_SINGLE * 1.25))
    sc = ax2.scatter(raw.loc[obs, m_key], inter, c=raw.loc[obs, mod_key], cmap=ps.shap_cmap(), s=11,
                     alpha=0.75, linewidths=0, rasterized=True)
    ax2.set_xscale("log")
    ax2.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax2.set_xlabel(f"{best['Metal']} (log scale)")
    ax2.set_ylabel(f"SHAP interaction: {best['Metal']} × {best['Modifier']}")
    cbar = fig2.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label(best["Modifier"], fontsize=8.5)
    ax2.text(0.04, 0.96, f"regression interaction: LRT p = {ac.fmt_p(best['p'])}, "
                         f"FDR q = {ac.fmt_p(best['FDR-adjusted p'])}",
             transform=ax2.transAxes, fontsize=7.2, va="top", ha="left", color="#333333")
    fig2.tight_layout()
    ps.save(fig2, FIGURES / "fig_top_effect_modification")


if __name__ == "__main__":
    main()
