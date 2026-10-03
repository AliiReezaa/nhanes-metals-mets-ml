"""
Extra novelty mining #1: does each metal's contribution to predicted MetS
risk depend on age, sex, or BMI (effect modification), using SHAP
interaction values from the already-fitted CatBoost model. Reuses
shared_state.joblib -- no refitting needed.
"""
import numpy as np
import pandas as pd
import joblib
import shap
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubstyle as ps

ps.apply_style()


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


METALS = {
    "LBXBPB": "Blood lead", "LBXBCD": "Blood cadmium", "LBXTHG": "Blood mercury",
    "LBXBSE": "Blood selenium", "LBXBMN": "Blood manganese",
}
MODIFIERS = {"RIDAGEYR": "Age", "sex_female": "Sex (female)", "BMXBMI": "BMI"}


def main():
    state = joblib.load("shared_state.joblib")
    cb_model = state["cb_model"]
    Xt_test_df = state["Xt_test_df"]
    feat_names = list(Xt_test_df.columns)

    explainer = shap.TreeExplainer(cb_model)
    iv = explainer.shap_interaction_values(Xt_test_df)
    print("Interaction tensor:", iv.shape)

    idx = {**{m: feat_names.index(m) for m in METALS}, **{m: feat_names.index(m) for m in MODIFIERS}}

    rows = []
    for metal_key, metal_label in METALS.items():
        for mod_key, mod_label in MODIFIERS.items():
            inter = iv[:, idx[metal_key], idx[mod_key]] * 2
            mod_val = Xt_test_df[mod_key].values
            rho, p = spearmanr(mod_val, inter)
            rows.append({
                "metal": metal_label, "modifier": mod_label,
                "mean_abs_interaction": float(np.mean(np.abs(inter))),
                "spearman_rho": round(rho, 4), "spearman_p": p,
            })
    table = pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False)
    n_tests = len(table)
    table["bonferroni_alpha"] = 0.05 / n_tests
    table["significant_after_bonferroni"] = table["spearman_p"] < table["bonferroni_alpha"]
    out_table = table.rename(columns={
        "metal": "Metal", "modifier": "Modifier",
        "mean_abs_interaction": "Mean |SHAP interaction|",
        "spearman_rho": "Spearman rho",
    }).copy()
    out_table["P value"] = table["spearman_p"].apply(_fmt_p)
    out_table["Bonferroni alpha"] = table["bonferroni_alpha"].round(5)
    out_table["Significant after Bonferroni"] = table["significant_after_bonferroni"].map({True: "Yes", False: "No"})
    out_table = out_table[["Metal", "Modifier", "Mean |SHAP interaction|", "Spearman rho",
                            "P value", "Bonferroni alpha", "Significant after Bonferroni"]]
    out_table.to_csv("table_metal_demographic_interaction.csv", index=False)
    pd.set_option("display.width", 120)
    print(out_table.to_string(index=False))

    # Heatmap: metals x {age, sex, BMI}
    mat = table.pivot(index="metal", columns="modifier", values="mean_abs_interaction")
    mat = mat.loc[list(METALS.values()), list(MODIFIERS.values())]
    fig, ax = plt.subplots(figsize=(ps.W_SINGLE * 1.5, ps.W_SINGLE * 1.4))
    im = ax.imshow(mat.values, cmap="cividis")
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            rgba = im.cmap(im.norm(mat.values[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "white" if luminance < 0.55 else "black"
            ax.text(j, i, f"{mat.values[i,j]:.4f}", ha="center", va="center",
                     fontsize=7.5, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean |SHAP interaction|", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    ps.save(fig, "fig_metal_demographic_interaction")
    print("Saved fig_metal_demographic_interaction.png/.pdf")

    # Highlight the most STATISTICALLY significant hit (not just largest raw
    # magnitude -- manganese x age turns out to be far more significant than
    # lead x age despite a smaller mean |interaction|)
    top = table.sort_values("spearman_p").iloc[0]
    metal_key = [k for k, v in METALS.items() if v == top["metal"]][0]
    mod_key = [k for k, v in MODIFIERS.items() if v == top["modifier"]][0]
    inter = iv[:, idx[metal_key], idx[mod_key]] * 2
    metal_val = Xt_test_df[metal_key].values
    mod_val = Xt_test_df[mod_key].values
    fig2, ax2 = plt.subplots(figsize=(ps.W_SINGLE * 1.55, ps.W_SINGLE * 1.25))
    sc = ax2.scatter(metal_val, inter, c=mod_val, cmap=ps.shap_cmap(), s=11, alpha=0.75,
                       linewidths=0, rasterized=True)
    ax2.axhline(0, color="#999999", lw=0.8, zorder=1)
    ax2.set_xlabel(f"{top['metal']} (standardized)")
    ax2.set_ylabel(f"SHAP interaction: {top['metal']} × {top['modifier']}")
    cbar = fig2.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label(top["modifier"], fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    ax2.text(0.04, 0.96, f"ρ = {top['spearman_rho']:.3f}, P = {_fmt_p(top['spearman_p'])}",
              transform=ax2.transAxes, fontsize=7.5, va="top", ha="left", color="#333333")
    fig2.tight_layout()
    ps.save(fig2, "fig_top_effect_modification")
    print("Saved fig_top_effect_modification.png/.pdf")


if __name__ == "__main__":
    main()
