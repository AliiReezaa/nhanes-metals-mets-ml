"""
Addition 2: test the toxic-essential metal antagonism hypothesis (selenium
protecting against cadmium/lead toxicity) using SHAP interaction values
from the fitted CatBoost model, computed on the full held-out test
partition (n=2,229) for consistency with Figures 3-5 of the manuscript.
"""
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubstyle as ps

ps.apply_style()

METALS = {
    "LBXBPB": "Blood lead",
    "LBXBCD": "Blood cadmium",
    "LBXTHG": "Blood mercury",
    "LBXBSE": "Blood selenium",
    "LBXBMN": "Blood manganese",
}


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def main():
    state = joblib.load("shared_state.joblib")
    cb_model = state["cb_model"]
    Xt_test_df = state["Xt_test_df"]
    test_df = state["test_df"].reset_index(drop=True)

    explainer = shap.TreeExplainer(cb_model)
    interaction_values = explainer.shap_interaction_values(Xt_test_df)
    print("Interaction tensor shape:", interaction_values.shape)

    feat_names = list(Xt_test_df.columns)
    idx = {name: feat_names.index(name) for name in METALS}

    # Mean |interaction| across all test rows, off-diagonal only, for every metal pair
    mean_abs_interaction = np.mean(np.abs(interaction_values), axis=0)  # (46, 46)
    metal_cols = list(METALS.keys())
    metal_idx = [idx[m] for m in metal_cols]
    submatrix = mean_abs_interaction[np.ix_(metal_idx, metal_idx)]
    pair_rows = []
    for i, m1 in enumerate(metal_cols):
        for j, m2 in enumerate(metal_cols):
            if j <= i:
                continue
            pair_rows.append({
                "Metal 1": METALS[m1], "Metal 2": METALS[m2],
                "Mean |SHAP interaction|": round(submatrix[i, j] * 2, 5),  # SHAP splits interaction across the pair symmetrically; report the combined (Lundberg convention: value at [i,j] and [j,i] are equal, each already half the joint contribution)
            })
    pair_table = pd.DataFrame(pair_rows).sort_values("Mean |SHAP interaction|", ascending=False)
    pair_table.to_csv("table_metal_interaction_ranking.csv", index=False)
    print(pair_table.to_string(index=False))

    # Heatmap of the 5x5 metal-metal interaction matrix (cividis: perceptually
    # uniform and colourblind-safe, consistent with the sequential-map choice
    # used elsewhere in this figure set)
    fig, ax = plt.subplots(figsize=(ps.W_SINGLE * 1.5, ps.W_SINGLE * 1.4))
    labels = [METALS[m] for m in metal_cols]
    disp = submatrix.copy()
    np.fill_diagonal(disp, np.nan)
    im = ax.imshow(disp, cmap="cividis")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                # Contrast text color from the ACTUAL rendered RGBA (perceived
                # luminance), not a guessed value threshold -- cividis is not
                # linear in lightness against the raw data value once vmin/vmax
                # auto-scale, so a value-based heuristic mis-colors several cells.
                rgba = im.cmap(im.norm(disp[i, j]))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "white" if luminance < 0.55 else "black"
                ax.text(j, i, f"{submatrix[i,j]:.4f}", ha="center", va="center",
                         fontsize=7.5, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean |SHAP interaction|", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    ps.save(fig, "fig_metal_interaction_heatmap")
    print("Saved fig_metal_interaction_heatmap.png/.pdf")

    # Dependence-style plots: selenium's SHAP contribution colored by cadmium, and by lead
    se_shap_main = interaction_values[:, idx["LBXBSE"], idx["LBXBSE"]]  # Se's own main effect
    se_val = Xt_test_df["LBXBSE"].values
    cd_val = Xt_test_df["LBXBCD"].values
    pb_val = Xt_test_df["LBXBPB"].values

    from scipy.stats import spearmanr

    fig2, axes = plt.subplots(1, 2, figsize=(ps.W_DOUBLE * 0.95, ps.W_DOUBLE * 0.42))
    panel_letters = ["A", "B"]
    summary_rows = []
    for ax, other_val, other_name, letter in zip(axes, [cd_val, pb_val], ["cadmium", "lead"], panel_letters):
        # interaction component specifically: Se x other metal
        other_key = "LBXBCD" if other_name == "cadmium" else "LBXBPB"
        se_other_interaction = interaction_values[:, idx["LBXBSE"], idx[other_key]] * 2
        rho, p = spearmanr(other_val, se_other_interaction)
        sc = ax.scatter(se_val, se_other_interaction, c=other_val, cmap=ps.shap_cmap(),
                          s=10, alpha=0.75, linewidths=0, rasterized=True)
        ax.axhline(0, color="#999999", lw=0.8, zorder=1)
        ax.set_xlabel("Selenium (standardized, imputed)")
        ax.set_ylabel(f"SHAP interaction: selenium × {other_name}", labelpad=4)
        ax.text(0.04, 0.96, f"ρ = {rho:.3f}, P = {_fmt_p(p)}", transform=ax.transAxes,
                 fontsize=7.5, va="top", ha="left", color="#333333")
        cbar = fig2.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(f"{other_name.capitalize()} (standardized)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ps.panel_label(ax, letter, dx=-0.24, dy=1.12)
        summary_rows.append({
            "Pair": f"Selenium × {other_name}",
            "Mean |SHAP interaction|": round(float(np.mean(np.abs(se_other_interaction))), 5),
            "Spearman rho": round(rho, 4),
            "P value": _fmt_p(p),
        })
    fig2.subplots_adjust(left=0.09, right=0.97, top=0.96, bottom=0.15, wspace=0.65)
    ps.save(fig2, "fig_selenium_interaction_dependence")
    print("Saved fig_selenium_interaction_dependence.png/.pdf")

    pd.DataFrame(summary_rows).to_csv("table_selenium_antagonism_test.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
