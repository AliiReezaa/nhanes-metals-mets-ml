"""
Addition 3: component-level models. Instead of only predicting the binary
MetS label, the same predictor set is fitted against each of the five MetS
components separately (waist circumference, triglycerides, HDL, blood
pressure, glucose -- each harmonized, i.e. measured value or treatment,
consistent with the primary outcome), to see whether each metal's
contribution concentrates in one physiological pathway or is spread across
all five.

Model: the primary pipeline's own model type and tuned hyperparameters
(best_params.json), fitted with the same leakage-safe preprocessing. The
components are close to balanced, so SMOTENC is not used here.

Caveat built into the output: BMI is a predictor and waist circumference
defines one component, so the waist model's AUROC mostly reflects that
overlap, not metals.

Run (after the main pipeline): python 07_Novelty_Additions/code/03_component_level.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_Code"))
import additions_common as ac  # noqa: E402

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline_utils as pu
import pubstyle as ps

ps.apply_style()
TABLES, FIGURES = ac.out_dirs("07_Novelty_Additions")

COMPONENTS = {
    "mets_waist": "Waist circumference",
    "mets_trigly": "Triglycerides",
    "mets_hdl": "HDL cholesterol",
    "mets_bp": "Blood pressure",
    "mets_glu": "Fasting glucose",
}


def main():
    st = ac.primary_state()
    name = st["model_name"]
    registry = pu.get_model_registry()
    params = ac.best_params(name)
    feats, cont, cat = st["feature_cols"], st["cont_cols"], st["cat_cols"]
    train_df, test_df = st["train_df"], st["test_df"]
    metal_idx = {m: st["feat_names"].index(m) for m in ac.METALS}

    auc_rows = []
    heat = np.zeros((len(ac.METALS), len(COMPONENTS)))
    for j, (comp_col, comp_label) in enumerate(COMPONENTS.items()):
        pipe = pu.build_pipeline(registry[name]["estimator"](), cont, cat, use_smote=False)
        if params:
            pipe.set_params(**params)
        pipe.fit(train_df[feats], train_df[comp_col])
        proba = pipe.predict_proba(test_df[feats])[:, 1]
        auc = roc_auc_score(test_df[comp_col], proba)
        auc_rows.append({"MetS component": comp_label, "Test AUROC": round(auc, 4),
                         "Test prevalence (%)": round(test_df[comp_col].mean() * 100, 1)})
        print(f"{comp_label}: test AUROC = {auc:.4f}")

        Xt = pd.DataFrame(np.asarray(pipe.named_steps["preprocess"].transform(test_df[feats])),
                          columns=st["feat_names"])
        sv = shap.TreeExplainer(pipe.named_steps["model"]).shap_values(Xt)
        sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        mean_abs = np.abs(sv).mean(axis=0)
        for i, m in enumerate(ac.METALS):
            heat[i, j] = mean_abs[metal_idx[m]]

    pd.DataFrame(auc_rows).to_csv(TABLES / "table_component_level_auc.csv", index=False)
    heat_df = pd.DataFrame(heat, index=[ac.METAL_NAMES[m] for m in ac.METALS],
                           columns=list(COMPONENTS.values()))
    heat_df.index.name = "Metal"
    heat_df.round(5).to_csv(TABLES / "table_metal_by_component_shap.csv")
    print(heat_df.round(4).to_string())

    fig, ax = plt.subplots(figsize=(ps.W_DOUBLE * 0.62, ps.W_DOUBLE * 0.5))
    im = ax.imshow(heat, cmap="cividis", aspect="auto")
    ax.set_xticks(range(len(COMPONENTS)))
    ax.set_xticklabels(list(COMPONENTS.values()), rotation=30, ha="right")
    ax.set_yticks(range(len(ac.METALS)))
    ax.set_yticklabels([ac.METAL_NAMES[m] for m in ac.METALS])
    for i in range(len(ac.METALS)):
        for j in range(len(COMPONENTS)):
            rgba = im.cmap(im.norm(heat[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, f"{heat[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                    color="white" if lum < 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Mean |SHAP value| ({name}, component model)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    ps.save(fig, FIGURES / "fig_metal_by_component_heatmap")


if __name__ == "__main__":
    main()
