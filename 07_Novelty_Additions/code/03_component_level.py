"""
Addition 3: instead of only predicting the collapsed binary MetS label,
fit the same predictor set against each of the five individual MetS
components separately (waist circumference, triglycerides, HDL, blood
pressure, glucose -- all harmonized, i.e. lab-or-treatment, consistent
with the primary outcome) to see whether each metal's association is
concentrated in a specific physiological pathway or diffuse across all
five. Components are near-balanced (39-57% prevalence) so SMOTE is
skipped, consistent with the manuscript's own finding that the imbalance
strategy is immaterial at this prevalence.
"""
import numpy as np
import pandas as pd
import joblib
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubstyle as ps

ps.apply_style()

COMPONENTS = {
    "mets_waist": "Waist circumference",
    "mets_trigly": "Triglycerides",
    "mets_hdl": "HDL cholesterol",
    "mets_bp": "Blood pressure",
    "mets_glu": "Fasting glucose",
}
METALS = {
    "LBXBPB": "Blood lead",
    "LBXBCD": "Blood cadmium",
    "LBXTHG": "Blood mercury",
    "LBXBSE": "Blood selenium",
    "LBXBMN": "Blood manganese",
}
RANDOM_STATE = 42


def main():
    state = joblib.load("shared_state.joblib")
    predictors = state["predictors"]
    continuous_cols = state["continuous_cols"]
    categorical_cols = state["categorical_cols"]
    preprocessor = state["preprocessor"]  # already fit on train (harmonized-outcome run); refit fresh per component below
    train_df = state["train_df"]
    test_df = state["test_df"]

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer, MissingIndicator
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fresh_preprocessor():
        cont_impute_scale = Pipeline(steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        cont_missing_flag = MissingIndicator(features="all")
        cat_impute = SimpleImputer(strategy="most_frequent")
        return ColumnTransformer(transformers=[
            ("cont", cont_impute_scale, continuous_cols),
            ("cont_flag", cont_missing_flag, continuous_cols),
            ("cat", cat_impute, categorical_cols),
        ])

    feat_names = state["feat_names"]
    metal_idx = {m: feat_names.index(m) for m in METALS}

    auc_rows = []
    heat = np.zeros((len(METALS), len(COMPONENTS)))
    for j, (comp_col, comp_label) in enumerate(COMPONENTS.items()):
        X_train, y_train = train_df[predictors], train_df[comp_col]
        X_test, y_test = test_df[predictors], test_df[comp_col]

        pp = fresh_preprocessor()
        Xt_train = pp.fit_transform(X_train)
        Xt_test = pp.transform(X_test)
        Xt_train_df = pd.DataFrame(Xt_train, columns=feat_names)
        Xt_test_df = pd.DataFrame(Xt_test, columns=feat_names)

        model = CatBoostClassifier(
            random_state=RANDOM_STATE, verbose=False,
            learning_rate=0.05, iterations=200, depth=6,
        )
        model.fit(Xt_train_df, y_train)
        proba = model.predict_proba(Xt_test_df)[:, 1]
        auc = roc_auc_score(y_test, proba)
        auc_rows.append({"component": comp_label, "test_auc": round(auc, 4),
                          "test_prevalence_pct": round(y_test.mean() * 100, 1)})
        print(f"{comp_label}: test AUROC = {auc:.4f}")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xt_test_df)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        for i, m in enumerate(METALS):
            heat[i, j] = mean_abs_shap[metal_idx[m]]

    auc_table = pd.DataFrame(auc_rows).rename(columns={
        "component": "MetS component", "test_auc": "Test AUROC",
        "test_prevalence_pct": "Test prevalence (%)",
    })
    auc_table.to_csv("table_component_level_auc.csv", index=False)
    print(auc_table.to_string(index=False))

    heat_df = pd.DataFrame(heat, index=list(METALS.values()), columns=list(COMPONENTS.values()))
    heat_df.index.name = "Metal"
    heat_df.round(5).to_csv("table_metal_by_component_shap.csv")
    print(heat_df.round(4).to_string())

    fig, ax = plt.subplots(figsize=(ps.W_DOUBLE * 0.62, ps.W_DOUBLE * 0.5))
    im = ax.imshow(heat, cmap="cividis", aspect="auto")
    ax.set_xticks(range(len(COMPONENTS))); ax.set_xticklabels(list(COMPONENTS.values()), rotation=30, ha="right")
    ax.set_yticks(range(len(METALS))); ax.set_yticklabels(list(METALS.values()))
    for i in range(len(METALS)):
        for j in range(len(COMPONENTS)):
            rgba = im.cmap(im.norm(heat[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = "white" if luminance < 0.55 else "black"
            ax.text(j, i, f"{heat[i,j]:.3f}", ha="center", va="center", fontsize=7.5,
                     color=txt_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean |SHAP value| (component-specific model)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    ps.save(fig, "fig_metal_by_component_heatmap")
    print("Saved fig_metal_by_component_heatmap.png/.pdf")


if __name__ == "__main__":
    main()
