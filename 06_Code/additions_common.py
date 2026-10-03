"""
Shared setup for the exploratory analyses in 07_Novelty_Additions/code and
08_Q1_Rigor_Additions/code (library module, not a run-script).

2026-10 audit (checklist C22): the earlier version of these analyses
  * read "SUBMISSION/05_Data_Processed/..." relative to the working folder, so
    they only ran inside a specially built scratch folder;
  * hard-coded the predictor list, the continuous-column list, the random
    seed and the CatBoost hyperparameters instead of reading them from the
    main pipeline (and quietly went stale when any of those changed);
  * refitted their own copy of the model, kept in two byte-identical
    00_setup_and_fit.py files, with three copies of pubstyle.py.
Every addition now imports this module, which resolves all paths from this
file's own location, reuses the main pipeline's config, feature lists and
FITTED model (outputs/models/<model>.joblib from 07_train_models.py), and
writes to outputs/07_Novelty_Additions or outputs/08_Q1_Rigor_Additions.
26_build_submission.py copies those outputs into the package.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))
import config as cfg  # noqa: E402
import pipeline_utils as pu  # noqa: E402

TREE_MODELS = ["CatBoost", "XGBoost", "LightGBM", "Gradient Boosting", "Random Forest", "Decision Tree"]
METALS = cfg.PREDICTOR_VARS["metals"]
METAL_NAMES = {"LBXBPB": "Blood lead", "LBXBCD": "Blood cadmium", "LBXTHG": "Blood mercury",
               "LBXBSE": "Blood selenium", "LBXBMN": "Blood manganese"}

# Covariates for every parametric (non-ML) model in the additions.
# 2026-10 audit (A6): the earlier set was age, BMI, sex and income only, which
# left smoking (the main non-dietary source of cadmium and a source of lead),
# survey cycle (blood lead fell over 2007-2018 while MetS rose), race and
# ethnicity, education and alcohol uncontrolled.
ADJUST_TERMS = ["RIDAGEYR", "BMXBMI", "sex_female", "INDFMPIR", "C(RIDRETH1)", "DMDEDUC2",
                "SMQ020", "SMQ040", "ALQ_PAST12", "C(cycle)"]
ADJUST_FORMULA = " + ".join(ADJUST_TERMS)
ADJUST_COLUMNS = ["RIDAGEYR", "BMXBMI", "sex_female", "INDFMPIR", "RIDRETH1", "DMDEDUC2",
                  "SMQ020", "SMQ040", "ALQ_PAST12", "cycle"]


def out_dirs(folder: str):
    """outputs/<folder>/tables and outputs/<folder>/figures, created if needed."""
    base = cfg.OUT_DIR / folder
    tables, figures = base / "tables", base / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def analysis_frame() -> pd.DataFrame:
    """Whole analytic cohort with the encoded predictors, the outcome, the
    components, the cycle and RIDRETH1 as a single categorical column (for
    regression formulas). Missing values are left as NaN."""
    df = pd.read_csv(cfg.DATA_PROC_DIR / "model_ready.csv", low_memory=False)
    raw = pd.read_csv(cfg.DATA_PROC_DIR / "nhanes_with_outcome.csv",
                      usecols=[cfg.ID_VAR, "RIDRETH1"], low_memory=False)
    return df.merge(raw, on=cfg.ID_VAR, how="left")


def model_name() -> str:
    """The model the explanation analyses use: the highest-AUROC tree model
    (the same rule as 09_explain_shap.py and 15_publication_figures.py)."""
    cmp_df = pd.read_csv(cfg.OUT_METRICS / "model_comparison.csv").sort_values("AUC", ascending=False)
    for m in cmp_df["Model"]:
        if m in TREE_MODELS and (cfg.OUT_MODELS / f"{m.replace(' ', '_')}.joblib").exists():
            return m
    raise SystemExit("No fitted tree model found; run the main pipeline (06-08) first.")


def best_params(name: str) -> dict:
    return json.loads((cfg.OUT_METRICS / "best_params.json").read_text())[name]["best_params"]


@lru_cache(maxsize=1)
def primary_state() -> dict:
    """Fitted primary model, the train/test split, transformed matrices and
    test-set SHAP values, shared by every addition."""
    import shap
    name = model_name()
    pipe = joblib.load(cfg.OUT_MODELS / f"{name.replace(' ', '_')}.joblib")
    train_df, test_df = pu.load_model_ready_split()
    feature_cols, cont_cols, cat_cols = pu.get_feature_lists()
    names = pu.transformed_feature_names(cont_cols, cat_cols)
    pre, est = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    Xt_train = pd.DataFrame(np.asarray(pre.transform(train_df[feature_cols])), columns=names)
    Xt_test = pd.DataFrame(np.asarray(pre.transform(test_df[feature_cols])), columns=names)
    explainer = shap.TreeExplainer(est)
    sv = explainer.shap_values(Xt_test)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return {
        "model_name": name, "pipe": pipe, "estimator": est, "explainer": explainer,
        "train_df": train_df, "test_df": test_df,
        "feature_cols": feature_cols, "cont_cols": cont_cols, "cat_cols": cat_cols,
        "feat_names": names, "Xt_train_df": Xt_train, "Xt_test_df": Xt_test,
        "X_train_raw": train_df[feature_cols], "X_test_raw": test_df[feature_cols],
        "y_train": train_df[cfg.OUTCOME_VAR], "y_test": test_df[cfg.OUTCOME_VAR],
        "shap_values": sv,
    }


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def fmt_p(p: float) -> str:
    """Journal-style p-value: three decimals, or scientific below 0.001."""
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"
