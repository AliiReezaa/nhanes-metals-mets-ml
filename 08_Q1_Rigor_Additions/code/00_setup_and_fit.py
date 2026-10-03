"""
Shared setup for the four novelty additions: loads the exact same
model_ready.csv / train_seqn.csv / test_seqn.csv the original pipeline used,
rebuilds the identical leakage-safe preprocessing pipeline (median/mode
impute + missing-flags + SMOTENC), refits CatBoost with the tuned
hyperparameters from best_params.json, and saves everything downstream
scripts need (fitted pipeline, transformed matrices, SHAP values) so each
addition doesn't have to repeat the fit.
"""
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import shap

RANDOM_STATE = 42

CONTINUOUS_AS_IS = [
    "RIDAGEYR", "INDFMPIR", "DMDHHSIZ", "BMXBMI",
    "ALQ130", "PAD645", "BPXPLS", "ALQ_EVER",
    "LBXBPB", "LBXBCD", "LBXTHG", "LBXBSE", "LBXBMN",
]

PREDICTORS = [
    "SMQ020", "PAQ605", "PAQ650", "PAQ665", "DBQ700", "DMDEDUC2", "SMQ040",
    "sex_female", "exam_summer", "RIDAGEYR", "INDFMPIR", "DMDHHSIZ", "BMXBMI",
    "ALQ130", "PAD645", "BPXPLS", "ALQ_EVER", "LBXBPB", "LBXBCD", "LBXTHG",
    "LBXBSE", "LBXBMN", "RIDRETH1_2.0", "RIDRETH1_3.0", "RIDRETH1_4.0",
    "RIDRETH1_5.0", "RIDRETH1_nan", "DMDMARTL_2.0", "DMDMARTL_3.0",
    "DMDMARTL_4.0", "DMDMARTL_5.0", "DMDMARTL_6.0", "DMDMARTL_nan",
]
CONTINUOUS_COLS = [c for c in PREDICTORS if c in CONTINUOUS_AS_IS]
CATEGORICAL_COLS = [c for c in PREDICTORS if c not in CONTINUOUS_AS_IS]


def build_preprocessor():
    cont_impute_scale = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cont_missing_flag = MissingIndicator(features="all")
    cat_impute = SimpleImputer(strategy="most_frequent")
    return ColumnTransformer(transformers=[
        ("cont", cont_impute_scale, CONTINUOUS_COLS),
        ("cont_flag", cont_missing_flag, CONTINUOUS_COLS),
        ("cat", cat_impute, CATEGORICAL_COLS),
    ])


def transformed_feature_names():
    return CONTINUOUS_COLS + [f"missingindicator_{c}" for c in CONTINUOUS_COLS] + CATEGORICAL_COLS


def smotenc_cat_indices():
    n_cont = len(CONTINUOUS_COLS)
    n_flag = len(CONTINUOUS_COLS)
    n_cat = len(CATEGORICAL_COLS)
    return list(range(n_cont, n_cont + n_flag + n_cat))


def main():
    df = pd.read_csv("SUBMISSION/05_Data_Processed/model_ready.csv", low_memory=False)
    train_seqn = pd.read_csv("SUBMISSION/05_Data_Processed/train_seqn.csv")["SEQN"]
    test_seqn = pd.read_csv("SUBMISSION/05_Data_Processed/test_seqn.csv")["SEQN"]
    train_df = df[df["SEQN"].isin(train_seqn)].reset_index(drop=True)
    test_df = df[df["SEQN"].isin(test_seqn)].reset_index(drop=True)
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    X_train, y_train = train_df[PREDICTORS], train_df["mets_status"]
    X_test, y_test = test_df[PREDICTORS], test_df["mets_status"]

    preprocessor = build_preprocessor()
    cat_idx = smotenc_cat_indices()
    model = CatBoostClassifier(
        random_state=RANDOM_STATE, verbose=False,
        learning_rate=0.05, iterations=200, depth=6,
    )
    pipe = ImbPipeline(steps=[
        ("preprocess", preprocessor),
        ("smote", SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    print(f"Sanity check -- CatBoost test AUROC (harmonized label): {auc:.4f}  "
          f"(manuscript reports 0.8567 for this same model/split)")

    # Transformed matrices (post-preprocess, pre-SMOTE) for downstream scripts
    Xt_train = preprocessor.fit_transform(X_train)  # refit alone is fine: same as inside pipe.fit
    Xt_test = preprocessor.transform(X_test)
    feat_names = transformed_feature_names()

    Xt_train_df = pd.DataFrame(Xt_train, columns=feat_names)
    Xt_test_df = pd.DataFrame(Xt_test, columns=feat_names)

    # SHAP (TreeExplainer) on the CatBoost step alone, applied to the
    # preprocessed (but not SMOTE-resampled) test matrix -- matches how
    # 09_explain_shap.py in the original pipeline computes it.
    cb_model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(cb_model)
    shap_values = explainer.shap_values(Xt_test_df)
    print(f"SHAP values shape: {shap_values.shape}")

    joblib.dump({
        "pipe": pipe,
        "preprocessor": preprocessor,
        "cb_model": cb_model,
        "feat_names": feat_names,
        "continuous_cols": CONTINUOUS_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "predictors": PREDICTORS,
        "X_train_raw": X_train, "y_train": y_train,
        "X_test_raw": X_test, "y_test": y_test,
        "Xt_train_df": Xt_train_df, "Xt_test_df": Xt_test_df,
        "shap_values": shap_values,
        "test_auc": auc,
        "train_df": train_df, "test_df": test_df,
    }, "shared_state.joblib")
    print("Saved shared_state.joblib")


if __name__ == "__main__":
    main()
