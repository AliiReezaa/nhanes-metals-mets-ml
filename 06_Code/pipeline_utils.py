"""
Steps 6-7 -- Shared preprocessing pipeline (Impute -> Flag -> Scale -> SMOTENC -> Model)
and the model registry used by 06_cv_tuning.py, 07_train_models.py,
08_evaluate.py, 09_explain_shap.py, and 10_sensitivity.py.

This is a library module, not a run-script. Centralizing the pipeline here
means every script builds preprocessing identically and fits it fresh per
fold/per training set -- never on the combined train+test data.

Key design point (Step 6 of the plan): SimpleImputer, MissingIndicator and
StandardScaler are built as *unfitted* transformers inside a Pipeline.
Calling .fit_resample / .fit on a training fold and .transform on a
validation/test fold is what prevents preprocessing leakage. Never call
.fit_transform on the full X before splitting -- that was the specific bug
in the prior pipeline.

REBUILD NOTE / FIX (two changes from the original pipeline_utils.py):

1. Missingness indicators. Several predictors here are not missing at random
   in the textbook sense -- e.g. selenium/manganese are structurally absent
   for two entire pooled cycles (2007-2010), and even the harmonized
   ALQ_EVER variable still carries some genuine non-response. Silently
   median/mode-imputing those and feeding the model a value indistinguishable
   from a real measurement both throws away real information (whether a
   value was observed at all can itself be predictive/confounded with
   cohort/era) and can manufacture the illusion of a stable per-feature
   signal that is really just "which imputed constant did this row get."
   Every continuous predictor now gets an explicit, always-present
   "was this value observed?" flag (via sklearn's MissingIndicator with
   features="all", not the default "missing-only" mode) alongside its
   imputed value, so the model can use both. "all" (not "missing-only") is
   deliberate: it guarantees a FIXED output width every single fold/fit,
   which the next fix depends on.

2. SMOTE -> SMOTENC. The original pipeline ran plain SMOTE across a feature
   matrix that mixes ~13 continuous columns with ~20 binary/one-hot/ordinal
   columns. Plain SMOTE interpolates linearly between a minority-class point
   and its nearest neighbor across ALL columns, which for a binary/dummy
   column manufactures fractional values (e.g. sex_female = 0.42) that
   correspond to no real category -- a well-known SMOTE-on-mixed-data pitfall
   that imbalanced-learn's SMOTENC exists specifically to avoid (it samples
   continuous features by interpolation as before, but categorical features
   by majority vote among the nearest neighbors, so synthetic rows always
   have valid, real categorical values). Because the preprocessor's output
   layout is now fully deterministic (continuous block, then their missing-
   flags, then categorical block -- see build_preprocessor), the indices to
   pass as SMOTENC's categorical_features are computable analytically from
   the column counts alone, with no risk of drifting out of sync between
   folds.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def get_feature_lists():
    """Returns (feature_cols, continuous_cols, categorical_cols) from the
    predictor list saved by 03_feature_engineering.py, so every script
    downstream uses the exact same, current feature set."""
    pred_path = cfg.OUT_METRICS / "predictors_used.csv"
    feature_cols = pd.read_csv(pred_path)["predictor"].tolist()
    continuous_cols = [c for c in feature_cols if c in cfg.CONTINUOUS_AS_IS]
    categorical_cols = [c for c in feature_cols if c not in cfg.CONTINUOUS_AS_IS]
    return feature_cols, continuous_cols, categorical_cols


def build_preprocessor(continuous_cols, categorical_cols) -> ColumnTransformer:
    """Continuous vars: median-impute + standardize, PLUS an always-present
    per-column "was this observed?" flag (see module docstring, FIX 1).
    Categorical/binary/one-hot vars: mode-impute, left on their 0/1 (or
    ordinal integer) scale -- scaling a 0/1 dummy doesn't help tree or
    linear models and just makes coefficients harder to read.

    Output column order (deterministic, same every fold): all continuous
    columns (imputed+scaled) -> one missing-flag per continuous column
    (always emitted, even if that fold has zero missing values for it) ->
    all categorical columns (mode-imputed). transformed_feature_names()
    below returns the matching clean names, and get_smotenc_categorical_indices()
    below returns the matching SMOTENC index list -- both derived from this
    exact layout, so they can never drift out of sync with it.
    """
    cont_impute_scale = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cont_missing_flag = MissingIndicator(features="all")  # fixed width: len(continuous_cols)
    cat_impute = SimpleImputer(strategy="most_frequent")

    return ColumnTransformer(transformers=[
        ("cont", cont_impute_scale, continuous_cols),
        ("cont_flag", cont_missing_flag, continuous_cols),
        ("cat", cat_impute, categorical_cols),
    ])


def transformed_feature_names(continuous_cols, categorical_cols) -> list[str]:
    """Clean, human/SHAP-friendly names matching build_preprocessor's fixed
    output layout exactly: continuous_cols, then 'missingindicator_<col>' for
    each continuous col (in the same order), then categorical_cols."""
    return (
        list(continuous_cols)
        + [f"missingindicator_{c}" for c in continuous_cols]
        + list(categorical_cols)
    )


def get_smotenc_categorical_indices(continuous_cols, categorical_cols) -> list[int]:
    """Indices, into the preprocessor's OUTPUT columns, of everything that is
    categorical for SMOTENC's purposes: the missing-flags (always binary 0/1)
    and the original categorical/binary/one-hot block. Computed analytically
    from the fixed layout in build_preprocessor -- never from a live fit --
    so it is guaranteed identical across every fold and every script that
    calls build_pipeline."""
    n_cont = len(continuous_cols)
    n_flag = len(continuous_cols)
    n_cat = len(categorical_cols)
    return list(range(n_cont, n_cont + n_flag + n_cat))


def build_pipeline(estimator, continuous_cols, categorical_cols, use_smote=True, random_state=None):
    """Full leakage-safe pipeline: preprocess -> (optional) SMOTENC -> model.
    Fit this on a training fold/set only; call .predict_proba etc. on
    validation/test data, never .fit or .fit_resample on it."""
    random_state = cfg.RANDOM_STATE if random_state is None else random_state
    preprocessor = build_preprocessor(continuous_cols, categorical_cols)
    steps = [("preprocess", preprocessor)]
    if use_smote:
        cat_idx = get_smotenc_categorical_indices(continuous_cols, categorical_cols)
        steps.append(("smote", SMOTENC(categorical_features=cat_idx, random_state=random_state)))
    steps.append(("model", estimator))
    return ImbPipeline(steps=steps)


def load_model_ready_split():
    """Loads model_ready.csv and the train/test SEQN lists (04_split.py),
    returns (train_df, test_df). Both still contain raw NaNs -- imputation
    happens inside build_pipeline, not here."""
    df = pd.read_csv(cfg.DATA_PROC_DIR / "model_ready.csv", low_memory=False)
    train_seqn = pd.read_csv(cfg.DATA_PROC_DIR / "train_seqn.csv")[cfg.ID_VAR]
    test_seqn = pd.read_csv(cfg.DATA_PROC_DIR / "test_seqn.csv")[cfg.ID_VAR]
    train_df = df[df[cfg.ID_VAR].isin(train_seqn)].reset_index(drop=True)
    test_df = df[df[cfg.ID_VAR].isin(test_seqn)].reset_index(drop=True)
    return train_df, test_df


# ----------------------------------------------------------------------
# Model registry. Constructors (not instances) so every fold/script gets a
# fresh, unfitted estimator. Hyperparameter grids used by 06_cv_tuning.py
# are intentionally small/sane defaults -- expand if you have compute budget.
# ----------------------------------------------------------------------
def get_model_registry(random_state=None):
    random_state = cfg.RANDOM_STATE if random_state is None else random_state

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    registry = {
        "Logistic Regression": {
            "estimator": lambda: LogisticRegression(max_iter=2000, random_state=random_state),
            "param_grid": {"model__C": [0.01, 0.1, 1, 10], "model__penalty": ["l2"]},
        },
        "Decision Tree": {
            "estimator": lambda: DecisionTreeClassifier(random_state=random_state),
            "param_grid": {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 5, 20]},
        },
        "Random Forest": {
            "estimator": lambda: RandomForestClassifier(random_state=random_state),
            "param_grid": {"model__n_estimators": [200, 500], "model__max_depth": [None, 8, 15],
                           "model__min_samples_leaf": [1, 5]},
        },
        "Gradient Boosting": {
            "estimator": lambda: GradientBoostingClassifier(random_state=random_state),
            "param_grid": {"model__n_estimators": [100, 200], "model__learning_rate": [0.01, 0.1],
                           "model__max_depth": [2, 3]},
        },
        "LightGBM": {
            "estimator": lambda: LGBMClassifier(random_state=random_state, verbose=-1),
            "param_grid": {"model__n_estimators": [200, 500], "model__learning_rate": [0.01, 0.05, 0.1],
                           "model__num_leaves": [15, 31]},
        },
        "XGBoost": {
            "estimator": lambda: XGBClassifier(random_state=random_state, eval_metric="logloss"),
            "param_grid": {"model__n_estimators": [200, 500], "model__learning_rate": [0.01, 0.05, 0.1],
                           "model__max_depth": [3, 5]},
        },
        "SVM": {
            "estimator": lambda: SVC(probability=True, random_state=random_state),
            "param_grid": {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]},
        },
        "Naive Bayes": {
            "estimator": lambda: GaussianNB(),
            "param_grid": {},
        },
        "MLP": {
            "estimator": lambda: MLPClassifier(max_iter=2000, random_state=random_state),
            "param_grid": {"model__hidden_layer_sizes": [(64,), (64, 32)], "model__alpha": [0.0001, 0.001]},
        },
        "KNN": {
            "estimator": lambda: KNeighborsClassifier(),
            "param_grid": {"model__n_neighbors": [5, 15, 25]},
        },
    }

    try:
        from catboost import CatBoostClassifier
        registry["CatBoost"] = {
            "estimator": lambda: CatBoostClassifier(random_state=random_state, verbose=False),
            "param_grid": {"model__iterations": [200, 500], "model__learning_rate": [0.01, 0.05, 0.1],
                           "model__depth": [4, 6]},
        }
    except ImportError:
        print("[NOTE] catboost not installed -- skipping CatBoost in the model registry")

    return registry
