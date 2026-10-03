"""
Q1 rigor addition 1: temporal (forward-in-time) validation. Every result so
far -- the manuscript's own AUROC 0.8567 and every novelty addition -- comes
from ONE random SEQN split that pools all six NHANES cycles (2007-2018)
together before splitting. That never tests whether the model generalizes
FORWARD in time; it only tests whether it generalizes to a random subsample
of the same pooled distribution it was trained on.

Design: train on the four earlier cycles (2007_2008 - 2013_2014, n~8,102),
test on the two most recent cycles the model has never seen in any form
(2015_2016 + 2017_2018, n~3,042). Same predictors, same leakage-safe
preprocessing, same tuned CatBoost hyperparameters as the primary model --
only the train/test partition changes, from random-by-SEQN to a hard
temporal cutoff.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubstyle as ps

ps.apply_style()

import importlib.util
spec = importlib.util.spec_from_file_location("setup00", "00_setup_and_fit.py")
setup00 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup00)

PREDICTORS = setup00.PREDICTORS
build_preprocessor = setup00.build_preprocessor
smotenc_cat_indices = setup00.smotenc_cat_indices

from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from catboost import CatBoostClassifier

RANDOM_STATE = 42
TRAIN_CYCLES = ["2007_2008", "2009_2010", "2011_2012", "2013_2014"]
TEST_CYCLES = ["2015_2016", "2017_2018"]


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"


def calibration_stats(y_true, p_pred):
    eps = 1e-6
    p_clip = np.clip(p_pred, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip))
    X = sm.add_constant(logit_p)
    m = sm.GLM(y_true, X, family=sm.families.Binomial()).fit()
    intercept, slope = m.params
    oe_ratio = y_true.mean() / p_pred.mean()
    brier = brier_score_loss(y_true, p_pred)
    return {"slope": slope, "intercept": intercept, "oe_ratio": oe_ratio, "brier": brier}


def main():
    df = pd.read_csv("SUBMISSION/05_Data_Processed/model_ready.csv", low_memory=False)
    train_df = df[df["cycle"].isin(TRAIN_CYCLES)].reset_index(drop=True)
    test_df = df[df["cycle"].isin(TEST_CYCLES)].reset_index(drop=True)
    print(f"Temporal train (cycles {TRAIN_CYCLES}): n={len(train_df)}, "
          f"prevalence={train_df['mets_status'].mean():.3f}")
    print(f"Temporal test  (cycles {TEST_CYCLES}): n={len(test_df)}, "
          f"prevalence={test_df['mets_status'].mean():.3f}")

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
    proba_temporal = pipe.predict_proba(X_test)[:, 1]
    auc_temporal = roc_auc_score(y_test, proba_temporal)
    cal_temporal = calibration_stats(y_test.values, proba_temporal)
    print(f"\nTemporal-holdout test AUROC: {auc_temporal:.4f}")
    print(f"Temporal-holdout calibration: slope={cal_temporal['slope']:.3f}, "
          f"intercept={cal_temporal['intercept']:.3f}, O:E={cal_temporal['oe_ratio']:.3f}, "
          f"Brier={cal_temporal['brier']:.4f}")

    # Reference: the manuscript's original random-SEQN-split model (already
    # fit and cached in shared_state.joblib) for a direct side-by-side
    state = joblib.load("shared_state.joblib")
    proba_random = state["pipe"].predict_proba(state["X_test_raw"])[:, 1]
    auc_random = roc_auc_score(state["y_test"], proba_random)
    cal_random = calibration_stats(state["y_test"].values, proba_random)
    print(f"\nOriginal random-split test AUROC (reference): {auc_random:.4f}")

    comparison = pd.DataFrame([
        {
            "Validation design": "Random SEQN split (original manuscript)",
            "Train n": len(state["X_train_raw"]), "Test n": len(state["X_test_raw"]),
            "Test AUROC": round(auc_random, 4),
            "Calibration slope": round(cal_random["slope"], 3),
            "Calibration intercept": round(cal_random["intercept"], 3),
            "O:E ratio": round(cal_random["oe_ratio"], 3),
            "Brier score": round(cal_random["brier"], 4),
        },
        {
            "Validation design": f"Temporal holdout (train {TRAIN_CYCLES[0]}-{TRAIN_CYCLES[-1]}, "
                                  f"test {TEST_CYCLES[0]}-{TEST_CYCLES[-1]})",
            "Train n": len(X_train), "Test n": len(X_test),
            "Test AUROC": round(auc_temporal, 4),
            "Calibration slope": round(cal_temporal["slope"], 3),
            "Calibration intercept": round(cal_temporal["intercept"], 3),
            "O:E ratio": round(cal_temporal["oe_ratio"], 3),
            "Brier score": round(cal_temporal["brier"], 4),
        },
    ])
    comparison.to_csv("table_temporal_validation.csv", index=False)
    print()
    print(comparison.to_string(index=False))

    # ROC curves, both designs overlaid
    fpr_r, tpr_r, _ = roc_curve(state["y_test"], proba_random)
    fpr_t, tpr_t, _ = roc_curve(y_test, proba_temporal)

    fig, ax = plt.subplots(figsize=(ps.W_SINGLE * 1.55, ps.W_SINGLE * 1.45))
    ax.plot(fpr_r, tpr_r, color=ps.OKABE_BLUE, lw=1.6,
            label=f"Random split (AUROC = {auc_random:.3f})", zorder=3)
    ax.plot(fpr_t, tpr_t, color=ps.OKABE_VERMIL, lw=1.6,
            label=f"Temporal holdout (AUROC = {auc_temporal:.3f})", zorder=3)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    ps.save(fig, "fig_temporal_validation_roc")
    print("Saved fig_temporal_validation_roc.png/.pdf")


if __name__ == "__main__":
    main()
