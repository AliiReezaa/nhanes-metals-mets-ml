"""
Q1 rigor addition 1: temporal (forward-in-time) validation. The main result
comes from one random split that pools all six NHANES cycles before
splitting, which tests generalization to a random subsample of the same
pooled population, not forward in time.

Design: develop on the four earlier cycles (2007-2008 to 2013-2014) and test
on the two most recent ones (2015-2016 and 2017-2018), which the model never
sees in any form. Same predictors, same leakage-safe pipeline, same model
type as the primary (leading) model.

2026-10 audit:
  * A7: the hyperparameters are RE-TUNED by stratified 10-fold cross-validation
    on the 2007-2014 cycles only, over the same grid as 06_cv_tuning.py. The
    earlier version reused the settings tuned on the random-split training
    set, which contained about 80% of the 2015-2018 participants it was then
    tested on.
  * C25: the calibration intercept is calibration-in-the-large (slope fixed
    at 1), the same quantity as in the main calibration table; the earlier
    version took the intercept from the joint slope-and-intercept model, so
    its "random split" row disagreed with the manuscript.

Run (after the main pipeline):
    python 08_Q1_Rigor_Additions/code/06_temporal_validation.py [--quick]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_Code"))
import additions_common as ac  # noqa: E402

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
import pipeline_utils as pu
import pubstyle as ps
import stats_utils as su

ps.apply_style()
TABLES, FIGURES = ac.out_dirs("08_Q1_Rigor_Additions")
TRAIN_CYCLES = ["2007_2008", "2009_2010", "2011_2012", "2013_2014"]
TEST_CYCLES = ["2015_2016", "2017_2018"]


def calibration_stats(y, p):
    slope, intercept = su.calibration_slope_intercept(y, p)
    return {"slope": slope, "intercept": intercept, "oe_ratio": float(np.mean(y) / np.mean(p)),
            "brier": brier_score_loss(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3-fold tuning (testing only)")
    args = ap.parse_args()

    st = ac.primary_state()
    name = st["model_name"]
    df = pd.read_csv(cfg.DATA_PROC_DIR / "model_ready.csv", low_memory=False)
    feats, cont, cat = st["feature_cols"], st["cont_cols"], st["cat_cols"]
    tr = df[df[cfg.CYCLE_VAR].isin(TRAIN_CYCLES)].reset_index(drop=True)
    te = df[df[cfg.CYCLE_VAR].isin(TEST_CYCLES)].reset_index(drop=True)
    print(f"Temporal development set (2007-2014): n={len(tr)}, prevalence={tr[cfg.OUTCOME_VAR].mean():.3f}")
    print(f"Temporal test set (2015-2018): n={len(te)}, prevalence={te[cfg.OUTCOME_VAR].mean():.3f}")

    registry = pu.get_model_registry()
    pipe = pu.build_pipeline(registry[name]["estimator"](), cont, cat, use_smote=True)
    grid = registry[name]["param_grid"]
    skf = StratifiedKFold(n_splits=3 if args.quick else 10, shuffle=True, random_state=cfg.RANDOM_STATE)
    if grid:
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=skf, n_jobs=1, refit=True)
        search.fit(tr[feats], tr[cfg.OUTCOME_VAR])
        best, cv_auc, model = search.best_params_, search.best_score_, search.best_estimator_
    else:
        best, cv_auc, model = {}, float("nan"), pipe.fit(tr[feats], tr[cfg.OUTCOME_VAR])
    print(f"Re-tuned on 2007-2014 ({name}): {best}, CV AUROC {cv_auc:.4f}")

    p_t = model.predict_proba(te[feats])[:, 1]
    y_t = te[cfg.OUTCOME_VAR].values
    auc_t = roc_auc_score(y_t, p_t)
    ci_t = su.bootstrap_ci(roc_auc_score, y_t, p_t)
    cal_t = calibration_stats(y_t, p_t)

    p_r = st["pipe"].predict_proba(st["X_test_raw"])[:, 1]
    y_r = st["y_test"].values
    auc_r = roc_auc_score(y_r, p_r)
    ci_r = su.bootstrap_ci(roc_auc_score, y_r, p_r)
    cal_r = calibration_stats(y_r, p_r)

    rows = []
    for label, ntr, nte, auc, ci, cal, params in [
        ("Random split (primary analysis)", len(st["X_train_raw"]), len(y_r), auc_r, ci_r, cal_r,
         json.loads((cfg.OUT_METRICS / "best_params.json").read_text())[name]["best_params"]),
        ("Temporal holdout (develop 2007-2014, test 2015-2018; re-tuned)", len(tr), len(y_t), auc_t,
         ci_t, cal_t, best)]:
        rows.append({"Validation design": label, "Model": name, "Train n": ntr, "Test n": nte,
                     "Test AUROC": round(auc, 4), "AUROC 95% CI low": round(ci[0], 4),
                     "AUROC 95% CI high": round(ci[1], 4),
                     "Calibration slope": round(cal["slope"], 3),
                     "Calibration intercept (in-the-large)": round(cal["intercept"], 3),
                     "O:E ratio": round(cal["oe_ratio"], 3), "Brier score": round(cal["brier"], 4),
                     "Hyperparameters": json.dumps(params)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table_temporal_validation.csv", index=False)
    print(out.drop(columns="Hyperparameters").to_string(index=False))

    fpr_r, tpr_r, _ = roc_curve(y_r, p_r)
    fpr_t, tpr_t, _ = roc_curve(y_t, p_t)
    fig, ax = plt.subplots(figsize=(ps.W_SINGLE * 1.55, ps.W_SINGLE * 1.45))
    ax.plot(fpr_r, tpr_r, color=ps.OKABE_BLUE, lw=1.6, label=f"Random split (AUROC = {auc_r:.3f})", zorder=3)
    ax.plot(fpr_t, tpr_t, color=ps.OKABE_VERMIL, lw=1.6, label=f"Temporal holdout (AUROC = {auc_t:.3f})",
            zorder=3)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    ps.save(fig, FIGURES / "fig_temporal_validation_roc")


if __name__ == "__main__":
    main()
