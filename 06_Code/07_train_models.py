"""
Steps 9-10 -- Model Training (full roster) and Threshold Selection

For each model:
  1. Load tuned hyperparameters from outputs/metrics/best_params.json if
     06_cv_tuning.py has been run for that model; otherwise fall back to
     the untuned default estimator (with a printed warning) so this script
     is still runnable standalone.
  2. Compute out-of-fold (OOF) predicted probabilities via stratified
     10-fold CV on the TRAINING partition only, and derive the Youden-
     optimal decision threshold from those OOF predictions. This is the
     leakage-safe way to pick a threshold -- never from the held-out test
     set (that was the specific bug in the prior pipeline's plain script).
  3. Refit the full pipeline on the entire training partition using the
     tuned hyperparameters.
  4. Save the fitted pipeline (joblib) and the chosen threshold.

Nothing in this script touches data_processed/test_seqn.csv rows. The test
set is opened for the first time in 08_evaluate.py.

REBUILD NOTE: unchanged in its own logic from the original; it picks up the
corrected predictor set, missing-flags, and SMOTENC automatically through
pipeline_utils.

Output: outputs/models/<model_name>.joblib
        outputs/metrics/thresholds.json

Run:
    python scripts/07_train_models.py
    python scripts/07_train_models.py --models "Logistic Regression,LightGBM"
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import pipeline_utils as pu


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", type=int, default=10)
    ap.add_argument("--models", type=str, default=None)
    return ap.parse_args()


def youden_cutoff(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def apply_best_params(pipe, best_params: dict):
    if best_params:
        pipe.set_params(**best_params)
    return pipe


def main():
    args = parse_args()
    train_df, _ = pu.load_model_ready_split()
    feature_cols, cont_cols, cat_cols = pu.get_feature_lists()
    X_train = train_df[feature_cols]
    y_train = train_df[cfg.OUTCOME_VAR]
    print(f"Training partition: {X_train.shape[0]} rows")

    tuned_path = cfg.OUT_METRICS / "best_params.json"
    tuned = {}
    if tuned_path.exists():
        with open(tuned_path) as f:
            tuned = json.load(f)
        print(f"Loaded tuned hyperparameters from {tuned_path}")
    else:
        print(f"[NOTE] {tuned_path} not found -- run 06_cv_tuning.py first for tuned models. "
              f"Proceeding with library defaults for now.")

    registry = pu.get_model_registry()
    if args.models:
        wanted = [m.strip() for m in args.models.split(",")]
        registry = {k: v for k, v in registry.items() if k in wanted}

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=cfg.RANDOM_STATE)

    thresholds = {}
    for name, spec in registry.items():
        print(f"\n=== {name} ===")
        best_params = tuned.get(name, {}).get("best_params", {})
        if best_params:
            print(f"  using tuned params: {best_params}")
        else:
            print("  using library defaults (no tuned params found)")

        pipe = pu.build_pipeline(spec["estimator"](), cont_cols, cat_cols, use_smote=True)
        pipe = apply_best_params(pipe, best_params)

        # OOF predictions on the training partition for threshold selection.
        # n_jobs=1: LightGBM/XGBoost/CatBoost/sklearn ensembles already multithread
        # internally, so joblib's process-based parallelism here just causes
        # oversubscription/hangs rather than speeding anything up.
        oof_prob = cross_val_predict(pipe, X_train, y_train, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
        cutoff = youden_cutoff(y_train, oof_prob)
        thresholds[name] = cutoff
        print(f"  OOF Youden cutoff: {cutoff:.4f}")

        # Refit on the full training partition
        pipe_final = pu.build_pipeline(spec["estimator"](), cont_cols, cat_cols, use_smote=True)
        pipe_final = apply_best_params(pipe_final, best_params)
        pipe_final.fit(X_train, y_train)

        model_path = cfg.OUT_MODELS / f"{name.replace(' ', '_')}.joblib"
        joblib.dump(pipe_final, model_path)
        print(f"  saved: {model_path}")

    thr_path = cfg.OUT_METRICS / "thresholds.json"
    # merge with any existing thresholds (so running a subset of models doesn't wipe the rest)
    existing = {}
    if thr_path.exists():
        with open(thr_path) as f:
            existing = json.load(f)
    existing.update(thresholds)
    with open(thr_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved thresholds: {thr_path}")


if __name__ == "__main__":
    main()
