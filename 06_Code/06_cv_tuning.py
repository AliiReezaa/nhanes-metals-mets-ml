"""
Step 8 -- Cross-Validation and Hyperparameter Tuning

Runs stratified k-fold CV on the TRAINING partition only (never the test
set from 04_split.py) to select hyperparameters for every model in
pipeline_utils.get_model_registry(). Each fold's preprocessing (impute/
scale/missing-flag) and SMOTENC are fit fresh inside that fold via the
shared Pipeline -- see pipeline_utils.build_pipeline for why that matters.

REBUILD NOTE: unchanged in its own logic from the original pipeline; it
automatically picks up the corrected predictor set (harmonized ALQ_EVER,
medication variables excluded), the missing-flags, and SMOTENC through
pipeline_utils, with no direct changes needed here.

Scoring metric for model selection is ROC AUC (imbalance-appropriate;
accuracy would be misleading at ~46% prevalence).

Output: outputs/metrics/best_params.json   (per-model best hyperparameters + CV AUC)
        outputs/metrics/tuning_summary.csv

Run (defaults to the full 10-fold search described in the plan -- this is
slow; expect anywhere from tens of minutes to a few hours depending on your
machine, mostly driven by SVM and MLP):
    python 06_Code/06_cv_tuning.py

Faster options while developing/debugging:
    python 06_Code/06_cv_tuning.py --quick                     # cv=3, n_iter=3, all models
    python 06_Code/06_cv_tuning.py --models "Logistic Regression,Decision Tree"
    python 06_Code/06_cv_tuning.py --cv 5 --n-iter 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import pipeline_utils as pu


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", type=int, default=10, help="number of stratified folds")
    ap.add_argument("--n-iter", type=int, default=20, help="RandomizedSearchCV iterations per model")
    ap.add_argument("--models", type=str, default=None, help="comma-separated subset of model names")
    ap.add_argument("--quick", action="store_true", help="shortcut for --cv 3 --n-iter 3 (debugging only)")
    args = ap.parse_args()
    if args.quick:
        args.cv = 3
        args.n_iter = 3
    return args


def main():
    args = parse_args()
    train_df, _ = pu.load_model_ready_split()
    feature_cols, cont_cols, cat_cols = pu.get_feature_lists()
    X_train = train_df[feature_cols]
    y_train = train_df[cfg.OUTCOME_VAR]
    print(f"Training partition: {X_train.shape[0]} rows, {len(feature_cols)} features")
    print(f"CV folds: {args.cv} | RandomizedSearchCV iterations: {args.n_iter}")

    registry = pu.get_model_registry()
    if args.models:
        wanted = [m.strip() for m in args.models.split(",")]
        registry = {k: v for k, v in registry.items() if k in wanted}
        print(f"Restricting to: {list(registry.keys())}")

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=cfg.RANDOM_STATE)

    results = {}
    summary_rows = []

    for name, spec in registry.items():
        t0 = time.time()
        pipe = pu.build_pipeline(spec["estimator"](), cont_cols, cat_cols, use_smote=True)
        param_grid = spec["param_grid"]

        # n_jobs=1: LightGBM/XGBoost/CatBoost/sklearn ensembles already multithread
        # internally; joblib's process-based parallelism on top of that causes
        # oversubscription and, in constrained/sandboxed environments, can hang.
        if param_grid:
            grid_size = _grid_size(param_grid)
            n_iter_actual = min(args.n_iter, grid_size)
            search = RandomizedSearchCV(
                pipe, param_distributions=param_grid, n_iter=n_iter_actual,
                scoring="roc_auc", cv=skf, random_state=cfg.RANDOM_STATE, n_jobs=1, refit=False,
            )
            search.fit(X_train, y_train)
            best_params = search.best_params_
            best_score = search.best_score_
            search_method = "RandomizedSearchCV"
        else:
            scores = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=skf, n_jobs=1)
            best_params = {}
            best_score = scores.mean()
            grid_size = 0
            n_iter_actual = None
            search_method = "cross_val_score (no hyperparameters to search)"

        elapsed = time.time() - t0
        tuning_meta = {
            "cv_folds": args.cv,
            "n_iter_requested": args.n_iter if param_grid else None,
            "n_iter_actual": n_iter_actual,
            "grid_size": grid_size,
            "search_method": search_method,
        }
        results[name] = {"best_params": best_params, "cv_auc": best_score, "seconds": round(elapsed, 1),
                          **tuning_meta}
        summary_rows.append({"model": name, "cv_auc": round(best_score, 4), "seconds": round(elapsed, 1),
                              "cv_folds": args.cv, "n_iter_actual": n_iter_actual, "search_method": search_method,
                              "best_params": json.dumps(best_params)})
        print(f"  {name}: CV AUC = {best_score:.4f}  ({elapsed:.1f}s)  cv={args.cv} n_iter={n_iter_actual}  "
              f"params = {best_params}")

    # Merge with any existing results so running a subset of models (e.g. one
    # at a time, across separate invocations) doesn't wipe out the rest.
    out_json = cfg.OUT_METRICS / "best_params.json"
    existing = {}
    if out_json.exists():
        with open(out_json) as f:
            existing = json.load(f)
    existing.update(results)
    with open(out_json, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved: {out_json}  ({len(existing)} models total on file)")

    summary_path = cfg.OUT_METRICS / "tuning_summary.csv"
    existing_summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame(columns=["model"])
    new_summary = pd.DataFrame(summary_rows)
    merged_summary = pd.concat([existing_summary[~existing_summary["model"].isin(new_summary["model"])], new_summary])
    merged_summary = merged_summary.sort_values("cv_auc", ascending=False)
    merged_summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")
    print("\nRanked by CV AUC (all models tuned so far):")
    print(merged_summary[["model", "cv_auc", "seconds"]].to_string(index=False))


def _grid_size(param_grid: dict) -> int:
    size = 1
    for v in param_grid.values():
        size *= len(v)
    return size


if __name__ == "__main__":
    main()
