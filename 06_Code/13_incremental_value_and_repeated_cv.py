"""
Step 13b -- Incremental value of the metals, simple baselines, and repeated
cross-validation (added in the 2026-10 audit, checklist A1-A4).

Part A. Incremental value (A1, A2, A3). The leading model (highest test AUROC
in model_comparison.csv) is refitted on the training partition, with its tuned
hyperparameters and the same leakage-safe pipeline, on four reduced predictor
sets, and every version is scored on the same held-out test participants:
    full            all predictors (the primary model)
    no_metals       all five blood metals removed     -> proposal question Q3
    lead_only       lead kept, the other four removed
    no_bmi          BMI removed (waist circumference defines one MetS component,
                    so part of BMI's importance is definitional)
    age_bmi_lr      logistic regression on age and BMI only -> tests the claim
                    that these two carry nearly all of the discrimination
For each reduced model: test AUROC with a bootstrap 95% CI, the paired
difference from the full model (paired-bootstrap 95% CI and DeLong p-value),
and the 10-fold cross-validated AUROC on the training partition.

Part B. Repeated cross-validation (A4). The main analysis uses one random
80/20 split under one seed. Here every model is re-evaluated with repeated
stratified 10-fold cross-validation (5 repeats, 50 fits per model) on the
training partition, with its tuned hyperparameters, to show how much the
AUROC moves from one data partition to another.

Output: outputs/metrics/incremental_value.csv
        outputs/metrics/repeated_cv.csv

Run:
    python 06_Code/13_incremental_value_and_repeated_cv.py
    python 06_Code/13_incremental_value_and_repeated_cv.py --quick   # 3 folds, 2 repeats
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import pipeline_utils as pu
import stats_utils as su

N_BOOT = 1000


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 folds and 2 repeats (testing only)")
    ap.add_argument("--skip-repeated", action="store_true", help="run Part A only")
    return ap.parse_args()


def tuned(name):
    return json.loads((cfg.OUT_METRICS / "best_params.json").read_text()).get(name, {}).get("best_params", {})


def make_pipe(name, cont, cat, registry):
    pipe = pu.build_pipeline(registry[name]["estimator"](), cont, cat, use_smote=True)
    params = tuned(name)
    if params:
        pipe.set_params(**params)
    return pipe


def part_a(args, registry, train_df, test_df, feats, cont, cat):
    cmp_df = pd.read_csv(cfg.OUT_METRICS / "model_comparison.csv").sort_values("AUC", ascending=False)
    lead = cmp_df["Model"].iloc[0]
    y_tr, y_te = train_df[cfg.OUTCOME_VAR], test_df[cfg.OUTCOME_VAR]
    metals = cfg.PREDICTOR_VARS["metals"]
    skf = StratifiedKFold(n_splits=3 if args.quick else 10, shuffle=True, random_state=cfg.RANDOM_STATE)

    arms = {
        "full": [c for c in feats],
        "no_metals": [c for c in feats if c not in metals],
        "lead_only": [c for c in feats if c not in metals or c == "LBXBPB"],
        "no_bmi": [c for c in feats if c != "BMXBMI"],
    }
    probs, rows = {}, []
    for arm, cols in arms.items():
        c_cont = [c for c in cont if c in cols]
        c_cat = [c for c in cat if c in cols]
        t0 = time.time()
        if arm == "full":
            pipe = joblib.load(cfg.OUT_MODELS / f"{lead.replace(' ', '_')}.joblib")
        else:
            pipe = make_pipe(lead, c_cont, c_cat, registry).fit(train_df[cols], y_tr)
        probs[arm] = pipe.predict_proba(test_df[cols])[:, 1]
        cv = cross_val_score(make_pipe(lead, c_cont, c_cat, registry), train_df[cols], y_tr,
                             scoring="roc_auc", cv=skf, n_jobs=1)
        rows.append({"model": arm, "algorithm": lead, "n_predictors": len(cols), "cv_auc": cv.mean(),
                     "cv_auc_sd": cv.std()})
        print(f"  {arm:10s} ({len(cols)} predictors): CV AUROC {cv.mean():.4f}  ({time.time() - t0:.0f} s)")

    # simple baseline: logistic regression on age and BMI only (median-imputed BMI)
    base_cols = ["RIDAGEYR", "BMXBMI"]
    lr_pipe = pu.build_pipeline(LogisticRegression(max_iter=2000), base_cols, [], use_smote=False)
    lr_pipe.fit(train_df[base_cols], y_tr)
    probs["age_bmi_lr"] = lr_pipe.predict_proba(test_df[base_cols])[:, 1]
    cv = cross_val_score(pu.build_pipeline(LogisticRegression(max_iter=2000), base_cols, [], use_smote=False),
                         train_df[base_cols], y_tr, scoring="roc_auc", cv=skf, n_jobs=1)
    rows.append({"model": "age_bmi_lr", "algorithm": "Logistic Regression", "n_predictors": 2,
                 "cv_auc": cv.mean(), "cv_auc_sd": cv.std()})

    y = y_te.values
    for r in rows:
        p = probs[r["model"]]
        r["test_auc"] = roc_auc_score(y, p)
        r["test_auc_ci_low"], r["test_auc_ci_high"] = su.bootstrap_ci(roc_auc_score, y, p, n_boot=N_BOOT)
        if r["model"] == "full":
            r.update(delta_vs_full=0.0, delta_ci_low=0.0, delta_ci_high=0.0, delong_p=1.0)
            continue
        _, _, d, _, _, pval = su.delong_test(y, p, probs["full"])
        rng = np.random.default_rng(cfg.RANDOM_STATE)
        diffs = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            diffs.append(roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], probs["full"][idx]))
        r.update(delta_vs_full=d, delta_ci_low=float(np.percentile(diffs, 2.5)),
                 delta_ci_high=float(np.percentile(diffs, 97.5)), delong_p=pval)
    out = pd.DataFrame(rows)
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(4) if c != "delong_p" else out[c].map(lambda v: float(f"{v:.3g}"))
    out.to_csv(cfg.OUT_METRICS / "incremental_value.csv", index=False)
    print("\n" + out.to_string(index=False))


def part_b(args, registry, train_df, feats, cont, cat):
    X, y = train_df[feats], train_df[cfg.OUTCOME_VAR]
    k, r = (3, 2) if args.quick else (10, 5)
    rskf = RepeatedStratifiedKFold(n_splits=k, n_repeats=r, random_state=cfg.RANDOM_STATE)
    rows = []
    for name in registry:
        t0 = time.time()
        scores = cross_val_score(make_pipe(name, cont, cat, registry), X, y, scoring="roc_auc",
                                 cv=rskf, n_jobs=1)
        per_repeat = scores.reshape(r, k).mean(axis=1)
        rows.append({"model": name, "n_fits": len(scores), "folds": k, "repeats": r,
                     "mean_auc": round(scores.mean(), 4), "sd_auc_folds": round(scores.std(), 4),
                     "p2_5_auc": round(float(np.percentile(scores, 2.5)), 4),
                     "p97_5_auc": round(float(np.percentile(scores, 97.5)), 4),
                     "min_repeat_mean": round(per_repeat.min(), 4),
                     "max_repeat_mean": round(per_repeat.max(), 4)})
        print(f"  {name:20s} mean {scores.mean():.4f}  SD {scores.std():.4f}  "
              f"repeat means {per_repeat.min():.4f}-{per_repeat.max():.4f}  ({time.time() - t0:.0f} s)")
    out = pd.DataFrame(rows).sort_values("mean_auc", ascending=False)
    out.to_csv(cfg.OUT_METRICS / "repeated_cv.csv", index=False)
    print("\n" + out.to_string(index=False))


def main():
    args = parse_args()
    registry = pu.get_model_registry()
    train_df, test_df = pu.load_model_ready_split()
    feats, cont, cat = pu.get_feature_lists()
    print("=== Part A: incremental value of the metals and simple baselines ===")
    part_a(args, registry, train_df, test_df, feats, cont, cat)
    if not args.skip_repeated:
        print("\n=== Part B: repeated stratified cross-validation ===")
        part_b(args, registry, train_df, feats, cont, cat)


if __name__ == "__main__":
    main()
