"""
Step 13 -- Sensitivity Analyses

All comparisons run via stratified CV on the TRAINING partition only.
Deliberately not touching the test set here: it was used once in
08_evaluate.py for the headline numbers, and repeatedly re-checking
exploratory variants against it would turn it into a de facto validation
set, undermining the point of holding it out at all.

  (a) Primary predictor set vs. predictor set + DIQ010 (diagnosed diabetes)
  (b) SMOTE vs. class-weighting, for the top few models
  (c) Blood lead only vs. all 5 metals
  (d) Outcome-definition arms: the harmonized (medication-adjusted) label used
      throughout this rebuild vs. the original pipeline's lab/exam-only
      label -- quantifies how much of a difference the Step-3 label fix
      (02_clean_and_outcome.py) makes to apparent model discrimination,
      holding the feature matrix and CV folds' construction identical and
      varying only which of the two label columns is fit against. This is
      the single most consequential fix in the rebuild (see
      outputs/metrics/outcome_definition_comparison.csv for the raw
      prevalence/reclassification numbers), so it gets its own dedicated
      sensitivity arm rather than being left as a footnote.
      2026-10 audit (A8): two further label arms for the two definition
      choices -- cholesterol medication NOT applied to the TG/HDL components
      (mets_status_nolipidmed), and blood pressure from the first reading
      only (mets_status_bpfirst).

2026-10 audit (C21): every arm now uses each model's TUNED hyperparameters
from best_params.json (the earlier version used library defaults, so its
"primary" arm was not the primary model), and Gradient Boosting, which has
no class_weight option, gets balanced sample weights in arm (b).

Output: outputs/metrics/sensitivity_diq010.csv
        outputs/metrics/sensitivity_smote_vs_classweight.csv
        outputs/metrics/sensitivity_metals_scope.csv
        outputs/metrics/sensitivity_outcome_definition.csv

Run:
    python 06_Code/10_sensitivity.py
    python 06_Code/10_sensitivity.py --cv 5 --top-n 3
"""
import argparse
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import pipeline_utils as pu
import json
from sklearn.utils.class_weight import compute_sample_weight

_TUNED = None


def tuned_params(name):
    global _TUNED
    if _TUNED is None:
        path = cfg.OUT_METRICS / "best_params.json"
        _TUNED = json.loads(path.read_text()) if path.exists() else {}
    return _TUNED.get(name, {}).get("best_params", {})


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=3, help="how many top models to use for (b)")
    ap.add_argument("--models", type=str, default=None,
                     help="comma-separated explicit model list, overrides --top-n auto-selection "
                          "(useful to force fast models when the top-ranked ones are slow, e.g. "
                          "CatBoost/XGBoost, and this is being run interactively)")
    return ap.parse_args()


def cv_auc(name, estimator_ctor, X, y, cont_cols, cat_cols, use_smote, skf, estimator=None, params=None):
    """Mean and SD of the CV AUROC for one model with its tuned hyperparameters."""
    pipe = pu.build_pipeline(estimator if estimator is not None else estimator_ctor(),
                             cont_cols, cat_cols, use_smote=use_smote)
    best = tuned_params(name)
    if best:
        pipe.set_params(**best)
    scores = cross_val_score(pipe, X, y, scoring="roc_auc", cv=skf, n_jobs=1, params=params)
    return scores.mean(), scores.std()


def encode_diq010(train_df: pd.DataFrame) -> pd.DataFrame:
    """DIQ010: 1=Yes, 2=No, 3=Borderline (7/9 already recoded to NaN in
    02_clean_and_outcome.py). One-hot with 'No' as reference."""
    out = pd.DataFrame(index=train_df.index)
    raw = train_df[cfg.DIQ_SENSITIVITY_VAR]
    out["diq010_yes"] = (raw == 1).astype(int)
    out["diq010_borderline"] = (raw == 3).astype(int)
    out.loc[raw.isna(), ["diq010_yes", "diq010_borderline"]] = np.nan
    return out


def top_models_by_ranked_metric(n):
    for candidate in [cfg.OUT_METRICS / "tuning_summary.csv", cfg.OUT_METRICS / "model_comparison.csv"]:
        if candidate.exists():
            df = pd.read_csv(candidate)
            metric_col = "cv_auc" if "cv_auc" in df.columns else "AUC"
            return df.sort_values(metric_col, ascending=False)["Model" if "Model" in df.columns else "model"].head(n).tolist()
    return list(pu.get_model_registry().keys())[:n]


def class_weighted_estimator(name, ctor, y_train):
    """Returns an estimator instance configured for class weighting, or None
    if this model has no simple class-weighting knob and should be skipped
    for comparison (b).

    FIX (round 1): the original version of this function checked
    `"class_weight" in est.get_params()` / `"scale_pos_weight" in est.get_params()`
    to decide how to weight each model. That works for genuine
    scikit-learn estimators, whose get_params() always reflects the full
    __init__ signature regardless of what was explicitly set. It silently
    fails for CatBoostClassifier: get_params() on an untouched CatBoost
    instance returns {} (CatBoost only reports params that were *explicitly*
    passed to the constructor, not scikit-learn's "every constructor
    argument, defaulted or not" contract) -- so CatBoost was always routed
    into the "no class_weight/scale_pos_weight support -- skipped" branch,
    even though CatBoostClassifier supports THREE such knobs
    (class_weights, auto_class_weights, scale_pos_weight; verified via
    `inspect.signature(CatBoostClassifier.__init__)`).

    FIX (round 2): switching to `inspect.signature(type(est).__init__)`
    alone (round 1's fix) fixed CatBoost but silently broke XGBoost.
    XGBClassifier's __init__ is `(self, objective=..., **kwargs)` --
    scale_pos_weight is a real, working constructor argument, but it is
    absorbed into **kwargs and therefore invisible to
    inspect.signature(); only get_params() (which XGBoost populates from
    the full effective parameter set, unlike CatBoost) reports it. So
    neither get_params() alone nor inspect.signature() alone is correct
    for every model in this registry -- get_params() misses CatBoost's
    unset params, inspect.signature() misses XGBoost's **kwargs-hidden
    params. Taking the UNION of both is required and handles all three
    gradient-boosting libraries correctly:
      - CatBoost:  signature has {class_weights, auto_class_weights,
                    scale_pos_weight}; get_params() has none of them.
      - XGBoost:   get_params() has {scale_pos_weight}; signature has
                    none of them (hidden behind **kwargs).
      - LightGBM:  LGBMClassifier has an explicit, non-**kwargs __init__
                    with class_weight (no scale_pos_weight) -- visible in
                    both signature and get_params(), so either alone
                    would have worked, but the union is still correct.
    """
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    est = ctor()
    sig_params = set(inspect.signature(type(est).__init__).parameters.keys())
    getparams_params = set(est.get_params().keys()) if hasattr(est, "get_params") else set()
    ctor_params = sig_params | getparams_params

    if "class_weight" in ctor_params:
        est.set_params(class_weight="balanced")
        return est
    if "auto_class_weights" in ctor_params:  # CatBoost's built-in balanced-weighting shortcut
        est.set_params(auto_class_weights="Balanced")
        return est
    if "class_weights" in ctor_params:  # CatBoost, if auto_class_weights weren't available
        est.set_params(class_weights=[1.0, n_neg / n_pos])
        return est
    if "scale_pos_weight" in ctor_params:  # XGBoost, LightGBM, CatBoost
        est.set_params(scale_pos_weight=n_neg / n_pos)
        return est
    return None


def main():
    args = parse_args()
    train_df, _ = pu.load_model_ready_split()
    feature_cols, cont_cols, cat_cols = pu.get_feature_lists()
    X_train = train_df[feature_cols]
    y_train = train_df[cfg.OUTCOME_VAR]
    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=cfg.RANDOM_STATE)
    registry = pu.get_model_registry()

    # ---------------------------------------------------------------
    # (a) primary vs. +DIQ010
    # ---------------------------------------------------------------
    print("=== (a) Primary predictors vs. + DIQ010 (diagnosed diabetes) ===")
    diq_dummies = encode_diq010(train_df)
    X_train_plus_diq = pd.concat([X_train, diq_dummies], axis=1)
    cont_plus = cont_cols  # DIQ010 dummies are categorical-like (0/1), imputed via most_frequent
    cat_plus = cat_cols + list(diq_dummies.columns)

    rows_a = []
    top3 = [m.strip() for m in args.models.split(",")] if args.models else top_models_by_ranked_metric(3)
    for name in top3:
        ctor = registry[name]["estimator"]
        auc_primary, sd_primary = cv_auc(name, ctor, X_train, y_train, cont_cols, cat_cols, True, skf)
        auc_plus, sd_plus = cv_auc(name, ctor, X_train_plus_diq, y_train, cont_plus, cat_plus, True, skf)
        rows_a.append({"model": name, "cv_auc_primary": round(auc_primary, 4), "cv_auc_plus_diq010": round(auc_plus, 4),
                        "delta": round(auc_plus - auc_primary, 4)})
        print(f"  {name}: primary={auc_primary:.4f}  +DIQ010={auc_plus:.4f}  delta={auc_plus-auc_primary:+.4f}")
    pd.DataFrame(rows_a).to_csv(cfg.OUT_METRICS / "sensitivity_diq010.csv", index=False)

    # ---------------------------------------------------------------
    # (b) SMOTE vs. class-weighting
    # ---------------------------------------------------------------
    print(f"\n=== (b) SMOTE vs. class-weighting (top {args.top_n} models) ===")
    top_models = [m.strip() for m in args.models.split(",")] if args.models else top_models_by_ranked_metric(args.top_n)
    rows_b = []
    for name in top_models:
        ctor = registry[name]["estimator"]
        auc_smote, _ = cv_auc(name, ctor, X_train, y_train, cont_cols, cat_cols, True, skf)

        weighted_est = class_weighted_estimator(name, ctor, y_train)
        fit_params = None
        if weighted_est is None:
            # no class-weight option (Gradient Boosting): balanced sample weights instead
            weighted_est = ctor()
            fit_params = {"model__sample_weight": compute_sample_weight("balanced", y_train)}
        auc_cw, _ = cv_auc(name, ctor, X_train, y_train, cont_cols, cat_cols, False, skf,
                           estimator=weighted_est, params=fit_params)
        rows_b.append({"model": name, "cv_auc_smote": round(auc_smote, 4), "cv_auc_class_weight": round(auc_cw, 4),
                        "delta": round(auc_cw - auc_smote, 4)})
        print(f"  {name}: SMOTE={auc_smote:.4f}  class-weight={auc_cw:.4f}  delta={auc_cw-auc_smote:+.4f}")
    pd.DataFrame(rows_b).to_csv(cfg.OUT_METRICS / "sensitivity_smote_vs_classweight.csv", index=False)

    # ---------------------------------------------------------------
    # (c) lead-only vs. all 5 metals
    # ---------------------------------------------------------------
    print("\n=== (c) Blood lead only vs. all 5 metals ===")
    all_metals = cfg.PREDICTOR_VARS["metals"]
    lead_only = ["LBXBPB"]
    non_metal_features = [c for c in feature_cols if c not in all_metals]

    feature_cols_full = feature_cols  # already has all 5 metals
    feature_cols_lead = non_metal_features + lead_only
    cont_lead = [c for c in cont_cols if c not in all_metals] + lead_only
    cat_lead = [c for c in cat_cols if c not in all_metals]

    rows_c = []
    for name in top3:
        ctor = registry[name]["estimator"]
        auc_full, _ = cv_auc(name, ctor, X_train[feature_cols_full], y_train, cont_cols, cat_cols, True, skf)
        auc_lead, _ = cv_auc(name, ctor, X_train[feature_cols_lead], y_train, cont_lead, cat_lead, True, skf)
        rows_c.append({"model": name, "cv_auc_all_5_metals": round(auc_full, 4),
                        "cv_auc_lead_only": round(auc_lead, 4), "delta": round(auc_lead - auc_full, 4)})
        print(f"  {name}: all-5-metals={auc_full:.4f}  lead-only={auc_lead:.4f}  delta={auc_lead-auc_full:+.4f}")
    pd.DataFrame(rows_c).to_csv(cfg.OUT_METRICS / "sensitivity_metals_scope.csv", index=False)

    # ---------------------------------------------------------------
    # (d) FIX (new): harmonized (medication-adjusted) outcome label vs.
    # the original pipeline's lab/exam-only label. Same X, same model, same
    # StratifiedKFold OBJECT -- but cross_val_score re-derives the actual
    # fold row-membership from whichever y is passed in that call (since a
    # fresh, unfitted StratifiedKFold holds no precomputed split), so each
    # comparison is still correctly stratified on its own label. This is a
    # between-label, not a paired-row, comparison -- appropriate here since
    # the point is "how much does the label definition itself move apparent
    # performance," not a hypothesis test of two models on identical rows.
    # ---------------------------------------------------------------
    print("\n=== (d) Outcome-definition arms: harmonized vs. lab-only, no lipid clause, first BP reading ===")
    if "mets_status_labonly" not in train_df.columns:
        print("  [SKIPPED] mets_status_labonly column not found in model_ready.csv -- "
              "re-run 02_clean_and_outcome.py / 03_feature_engineering.py to regenerate it.")
    else:
        y_train_labonly = train_df["mets_status_labonly"]
        y_nolipid = train_df["mets_status_nolipidmed"]
        y_bpfirst = train_df["mets_status_bpfirst"]
        rows_d = []
        for name in top3:
            ctor = registry[name]["estimator"]
            auc_harmonized, _ = cv_auc(name, ctor, X_train, y_train, cont_cols, cat_cols, True, skf)
            auc_labonly, _ = cv_auc(name, ctor, X_train, y_train_labonly, cont_cols, cat_cols, True, skf)
            auc_nolipid, _ = cv_auc(name, ctor, X_train, y_nolipid, cont_cols, cat_cols, True, skf)
            auc_bpfirst, _ = cv_auc(name, ctor, X_train, y_bpfirst, cont_cols, cat_cols, True, skf)
            rows_d.append({
                "model": name,
                "cv_auc_harmonized_label": round(auc_harmonized, 4),
                "cv_auc_labonly_label": round(auc_labonly, 4),
                "delta": round(auc_harmonized - auc_labonly, 4),
                "cv_auc_nolipidmed_label": round(auc_nolipid, 4),
                "cv_auc_bpfirst_label": round(auc_bpfirst, 4),
                "harmonized_prevalence_pct": round(y_train.mean() * 100, 2),
                "labonly_prevalence_pct": round(y_train_labonly.mean() * 100, 2),
                "nolipidmed_prevalence_pct": round(y_nolipid.mean() * 100, 2),
                "bpfirst_prevalence_pct": round(y_bpfirst.mean() * 100, 2),
            })
            print(f"  {name}: harmonized={auc_harmonized:.4f}  lab-only={auc_labonly:.4f}  "
                  f"no-lipid-clause={auc_nolipid:.4f}  first-BP-reading={auc_bpfirst:.4f}")
        pd.DataFrame(rows_d).to_csv(cfg.OUT_METRICS / "sensitivity_outcome_definition.csv", index=False)
        print(f"  NOTE: this compares apparent AUC under two label definitions on the SAME features/model; "
              f"it is not a claim that one AUC is 'better' in isolation -- prevalence differs between the two "
              f"labels ({y_train.mean()*100:.2f}% vs {y_train_labonly.mean()*100:.2f}%), which alone can shift "
              f"AUC somewhat. See outputs/metrics/outcome_definition_comparison.csv for the underlying "
              f"reclassification counts driving this.")

    print(f"\nSaved all sensitivity tables to {cfg.OUT_METRICS}")


if __name__ == "__main__":
    main()
