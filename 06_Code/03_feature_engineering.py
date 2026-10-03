"""
Step 4 -- Feature Engineering and Predictor Set

Reads data_processed/nhanes_with_outcome.csv (from 02_clean_and_outcome.py) and:
  1. Verifies every predictor in config.PREDICTOR_VARS has an explicit encoding
     rule (fails loudly if anything was missed -- no variable is encoded by
     silent default)
  2. Recodes binary yes/no variables (raw 1/2) to 1/0
  3. Recodes the one variable whose raw ordinal direction runs backwards
     (SMQ040) so higher = more exposure, consistent with everything else
  4. One-hot encodes true unordered categoricals (RIDRETH1, DMDMARTL), with
     an explicit "missing" indicator column rather than silently folding
     missing into the reference category
  5. Renames the two 2-level variables to clearly-labeled binary columns
  6. Leaves continuous variables and NaNs untouched -- imputation and scaling
     happen later, inside the training pipeline, fit per-fold (Step 6). Doing
     that here would leak test-set information into training.

REBUILD NOTE: the numeric collinearity screen that used to run here (on the
full pre-split dataframe) has been MOVED to 04_split.py, and now runs on the
training partition only, after the split -- see the FIX note there. Computing
correlations on train+test combined was a real, if small, breach of the
project's own leakage-safety rule; this file no longer touches anything that
depends on the full sample's numeric distribution.

Output: data_processed/model_ready.csv
        outputs/metrics/predictors_used.csv

Run:
    python scripts/03_feature_engineering.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg


def all_predictors():
    return sorted(set(sum(cfg.PREDICTOR_VARS.values(), [])))


def check_encoding_completeness():
    predictors = set(all_predictors())
    covered = (
        set(cfg.BINARY_YESNO_1_2)
        | set(cfg.ORDINAL_AS_IS)
        | set(cfg.ORDINAL_REMAP.keys())
        | set(cfg.NOMINAL_ONEHOT)
        | set(cfg.BINARY_RENAME.keys())
        | set(cfg.CONTINUOUS_AS_IS)
    )
    missing = predictors - covered
    extra = covered - predictors
    if missing:
        raise SystemExit(f"ENCODING GAP: these predictors have no encoding rule in config.py: {sorted(missing)}")
    if extra:
        print(f"[NOTE] encoding rules exist for variables not in PREDICTOR_VARS (harmless, just unused): {sorted(extra)}")
    print(f"Encoding completeness check passed: all {len(predictors)} predictors covered.")


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=df.index)
    feature_cols = []

    # binary yes/no (1=Yes,2=No -> 1/0)
    for col in cfg.BINARY_YESNO_1_2:
        s = df[col].map({1: 1, 2: 0})
        out[col] = s
        feature_cols.append(col)

    # ordinal, kept as-is
    for col in cfg.ORDINAL_AS_IS:
        out[col] = df[col]
        feature_cols.append(col)

    # ordinal, remapped
    for col, mapping in cfg.ORDINAL_REMAP.items():
        out[col] = df[col].map(mapping)
        feature_cols.append(col)

    # binary rename
    for col, spec in cfg.BINARY_RENAME.items():
        new_name = spec["new_name"]
        out[new_name] = (df[col] == spec["positive_raw_value"]).astype("Int64")
        out.loc[df[col].isna(), new_name] = pd.NA
        feature_cols.append(new_name)

    # continuous, untouched (includes ALQ_EVER, already a clean 0/1 built in
    # 02_clean_and_outcome.py from the harmonized ALQ101/ALQ111 source)
    for col in cfg.CONTINUOUS_AS_IS:
        out[col] = df[col]
        feature_cols.append(col)

    # nominal one-hot, with explicit missing indicator (dummy_na=True) --
    # safe to do globally (not fold-by-fold) because these are fixed,
    # known category sets, not data-driven groupings, so no distributional
    # information leaks from test into train by doing this before the split.
    for col in cfg.NOMINAL_ONEHOT:
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True, drop_first=True)
        dummies = dummies.astype(int)
        out = pd.concat([out, dummies], axis=1)
        feature_cols.extend(dummies.columns.tolist())
        print(f"  {col}: one-hot encoded into {list(dummies.columns)} (reference category = lowest code, dropped)")

    return out, feature_cols


def main():
    in_path = cfg.DATA_PROC_DIR / "nhanes_with_outcome.csv"
    df = pd.read_csv(in_path, low_memory=False)
    print(f"Loaded {in_path.name}: {df.shape[0]} rows, {df.shape[1]} cols")

    check_encoding_completeness()

    print("\nEncoding predictors:")
    X, feature_cols = encode(df)
    print(f"Final predictor matrix: {X.shape[0]} rows, {len(feature_cols)} columns")

    # Assemble the file passed to downstream scripts: ID/cycle/outcome +
    # MetS components & survey design vars (kept for Table 1 / sensitivity
    # analyses, NOT used as predictors) + DIQ010 (sensitivity-only) + features.
    # FIX: also carries the label-only MetS score/status, the individual
    # medication flags, and the raw medication variables, so 10_sensitivity.py
    # can directly quantify the old-vs-new outcome definition's downstream
    # effect on model performance, and so Table 1 / STROBE reporting has
    # everything needed for a full audit trail.
    keep_context_cols = (
        [cfg.ID_VAR, cfg.CYCLE_VAR, cfg.OUTCOME_VAR]
        + cfg.METS_COMPONENT_VARS
        + cfg.MEDICATION_VARS_FOR_LABEL
        + ["mets_waist", "mets_trigly", "mets_hdl", "mets_bp", "mets_glu", "mets_score"]
        + ["mets_trigly_labonly", "mets_hdl_labonly", "mets_bp_labonly", "mets_glu_labonly",
           "mets_score_labonly", "mets_status_labonly"]
        + ["on_bp_med", "on_lipid_med", "on_glu_med"]
        + [cfg.DIQ_SENSITIVITY_VAR]
        + [cfg.SURVEY_WEIGHT_MEC, cfg.SURVEY_WEIGHT_FASTING, cfg.METALS_WEIGHT_VAR,
           cfg.SURVEY_PSU, cfg.SURVEY_STRATA]
    )
    keep_context_cols = [c for c in keep_context_cols if c in df.columns]

    model_ready = pd.concat([df[keep_context_cols].reset_index(drop=True), X.reset_index(drop=True)], axis=1)

    out_path = cfg.DATA_PROC_DIR / "model_ready.csv"
    model_ready.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({model_ready.shape[0]} rows, {model_ready.shape[1]} cols)")

    pred_path = cfg.OUT_METRICS / "predictors_used.csv"
    pd.Series(feature_cols, name="predictor").to_csv(pred_path, index=False)
    print(f"Saved predictor list ({len(feature_cols)} columns): {pred_path}")

    # Sanity printout: value ranges after encoding
    print("\nSanity check -- encoded value ranges (should look right, not raw NHANES codes):")
    for col in ["sex_female", "exam_summer", "SMQ040", "ALQ_EVER"] + cfg.BINARY_YESNO_1_2:
        if col in X.columns:
            print(f"  {col}: {sorted(X[col].dropna().unique().tolist())}")


if __name__ == "__main__":
    main()
