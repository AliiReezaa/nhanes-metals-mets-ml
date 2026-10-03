"""
Step 5 -- Train/Test Split (before any fitting)

This script does ONLY the split -- no imputing, no encoding beyond what
03_feature_engineering.py already did, no scaling, no SMOTE. It saves the
SEQN values belonging to each partition so every later script loads the
same split from disk instead of re-splitting (re-splitting with a different
random call anywhere downstream is a common, subtle way to silently leak
test rows into training).

Test set is set aside here and not touched again until 08_evaluate.py.

REBUILD NOTE / FIX: the numeric predictor-collinearity screen (Pearson r
among the encoded predictors, used to sanity-check that nothing above the
0.7 threshold survived into the final 33-predictor set) used to run in
03_feature_engineering.py on the FULL pre-split dataframe -- i.e. on
train+test combined. That is a real, if small, breach of this project's own
"nothing touches the test partition before 08_evaluate.py" rule: it is a
statistic, and a modeling decision (dropping a collinear predictor) could in
principle have been made from it, using test-set information. It has been
moved here, AFTER the split, and now runs on the training partition only.

Output: data_processed/train_seqn.csv
        data_processed/test_seqn.csv
        outputs/metrics/collinearity_flags.csv

Run:
    python 06_Code/04_split.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

TEST_SIZE = 0.20


def collinearity_screen(X: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    numeric = X.select_dtypes(include=[np.number])
    corr = numeric.corr(method="pearson", min_periods=30)
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) >= threshold:
                pairs.append({"var_1": cols[i], "var_2": cols[j], "pearson_r": round(r, 3)})
    if not pairs:
        return pd.DataFrame(columns=["var_1", "var_2", "pearson_r"])
    return pd.DataFrame(pairs).sort_values("pearson_r", key=abs, ascending=False)


def main():
    in_path = cfg.DATA_PROC_DIR / "model_ready.csv"
    df = pd.read_csv(in_path, low_memory=False)
    print(f"Loaded {in_path.name}: {df.shape[0]} rows")

    train_df, test_df = train_test_split(
        df[[cfg.ID_VAR, cfg.OUTCOME_VAR]],
        test_size=TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=df[cfg.OUTCOME_VAR],
    )

    print(f"\nTrain: {len(train_df)} rows, MetS prevalence {train_df[cfg.OUTCOME_VAR].mean()*100:.2f}%")
    print(f"Test:  {len(test_df)} rows, MetS prevalence {test_df[cfg.OUTCOME_VAR].mean()*100:.2f}%")

    overlap = set(train_df[cfg.ID_VAR]) & set(test_df[cfg.ID_VAR])
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} SEQNs appear in both train and test"
    print("\nVerified: zero overlap between train and test SEQNs.")

    train_df[[cfg.ID_VAR]].to_csv(cfg.DATA_PROC_DIR / "train_seqn.csv", index=False)
    test_df[[cfg.ID_VAR]].to_csv(cfg.DATA_PROC_DIR / "test_seqn.csv", index=False)
    print(f"\nSaved: {cfg.DATA_PROC_DIR / 'train_seqn.csv'} ({len(train_df)} SEQNs)")
    print(f"Saved: {cfg.DATA_PROC_DIR / 'test_seqn.csv'} ({len(test_df)} SEQNs)")

    # FIX: collinearity screen now runs on the training partition only.
    print("\nCollinearity screen (|Pearson r| >= 0.7 among numeric predictors, TRAINING PARTITION ONLY):")
    predictors = pd.read_csv(cfg.OUT_METRICS / "predictors_used.csv")["predictor"].tolist()
    train_full = df[df[cfg.ID_VAR].isin(train_df[cfg.ID_VAR])]
    flags = collinearity_screen(train_full[predictors])
    if len(flags):
        print(flags.to_string(index=False))
    else:
        print("  none found (as expected -- this screens the FINAL 33-predictor")
        print("  set for anything that slipped through; predictors already")
        print("  excluded on collinearity grounds, e.g. BMXHIP below, never")
        print("  reach this list in the first place)")

    # FIX (found during the "check every step" precision review): the comment
    # in config.py justifying BMXHIP's exclusion ("r=0.935 with BMXBMI ...
    # and 85.8% missing") was a NUMBER ASSERTED IN A COMMENT, not a number
    # this pipeline ever actually computed -- collinearity_screen() above
    # only screens the FINAL predictor list from predictors_used.csv, and
    # BMXHIP was excluded from PREDICTOR_VARS in config.py before that list
    # was ever built, so the screen could never have produced this number.
    # It silently could not be reproduced from anything in this codebase.
    # Fixed by explicitly re-deriving it here, from real data, on the
    # training partition, BEFORE any exclusion decision is asserted as fact
    # -- this is now a real, reproducible, pre-registered candidate-variable
    # check rather than a hardcoded, unverifiable claim.
    print("\nCandidate-variable screen (BMXHIP vs. BMXBMI, TRAINING PARTITION ONLY,")
    print("re-derived from raw data -- this is what actually justifies excluding"
          " BMXHIP from the predictor set):")
    with_outcome = pd.read_csv(cfg.DATA_PROC_DIR / "nhanes_with_outcome.csv", low_memory=False,
                                usecols=[cfg.ID_VAR, "BMXHIP", "BMXBMI"])
    hip_train = with_outcome[with_outcome[cfg.ID_VAR].isin(train_df[cfg.ID_VAR])]
    hip_missing_pct = hip_train["BMXHIP"].isna().mean() * 100
    hip_pairs = hip_train[["BMXHIP", "BMXBMI"]].dropna()
    hip_r = hip_pairs["BMXHIP"].corr(hip_pairs["BMXBMI"])
    print(f"  BMXHIP missing: {hip_missing_pct:.1f}% of training partition (n={len(hip_train)})")
    print(f"  Pearson r(BMXHIP, BMXBMI) on the {len(hip_pairs)} training rows with both observed: {hip_r:.4f}")
    candidate_row = pd.DataFrame([{
        "var_1": "BMXHIP", "var_2": "BMXBMI", "pearson_r": round(hip_r, 4),
        "pct_missing_var_1_train": round(hip_missing_pct, 2),
        "decision": "excluded from predictor set (PREDICTOR_VARS in config.py): "
                    "redundant with BMXBMI and majority-missing",
    }])
    flags_out = pd.concat([flags.assign(pct_missing_var_1_train=pd.NA, decision=pd.NA), candidate_row],
                           ignore_index=True)
    flags_out.to_csv(cfg.OUT_METRICS / "collinearity_flags.csv", index=False)
    print(f"\nSaved: {cfg.OUT_METRICS / 'collinearity_flags.csv'}")

    print("\nFrom here on: the test set is off-limits until 08_evaluate.py.")


if __name__ == "__main__":
    main()
