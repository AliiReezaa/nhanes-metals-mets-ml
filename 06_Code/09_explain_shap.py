"""
Step 12 -- Explainable AI (SHAP + permutation importance)

Explains ONE model at a time -- by default, the best-performing tree-based
model on the test set (tree models get exact, fast SHAP via TreeExplainer;
non-tree models would need the much slower Kernel/Permutation explainer).
Override with --model "Model Name" to explain a different one.

Uses the actual fitted pipeline saved by 07_train_models.py -- SHAP values
are computed for the model you actually evaluated, not a fresh ad hoc refit
on the full dataset (that was a wrinkle in the prior notebook: several
cells refit LightGBM on all the data just for SHAP, which is a different
model than the one whose metrics got reported).

Also computes permutation importance (model-agnostic) as a cross-check:
if SHAP and permutation importance agree on the top features, that's a
much stronger signal than either one alone.

REBUILD NOTE / FIX (two changes from the original 09_explain_shap.py):

1. Column layout. The original script assumed the ColumnTransformer's
   output columns were simply `cont_cols + cat_cols`. That stopped being
   true the moment pipeline_utils.py added the missing-value flag block
   (see that module's docstring, point 1): the real output layout is now
   `cont_cols + [missingindicator_<c> for c in flagged cont_cols] + cat_cols`. Silently keeping the old
   `transformed_cols = cont_cols + cat_cols` here would have handed SHAP's
   arrays column labels that no longer line up with the columns those
   arrays actually contain (an N-column array labeled with fewer names,
   silently mis-assigning every label after the continuous block). This
   script now gets the layout from the single source of truth,
   pipeline_utils.transformed_feature_names(), the same function
   pipeline_utils.build_preprocessor's docstring points to.

2. "Best model" overclaiming. 08_evaluate.py's paired bootstrap test
   frequently finds several models statistically indistinguishable from
   the top point-estimate AUC (a 2,229-row test set has real sampling
   noise). This script now reads
   outputs/metrics/model_comparison_significance.csv (if present) and
   prints, alongside the chosen model's SHAP results, how many other
   models are statistically tied with it -- so a reader of the SHAP
   outputs is not left assuming the explained model is unambiguously the
   single best performer.

Output: outputs/figures/shap_*.png
        outputs/metrics/shap_importance.csv
        outputs/metrics/permutation_importance.csv

Run:
    python 06_Code/09_explain_shap.py
    python 06_Code/09_explain_shap.py --model "Random Forest"
"""
import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import pipeline_utils as pu

TREE_MODELS = {"Decision Tree", "Random Forest", "Gradient Boosting", "LightGBM", "XGBoost", "CatBoost"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="model name to explain (default: best tree model by test AUC)")
    ap.add_argument("--n-perm-repeats", type=int, default=10)
    return ap.parse_args()


def candidate_models(explicit_name):
    """Returns an ordered list of model names to try. If the user passed
    --model explicitly, that's the only candidate. Otherwise: all tree
    models ranked by test AUC, then everything else ranked by test AUC --
    so if the top choice fails (e.g. a SHAP/XGBoost version mismatch on
    base_score parsing, a real issue hit while building this), we fall back
    to the next-best rather than hard-failing."""
    if explicit_name:
        return [explicit_name]
    cmp_path = cfg.OUT_METRICS / "model_comparison.csv"
    if not cmp_path.exists():
        raise SystemExit("outputs/metrics/model_comparison.csv not found -- run 08_evaluate.py first, "
                          "or pass --model explicitly.")
    df = df_full = pd.read_csv(cmp_path).sort_values("AUC", ascending=False)
    tree_first = pd.concat([df[df["Model"].isin(TREE_MODELS)], df[~df["Model"].isin(TREE_MODELS)]])
    return tree_first["Model"].tolist()


def print_tie_caveat(model_name):
    """FIX: surface 08_evaluate.py's paired-bootstrap tie information right
    where a reader will see which model got explained, so this script never
    implies "this is unambiguously the best model" when the significance
    test says otherwise."""
    sig_path = cfg.OUT_METRICS / "model_comparison_significance.csv"
    if not sig_path.exists():
        print("  [NOTE] outputs/metrics/model_comparison_significance.csv not found -- "
              "run 08_evaluate.py to get the paired-bootstrap tie test before treating "
              "this model's AUC ranking as definitive.")
        return
    sig_df = pd.read_csv(sig_path)
    n_tied = int(sig_df["statistically_indistinguishable_from_top"].sum())
    this_row = sig_df[sig_df["model"] == model_name]
    if len(this_row) and not bool(this_row.iloc[0]["is_top_model"]) and not bool(this_row.iloc[0]["statistically_indistinguishable_from_top"]):
        print(f"  [CAVEAT] '{model_name}' is NOT the top point-estimate-AUC model and IS statistically "
              f"distinguishable from it (paired bootstrap, 08_evaluate.py) -- explaining it anyway because "
              f"it was requested via --model or was the best available TREE model for exact SHAP.")
    print(f"  [CAVEAT] {n_tied} of {len(sig_df)} models (per 08_evaluate.py's paired bootstrap test) are "
          f"statistically indistinguishable from the top-ranked model at the 95% level -- the feature "
          f"importances below describe '{model_name}' specifically, not a uniquely-best model.")


def _dynamic_plot_size(feature_labels, n_rows=None, row_height=0.45, base_height=2.2,
                        char_width=0.105, min_width=12.0, max_width=22.0):
    """SHAP's own auto-sizing hardcodes an 8-inch-wide figure no matter how
    long the feature labels are (see plot_size='auto' in shap/plots/_beeswarm.py
    and auto_size_plot in shap/plots/_decision.py). With full clinical names
    like 'Current smoking intensity (0=none, 1=some days, 2=every day)' that
    leaves the actual chart squeezed into a sliver on the right and clips the
    x-axis title / colorbar. Size the figure from the longest label instead."""
    n_rows = n_rows or len(feature_labels)
    max_label_len = max((len(str(lbl)) for lbl in feature_labels), default=20)
    width = min(max_width, max(min_width, 6.5 + max_label_len * char_width))
    height = n_rows * row_height + base_height
    return (width, height)


def extract_shap_positive_class(shap_values):
    """Normalizes SHAP's binary-classification output across library/version
    differences (list-of-arrays, 3D array, or already 2D) into one 2D array
    of shape (n_samples, n_features) for the positive class."""
    if isinstance(shap_values, list):
        return shap_values[1]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def build_raw_display(X_test, X_test_t, cont_cols, cat_cols, transformed_cols, display_cols):
    """FIX: raw/natural-units companion frame for the SHAP plots, updated for
    the new cont -> flags -> cat layout (see module docstring, FIX 1).

    - continuous columns: original, un-imputed, un-scaled values from
      X_test (so plots show '55 years', not a StandardScaler z-score).
    - missingindicator_<c> columns: these have no separate 'raw' form --
      MissingIndicator's own 0/1 output IS already the natural-units value
      (1 = the value was missing and imputed), so they're taken directly from the transformed
      frame X_test_t, not reconstructed from X_test.
    - categorical/binary columns: original values from X_test (only mode-
      imputation happens to these downstream; the original code already
      used the raw values here and that's unchanged).
    """
    flag_cols = [f"missingindicator_{c}" for c in pu.get_flag_cols(cont_cols)]
    parts = [X_test[cont_cols], X_test_t[flag_cols], X_test[cat_cols]]
    raw = pd.concat(parts, axis=1)[transformed_cols]  # enforce exact column order
    raw.columns = display_cols
    return raw.round(2)


def main():
    args = parse_args()
    candidates = candidate_models(args.model)

    _, test_df = pu.load_model_ready_split()
    feature_cols, cont_cols, cat_cols = pu.get_feature_lists()
    X_test = test_df[feature_cols]
    y_test = test_df[cfg.OUTCOME_VAR]
    # FIX: was `cont_cols + cat_cols`; the real ColumnTransformer output
    # layout also has the missing-flag block in between -- see
    # pipeline_utils.build_preprocessor / transformed_feature_names.
    transformed_cols = pu.transformed_feature_names(cont_cols, cat_cols)

    model_name = pipe = X_test_t = explainer = shap_values = expected_value = is_tree = None
    for candidate in candidates:
        model_path = cfg.OUT_MODELS / f"{candidate.replace(' ', '_')}.joblib"
        if not model_path.exists():
            print(f"  [skip] {candidate}: no saved model at {model_path}")
            continue
        print(f"Trying: {candidate}")
        try:
            pipe_c = joblib.load(model_path)
            preprocessor = pipe_c.named_steps["preprocess"]
            estimator = pipe_c.named_steps["model"]
            X_test_t_c = preprocessor.transform(X_test)
            X_test_t_c = pd.DataFrame(np.asarray(X_test_t_c), columns=transformed_cols, index=X_test.index)

            is_tree_c = candidate in TREE_MODELS
            if is_tree_c:
                explainer_c = shap.TreeExplainer(estimator)
                shap_values_c = extract_shap_positive_class(explainer_c.shap_values(X_test_t_c))
                expected_value_c = explainer_c.expected_value
                if isinstance(expected_value_c, (list, np.ndarray)):
                    expected_value_c = np.asarray(expected_value_c).flatten()[-1]
            else:
                print("  [NOTE] non-tree model -- using model-agnostic Explainer on a background sample "
                      "of 100 rows (slower).")
                background = X_test_t_c.sample(min(100, len(X_test_t_c)), random_state=cfg.RANDOM_STATE)
                explainer_c = shap.Explainer(estimator.predict_proba, background)
                sv = explainer_c(X_test_t_c)
                shap_values_c = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values
                expected_value_c = sv.base_values[0] if np.ndim(sv.base_values) else sv.base_values

            model_name, pipe, X_test_t = candidate, pipe_c, X_test_t_c
            explainer, shap_values, expected_value, is_tree = explainer_c, shap_values_c, expected_value_c, is_tree_c
            print(f"Explaining model: {model_name}")
            print_tie_caveat(model_name)
            break
        except Exception as e:
            print(f"  [FAILED] {candidate}: {type(e).__name__}: {e} -- trying next candidate")
            continue

    if model_name is None:
        raise SystemExit("Could not build a SHAP explainer for any saved model. "
                          "Run 07_train_models.py first, or check the errors above.")

    # Two "display" copies with clinical names attached:
    #   X_test_display     -- model-space (imputed/scaled) values. Kept only
    #                         because shap_values indexing assumes this
    #                         column order; not used for plotting anymore.
    #   X_test_raw_display -- the *original*, unscaled values in their
    #                         natural units (years, kg/m^2, ug/L, ...) for
    #                         cont/cat columns, and the flag's own 0/1 value
    #                         for the missing-indicator columns (see
    #                         build_raw_display). Only the *displayed* value
    #                         changes here; shap_values (computed in
    #                         model/scaled space) is untouched, so the SHAP
    #                         numbers are still exact.
    display_cols = cfg.label_features(transformed_cols)
    X_test_display = X_test_t.copy()
    X_test_display.columns = display_cols

    X_test_raw_display = build_raw_display(X_test, X_test_t, cont_cols, cat_cols, transformed_cols, display_cols)

    shap_df = pd.DataFrame(shap_values, columns=transformed_cols)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    shap_importance_out = mean_abs_shap.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature_code"})
    shap_importance_out["clinical_name"] = shap_importance_out["feature_code"].map(cfg.label_feature)
    shap_importance_out.to_csv(cfg.OUT_METRICS / "shap_importance.csv", index=False)
    print("\nTop 10 features by mean |SHAP|:")
    print(shap_importance_out.head(10).to_string(index=False))

    # --- Summary bar plot ---
    # plot_size overrides shap's hardcoded 8"-wide default (see
    # _dynamic_plot_size above) -- this is what actually fixes the squeezed
    # plot area / clipped x-axis title, not plt.figure()+tight_layout(),
    # which summary_plot ignores and overwrites internally.
    max_display = min(20, len(display_cols))
    top_labels_for_sizing = mean_abs_shap.head(max_display).index.map(cfg.label_feature)
    bar_size = _dynamic_plot_size(top_labels_for_sizing, n_rows=max_display)
    shap.summary_plot(shap_values, X_test_raw_display, plot_type="bar",
                       max_display=max_display, plot_size=bar_size, show=False)
    plt.title(f"SHAP Feature Importance -- {model_name}")
    plt.savefig(cfg.OUT_FIGURES / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Beeswarm ---
    beeswarm_size = _dynamic_plot_size(top_labels_for_sizing, n_rows=max_display)
    shap.summary_plot(shap_values, X_test_raw_display, max_display=max_display,
                       plot_size=beeswarm_size, show=False)
    plt.title(f"SHAP Beeswarm -- {model_name}")
    plt.savefig(cfg.OUT_FIGURES / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Dependence plots, top 5 features (select by raw code, plot with clinical label) ---
    top5_codes = mean_abs_shap.head(5).index.tolist()
    for code in top5_codes:
        label = cfg.label_feature(code)
        shap.dependence_plot(label, shap_values, X_test_raw_display, interaction_index=None, show=False)
        plt.title(f"SHAP Dependence -- {label}")
        plt.tight_layout()
        safe_name = code.replace("/", "_")
        plt.savefig(cfg.OUT_FIGURES / f"shap_dependence_{safe_name}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # --- Metals panel: all 5 blood metals together, regardless of global rank ---
    # The top-5-by-importance dependence plots above are driven by whichever
    # features happen to matter most overall, which can leave out metals that
    # rank low on THIS outcome but are still the core exposures of the study
    # (e.g. lead/cadmium/manganese did not make the global top 5 in the real
    # run). Readers evaluating the metals hypothesis specifically need to see
    # all 5 side by side on a common, comparable layout, not just whichever
    # ones happened to be most predictive.
    metal_codes = [c for c in cfg.PREDICTOR_VARS["metals"] if c in transformed_cols]
    if metal_codes:
        fig, axes = plt.subplots(1, len(metal_codes), figsize=(5 * len(metal_codes), 4.5))
        axes = np.atleast_1d(axes)
        for ax, code in zip(axes, metal_codes):
            label = cfg.label_feature(code)
            rank = int(mean_abs_shap.index.get_loc(code)) + 1
            shap.dependence_plot(label, shap_values, X_test_raw_display, interaction_index=None,
                                  ax=ax, show=False)
            ax.set_title(f"{label}\n(overall importance rank {rank}/{len(transformed_cols)})", fontsize=10)
        plt.suptitle(f"SHAP Dependence -- All 5 Blood Metals -- {model_name}", fontsize=13, fontweight="bold", y=1.04)
        plt.tight_layout()
        plt.savefig(cfg.OUT_FIGURES / "shap_metals_panel.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {cfg.OUT_FIGURES / 'shap_metals_panel.png'}")

    # --- Force plot, single instance ---
    # matplotlib=True builds its own figure sized by `figsize` (a
    # pre-created plt.figure() call is ignored, so pass sizing directly).
    # Long clinical labels + many decimal places overlapped badly at shap's
    # default figsize=(20,3)/text_rotation=0 -- widen it and angle the text.
    i = 0
    shap.force_plot(expected_value, shap_values[i, :], X_test_raw_display.iloc[i, :],
                     matplotlib=True, show=False, figsize=(26, 5), text_rotation=45,
                     contribution_threshold=0.06)
    plt.savefig(cfg.OUT_FIGURES / "shap_force_single.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Decision plot, sample of 50 ---
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    sample_idx = rng.choice(X_test_t.shape[0], size=min(50, X_test_t.shape[0]), replace=False)
    decision_size = _dynamic_plot_size(display_cols, n_rows=len(display_cols))
    plt.figure(figsize=decision_size)
    shap.decision_plot(expected_value, shap_values[sample_idx], X_test_raw_display.iloc[sample_idx],
                        auto_size_plot=False, show=False)
    plt.title(f"SHAP Decision Plot (n=50) -- {model_name}")
    plt.savefig(cfg.OUT_FIGURES / "shap_decision.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved SHAP figures to {cfg.OUT_FIGURES}")

    # --- Permutation importance (model-agnostic cross-check) ---
    print(f"\nComputing permutation importance ({args.n_perm_repeats} repeats, scoring=roc_auc)...")
    perm = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                   n_repeats=args.n_perm_repeats, random_state=cfg.RANDOM_STATE, n_jobs=1)
    perm_df = pd.DataFrame({
        "feature_code": feature_cols,
        "clinical_name": cfg.label_features(feature_cols),
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
    perm_df.to_csv(cfg.OUT_METRICS / "permutation_importance.csv", index=False)
    print("\nTop 10 by permutation importance:")
    print(perm_df.head(10).to_string(index=False))

    # NOTE: permutation_importance above uses `pipe` (the FULL pipeline
    # including preprocess+SMOTE+model) and `feature_cols` (the original,
    # pre-transform predictor names) -- it permutes each raw predictor once
    # and re-scores through the whole pipeline, so it is unaffected by the
    # missing-indicator layout change and needed no fix. It naturally has no
    # separate entries for the missingindicator_* columns (those don't exist
    # at the raw-feature level this function operates on) -- SHAP's
    # importances DO include them, which is expected and worth noting when
    # comparing the two top-10 lists below.
    shap_top10 = set(mean_abs_shap.head(10).index)
    perm_top10 = set(perm_df.head(10)["feature_code"])
    overlap = shap_top10 & perm_top10
    print(f"\nAgreement check: {len(overlap)}/10 features appear in both SHAP and permutation-importance top 10.")
    print(f"  In both: {sorted(cfg.label_feature(c) for c in overlap)}")


if __name__ == "__main__":
    main()
