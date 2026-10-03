"""
Step 11 -- Model Evaluation

This is the FIRST script that touches the held-out test set
(data_processed/test_seqn.csv) since 04_split.py set it aside. It is used
exactly once, here, purely for reporting.

For every fitted model in outputs/models/*.joblib:
  - predict_proba on the test set
  - apply the OOF-derived threshold from outputs/metrics/thresholds.json
    (not a threshold picked on the test set itself)
  - compute AUROC, AUPRC, accuracy, sensitivity, specificity, PPV, NPV, F1,
    kappa and Brier score, each with a 1,000-resample percentile bootstrap
    95% CI (2026-10 audit, P14: the Methods promised intervals for the
    threshold metrics, but only AUROC had one)
  - also report metrics at a fixed 0.5 cutoff for transparency

REBUILD NOTE / FIX: the original pipeline picked a single "best" model by the
top test-AUC point estimate and treated the resulting 1-through-11 ranking as
reliable, even though the top handful of models sat within about 0.005 AUC of
each other -- well inside plausible sampling noise for a test set of about 2,200 participants.
This script compares the top-ranked model with every other model on the
same test participants: the p-value comes from DeLong's test for correlated
AUROCs (2026-10 audit, C8: the earlier bootstrap p-value could not resolve
values below about 0.002 but was printed as "p < 0.001"), and the 95% CI of
the AUROC difference from a paired bootstrap (same resampled indices for both
models in each draw). Models whose difference CI includes zero are reported
as statistically indistinguishable from the top model. Downstream scripts (09_explain_shap.py in particular) use this to
avoid overclaiming that whichever model won by a coin-flip-sized margin is
definitively "the best."

Output: outputs/metrics/model_comparison.csv
        outputs/metrics/model_comparison_significance.csv
        outputs/figures/roc_curves.png
        outputs/figures/calibration_curves.png

Run:
    python 06_Code/08_evaluate.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score, confusion_matrix,
    accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score,
    brier_score_loss,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import pipeline_utils as pu
import stats_utils as su

N_BOOTSTRAP = 1000


def metrics_at_cutoff(y_true, y_prob, cutoff):
    y_pred = (y_prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = precision_score(y_true, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "Cutoff": cutoff,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV": ppv,
        "NPV": npv,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "Youden": sens + spec - 1,
    }


def bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, seed=None):
    seed = cfg.RANDOM_STATE if seed is None else seed
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_auc_diff(y_true, prob_top, prob_other, n_boot=N_BOOTSTRAP, seed=None):
    """Paired comparison of two models on the same participants: the
    difference and its paired-bootstrap 95% CI (same resampled indices for
    both models in every draw), and the two-sided DeLong p-value."""
    seed = cfg.RANDOM_STATE if seed is None else seed
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    prob_top = np.asarray(prob_top)
    prob_other = np.asarray(prob_other)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(roc_auc_score(yt, prob_top[idx]) - roc_auc_score(yt, prob_other[idx]))
    diffs = np.array(diffs)
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    _, _, observed, _, _, p = su.delong_test(y_true, prob_top, prob_other)
    return observed, ci_lo, ci_hi, p


def bootstrap_metric_cis(y_true, y_prob, cutoff, n_boot=N_BOOTSTRAP, seed=None):
    """Percentile bootstrap 95% CIs for AUPRC, Brier and every threshold
    metric at the given cutoff, from one shared set of resamples."""
    seed = cfg.RANDOM_STATE if seed is None else seed
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    n = len(y_true)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = metrics_at_cutoff(yt, yp, cutoff)
        m["AUPRC"] = average_precision_score(yt, yp)
        m["Brier"] = brier_score_loss(yt, yp)
        draws.append(m)
    d = pd.DataFrame(draws).drop(columns=["Cutoff"])
    return {k: (float(d[k].quantile(0.025)), float(d[k].quantile(0.975))) for k in d.columns}


def main():
    _, test_df = pu.load_model_ready_split()
    feature_cols, _, _ = pu.get_feature_lists()
    X_test = test_df[feature_cols]
    y_test = test_df[cfg.OUTCOME_VAR].values
    print(f"TEST SET -- touched for the first time in this pipeline: {X_test.shape[0]} rows, "
          f"prevalence {y_test.mean()*100:.2f}%")

    thr_path = cfg.OUT_METRICS / "thresholds.json"
    thresholds = {}
    if thr_path.exists():
        with open(thr_path) as f:
            thresholds = json.load(f)

    # FIX (2026-10 audit, C17): evaluate only the models trained in the
    # current run (the ones 07_train_models.py wrote thresholds for), never a
    # stale .joblib left over from an earlier run.
    model_files = [cfg.OUT_MODELS / f"{n.replace(' ', '_')}.joblib" for n in sorted(thresholds)]
    model_files = [p for p in model_files if p.exists()]
    if not model_files:
        raise SystemExit("No fitted models found in outputs/models/. Run 07_train_models.py first.")

    rows = []
    roc_data = {}
    calib_data = {}
    prob_by_model = {}

    for path in model_files:
        name = path.stem.replace("_", " ")
        pipe = joblib.load(path)
        prob = pipe.predict_proba(X_test)[:, 1]
        prob_by_model[name] = prob

        auc = roc_auc_score(y_test, prob)
        auc_lo, auc_hi = bootstrap_auc_ci(y_test, prob)
        auprc = average_precision_score(y_test, prob)
        brier = brier_score_loss(y_test, prob)

        cutoff = thresholds.get(name, 0.5)
        m_tuned = metrics_at_cutoff(y_test, prob, cutoff)
        m_default = metrics_at_cutoff(y_test, prob, 0.5)

        row = {
            "Model": name,
            "AUC": round(auc, 4),
            "AUC_95CI_low": round(auc_lo, 4),
            "AUC_95CI_high": round(auc_hi, 4),
            "AUPRC": round(auprc, 4),
            "Brier": round(brier, 4),
        }
        row.update({f"{k}_tuned_cutoff": round(v, 4) if isinstance(v, float) else v for k, v in m_tuned.items()})
        row.update({f"{k}_at_0.5": round(v, 4) if isinstance(v, float) else v for k, v in m_default.items()})
        cis = bootstrap_metric_cis(y_test, prob, cutoff)
        for k, (lo, hi) in cis.items():
            suffix = "" if k in ("AUPRC", "Brier") else "_tuned_cutoff"
            row[f"{k}{suffix}_95CI_low"] = round(lo, 4)
            row[f"{k}{suffix}_95CI_high"] = round(hi, 4)
        rows.append(row)

        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_data[name] = (fpr, tpr, auc)
        prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10, strategy="uniform")
        calib_data[name] = (prob_pred, prob_true)

        print(f"  {name}: AUC={auc:.4f} (95% CI {auc_lo:.4f}-{auc_hi:.4f})  AUPRC={auprc:.4f}  "
              f"Brier={brier:.4f}  Sens={m_tuned['Sensitivity']:.3f}  Spec={m_tuned['Specificity']:.3f}")

    results_df = pd.DataFrame(rows).sort_values("AUC", ascending=False).reset_index(drop=True)

    # FIX: paired bootstrap significance test, top model vs. every other model
    top_name = results_df.iloc[0]["Model"]
    prob_top = prob_by_model[top_name]
    sig_rows = []
    print(f"\nPaired comparison: '{top_name}' (top by point-estimate AUC) vs. every other model "
          f"(DeLong p-value; paired bootstrap CI, n_boot={N_BOOTSTRAP}):")
    for name in results_df["Model"]:
        if name == top_name:
            sig_rows.append({"model": name, "is_top_model": True, "auc_diff_vs_top": 0.0,
                              "diff_95ci_low": 0.0, "diff_95ci_high": 0.0, "p_value": 1.0,
                              "statistically_indistinguishable_from_top": True})
            continue
        diff, lo, hi, p = paired_bootstrap_auc_diff(y_test, prob_top, prob_by_model[name])
        # 2026-10 review: one tie rule everywhere. A model is "indistinguishable"
        # from the top model when DeLong's test does not reject at 0.05; the
        # paired-bootstrap interval is reported alongside.
        indistinguishable = bool(p >= 0.05)
        sig_rows.append({"model": name, "is_top_model": False, "auc_diff_vs_top": round(diff, 4),
                          "diff_95ci_low": round(lo, 4), "diff_95ci_high": round(hi, 4),
                          "p_value": float(f"{p:.3g}"), "p_method": "DeLong",
                          "statistically_indistinguishable_from_top": indistinguishable})
        flag = "  <-- NOT distinguishable from top model at 95%" if indistinguishable else ""
        print(f"  {top_name} - {name}: {diff:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}), DeLong p={p:.3g}{flag}")

    sig_df = pd.DataFrame(sig_rows)
    sig_path = cfg.OUT_METRICS / "model_comparison_significance.csv"
    sig_df.to_csv(sig_path, index=False)
    n_tied = int(sig_df["statistically_indistinguishable_from_top"].sum())
    print(f"\n{n_tied} of {len(sig_df)} models (including the top model itself) are statistically "
          f"indistinguishable from the top-ranked model at the 95% level. Saved: {sig_path}")

    results_df = results_df.merge(
        sig_df[["model", "statistically_indistinguishable_from_top"]].rename(columns={"model": "Model"}),
        on="Model", how="left",
    )

    out_csv = cfg.OUT_METRICS / "model_comparison.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print("\nRanked by test AUC:")
    print(results_df[["Model", "AUC", "AUC_95CI_low", "AUC_95CI_high", "AUPRC", "Brier",
                       "statistically_indistinguishable_from_top"]].to_string(index=False))

    # ROC plot
    plt.figure(figsize=(9, 7))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test-Set ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(cfg.OUT_FIGURES / "roc_curves.png", dpi=150)
    plt.close()

    # Calibration plot
    n_models = len(calib_data)
    n_cols = 3
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    for i, (name, (prob_pred, prob_true)) in enumerate(calib_data.items()):
        ax = axes[i]
        ax.plot(prob_pred, prob_true, marker="o", lw=1.5)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
    for j in range(len(calib_data), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(cfg.OUT_FIGURES / "calibration_curves.png", dpi=150)
    plt.close()

    print(f"\nSaved figures: {cfg.OUT_FIGURES / 'roc_curves.png'}, {cfg.OUT_FIGURES / 'calibration_curves.png'}")


if __name__ == "__main__":
    main()
