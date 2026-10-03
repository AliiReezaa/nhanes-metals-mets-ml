"""
Q1 rigor addition 3: precision-recall curve. AUPRC (0.8114 for CatBoost) is
already reported in the manuscript's model_comparison.csv, but there is no
accompanying PR curve figure -- only the ROC curve (Figure 2). At 46%
prevalence class imbalance is mild, so this is not expected to change the
story, but a PR curve is the standard companion to a ROC curve and its
absence is the kind of thing a reviewer flags reflexively.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubstyle as ps

ps.apply_style()


def main():
    state = joblib.load("shared_state.joblib")
    y_test = state["y_test"]
    proba = state["pipe"].predict_proba(state["X_test_raw"])[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, proba)
    auprc = average_precision_score(y_test, proba)
    auroc = roc_auc_score(y_test, proba)
    prevalence = y_test.mean()
    print(f"AUPRC = {auprc:.4f} (manuscript model_comparison.csv reports 0.8114 for CatBoost)")
    print(f"Test-set prevalence (no-skill baseline) = {prevalence:.4f}")

    fig, ax = plt.subplots(figsize=(ps.W_SINGLE * 1.55, ps.W_SINGLE * 1.45))
    ax.plot(recall, precision, color=ps.OKABE_BLUE, lw=1.6, zorder=3,
            label=f"CatBoost (AUPRC = {auprc:.3f})")
    ax.axhline(prevalence, color="#999999", lw=0.8, ls="--", zorder=1,
               label=f"No-skill baseline (prevalence = {prevalence:.3f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    ps.save(fig, "fig_precision_recall_curve")
    print("Saved fig_precision_recall_curve.png/.pdf")

    pd.DataFrame([{
        "Model": "CatBoost (primary)", "N test": len(y_test),
        "Test prevalence": round(prevalence, 4),
        "AUROC": round(auroc, 4), "AUPRC": round(auprc, 4),
    }]).to_csv("table_precision_recall_summary.csv", index=False)


if __name__ == "__main__":
    main()
