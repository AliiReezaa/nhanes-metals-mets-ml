"""
Q1 rigor addition 4: individual-level SHAP waterfall plots. Every SHAP figure
so far (Figure 3 summary, Figure 5 dependence, and everything in
07_Novelty_Additions) is population-level -- averaged or aggregated across
the whole test set. A waterfall plot for one true positive and one true
negative makes the interpretability story concrete: for THIS person, here is
exactly which features pushed the prediction up or down from the model's
average, and by how much.

Run (after the main pipeline): python 08_Q1_Rigor_Additions/code/08_shap_waterfall.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_Code"))
import additions_common as ac  # noqa: E402
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubstyle as ps
import feature_labels as fl

ps.apply_style()
TABLES, FIGURES = ac.out_dirs("08_Q1_Rigor_Additions")


def main():
    state = ac.primary_state()
    cb_model = state["estimator"]
    Xt_test_df = state["Xt_test_df"]
    y_test = state["y_test"].reset_index(drop=True)
    X_test_raw = state["X_test_raw"].reset_index(drop=True)

    explainer = state["explainer"]
    proba = cb_model.predict_proba(Xt_test_df)[:, 1]

    # Most-confident correctly classified true positive and true negative
    tp_mask = (y_test.values == 1)
    tn_mask = (y_test.values == 0)
    tp_idx = np.argmax(np.where(tp_mask, proba, -1))
    tn_idx = np.argmin(np.where(tn_mask, proba, 2))
    print(f"True positive example: row {tp_idx}, P(MetS)={proba[tp_idx]:.4f}, actual=1")
    print(f"True negative example: row {tn_idx}, P(MetS)={proba[tn_idx]:.4f}, actual=0")

    feat_names_short = fl.abbrevs(list(Xt_test_df.columns))

    summary_rows = []
    for idx, label, fname in [(tp_idx, "true positive", "fig_shap_waterfall_true_positive"),
                                (tn_idx, "true negative", "fig_shap_waterfall_true_negative")]:
        sv_row = state["shap_values"][idx]
        base_value = explainer.expected_value
        if np.ndim(base_value) > 0:
            base_value = np.ravel(base_value)[-1]
        row_data = Xt_test_df.iloc[idx].values

        expl = shap.Explanation(
            values=sv_row, base_values=base_value, data=row_data,
            feature_names=feat_names_short,
        )
        fig = plt.figure(figsize=(ps.W_SINGLE * 1.7, ps.W_SINGLE * 1.55))
        shap.plots.waterfall(expl, max_display=12, show=False)
        fig = plt.gcf()
        fig.set_size_inches(ps.W_SINGLE * 1.7, ps.W_SINGLE * 1.55)
        for ax in fig.get_axes():
            for txt in ax.texts:
                txt.set_fontsize(7)
            ax.tick_params(labelsize=7.5)
            ax.xaxis.label.set_fontsize(8.5)
        fig.tight_layout()
        ps.save(fig, FIGURES / fname)
        plt.close(fig)
        print(f"Saved {fname}.png/.pdf")

        summary_rows.append({
            "Example": label.capitalize(),
            "Actual MetS status": int(y_test.iloc[idx]),
            "Predicted P(MetS)": round(float(proba[idx]), 4),
            "Model base value (log-odds, expected value)": round(float(base_value), 4),
            "Top contributing feature (|SHAP|)": feat_names_short[int(np.argmax(np.abs(sv_row)))],
            "Top feature |SHAP| value (log-odds)": round(float(np.max(np.abs(sv_row))), 4),
        })

    pd.DataFrame(summary_rows).to_csv(TABLES / "table_shap_waterfall_examples.csv", index=False)
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
