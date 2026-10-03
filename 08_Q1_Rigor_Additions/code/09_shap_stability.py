"""
Q1 rigor addition 5: bootstrap stability of feature importance. Every SHAP
importance ranking reported so far (Figure 3/4, permutation_importance.csv,
shap_importance.csv) is a single point estimate on the n=2,229 held-out test
set, with no uncertainty attached -- so there is no way to tell whether the
5th-ranked feature is meaningfully different from the 8th-ranked one, or
whether that's noise from this particular test partition. This resamples the
test set with replacement (does not refit the model -- the already-computed
SHAP value matrix is resampled row-wise, which is the standard nonparametric
bootstrap for characterizing sampling variability of an importance ranking
computed on a fixed model), recomputes mean |SHAP| per feature each time,
and reports both a 95% CI per feature and how often each feature lands in
the top 10 across resamples (a ranking-stability measure that a single point
estimate cannot show).
"""
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubstyle as ps
import sys

sys.path.insert(0, "SUBMISSION/06_Code")
import feature_labels as fl

ps.apply_style()

RNG = np.random.default_rng(42)
N_BOOT = 2000
TOP_N_DISPLAY = 15
TOP_N_STABILITY = 10


def main():
    state = joblib.load("shared_state.joblib")
    shap_values = state["shap_values"]  # (n_test, n_features)
    feat_names = list(state["Xt_test_df"].columns)
    n, p = shap_values.shape
    print(f"Bootstrapping mean |SHAP| over {N_BOOT} resamples of n={n} test rows...")

    abs_shap = np.abs(shap_values)
    point_estimate = abs_shap.mean(axis=0)

    boot_means = np.empty((N_BOOT, p))
    top10_counts = np.zeros(p)
    for b in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        m = abs_shap[idx].mean(axis=0)
        boot_means[b] = m
        top10 = np.argsort(-m)[:TOP_N_STABILITY]
        top10_counts[top10] += 1

    ci_low = np.percentile(boot_means, 2.5, axis=0)
    ci_high = np.percentile(boot_means, 97.5, axis=0)
    top10_pct = 100 * top10_counts / N_BOOT

    order = np.argsort(-point_estimate)
    labels_short = fl.abbrevs(feat_names)

    table = pd.DataFrame({
        "Feature": [labels_short[i] for i in order],
        "Mean |SHAP|": point_estimate[order].round(5),
        "95% CI low": ci_low[order].round(5),
        "95% CI high": ci_high[order].round(5),
        "Top-10 inclusion rate across bootstraps (%)": top10_pct[order].round(1),
    })
    table.to_csv("table_shap_importance_stability.csv", index=False)
    print(table.head(TOP_N_DISPLAY).to_string(index=False))

    # Plot: top-N features, mean |SHAP| with 95% bootstrap CI
    top_idx = order[:TOP_N_DISPLAY]
    y_pos = np.arange(TOP_N_DISPLAY)[::-1]
    means = point_estimate[top_idx]
    err_low = means - ci_low[top_idx]
    err_high = ci_high[top_idx] - means

    fig, ax = plt.subplots(figsize=(ps.W_SINGLE * 1.6, ps.W_SINGLE * 1.75))
    ax.barh(y_pos, means, xerr=[err_low, err_high], color=ps.OKABE_BLUE,
            height=0.62, error_kw=dict(elinewidth=0.9, ecolor="#333333", capsize=2.2))
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels_short[i] for i in top_idx], fontsize=8)
    ax.set_xlabel("Mean |SHAP value| (test set, bootstrap 95% CI)")
    for y, i in zip(y_pos, top_idx):
        ax.text(ci_high[i] + means.max() * 0.015, y, f"{top10_pct[i]:.0f}% top-10",
                 va="center", fontsize=6.8, color="#555555")
    ax.set_xlim(0, ci_high[top_idx].max() * 1.28)
    fig.tight_layout()
    ps.save(fig, "fig_shap_importance_stability")
    print("Saved fig_shap_importance_stability.png/.pdf")


if __name__ == "__main__":
    main()
