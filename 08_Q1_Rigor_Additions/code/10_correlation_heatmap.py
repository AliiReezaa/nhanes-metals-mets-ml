"""
Q1 rigor addition 6: full predictor correlation heatmap. The pipeline
already screens for collinearity (collinearity_flags.csv: BMXHIP dropped for
its high correlation with BMXBMI before predictor selection), but that table
only shows the ONE pair that got flagged and excluded -- there is no figure
showing the correlation structure of the predictors that were actually kept.
This computes the full Spearman correlation matrix on those (continuous +
categorical/dummy) predictors as they enter the model (post-impute, i.e. the
same matrix the model trains on), hierarchically clustered so that any
residual moderate correlation blocks are visually grouped rather than
scattered across an arbitrarily-ordered grid.

Run (after the main pipeline): python 08_Q1_Rigor_Additions/code/10_correlation_heatmap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "06_Code"))
import additions_common as ac  # noqa: E402
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pubstyle as ps
import feature_labels as fl

ps.apply_style()
TABLES, FIGURES = ac.out_dirs("08_Q1_Rigor_Additions")


def main():
    state = ac.primary_state()
    predictors_all = state["feature_cols"]  # the model-input columns, no missing-flags
    Xt_train_full = state["Xt_train_df"][predictors_all]

    # Drop any predictor with zero variance in the training partition (a
    # constant column has an undefined correlation with everything else,
    # e.g. RIDRETH1_nan when no participant in the train split has missing
    # race/ethnicity) -- report it explicitly rather than let it silently
    # propagate NaNs into the clustering.
    stds = Xt_train_full.std()
    constant_cols = stds[stds == 0].index.tolist()
    predictors = [c for c in predictors_all if c not in constant_cols]
    if constant_cols:
        print(f"Dropped {len(constant_cols)} zero-variance predictor(s) from the correlation "
              f"analysis (undefined correlation): {[fl.abbrev(c) for c in constant_cols]}")
    Xt_train_df = Xt_train_full[predictors]

    corr, _ = spearmanr(Xt_train_df.values)
    corr = np.asarray(corr)
    np.fill_diagonal(corr, 1.0)
    corr_df = pd.DataFrame(corr, index=predictors, columns=predictors)

    # Flag every off-diagonal pair above |rho| = 0.5 (a conventional
    # "moderate-or-higher" collinearity threshold) for the table, since the
    # figure alone can be hard to read precisely off small cells.
    flagged = []
    for i in range(len(predictors)):
        for j in range(i + 1, len(predictors)):
            r = corr[i, j]
            if abs(r) >= 0.5:
                flagged.append({
                    "Variable 1": fl.abbrev(predictors[i]), "Variable 2": fl.abbrev(predictors[j]),
                    "Spearman rho": round(float(r), 3),
                })
    flagged_df = pd.DataFrame(flagged).sort_values("Spearman rho", key=abs, ascending=False) \
        if flagged else pd.DataFrame(columns=["Variable 1", "Variable 2", "Spearman rho"])
    flagged_df.to_csv(TABLES / "table_predictor_correlation_flags.csv", index=False)
    corr_df.round(4).to_csv(TABLES / "table_predictor_correlation_matrix.csv")
    print(f"{len(flagged)} pairs with |Spearman rho| >= 0.5 among the {len(predictors)} retained predictors:")
    print(flagged_df.to_string(index=False) if flagged else "  (none)")

    # Hierarchical clustering order (average linkage on 1-|rho| distance)
    dist = 1 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    order = dendrogram(Z, no_plot=True)["leaves"]

    ordered_corr = corr[np.ix_(order, order)]
    ordered_labels = [fl.abbrev(predictors[i]) for i in order]

    fig, ax = plt.subplots(figsize=(ps.W_DOUBLE * 0.92, ps.W_DOUBLE * 0.92))
    im = ax.imshow(ordered_corr, cmap=ps.shap_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels, rotation=90, fontsize=5.6)
    ax.set_yticks(range(len(ordered_labels)))
    ax.set_yticklabels(ordered_labels, fontsize=5.6)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Spearman rho", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    ps.save(fig, FIGURES / "fig_predictor_correlation_heatmap")
    print("Saved fig_predictor_correlation_heatmap.png/.pdf")


if __name__ == "__main__":
    main()
