"""
Step 14 -- Publication-grade extras.

Six additional figures aimed specifically at what a journal reviewer looks
for beyond a metrics table: a proper participant flow diagram, uncertainty
on the model ranking, evidence that the top features aren't an artifact of
one model, a clinical-utility check (does the model actually help decisions,
not just rank patients), a generalizability check across demographic
subgroups, and a multicollinearity check across the full predictor set.

  A. outputs/figures/strobe_flow.png              -- participant flow diagram
  B. outputs/figures/auc_forest_plot.png           -- all 11 models, 95% CI
  C. outputs/figures/importance_concordance.png    -- top features across 4 models
  D. outputs/figures/decision_curve.png            -- net benefit vs. threshold
  E. outputs/figures/subgroup_auc_forest.png       -- AUC by sex/age/race, best model
  F. outputs/figures/predictor_correlation.png     -- clustered correlation heatmap

Reads only already-saved outputs (model_comparison.csv, exclusion_flow.csv,
the fitted pipelines in outputs/models/) -- run 02_clean_and_outcome.py (for
exclusion_flow.csv) and 08_evaluate.py (for model_comparison.csv and the
fitted models) first.

REBUILD NOTE / FIX: (B) and (E) rank/select models by point-estimate test
AUC alone, which is exactly the "implicitly overclaims a single best model"
pattern corrected in 08_evaluate.py (paired bootstrap significance test),
09_explain_shap.py, and 12_visualize.py. This script now applies the same
correction here: the forest plot marks every model statistically
indistinguishable from the top with "*", and the subgroup panel's title
notes when the model it's dissecting is itself part of a statistical tie
at the top, rather than presenting it as uncontested. (A)-(D), (F) were
otherwise unaffected by the pipeline_utils column-layout fix (missing-flag
block) -- they all operate on raw/model_ready feature columns or on
already-aggregated model_comparison.csv, never on the transformed
ColumnTransformer output that changed.

Run:
    python 06_Code/13_publication_extras.py
"""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import pipeline_utils as pu

N_BOOTSTRAP = 1000
TOP_N_MODELS = 4
TIE_COL = "statistically_indistinguishable_from_top"


def display_label(name, cmp_df):
    """FIX: append '*' when 08_evaluate.py's paired bootstrap test found
    this model statistically indistinguishable from the top-ranked model --
    same convention used in 12_visualize.py, applied here too."""
    if TIE_COL in cmp_df.columns:
        row = cmp_df.loc[cmp_df["Model"] == name]
        if len(row) and bool(row.iloc[0][TIE_COL]):
            return f"{name} *"
    return name


def bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOTSTRAP, seed=None):
    """Same procedure as 08_evaluate.py -- kept local so this script can run
    standalone without importing a run-script as a module."""
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
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def load_test_and_models():
    _, test_df = pu.load_model_ready_split()
    feature_cols, cont_cols, cat_cols = pu.get_feature_lists()
    X_test = test_df[feature_cols]
    y_test = test_df[cfg.OUTCOME_VAR].values
    cmp_df = pd.read_csv(cfg.OUT_METRICS / "model_comparison.csv").sort_values("AUC", ascending=False).reset_index(drop=True)
    return test_df, X_test, y_test, feature_cols, cont_cols, cat_cols, cmp_df


# ============================================================
# A. STROBE-style participant flow diagram
# ============================================================
def plot_strobe_flow():
    flow_path = cfg.OUT_METRICS / "exclusion_flow.csv"
    if not flow_path.exists():
        print("  [skip] exclusion_flow.csv not found -- run 02_clean_and_outcome.py first")
        return
    flow = pd.read_csv(flow_path)

    fig, ax = plt.subplots(figsize=(9, 2.6 * len(flow)))
    ax.axis("off")
    box_w, box_h = 0.62, 0.6
    y_positions = np.linspace(0.92, 0.08, len(flow))

    for i, row in flow.iterrows():
        y = y_positions[i]
        label = row["step"]
        n_remaining = int(row["n_remaining"])
        box = mpatches.FancyBboxPatch((0.19, y - box_h / 2 / len(flow)), box_w, box_h / len(flow),
                                       boxstyle="round,pad=0.01,rounding_size=0.02",
                                       linewidth=1.5, edgecolor="#1F3864", facecolor="#DCE6F5",
                                       transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(0.5, y, f"{label}\nN = {n_remaining:,}", transform=ax.transAxes,
                ha="center", va="center", fontsize=10.5, fontweight="bold" if i in (0, len(flow) - 1) else "normal")

        n_excl = row.get("n_excluded_this_step", 0)
        if i > 0 and n_excl and n_excl > 0:
            y_prev = y_positions[i - 1]
            y_mid = (y_prev + y) / 2
            ax.annotate("", xy=(0.5, y + box_h / 2 / len(flow)), xytext=(0.5, y_prev - box_h / 2 / len(flow)),
                        arrowprops=dict(arrowstyle="-|>", color="#1F3864", lw=1.5), transform=ax.transAxes)
            excl_box = mpatches.FancyBboxPatch((0.84, y_mid - 0.035), 0.5, 0.07,
                                                boxstyle="round,pad=0.01,rounding_size=0.015",
                                                linewidth=1.2, edgecolor="#A33", facecolor="#FBE7E7",
                                                transform=ax.transAxes, clip_on=False)
            ax.add_patch(excl_box)
            ax.text(1.09, y_mid, f"Excluded: {int(n_excl):,}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#A33")
        elif i > 0:
            y_prev = y_positions[i - 1]
            ax.annotate("", xy=(0.5, y + box_h / 2 / len(flow)), xytext=(0.5, y_prev - box_h / 2 / len(flow)),
                        arrowprops=dict(arrowstyle="-|>", color="#1F3864", lw=1.5), transform=ax.transAxes)

    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 1)
    ax.set_title("Participant Flow -- NHANES 2007-2018 (6 cycles pooled)", fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(cfg.OUT_FIGURES / "strobe_flow.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {cfg.OUT_FIGURES / 'strobe_flow.png'}")


# ============================================================
# B. AUC forest plot, all models, 95% CI
# ============================================================
def plot_auc_forest(cmp_df):
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(cmp_df) + 2))
    y = np.arange(len(cmp_df))[::-1]
    err_lo = cmp_df["AUC"] - cmp_df["AUC_95CI_low"]
    err_hi = cmp_df["AUC_95CI_high"] - cmp_df["AUC"]
    colors = ["#1F3864" if i == 0 else "#5B84C4" for i in range(len(cmp_df))]

    ax.errorbar(cmp_df["AUC"], y, xerr=[err_lo, err_hi], fmt="o", markersize=8,
                capsize=4, elinewidth=1.5, color="#333")
    for i, (yy, auc, c) in enumerate(zip(y, cmp_df["AUC"], colors)):
        ax.scatter([auc], [yy], s=90, color=c, zorder=5)
    ax.axvline(cmp_df["AUC"].iloc[0], color="#1F3864", linestyle=":", lw=1, alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", lw=1, alpha=0.6, label="Chance (AUC=0.5)")
    ax.set_yticks(y)
    ax.set_yticklabels([display_label(m, cmp_df) for m in cmp_df["Model"]])
    ax.set_xlabel("Test-set AUROC (95% bootstrap CI, n=1000)")
    title = "Model Ranking with Uncertainty -- Test-Set AUROC"
    if TIE_COL in cmp_df.columns:
        title += "\n(* = statistically indistinguishable from top model, paired bootstrap 95% CI)"
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_xlim(0.45, min(1.0, cmp_df["AUC_95CI_high"].max() + 0.03))
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(cfg.OUT_FIGURES / "auc_forest_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {cfg.OUT_FIGURES / 'auc_forest_plot.png'}")


# ============================================================
# C. Cross-model feature importance concordance
# ============================================================
def plot_importance_concordance(cmp_df, X_test, y_test, top_n_models=TOP_N_MODELS, top_n_features=15):
    model_names = cmp_df["Model"].head(top_n_models).tolist()
    importances = {}
    for name in model_names:
        path = cfg.OUT_MODELS / f"{name.replace(' ', '_')}.joblib"
        if not path.exists():
            continue
        print(f"  computing permutation importance for {name}...")
        pipe = joblib.load(path)
        perm = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                       n_repeats=10, random_state=cfg.RANDOM_STATE, n_jobs=1)
        importances[name] = pd.Series(perm.importances_mean, index=X_test.columns)

    if len(importances) < 2:
        print("  [skip] fewer than 2 models available for concordance check")
        return

    imp_df = pd.DataFrame(importances)
    imp_df["__mean__"] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values("__mean__", ascending=False).head(top_n_features)
    imp_df = imp_df.drop(columns="__mean__")
    imp_df.columns = [display_label(c, cmp_df) for c in imp_df.columns]
    imp_df.index = cfg.label_features(imp_df.index)

    # Rank matrix (1 = most important) is what actually shows concordance --
    # raw importance scales differ a lot by model family (tree impurity-free
    # permutation drop vs. a linear model's), ranks don't.
    rank_df = imp_df.rank(ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(1.8 * len(imp_df.columns) + 3, 0.5 * len(imp_df) + 2))
    sns.heatmap(rank_df, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax,
                cbar_kws={"label": "Importance rank (1 = most important)"},
                linewidths=0.5, linecolor="white", vmin=1, vmax=max(15, rank_df.values.max()))
    ax.set_title(f"Feature Importance Concordance Across Top {len(imp_df.columns)} Models\n"
                 f"(permutation importance, ROC AUC, test set)", fontsize=12, fontweight="bold")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(cfg.OUT_FIGURES / "importance_concordance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {cfg.OUT_FIGURES / 'importance_concordance.png'}")


# ============================================================
# D. Decision curve analysis
# ============================================================
def plot_decision_curve(cmp_df, X_test, y_test, top_n_models=TOP_N_MODELS):
    model_names = cmp_df["Model"].head(top_n_models).tolist()
    thresholds = np.arange(0.02, 0.51, 0.01)
    prevalence = y_test.mean()
    n = len(y_test)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    palette = sns.color_palette("tab10", top_n_models)
    for color, name in zip(palette, model_names):
        path = cfg.OUT_MODELS / f"{name.replace(' ', '_')}.joblib"
        if not path.exists():
            continue
        pipe = joblib.load(path)
        prob = pipe.predict_proba(X_test)[:, 1]
        net_benefit = []
        for pt in thresholds:
            pred_pos = prob >= pt
            tp = np.sum(pred_pos & (y_test == 1))
            fp = np.sum(pred_pos & (y_test == 0))
            nb = (tp / n) - (fp / n) * (pt / (1 - pt))
            net_benefit.append(nb)
        ax.plot(thresholds, net_benefit, label=display_label(name, cmp_df), color=color, lw=2)

    treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
    ax.plot(thresholds, treat_all, color="black", linestyle="--", lw=1.5, label="Treat all")
    ax.axhline(0, color="gray", linestyle=":", lw=1.5, label="Treat none")

    ax.set_xlabel("Risk threshold (probability at which a clinician would act)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis -- Top Models vs. Treat-All / Treat-None", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, max(0.4, prevalence + 0.05))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.OUT_FIGURES / "decision_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {cfg.OUT_FIGURES / 'decision_curve.png'}")


# ============================================================
# E. Subgroup performance forest plot (best model)
# ============================================================
def plot_subgroup_auc(cmp_df, test_df, X_test, y_test):
    best_name = cmp_df["Model"].iloc[0]
    path = cfg.OUT_MODELS / f"{best_name.replace(' ', '_')}.joblib"
    if not path.exists():
        print("  [skip] best model file not found")
        return
    pipe = joblib.load(path)
    prob = pipe.predict_proba(X_test)[:, 1]

    subgroups = {}
    subgroups["Overall"] = np.ones(len(y_test), dtype=bool)

    if "sex_female" in test_df.columns:
        subgroups["Sex: Male"] = (test_df["sex_female"] == 0).values
        subgroups["Sex: Female"] = (test_df["sex_female"] == 1).values

    if "RIDAGEYR" in test_df.columns:
        age = test_df["RIDAGEYR"].values
        q1, q2 = np.nanpercentile(age, [33.3, 66.7])
        subgroups[f"Age: 20-{int(q1)}"] = age <= q1
        subgroups[f"Age: {int(q1)+1}-{int(q2)}"] = (age > q1) & (age <= q2)
        subgroups[f"Age: {int(q2)+1}+"] = age > q2

    race_cols = {c: cfg.label_feature(c) for c in test_df.columns if c.startswith("RIDRETH1_") and not c.endswith("nan")}
    if race_cols:
        any_dummy = np.zeros(len(test_df), dtype=bool)
        for c in race_cols:
            any_dummy |= (test_df[c] == 1).values
        subgroups["Race/ethnicity: Mexican American (ref.)"] = ~any_dummy & (test_df[list(race_cols)].sum(axis=1) == 0).values
        for c, label in race_cols.items():
            subgroups[label] = (test_df[c] == 1).values

    rows = []
    for label, mask in subgroups.items():
        yt, yp = y_test[mask], prob[mask]
        n_pos = int((yt == 1).sum())
        if mask.sum() < 30 or n_pos < 5 or n_pos > mask.sum() - 5:
            continue
        mean_auc, lo, hi = bootstrap_auc_ci(yt, yp, n_boot=500)
        rows.append({"subgroup": label, "n": int(mask.sum()), "auc": mean_auc, "lo": lo, "hi": hi})

    if not rows:
        print("  [skip] no usable subgroups (too small / too imbalanced)")
        return
    sub_df = pd.DataFrame(rows)
    overall_auc = sub_df.loc[sub_df["subgroup"] == "Overall", "auc"].values[0]
    sub_df = pd.concat([sub_df[sub_df["subgroup"] == "Overall"],
                         sub_df[sub_df["subgroup"] != "Overall"].sort_values("auc", ascending=False)])

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(sub_df) + 2))
    y = np.arange(len(sub_df))[::-1]
    err_lo = sub_df["auc"] - sub_df["lo"]
    err_hi = sub_df["hi"] - sub_df["auc"]
    colors = ["#1F3864" if s == "Overall" else "#5B84C4" for s in sub_df["subgroup"]]
    ax.errorbar(sub_df["auc"], y, xerr=[err_lo, err_hi], fmt="o", markersize=0, capsize=4, elinewidth=1.5, color="#333")
    ax.scatter(sub_df["auc"], y, s=[110 if s == "Overall" else 80 for s in sub_df["subgroup"]], color=colors, zorder=5)
    for yy, n_val in zip(y, sub_df["n"]):
        ax.text(sub_df["hi"].max() + 0.02, yy, f"n={n_val}", va="center", fontsize=8.5, color="#555")
    ax.axvline(overall_auc, color="#1F3864", linestyle=":", lw=1, alpha=0.6)
    ax.axvline(0.5, color="gray", linestyle="--", lw=1, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(sub_df["subgroup"])
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    title = f"Subgroup Performance -- {display_label(best_name, cmp_df)} (Test Set)"
    if TIE_COL in cmp_df.columns and cmp_df.loc[cmp_df["Model"] == best_name, TIE_COL].any():
        pass  # best model's own row is always is_top_model in 08_evaluate.py's output, not itself "tied"
    n_tied = int(cmp_df[TIE_COL].sum()) - 1 if TIE_COL in cmp_df.columns else None
    if n_tied is not None and n_tied > 0:
        title += f"\n(this is the top point-estimate model; {n_tied} other model(s) are statistically tied with it -- see auc_forest_plot.png)"
    ax.set_title(title, fontsize=12.5, fontweight="bold")
    ax.set_xlim(0.4, min(1.05, sub_df["hi"].max() + 0.08))
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.OUT_FIGURES / "subgroup_auc_forest.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {cfg.OUT_FIGURES / 'subgroup_auc_forest.png'}")


# ============================================================
# F. Predictor correlation / clustering heatmap
# ============================================================
def plot_predictor_correlation(X_test, feature_cols):
    X_num = X_test.copy()
    for c in X_num.columns:
        X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
        X_num[c] = X_num[c].fillna(X_num[c].median())
    zero_var = X_num.columns[X_num.std(numeric_only=True) == 0].tolist()
    if zero_var:
        print(f"  dropping {len(zero_var)} zero-variance column(s) from correlation matrix: {zero_var}")
        X_num = X_num.drop(columns=zero_var)
    corr = X_num.corr(method="pearson").fillna(0.0)
    corr.index = cfg.label_features(corr.index)
    corr.columns = cfg.label_features(corr.columns)

    g = sns.clustermap(corr, cmap="RdBu_r", vmin=-1, vmax=1, figsize=(15, 14),
                        cbar_kws={"label": "Pearson r"}, linewidths=0.3, linecolor="white",
                        xticklabels=True, yticklabels=True)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize=7.5, rotation=90)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize=7.5)
    g.fig.suptitle("Predictor Correlation Structure (Test Set, Hierarchically Clustered)",
                    fontsize=13, fontweight="bold", y=1.02)
    g.savefig(cfg.OUT_FIGURES / "predictor_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  saved: {cfg.OUT_FIGURES / 'predictor_correlation.png'}")


def main():
    print("Loading test set, fitted models, and model_comparison.csv...")
    test_df, X_test, y_test, feature_cols, cont_cols, cat_cols, cmp_df = load_test_and_models()

    print("\n[A] STROBE participant flow diagram...")
    plot_strobe_flow()

    print("\n[B] AUC forest plot (all models, 95% CI)...")
    plot_auc_forest(cmp_df)

    print(f"\n[C] Feature importance concordance (top {TOP_N_MODELS} models)...")
    plot_importance_concordance(cmp_df, X_test, y_test)

    print(f"\n[D] Decision curve analysis (top {TOP_N_MODELS} models)...")
    plot_decision_curve(cmp_df, X_test, y_test)

    print("\n[E] Subgroup performance forest plot (best model)...")
    plot_subgroup_auc(cmp_df, test_df, X_test, y_test)

    print("\n[F] Predictor correlation / clustering heatmap...")
    plot_predictor_correlation(X_test, feature_cols)

    print("\nDone -- all publication extras saved to outputs/figures/")


if __name__ == "__main__":
    main()
