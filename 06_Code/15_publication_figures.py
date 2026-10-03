"""
Step 15 -- Publication figure set for the manuscript (Methods + Results).

Regenerates every figure cited in the manuscript at journal standard, and --
critically -- writes every number that the manuscript text quotes into a single
machine-readable file (outputs/metrics/publication_numbers.json) so the prose can
be verified against the figures programmatically rather than by eye.

All figures derive from ONE set of test-set predicted probabilities
(outputs/metrics/test_predictions.csv), so no two figures can silently disagree.
2026-10 audit (C17): the predictions and the per-model permutation importances
are recomputed from the current models on every run; the earlier version
reused any cached file whose model names matched, even after the models had
been retrained.

Outputs
-------
outputs/figures_pub/fig1_participant_flow.{png,pdf}
outputs/figures_pub/fig2_roc_curves.{png,pdf}
outputs/figures_pub/fig3_auc_forest.{png,pdf}
outputs/figures_pub/fig4_calibration.{png,pdf}
outputs/figures_pub/fig5_decision_curve.{png,pdf}
outputs/figures_pub/fig6_shap_beeswarm.{png,pdf}
outputs/figures_pub/fig7_importance_concordance.{png,pdf}
outputs/figures_pub/fig8_metals_shap_dependence.{png,pdf}
outputs/figures_pub/fig9_subgroup_auc.{png,pdf}
outputs/figures_pub/figM1_analysis_pipeline.{png,pdf}
outputs/metrics/test_predictions.csv
outputs/metrics/permutation_importance_by_model.csv
outputs/metrics/subgroup_auc.csv
outputs/metrics/calibration_metrics.csv
outputs/metrics/decision_curve_data.csv
outputs/metrics/publication_numbers.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import feature_labels as fl
import pipeline_utils as pu
import pubstyle as ps
import stats_utils as su

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

ps.apply_style()

OUT_FIG = cfg.OUT_DIR / "figures_pub"
OUT_FIG.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
TOP_N_MODELS = 4
RNG_SEED = cfg.RANDOM_STATE

NUMBERS: dict = {}


def _record_abbrevs(figure_key: str, codes) -> None:
    """Store the caption glossary for one figure alongside its numbers.

    The captions in the Results and Supplement documents read these strings rather than
    restating them, so an abbreviation can never appear on an axis without the matching
    caption expanding it, and the two cannot drift apart when a figure is regenerated.
    """
    NUMBERS.setdefault("figure_abbreviations", {})[figure_key] = fl.glossary(list(codes))


# ======================================================================
# Data / prediction cache
# ======================================================================
def load_everything():
    train_df, test_df = pu.load_model_ready_split()
    feature_cols, cont_cols, cat_cols = pu.get_feature_lists()
    X_test = test_df[feature_cols]
    y_test = test_df[cfg.OUTCOME_VAR].values
    cmp_df = pd.read_csv(cfg.OUT_METRICS / "model_comparison.csv")
    sig_df = pd.read_csv(cfg.OUT_METRICS / "model_comparison_significance.csv")
    return train_df, test_df, X_test, y_test, cmp_df, sig_df, feature_cols, cont_cols, cat_cols


def build_prediction_cache(cmp_df, X_test, y_test):
    """Predicted probabilities for every model on the held-out test set,
    recomputed from the current models every run and saved to CSV so every
    downstream figure uses byte-identical inputs."""
    cache_path = cfg.OUT_METRICS / "test_predictions.csv"
    preds = pd.DataFrame({"y_true": y_test})
    for name in cmp_df["Model"]:
        path = cfg.OUT_MODELS / f"{name.replace(' ', '_')}.joblib"
        if not path.exists():
            print(f"  [warn] missing model artifact: {path.name}")
            continue
        pipe = joblib.load(path)
        preds[name] = pipe.predict_proba(X_test)[:, 1]
        print(f"  predicted: {name}")
    preds.to_csv(cache_path, index=False)
    return preds


# ======================================================================
# Figure 1 -- participant flow
# ======================================================================
def fig1_participant_flow():
    flow = pd.read_csv(cfg.OUT_METRICS / "exclusion_flow.csv")
    strobe = pd.read_csv(cfg.OUT_METRICS / "strobe_flow.csv")
    cycles = strobe[strobe["stage"].str.startswith("Cycle")]

    # 2026-10 audit: the flow now has five exclusion steps (pregnancy and the
    # fasting criterion were added), read from exclusion_flow.csv by position.
    n_at = lambda i: int(flow.loc[i, "n_remaining"])
    exc_at = lambda i: int(flow.loc[i, "n_excluded_this_step"])
    labels = [
        ("NHANES 2007–2018 pooled sample\nsix consecutive two-year cycles", n_at(0)),
        ("Adults aged ≥ 20 years", n_at(1)),
        ("Not pregnant", n_at(2)),
        ("Complete data on all five\nMetS-defining components", n_at(3)),
        ("Met the fasting criterion\n(8 to 24 h)", n_at(4)),
        ("≥ 1 of 5 blood metals measured\n(final analytic cohort)", n_at(5)),
    ]
    exclusions = [
        f"Excluded: {exc_at(1):,}\naged < 20 years",
        f"Excluded: {exc_at(2):,}\npregnant",
        f"Excluded: {exc_at(3):,}\nmissing ≥ 1 MetS component",
        f"Excluded: {exc_at(4):,}\ndid not meet fasting criterion",
        f"Excluded: {exc_at(5):,}\nno blood metal measured",
    ]

    fig, ax = plt.subplots(figsize=(ps.W_DOUBLE, 7.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 14.1); ax.axis("off")

    box_x, box_w = 0.55, 5.1
    exc_x, exc_w = 6.35, 3.2
    ys = [12.15, 10.15, 8.15, 6.15, 4.15, 2.15]
    box_h = 1.05
    fig_center = (box_x + exc_x + exc_w) / 2

    for i, ((text, n), y) in enumerate(zip(labels, ys)):
        face = "#EAF2F8" if i < len(labels) - 1 else "#D6EAF8"
        edge = ps.OKABE_BLUE
        lw = 1.0 if i < len(labels) - 1 else 1.6
        ax.add_patch(FancyBboxPatch((box_x, y), box_w, box_h,
                                    boxstyle="round,pad=0.06,rounding_size=0.10",
                                    facecolor=face, edgecolor=edge, linewidth=lw))
        ax.text(box_x + box_w / 2, y + box_h * 0.63, text, ha="center", va="center",
                fontsize=9, linespacing=1.35)
        ax.text(box_x + box_w / 2, y + box_h * 0.19, f"n = {n:,}", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#0B3C5D")

    for i, (y_from, y_to, exc) in enumerate(zip(ys[:-1], ys[1:], exclusions)):
        x_mid = box_x + box_w / 2
        ax.add_patch(FancyArrowPatch((x_mid, y_from), (x_mid, y_to + box_h),
                                     arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.0, color="#333333"))
        y_branch = y_to + box_h + (y_from - y_to - box_h) / 2
        ax.plot([x_mid, exc_x], [y_branch, y_branch], color="#999999", lw=0.9)
        ax.add_patch(FancyBboxPatch((exc_x, y_branch - 0.44), exc_w, 0.88,
                                    boxstyle="round,pad=0.05,rounding_size=0.08",
                                    facecolor="#FDF2E9", edgecolor="#B9770E", linewidth=0.8))
        ax.text(exc_x + exc_w / 2, y_branch, exc, ha="center", va="center",
                fontsize=8.2, linespacing=1.3, color="#7E5109")

    cyc_txt = "  ·  ".join(
        f"{r['stage'].split()[1].replace('_', '–')}: {int(r['n']):,}" for _, r in cycles.iterrows()
    )
    ax.text(fig_center, 14.02, "Cycle contributions to the pooled sample",
            ha="center", va="top", fontsize=8.2, style="italic", color="#555555")
    ax.text(fig_center, 13.70, cyc_txt, ha="center", va="top",
            fontsize=7.5, color="#555555")

    train_n = len(pd.read_csv(cfg.DATA_PROC_DIR / "train_seqn.csv"))
    test_n = len(pd.read_csv(cfg.DATA_PROC_DIR / "test_seqn.csv"))
    sub_w, sub_gap = 2.45, 0.20
    ax.add_patch(FancyBboxPatch((box_x, 0.28), sub_w, 0.95,
                                boxstyle="round,pad=0.05,rounding_size=0.08",
                                facecolor="#E8F8F5", edgecolor=ps.OKABE_GREEN, linewidth=1.0))
    ax.text(box_x + sub_w / 2, 0.91, "Training set (80%)", ha="center", va="center", fontsize=8.2)
    ax.text(box_x + sub_w / 2, 0.55, f"n = {train_n:,}", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#0E6251")
    x2 = box_x + sub_w + sub_gap
    ax.add_patch(FancyBboxPatch((x2, 0.28), sub_w, 0.95,
                                boxstyle="round,pad=0.05,rounding_size=0.08",
                                facecolor="#FDEDEC", edgecolor=ps.OKABE_VERMIL, linewidth=1.0))
    ax.text(x2 + sub_w / 2, 0.91, "Held-out test set (20%)", ha="center", va="center", fontsize=8.2)
    ax.text(x2 + sub_w / 2, 0.55, f"n = {test_n:,}", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#922B21")
    x_mid = box_x + box_w / 2
    ax.add_patch(FancyArrowPatch((x_mid, ys[-1]), (x_mid, 1.28),
                                 arrowstyle="-|>", mutation_scale=11, linewidth=1.0, color="#333333"))
    ax.text(x_mid + 0.14, (ys[-1] + 1.28) / 2, "stratified split on MetS status",
            ha="left", va="center", fontsize=7.8, style="italic", color="#555555")

    NUMBERS["flow"] = {
        "pooled": n_at(0), "adults": n_at(1), "not_pregnant": n_at(2),
        "complete_mets": n_at(3), "fasting_met": n_at(4), "final_cohort": n_at(5),
        "excluded_age": exc_at(1), "excluded_pregnant": exc_at(2),
        "excluded_mets": exc_at(3), "excluded_fasting": exc_at(4),
        "excluded_metals": exc_at(5),
        "train_n": train_n, "test_n": test_n,
    }
    return ps.save(fig, OUT_FIG / "fig1_participant_flow")


# ======================================================================
# Figure 2 -- ROC curves
# ======================================================================
def fig2_roc(preds, cmp_df):
    y = preds["y_true"].values
    fig, axes = plt.subplots(1, 2, figsize=(ps.W_DOUBLE, 3.5),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.28})
    ax, axz = axes

    for rank, row in enumerate(cmp_df.itertuples(), start=1):
        name = row.Model
        if name not in preds.columns:
            continue
        st = ps.model_style(name, rank)
        fpr, tpr, _ = roc_curve(y, preds[name].values)
        label = f"{name} — {row.AUC:.3f}"
        for a in (ax, axz):
            a.plot(fpr, tpr, label=label, **st)

    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color="#999999", lw=0.9, zorder=1)
    ax.set_xlabel("1 − specificity (false-positive rate)")
    ax.set_ylabel("Sensitivity (true-positive rate)")
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ps.panel_label(ax, "A", dx=-0.16)
    # legend goes in panel A's empty lower-right triangle, never over the curves
    ax.legend(loc="lower right", ncol=1, handlelength=1.4, labelspacing=0.26,
              borderpad=0.15, borderaxespad=0.25, fontsize=6.9)

    axz.set_xlim(0.05, 0.40); axz.set_ylim(0.50, 0.95)
    axz.set_xlabel("1 − specificity")
    axz.set_ylabel("Sensitivity")
    axz.grid(True, alpha=0.6)
    ps.panel_label(axz, "B", dx=-0.18)

    NUMBERS["roc"] = {r.Model: {"auc": float(r.AUC),
                                "ci": [float(r.AUC_95CI_low), float(r.AUC_95CI_high)]}
                      for r in cmp_df.itertuples()}
    return ps.save(fig, OUT_FIG / "fig2_roc_curves")


# ======================================================================
# Figure 3 -- AUC forest + paired-bootstrap difference vs top model
# ======================================================================
def fig3_auc_forest(cmp_df, sig_df):
    # both frames carry the tie flag; keep sig_df's copy only, so the merge
    # cannot silently produce _x/_y suffixes
    left = cmp_df.drop(columns=["statistically_indistinguishable_from_top"], errors="ignore")
    d = left.merge(sig_df, left_on="Model", right_on="model", how="left")
    d = d.sort_values("AUC", ascending=True).reset_index(drop=True)
    ypos = np.arange(len(d))
    top_name = sig_df.loc[sig_df["is_top_model"], "model"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(ps.W_DOUBLE, 4.0),
                             gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.06})
    axL, axR = axes

    for i, row in d.iterrows():
        tied = bool(row["statistically_indistinguishable_from_top"])
        col = ps.TIER1_COLORS.get(row["Model"], "#666666")
        mk = "o" if tied else "s"
        axL.plot([row["AUC_95CI_low"], row["AUC_95CI_high"]], [i, i],
                 color=col, lw=1.5, solid_capstyle="round", zorder=2)
        axL.plot(row["AUC"], i, mk, color=col, ms=5.0, zorder=3,
                 markeredgecolor="white", markeredgewidth=0.6)

    axL.set_yticks(ypos)
    axL.set_yticklabels([f"{m}" + (" *" if t else "")
                         for m, t in zip(d["Model"], d["statistically_indistinguishable_from_top"])])
    axL.set_xlabel("Area under the ROC curve (95% CI)")
    axL.set_xlim(0.69, 0.90)
    axL.grid(True, axis="x", alpha=0.55)
    axL.set_ylim(-0.7, len(d) - 0.3)
    ps.panel_label(axL, "A", dx=-0.52)

    for i, row in d.iterrows():
        if row["Model"] == top_name:
            axR.plot(0, i, "D", color=ps.TIER1_COLORS.get(top_name, "#333333"),
                     ms=5.0, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
            continue
        tied = bool(row["statistically_indistinguishable_from_top"])
        col = ps.TIER1_COLORS.get(row["Model"], "#666666")
        axR.plot([row["diff_95ci_low"], row["diff_95ci_high"]], [i, i],
                 color=col, lw=1.5, solid_capstyle="round", zorder=2)
        axR.plot(row["auc_diff_vs_top"], i, "o" if tied else "s", color=col, ms=5.0,
                 markeredgecolor="white", markeredgewidth=0.6, zorder=3)

    axR.axvline(0, color="#333333", lw=0.9, ls=(0, (4, 3)), zorder=1)
    axR.set_yticks(ypos); axR.set_yticklabels([])
    axR.set_xlabel(f"Δ AUC vs. {top_name} (95% CI)")
    axR.grid(True, axis="x", alpha=0.55)
    axR.set_ylim(-0.7, len(d) - 0.3)
    ps.panel_label(axR, "B", dx=-0.06)

    for i, row in d.iterrows():
        if row["Model"] == top_name:
            txt = "reference"
        else:
            txt = su.format_p(row["p_value"])
        axR.text(1.02, i, txt, transform=axR.get_yaxis_transform(),
                 va="center", ha="left", fontsize=7.6,
                 color="#333333" if row["statistically_indistinguishable_from_top"] else "#777777")

    NUMBERS["significance"] = {
        r.model: {"auc_diff": float(r.auc_diff_vs_top),
                  "ci": [float(r.diff_95ci_low), float(r.diff_95ci_high)],
                  "p": float(r.p_value),
                  "tied": bool(r.statistically_indistinguishable_from_top)}
        for r in sig_df.itertuples()}
    NUMBERS["top_model"] = top_name
    NUMBERS["n_tied"] = int(sig_df["statistically_indistinguishable_from_top"].sum())
    return ps.save(fig, OUT_FIG / "fig3_auc_forest")


# ======================================================================
# Figure 4 -- calibration
# ======================================================================
def calibration_slope_intercept(y, p):
    """Standard recalibration model: logit(p) as the sole predictor.
    Slope 1 / intercept 0 = perfect calibration."""
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps))).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lp, y)
    slope = float(lr.coef_[0][0])
    # calibration-in-the-large: intercept with slope fixed at 1 (offset model)
    from scipy.optimize import minimize_scalar
    def nll(a):
        z = a + lp.ravel()
        pr = 1 / (1 + np.exp(-z))
        pr = np.clip(pr, eps, 1 - eps)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    res = minimize_scalar(nll, bounds=(-5, 5), method="bounded")
    intercept = float(res.x)
    return slope, intercept


def fig4_calibration(preds, cmp_df):
    y = preds["y_true"].values
    plotted = [m for m in cmp_df["Model"].head(TOP_N_MODELS) if m in preds.columns]
    # calibration metrics are computed for EVERY model (they populate the results
    # table); only the leading models are drawn, to keep the panel readable.
    all_models = [m for m in cmp_df["Model"] if m in preds.columns]

    fig, axes = plt.subplots(1, 2, figsize=(ps.W_DOUBLE, 3.4),
                             gridspec_kw={"width_ratios": [1, 0.85], "wspace": 0.3})
    ax, axh = axes
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), color="#999999", lw=0.9, zorder=1)

    rows = []
    for rank, name in enumerate(all_models, start=1):
        p = preds[name].values
        slope, intercept = calibration_slope_intercept(y, p)
        brier = brier_score_loss(y, p)
        oe = float(y.mean() / p.mean())
        rows.append({"model": name, "calibration_slope": round(slope, 3),
                     "calibration_intercept": round(intercept, 3),
                     "brier": round(brier, 4), "observed_expected_ratio": round(oe, 3)})
        if name not in plotted:
            continue
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        st = ps.model_style(name, rank)
        ax.plot(mean_pred, frac_pos, marker="o", ms=3.4, markeredgecolor="white",
                markeredgewidth=0.4, **st)
        ax.plot([], [], **st, label=f"{name} — slope {slope:.2f}")
    models = plotted

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed proportion with MetS")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", handlelength=1.5, labelspacing=0.3, fontsize=7.5)
    ps.panel_label(ax, "A", dx=-0.17)

    top = models[0]
    p_top = preds[top].values
    bins = np.linspace(0, 1, 26)
    axh.hist(p_top[y == 0], bins=bins, color="#B0B0B0", alpha=0.85, label="No MetS", lw=0)
    axh.hist(p_top[y == 1], bins=bins, color=ps.TIER1_COLORS.get(top, ps.OKABE_BLUE),
             alpha=0.72, label="MetS", lw=0)
    axh.set_xlabel(f"Predicted probability ({top})")
    axh.set_ylabel("Participants in test set")
    axh.legend(loc="upper right", handlelength=1.2, fontsize=7.8)
    ps.panel_label(axh, "B", dx=-0.2)

    cal = pd.DataFrame(rows)
    cal.to_csv(cfg.OUT_METRICS / "calibration_metrics.csv", index=False)
    NUMBERS["calibration"] = {r["model"]: r for r in rows}
    return ps.save(fig, OUT_FIG / "fig4_calibration")


# ======================================================================
# Figure 5 -- decision curve analysis
# ======================================================================
def fig5_decision_curve(preds, cmp_df):
    y = preds["y_true"].values
    n = len(y)
    prevalence = float(y.mean())
    # 2026-10 audit (C23): the range now extends to 0.80. With a prevalence
    # near 47% and Youden cut-offs near 0.5, stopping at 0.50 hid the part of
    # the curve where the model's thresholds actually sit.
    thresholds = np.arange(0.02, 0.81, 0.01)
    models = [m for m in cmp_df["Model"].head(TOP_N_MODELS) if m in preds.columns]

    fig, ax = plt.subplots(figsize=(ps.W_DOUBLE * 0.62, 3.4))
    rows = []
    for rank, name in enumerate(models, start=1):
        p = preds[name].values
        nb = []
        for pt in thresholds:
            pred_pos = p >= pt
            tp = np.sum(pred_pos & (y == 1))
            fp = np.sum(pred_pos & (y == 0))
            nb.append(tp / n - (fp / n) * (pt / (1 - pt)))
        st = ps.model_style(name, rank)
        ax.plot(thresholds, nb, label=name, **st)
        for pt, v in zip(thresholds, nb):
            rows.append({"model": name, "threshold": round(float(pt), 2), "net_benefit": round(float(v), 5)})

    treat_all = [prevalence - (1 - prevalence) * (pt / (1 - pt)) for pt in thresholds]
    ax.plot(thresholds, treat_all, color="#333333", ls=(0, (5, 3)), lw=1.1, label="Treat all")
    ax.axhline(0, color="#999999", ls=(0, (1, 2)), lw=1.0, label="Treat none")
    for pt, v in zip(thresholds, treat_all):
        rows.append({"model": "Treat all", "threshold": round(float(pt), 2), "net_benefit": round(float(v), 5)})

    ax.set_xlabel("Risk threshold (probability at which intervention is warranted)")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0.02, 0.80)
    ax.set_ylim(-0.02, max(0.50, prevalence + 0.02))
    ax.grid(True, alpha=0.5)
    ax.legend(loc="lower left", handlelength=1.6, labelspacing=0.32, fontsize=7.6)

    dca = pd.DataFrame(rows)
    dca.to_csv(cfg.OUT_METRICS / "decision_curve_data.csv", index=False)

    top = models[0]
    sel = {}
    for pt in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
        nb_m = dca[(dca.model == top) & (np.isclose(dca.threshold, pt))]["net_benefit"].iloc[0]
        nb_a = dca[(dca.model == "Treat all") & (np.isclose(dca.threshold, pt))]["net_benefit"].iloc[0]
        # 2026-10 audit (P11): the gain over "treat all" is a net REDUCTION IN
        # UNNECESSARY INTERVENTIONS, (NB_model - NB_all) / (pt / (1 - pt)), at
        # the same number of true cases found (Vickers & Elkin 2006). The
        # earlier version called (NB_model - NB_all) x 1000 "additional true
        # cases", which is wrong: against treat-all the model finds fewer cases.
        delta = float(nb_m - nb_a)
        sel[f"{pt:.2f}"] = {"model_nb": float(nb_m), "treat_all_nb": float(nb_a),
                            "delta": round(delta, 4),
                            "net_interventions_avoided_per_1000": round(delta / (pt / (1 - pt)) * 1000, 0),
                            "net_true_positives_per_1000_vs_treat_none": round(float(nb_m) * 1000, 0)}
    NUMBERS["dca"] = {"prevalence_test": round(prevalence, 4), "top_model": top, "points": sel}
    return ps.save(fig, OUT_FIG / "fig5_decision_curve")


# ======================================================================
# Figure 6 -- SHAP beeswarm
# ======================================================================
TREE_MODELS = ["CatBoost", "XGBoost", "LightGBM", "Gradient Boosting", "Random Forest", "Decision Tree"]


def shap_model_name(cmp_df):
    """The model SHAP is computed for: the highest-AUROC tree model (the same
    rule 09_explain_shap.py uses). 2026-10 audit (C23): figures 6 to 8 used to
    hard-code 'CatBoost' and crashed if CatBoost was absent or not the top."""
    for m in cmp_df["Model"]:
        if m in TREE_MODELS and (cfg.OUT_MODELS / f"{m.replace(' ', '_')}.joblib").exists():
            return m
    raise SystemExit("No fitted tree model available for SHAP.")


def fig6_shap_beeswarm(X_test, cont_cols, cat_cols, name):
    import shap
    pipe = joblib.load(cfg.OUT_MODELS / f"{name.replace(' ', '_')}.joblib")
    pre = pipe.named_steps["preprocess"]
    est = pipe.named_steps["model"]
    Xt = pd.DataFrame(pre.transform(X_test),
                      columns=pu.transformed_feature_names(cont_cols, cat_cols))
    explainer = shap.TreeExplainer(est)
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]

    # Short labels on the y axis; every one of them is expanded in the figure caption
    # via feature_labels.glossary(), so nothing is abbreviated without being defined.
    MAX_DISPLAY = 18
    disp = pd.DataFrame(Xt.values, columns=fl.abbrevs(Xt.columns))
    # Only the rows the reader can actually see get glossed. Recording all predictors
    # would pad the caption with expansions for labels that never appear in the panel.
    shown = (pd.Series(np.abs(sv).mean(axis=0), index=Xt.columns)
             .sort_values(ascending=False).head(MAX_DISPLAY).index)
    _record_abbrevs("fig6_shap_beeswarm", shown)
    fig = plt.figure(figsize=(ps.W_DOUBLE * 0.78, 4.6))
    shap.summary_plot(sv, disp, max_display=MAX_DISPLAY, show=False, plot_size=None,
                      color_bar=True, cmap=ps.shap_cmap())
    ax = plt.gca()
    ax.set_xlabel("SHAP value (impact on log-odds of metabolic syndrome)", fontsize=9.5)
    ax.tick_params(labelsize=8.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches(ps.W_DOUBLE * 0.82, 4.8)
    return ps.save(fig, OUT_FIG / "fig6_shap_beeswarm")


# ======================================================================
# Figure 7 -- SHAP vs permutation concordance + cross-model rank matrix
# ======================================================================
def compute_perm_by_model(cmp_df, X_test, y_test):
    cache = cfg.OUT_METRICS / "permutation_importance_by_model.csv"
    out = {}
    for name in cmp_df["Model"].head(TOP_N_MODELS):
        path = cfg.OUT_MODELS / f"{name.replace(' ', '_')}.joblib"
        if not path.exists():
            continue
        print(f"  permutation importance: {name}")
        pipe = joblib.load(path)
        perm = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                      n_repeats=10, random_state=RNG_SEED, n_jobs=1)
        out[name] = pd.Series(perm.importances_mean, index=X_test.columns)
    df = pd.DataFrame(out)
    df.to_csv(cache)
    return df


def fig7_concordance(perm_by_model, shap_name):
    shap_imp_all = pd.read_csv(cfg.OUT_METRICS / "shap_importance.csv")
    perm_imp = pd.read_csv(cfg.OUT_METRICS / "permutation_importance.csv")

    # SHAP is computed in the TRANSFORMED feature space (the predictors plus the
    # missing-value flags), whereas permutation importance is computed on the
    # original predictor columns. Ranking SHAP over both and permutation over
    # the predictors only would put the two methods on different denominators,
    # so SHAP is ranked among the same original predictors here. The
    # missing-flag SHAP values are retained separately below.
    miss_mask = shap_imp_all["feature_code"].str.startswith("missingindicator")
    shap_missing = shap_imp_all[miss_mask].copy()
    shap_imp = shap_imp_all[~miss_mask].copy()
    shap_imp["shap_rank"] = shap_imp["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    perm_imp["perm_rank"] = perm_imp["perm_importance_mean"].rank(ascending=False, method="min").astype(int)
    n_pred = len(pd.read_csv(cfg.OUT_METRICS / "predictors_used.csv"))
    assert len(shap_imp) == len(perm_imp) == n_pred, (len(shap_imp), len(perm_imp), n_pred)
    merged = shap_imp.merge(perm_imp[["feature_code", "perm_importance_mean", "perm_rank"]],
                            on="feature_code", how="inner")

    # Stacked, not side-by-side: the heatmap's feature names are long enough that
    # a two-column layout pushes them across the neighbouring panel.
    fig, axes = plt.subplots(2, 1, figsize=(ps.W_DOUBLE, 7.5),
                             gridspec_kw={"height_ratios": [1.0, 1.32], "hspace": 0.26})
    axS, axM = axes
    axS.set_box_aspect(0.62)

    metals = set(cfg.PREDICTOR_VARS["metals"])
    top20 = merged.nsmallest(20, "shap_rank")
    for _, r in merged.iterrows():
        is_metal = r["feature_code"] in metals
        in_top = r["feature_code"] in set(top20["feature_code"])
        if is_metal:
            axS.scatter(r["shap_rank"], r["perm_rank"], s=42, color=ps.OKABE_VERMIL,
                        edgecolor="white", linewidth=0.7, zorder=4)
        elif in_top:
            axS.scatter(r["shap_rank"], r["perm_rank"], s=22, color=ps.OKABE_BLUE,
                        edgecolor="white", linewidth=0.5, alpha=0.85, zorder=3)
        else:
            axS.scatter(r["shap_rank"], r["perm_rank"], s=14, color="#BBBBBB", zorder=2)

    lim = merged[["shap_rank", "perm_rank"]].max().max() + 1
    axS.plot([0, lim], [0, lim], ls=(0, (4, 3)), color="#999999", lw=0.9, zorder=1)
    offsets = {"LBXBPB": (6, 13), "LBXBSE": (13, -16), "LBXBCD": (11, 1),
               "LBXBMN": (10, 2), "LBXTHG": (-10, -17)}
    for _, r in merged[merged["feature_code"].isin(metals)].iterrows():
        lab = fl.abbrev(r["feature_code"]).replace("Blood ", "")
        axS.annotate(lab, (r["shap_rank"], r["perm_rank"]), textcoords="offset points",
                     xytext=offsets.get(r["feature_code"], (7, 3)),
                     fontsize=7.6, color="#7E2F0E", fontweight="bold")
    rho = merged[["shap_rank", "perm_rank"]].corr(method="spearman").iloc[0, 1]
    axS.text(0.03, 0.94, f"Spearman ρ = {rho:.2f}", transform=axS.transAxes,
             ha="left", va="top", fontsize=8.4)
    axS.set_xlabel(f"Rank by mean |SHAP| ({shap_name})")
    axS.set_ylabel(f"Rank by permutation\nimportance ({shap_name})")
    axS.set_xlim(lim, 0.2); axS.set_ylim(lim, 0.2)
    axS.grid(True, alpha=0.4)
    ps.panel_label(axS, "A", dx=-0.115, dy=1.06)

    imp = perm_by_model.copy()
    imp["__mean__"] = imp.mean(axis=1)
    imp = imp.sort_values("__mean__", ascending=False).head(15).drop(columns="__mean__")
    ranks = perm_by_model.rank(ascending=False, method="min").loc[imp.index]
    _record_abbrevs("fig7_importance_concordance",
                    list(metals) + list(imp.index))
    ranks.index = [fl.abbrev(i) for i in ranks.index]

    # vmax follows the data rather than a fixed 25: a hard cap silently clipped any rank
    # above it to the darkest colour, which made a rank of 27 indistinguishable from 25.
    rank_max = int(ranks.values.max())
    im = axM.imshow(ranks.values, cmap="YlGnBu", vmin=1, vmax=rank_max, aspect="auto")
    axM.set_xticks(range(ranks.shape[1]))
    axM.set_xticklabels(ranks.columns, rotation=32, ha="right", fontsize=8)
    axM.set_yticks(range(ranks.shape[0]))
    axM.set_yticklabels(ranks.index, fontsize=7.8)
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            v = int(ranks.values[i, j])
            # YlGnBu runs pale (low rank) -> dark navy (high rank), so dark text
            # belongs on the pale cells and white text on the dark ones.
            # YlGnBu runs pale -> dark navy, so the switch to white text has to track
            # vmax rather than sit at a fixed rank.
            axM.text(j, i, f"{v}", ha="center", va="center", fontsize=7.6,
                     color="white" if v >= 0.62 * rank_max else "#1A1A1A")
    axM.set_xticks(np.arange(-0.5, ranks.shape[1], 1), minor=True)
    axM.set_yticks(np.arange(-0.5, ranks.shape[0], 1), minor=True)
    axM.grid(which="minor", color="white", linewidth=1.2)
    axM.tick_params(which="minor", length=0)
    for s in axM.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=axM, fraction=0.030, pad=0.015)
    cb.set_label("Permutation-importance rank\n(1 = most important)", fontsize=7.6)
    cb.ax.tick_params(labelsize=7.4)
    ps.panel_label(axM, "B", dx=-0.115, dy=1.06)

    metal_ranks = {}
    for m in cfg.PREDICTOR_VARS["metals"]:
        row = merged[merged["feature_code"] == m]
        if len(row):
            metal_ranks[m] = {
                "clinical_name": cfg.label_feature(m),
                "shap_rank": int(row["shap_rank"].iloc[0]),
                "perm_rank": int(row["perm_rank"].iloc[0]),
                "mean_abs_shap": float(row["mean_abs_shap"].iloc[0]),
            }
        if m in perm_by_model.index:
            per_model = perm_by_model.rank(ascending=False, method="min").loc[m]
            metal_ranks.setdefault(m, {})["per_model_perm_rank"] = {
                k: int(v) for k, v in per_model.items()}
    NUMBERS["metal_ranks"] = metal_ranks
    NUMBERS["spearman_shap_perm"] = round(float(rho), 3)
    NUMBERS["n_predictors_ranked"] = int(len(merged))
    NUMBERS["shap_top"] = [
        {"feature": r.feature_code, "name": r.clinical_name,
         "mean_abs_shap": float(r.mean_abs_shap), "rank": int(r.shap_rank)}
        for r in shap_imp.nsmallest(12, "shap_rank").itertuples()]
    NUMBERS["perm_top"] = [
        {"feature": r.feature_code, "name": r.clinical_name,
         "perm_importance": float(r.perm_importance_mean), "rank": int(r.perm_rank)}
        for r in perm_imp.nsmallest(12, "perm_rank").itertuples()]
    # missing-flag signal, reported separately from the predictor ranking
    sm = shap_missing.sort_values("mean_abs_shap", ascending=False).head(4)
    NUMBERS["missingness_indicator_shap"] = [
        {"feature": r.feature_code, "name": r.clinical_name,
         "mean_abs_shap": round(float(r.mean_abs_shap), 5)} for r in sm.itertuples()]
    NUMBERS["n_transformed_columns_shap"] = int(len(shap_imp_all))
    return ps.save(fig, OUT_FIG / "fig7_importance_concordance")


# ======================================================================
# Figure 8 -- metals SHAP dependence
# ======================================================================
def fig8_metals_dependence(X_test, cont_cols, cat_cols, name):
    import shap
    pipe = joblib.load(cfg.OUT_MODELS / f"{name.replace(' ', '_')}.joblib")
    pre = pipe.named_steps["preprocess"]
    est = pipe.named_steps["model"]
    names = pu.transformed_feature_names(cont_cols, cat_cols)
    Xt = pd.DataFrame(pre.transform(X_test), columns=names)
    sv = shap.TreeExplainer(est).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    sv = pd.DataFrame(sv, columns=names)

    metals = cfg.PREDICTOR_VARS["metals"]
    fig, axes = plt.subplots(1, 5, figsize=(ps.W_DOUBLE, 2.45), sharey=True)
    summary = {}
    for ax, m in zip(axes, metals):
        raw = X_test[m].values
        s = sv[m].values
        ok = ~np.isnan(raw)
        raw_o, s_o = raw[ok], s[ok]
        lo, hi = np.nanpercentile(raw_o, [1, 98])
        ax.scatter(raw_o, s_o, s=5, color="#B9C9D6", alpha=0.55, lw=0, zorder=2)
        qs = np.linspace(0, 100, 13)
        edges = np.percentile(raw_o, qs)
        edges = np.unique(edges)
        centers, means = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (raw_o >= a) & (raw_o <= b)
            if sel.sum() >= 15:
                centers.append(np.median(raw_o[sel]))
                means.append(np.mean(s_o[sel]))
        ax.plot(centers, means, color=ps.OKABE_VERMIL, lw=1.7, zorder=4)
        ax.scatter(centers, means, s=13, color=ps.OKABE_VERMIL, zorder=5,
                   edgecolor="white", linewidth=0.5)
        ax.axhline(0, color="#999999", lw=0.8, ls=(0, (3, 3)), zorder=1)
        ax.set_xlim(lo, hi)
        units = cfg.label_feature(m).split("(")[-1].rstrip(")") if "(" in cfg.label_feature(m) else ""
        label = fl.abbrev(m) + (f" ({units})" if units else "")
        ax.set_xlabel(label, fontsize=8.2)
        ax.tick_params(labelsize=7.4)
        summary[m] = {"binned_centers": [round(float(c), 3) for c in centers],
                      "binned_mean_shap": [round(float(v), 4) for v in means],
                      "n_observed": int(ok.sum())}
    axes[0].set_ylabel("SHAP value (log-odds)", fontsize=8.6)
    _record_abbrevs("fig8_metals_shap_dependence", metals)
    NUMBERS["metals_dependence"] = summary
    return ps.save(fig, OUT_FIG / "fig8_metals_shap_dependence")


# ======================================================================
# Figure 9 -- subgroup AUC
# ======================================================================
def bootstrap_auc_ci(y, p, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def bootstrap_calibration_ci(y, p, n_boot=N_BOOTSTRAP, seed=RNG_SEED):
    """Percentile bootstrap intervals for the calibration slope, the calibration
    intercept and the observed-to-expected ratio, on the same 1,000 resamples
    used for AUROC so that every interval in the subgroup table rests on the
    same resampling scheme. Slope speaks to dispersion and O:E to
    calibration-in-the-large; they answer different questions and an interval on
    one does not license a claim about the other."""
    rng = np.random.default_rng(seed)
    n = len(y)
    slopes, intercepts, oes = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            s, i = calibration_slope_intercept(y[idx], p[idx])
        except Exception:
            continue
        slopes.append(s)
        intercepts.append(i)
        oes.append(y[idx].mean() / p[idx].mean())
    if len(slopes) < 50:
        nan = float("nan")
        return (nan, nan), (nan, nan), (nan, nan)
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(slopes), q(intercepts), q(oes)


def fig9_subgroups(preds, test_df, cmp_df):
    y = preds["y_true"].values
    top = cmp_df["Model"].iloc[0]
    p = preds[top].values

    age = test_df["RIDAGEYR"].values
    q1, q2 = np.percentile(age, [33.33, 66.67])
    groups = [("Overall", np.ones(len(y), bool))]
    groups += [("Male", (test_df["sex_female"] == 0).values),
               ("Female", (test_df["sex_female"] == 1).values)]
    groups += [(f"Age {int(age.min())}–{int(q1)} y", age <= q1),
               (f"Age {int(q1) + 1}–{int(q2)} y", (age > q1) & (age <= q2)),
               (f"Age > {int(q2)} y", age > q2)]
    eth = {"RIDRETH1_3.0": "Non-Hispanic White", "RIDRETH1_4.0": "Non-Hispanic Black",
           "RIDRETH1_2.0": "Other Hispanic", "RIDRETH1_5.0": "Other/Multi-racial"}
    for col, lab in eth.items():
        if col in test_df.columns:
            groups.append((lab, (test_df[col] == 1).values))
    mex = np.ones(len(y), bool)
    for col in eth:
        if col in test_df.columns:
            mex &= (test_df[col] == 0).values
    groups.insert(-3, ("Mexican American", mex))

    rows = []
    for lab, mask in groups:
        if mask.sum() < 40 or len(np.unique(y[mask])) < 2:
            continue
        auc = roc_auc_score(y[mask], p[mask])
        lo, hi = bootstrap_auc_ci(y[mask], p[mask])
        # subgroup CALIBRATION, not only discrimination: a model can rank well
        # inside a stratum while systematically mis-stating absolute risk there,
        # and reporting AUC alone would hide that. Slope gets its own bootstrap
        # interval because several strata are small enough that a point estimate
        # on its own would invite over-reading.
        slope, intercept = calibration_slope_intercept(y[mask], p[mask])
        (s_lo, s_hi), (i_lo, i_hi), (oe_lo, oe_hi) = bootstrap_calibration_ci(y[mask], p[mask])
        rows.append({"subgroup": lab, "n": int(mask.sum()),
                     "n_positive": int(y[mask].sum()),
                     "prevalence_pct": round(float(y[mask].mean() * 100), 1),
                     "auc": round(float(auc), 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "calibration_slope": round(float(slope), 3),
                     "slope_ci_low": round(s_lo, 3), "slope_ci_high": round(s_hi, 3),
                     "calibration_intercept": round(float(intercept), 3),
                     "intercept_ci_low": round(i_lo, 3), "intercept_ci_high": round(i_hi, 3),
                     "observed_expected_ratio": round(float(y[mask].mean() / p[mask].mean()), 3),
                     "oe_ci_low": round(oe_lo, 3), "oe_ci_high": round(oe_hi, 3)})
    sg = pd.DataFrame(rows)
    sg.to_csv(cfg.OUT_METRICS / "subgroup_auc.csv", index=False)

    fig, ax = plt.subplots(figsize=(ps.W_DOUBLE * 0.72, 3.9))
    order = sg.iloc[::-1].reset_index(drop=True)
    for i, r in order.iterrows():
        is_overall = r["subgroup"] == "Overall"
        col = ps.OKABE_BLUE if is_overall else "#666666"
        ax.plot([r["ci_low"], r["ci_high"]], [i, i], color=col, lw=1.4, solid_capstyle="round")
        ax.plot(r["auc"], i, "D" if is_overall else "o", color=col, ms=5.2 if is_overall else 4.4,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
        ax.text(1.02, i, f"{int(r['n']):,}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7.6, color="#444444")
    overall_auc = sg.loc[sg.subgroup == "Overall", "auc"].iloc[0]
    ax.axvline(overall_auc, color=ps.OKABE_BLUE, lw=0.9, ls=(0, (4, 3)), alpha=0.7, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order["subgroup"], fontsize=8.2)
    ax.set_xlabel(f"Area under the ROC curve, {top} (95% CI)")
    ax.set_xlim(0.60, 1.0)
    ax.grid(True, axis="x", alpha=0.5)
    ax.text(1.02, len(order) - 0.15, "n", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=7.8, style="italic", color="#444444")

    NUMBERS["subgroups"] = rows
    NUMBERS["subgroup_model"] = top
    return ps.save(fig, OUT_FIG / "fig9_subgroup_auc")


# ======================================================================
# Methods Figure M1 -- analysis pipeline schematic
# ======================================================================
N_PRED = None


def figM1_pipeline():
    fig, ax = plt.subplots(figsize=(ps.W_DOUBLE, 3.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")

    stages = [
        ("1. Data assembly", "Six NHANES cycles\n2007–2018, merged on\nSEQN (13 modules)", "#EAF2F8", ps.OKABE_BLUE),
        ("2. Outcome", "Harmonized MetS\n(Alberti 2009; ≥3 of 5,\nmeasured value or\ndrug treatment)", "#E8F6F3", ps.OKABE_GREEN),
        ("3. Predictors", f"{N_PRED} predictor columns\nacross metal,\ndemographic, body-size\nand lifestyle domains", "#FEF9E7", "#B9770E"),
        ("4. Partition", "80/20 stratified split;\ntest set sealed until\nfinal evaluation", "#FDEDEC", ps.OKABE_VERMIL),
    ]
    x = 2.0; w = 22.6; y = 25.0; h = 15.0
    for i, (title, body, face, edge) in enumerate(stages):
        xi = x + i * (w + 2.4)
        ax.add_patch(FancyBboxPatch((xi, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.8",
                                    facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(xi + w / 2, y + h - 2.8, title, ha="center", va="center",
                fontsize=8.4, fontweight="bold", color=edge)
        ax.text(xi + w / 2, y + h / 2 - 2.6, body, ha="center", va="center",
                fontsize=6.9, linespacing=1.42, color="#333333")
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((xi + w, y + h / 2), (xi + w + 2.2, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#555555"))

    ax.add_patch(FancyBboxPatch((2.0, 3.2), 96.0, 17.8,
                                boxstyle="round,pad=0.4,rounding_size=0.8",
                                facecolor="#F8F9F9", edgecolor="#5D6D7E", linewidth=1.0))
    ax.text(50.0, 18.5, "5. Leakage-safe model development — every step below refitted inside each training fold",
            ha="center", va="center", fontsize=8.2, fontweight="bold", color="#2C3E50")

    inner = [
        ("Median / mode\nimputation", "#FFFFFF"),
        ("Standardization\n+ missingness flags", "#FFFFFF"),
        ("SMOTENC\noversampling", "#FFFFFF"),
        ("11 classifiers\n(10-fold tuning)", "#FFFFFF"),
        ("Threshold from\nout-of-fold preds", "#FFFFFF"),
    ]
    iw = 16.6; ix = 5.0; iy = 5.6; ih = 9.8
    for i, (txt, face) in enumerate(inner):
        xi = ix + i * (iw + 2.6)
        ax.add_patch(FancyBboxPatch((xi, iy), iw, ih, boxstyle="round,pad=0.3,rounding_size=0.6",
                                    facecolor=face, edgecolor="#85929E", linewidth=0.9))
        ax.text(xi + iw / 2, iy + ih / 2, txt, ha="center", va="center",
                fontsize=7.4, linespacing=1.45, color="#2C3E50")
        if i < len(inner) - 1:
            ax.add_patch(FancyArrowPatch((xi + iw, iy + ih / 2), (xi + iw + 2.4, iy + ih / 2),
                                         arrowstyle="-|>", mutation_scale=8, lw=0.85, color="#85929E"))

    ax.add_patch(FancyArrowPatch((50.0, 25.0), (50.0, 21.2),
                                 arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#555555"))
    return ps.save(fig, OUT_FIG / "figM1_analysis_pipeline")


# ======================================================================
def main():
    print("Loading data and model artifacts...")
    train_df, test_df, X_test, y_test, cmp_df, sig_df, feature_cols, cont_cols, cat_cols = load_everything()
    print(f"  test set: n={len(y_test)}, positives={int(y_test.sum())}")

    print("\nBuilding prediction cache...")
    preds = build_prediction_cache(cmp_df, X_test, y_test)

    print("\nFigure 1: participant flow")
    fig1_participant_flow()
    print("Figure 2: ROC curves")
    fig2_roc(preds, cmp_df)
    print("Figure 3: AUC forest")
    fig3_auc_forest(cmp_df, sig_df)
    print("Figure 4: calibration")
    fig4_calibration(preds, cmp_df)
    print("Figure 5: decision curve")
    fig5_decision_curve(preds, cmp_df)
    shap_name = shap_model_name(cmp_df)
    NUMBERS["shap_model"] = shap_name
    print(f"Figure 6: SHAP beeswarm ({shap_name})")
    fig6_shap_beeswarm(X_test, cont_cols, cat_cols, shap_name)
    print("Figure 7: importance concordance")
    perm_by_model = compute_perm_by_model(cmp_df, X_test, y_test)
    fig7_concordance(perm_by_model, shap_name)
    print("Figure 8: metals SHAP dependence")
    fig8_metals_dependence(X_test, cont_cols, cat_cols, shap_name)
    print("Figure 9: subgroup AUC")
    fig9_subgroups(preds, test_df, cmp_df)
    print("Figure M1: analysis pipeline")
    global N_PRED
    N_PRED = len(feature_cols)
    figM1_pipeline()

    # ---- extra numbers the manuscript text quotes ----
    NUMBERS["performance"] = cmp_df.set_index("Model").round(4).to_dict(orient="index")
    NUMBERS["tuning"] = pd.read_csv(cfg.OUT_METRICS / "tuning_summary.csv").set_index("model").to_dict(orient="index")
    NUMBERS["thresholds"] = json.loads((cfg.OUT_METRICS / "thresholds.json").read_text())
    NUMBERS["outcome_definition"] = pd.read_csv(
        cfg.OUT_METRICS / "outcome_definition_comparison.csv").to_dict(orient="records")
    for f in ["sensitivity_diq010", "sensitivity_metals_scope",
              "sensitivity_smote_vs_classweight", "sensitivity_outcome_definition"]:
        NUMBERS[f] = pd.read_csv(cfg.OUT_METRICS / f"{f}.csv").to_dict(orient="records")
    NUMBERS["collinearity"] = pd.read_csv(cfg.OUT_METRICS / "collinearity_flags.csv").to_dict(orient="records")
    NUMBERS["missingness"] = pd.read_csv(cfg.OUT_METRICS / "missingness_report.csv").to_dict(orient="records")
    NUMBERS["n_predictors"] = int(len(pd.read_csv(cfg.OUT_METRICS / "predictors_used.csv")))
    NUMBERS["n_transformed_features"] = int(len(pu.transformed_feature_names(cont_cols, cat_cols)))
    NUMBERS["train_prevalence_pct"] = round(float(train_df[cfg.OUTCOME_VAR].mean() * 100), 4)
    NUMBERS["test_prevalence_pct"] = round(float(test_df[cfg.OUTCOME_VAR].mean() * 100), 4)

    out = cfg.OUT_METRICS / "publication_numbers.json"
    out.write_text(json.dumps(NUMBERS, indent=2, default=str))
    print(f"\nWrote {out}")
    print(f"Figures in {OUT_FIG}")


if __name__ == "__main__":
    main()
