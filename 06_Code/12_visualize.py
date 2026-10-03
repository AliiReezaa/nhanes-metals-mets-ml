"""
Visualization suite -- traditional + modern views of the full 11-model comparison.

Traditional / conventional:
  - outputs/figures/metrics_bar_comparison.png   (grouped bar chart, all models x key metrics)
  - outputs/figures/confusion_matrix_grid.png    (one confusion matrix per model, tuned cutoff)
  - (roc_curves.png and calibration_curves.png already exist from 08_evaluate.py)

Modern / creative:
  - outputs/figures/metrics_heatmap.png          (models x metrics, color-coded, annotated)
  - outputs/figures/radar_chart.png              (top 6 models, multi-metric radar/spider)
  - outputs/dashboard.html                       (self-contained interactive Plotly dashboard:
                                                    sortable metrics table, ROC curves, AUC ranking
                                                    with 95% CI, radar comparison, feature importance)

Reads only already-saved outputs (model_comparison.csv, the fitted pipelines in outputs/models/,
shap_importance.csv, permutation_importance.csv) -- run 08_evaluate.py and 09_explain_shap.py first.

REBUILD NOTE / FIX: this script's charts and dashboard rank models purely
by point-estimate test AUC, which on its own invites a reader to treat the
#1 model as unambiguously "the best" -- 08_evaluate.py's paired bootstrap
test (model_comparison_significance.csv) frequently finds several
top-ranked models statistically indistinguishable from each other on a
2,229-row test set. This script now reads that column (merged into
model_comparison.csv by 08_evaluate.py) where available and marks tied
models with a "*" in every chart/table that ranks models, plus a caption
on the dashboard -- so the same "don't overclaim a single best model"
correction made in 08_evaluate.py/09_explain_shap.py is visible everywhere
these results get looked at, not just in the raw CSV.

Run:
    python scripts/12_visualize.py
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
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
import pipeline_utils as pu

KEY_METRICS = ["AUC", "AUPRC", "Accuracy_tuned_cutoff", "Sensitivity_tuned_cutoff",
               "Specificity_tuned_cutoff", "F1_tuned_cutoff", "Kappa_tuned_cutoff"]
KEY_METRICS_LABELS = ["AUC", "AUPRC", "Accuracy", "Sensitivity", "Specificity", "F1", "Kappa"]

MODEL_COLORS = {}  # filled in main() with a consistent color per model across all charts

TIE_COL = "statistically_indistinguishable_from_top"


def display_label(name, cmp_df):
    """FIX: append a '*' to a model's display label when 08_evaluate.py's
    paired bootstrap test found it statistically indistinguishable from the
    top-ranked model -- see module docstring."""
    if TIE_COL in cmp_df.columns:
        row = cmp_df.loc[cmp_df["Model"] == name]
        if len(row) and bool(row.iloc[0][TIE_COL]):
            return f"{name} *"
    return name


def load_everything():
    cmp_df = pd.read_csv(cfg.OUT_METRICS / "model_comparison.csv").sort_values("AUC", ascending=False).reset_index(drop=True)
    with open(cfg.OUT_METRICS / "thresholds.json") as f:
        thresholds = json.load(f)

    _, test_df = pu.load_model_ready_split()
    feature_cols, _, _ = pu.get_feature_lists()
    X_test = test_df[feature_cols]
    y_test = test_df[cfg.OUTCOME_VAR].values

    per_model = {}
    for name in cmp_df["Model"]:
        path = cfg.OUT_MODELS / f"{name.replace(' ', '_')}.joblib"
        if not path.exists():
            continue
        pipe = joblib.load(path)
        prob = pipe.predict_proba(X_test)[:, 1]
        cutoff = thresholds.get(name, 0.5)
        y_pred = (prob >= cutoff).astype(int)
        fpr, tpr, _ = roc_curve(y_test, prob)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        per_model[name] = {"prob": prob, "fpr": fpr, "tpr": tpr, "cm": cm, "cutoff": cutoff}

    if TIE_COL not in cmp_df.columns:
        print(f"  [NOTE] '{TIE_COL}' column not found in model_comparison.csv -- "
              f"re-run 08_evaluate.py to get the paired-bootstrap tie test; "
              f"proceeding without tie annotations for now.")

    return cmp_df, per_model, y_test


# ============================================================
# TRADITIONAL: grouped bar chart
# ============================================================
def plot_metrics_bar(cmp_df):
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(cmp_df))
    n_metrics = len(KEY_METRICS)
    width = 0.8 / n_metrics
    palette = sns.color_palette("colorblind", n_metrics)

    for i, (metric, label) in enumerate(zip(KEY_METRICS, KEY_METRICS_LABELS)):
        vals = cmp_df[metric].values
        ax.bar(x + i * width, vals, width=width, label=label, color=palette[i])

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels([display_label(m, cmp_df) for m in cmp_df["Model"]], rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    title = "Model Comparison Across Key Metrics (Test Set)"
    if TIE_COL in cmp_df.columns:
        title += "\n(* = statistically indistinguishable from top model, paired bootstrap 95% CI)"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=n_metrics, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.OUT_FIGURES / "metrics_bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# TRADITIONAL: confusion matrix grid
# ============================================================
def plot_confusion_grid(cmp_df, per_model):
    models = [m for m in cmp_df["Model"] if m in per_model]
    n = len(models)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(models):
        ax = axes[i]
        cm = per_model[name]["cm"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["No MetS", "MetS"], yticklabels=["No MetS", "MetS"], vmin=0, vmax=1)
        ax.set_title(f"{display_label(name, cmp_df)}\n(cutoff={per_model[name]['cutoff']:.3f})", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Confusion Matrices -- All Models (Test Set, Tuned Cutoff)", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(cfg.OUT_FIGURES / "confusion_matrix_grid.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# MODERN: metrics heatmap
# ============================================================
def plot_metrics_heatmap(cmp_df):
    data = cmp_df.set_index("Model")[KEY_METRICS].copy()
    data.columns = KEY_METRICS_LABELS
    data.index = [display_label(m, cmp_df) for m in data.index]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(data) + 2))
    sns.heatmap(data, annot=True, fmt=".3f", cmap="viridis", cbar_kws={"label": "Score"}, ax=ax,
                linewidths=0.5, linecolor="white")
    title = "Model Performance Heatmap (Test Set)"
    if TIE_COL in cmp_df.columns:
        title += "\n(* = statistically indistinguishable from top model)"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(cfg.OUT_FIGURES / "metrics_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# MODERN: radar chart, top 6 models
# ============================================================
def _radar_axis_range(top_df):
    """Top-N models on AUC etc. tend to cluster tightly (e.g. 0.80-0.81) --
    a fixed 0-1 radial axis flattens all of that into an indistinguishable
    blob. Zoom to the actual data range (with padding) so differences
    between close performers are actually visible, same idea as not
    starting a bar chart's y-axis at zero when the signal is in small
    differences between large numbers."""
    vals = top_df[KEY_METRICS].values
    lo, hi = vals.min(), vals.max()
    pad = max((hi - lo) * 0.15, 0.02)
    return max(0, lo - pad), min(1, hi + pad)


def plot_radar(cmp_df, top_n=6):
    top = cmp_df.head(top_n)
    labels = KEY_METRICS_LABELS
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    r_min, r_max = _radar_axis_range(top)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    palette = sns.color_palette("tab10", len(top))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for i, (_, row) in enumerate(top.iterrows()):
        values = row[KEY_METRICS].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=display_label(row["Model"], cmp_df), color=palette[i],
                 marker=markers[i % len(markers)], markersize=7)
        ax.fill(angles, values, alpha=0.06, color=palette[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(r_min, r_max)
    subtitle = f"(axis zoomed to {r_min:.2f}-{r_max:.2f} to show differences between close performers)"
    if TIE_COL in cmp_df.columns:
        subtitle += "\n(* = statistically indistinguishable from top model, paired bootstrap 95% CI)"
    ax.set_title(f"Top {top_n} Models -- Multi-Metric Radar Comparison\n{subtitle}", fontsize=13, fontweight="bold", pad=44)
    ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1.0), fontsize=10)
    plt.tight_layout()
    plt.savefig(cfg.OUT_FIGURES / "radar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# MODERN/CREATIVE: interactive Plotly dashboard
# ============================================================
def build_dashboard(cmp_df, per_model, y_test):
    global MODEL_COLORS
    colors = sns.color_palette("tab10", len(cmp_df)).as_hex()
    MODEL_COLORS = dict(zip(cmp_df["Model"], colors))
    has_tie_col = TIE_COL in cmp_df.columns

    sections = []

    # --- AUC ranking with 95% CI ---
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=cmp_df["AUC"], y=[display_label(m, cmp_df) for m in cmp_df["Model"]], orientation="h",
        error_x=dict(type="data", symmetric=False,
                     array=cmp_df["AUC_95CI_high"] - cmp_df["AUC"],
                     arrayminus=cmp_df["AUC"] - cmp_df["AUC_95CI_low"]),
        marker_color=[MODEL_COLORS[m] for m in cmp_df["Model"]],
        text=cmp_df["AUC"].round(3), textposition="outside",
    ))
    auc_title = "Test AUC by Model (95% bootstrap CI)"
    if has_tie_col:
        auc_title += " -- * = statistically indistinguishable from top model (paired bootstrap)"
    fig1.update_layout(title=auc_title, xaxis_title="AUC", yaxis_title="",
                        yaxis=dict(autorange="reversed"), height=500, template="plotly_white")

    # --- Interactive ROC curves ---
    fig2 = go.Figure()
    for name in cmp_df["Model"]:
        if name not in per_model:
            continue
        d = per_model[name]
        auc = cmp_df.loc[cmp_df["Model"] == name, "AUC"].values[0]
        fig2.add_trace(go.Scatter(x=d["fpr"], y=d["tpr"], mode="lines",
                                   name=f"{display_label(name, cmp_df)} (AUC={auc:.3f})",
                                   line=dict(color=MODEL_COLORS[name])))
    fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                               line=dict(color="gray", dash="dash")))
    fig2.update_layout(title="ROC Curves (interactive -- hover, zoom, click legend to toggle)",
                        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                        height=550, template="plotly_white")

    # --- Metrics heatmap (interactive) ---
    heat_data = cmp_df.set_index("Model")[KEY_METRICS]
    heat_data.columns = KEY_METRICS_LABELS
    heat_data.index = [display_label(m, cmp_df) for m in heat_data.index]
    fig3 = go.Figure(data=go.Heatmap(
        z=heat_data.values, x=heat_data.columns, y=heat_data.index,
        colorscale="Viridis", text=np.round(heat_data.values, 3), texttemplate="%{text}",
        hovertemplate="%{y} / %{x}: %{z:.4f}<extra></extra>",
    ))
    fig3.update_layout(title="Metrics Heatmap (hover for exact values)", height=500, template="plotly_white")

    # --- Radar (top 6, interactive) ---
    top6 = cmp_df.head(6)
    r_min, r_max = _radar_axis_range(top6)
    fig4 = go.Figure()
    for _, row in top6.iterrows():
        vals = row[KEY_METRICS].tolist()
        fig4.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=KEY_METRICS_LABELS + [KEY_METRICS_LABELS[0]],
                                        fill="toself", name=display_label(row["Model"], cmp_df),
                                        line=dict(color=MODEL_COLORS[row["Model"]])))
    fig4.update_layout(title=f"Top 6 Models -- Multi-Metric Radar (axis zoomed to {r_min:.2f}-{r_max:.2f})",
                        polar=dict(radialaxis=dict(range=[r_min, r_max])),
                        height=550, template="plotly_white")

    # --- Feature importance (SHAP, if available) ---
    shap_path = cfg.OUT_METRICS / "shap_importance.csv"
    fig5 = None
    if shap_path.exists():
        shap_df = pd.read_csv(shap_path).head(15).sort_values("mean_abs_shap")
        fig5 = go.Figure(go.Bar(x=shap_df["mean_abs_shap"], y=shap_df["clinical_name"], orientation="h",
                                 marker_color="teal"))
        fig5.update_layout(title="Top 15 Features by Mean |SHAP Value|", xaxis_title="Mean |SHAP value|",
                            height=550, template="plotly_white")

    # --- Sortable metrics table ---
    table_cols = ["Model"] + KEY_METRICS
    table_labels = ["Model"] + KEY_METRICS_LABELS
    table_values = [[display_label(m, cmp_df) for m in cmp_df["Model"]]] + \
                   [cmp_df[c].round(4) for c in KEY_METRICS]
    if has_tie_col:
        table_labels = table_labels + ["Tied w/ top (95%)?"]
        table_values = table_values + [cmp_df[TIE_COL].map({True: "Yes", False: "No"})]
    fig6 = go.Figure(data=[go.Table(
        header=dict(values=table_labels, fill_color="#2E5395", font=dict(color="white", size=12), align="left"),
        cells=dict(values=table_values,
                   fill_color=[["#f5f5fa" if i % 2 == 0 else "white" for i in range(len(cmp_df))]],
                   align="left", height=26),
    )])
    fig6.update_layout(title="Full Metrics Table (Test Set)", height=120 + 30 * len(cmp_df))

    figs = [fig6, fig1, fig2, fig3, fig4] + ([fig5] if fig5 is not None else [])

    caveat_html = ""
    if has_tie_col:
        n_tied = int(cmp_df[TIE_COL].sum())
        caveat_html = (
            f"<p><b>Statistical caveat:</b> {n_tied} of {len(cmp_df)} models (marked with *) are "
            f"statistically indistinguishable from the top-ranked model at the 95% level, per a paired "
            f"bootstrap test on the test set (see outputs/metrics/model_comparison_significance.csv). "
            f"The point-estimate ranking above should not be read as proof that the #1 model is the "
            f"uniquely best-performing one.</p>"
        )

    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Metabolic Syndrome Model Comparison Dashboard</title>",
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
        "<style>body{font-family:-apple-system,Segoe UI,Arial,sans-serif;background:#fafafa;margin:0;padding:24px;}",
        "h1{color:#1F3864;} .chart{background:white;border-radius:10px;padding:12px;margin-bottom:28px;",
        "box-shadow:0 1px 4px rgba(0,0,0,0.08);}</style></head><body>",
        "<h1>Metabolic Syndrome Prediction -- Model Comparison Dashboard</h1>",
        "<p>All charts are interactive: hover for exact values, click legend entries to toggle, "
        "drag to zoom. Data from outputs/metrics/model_comparison.csv.</p>",
        caveat_html,
    ]
    for i, fig in enumerate(figs):
        html_parts.append(f"<div class='chart'>{fig.to_html(full_html=False, include_plotlyjs=False, div_id=f'chart{i}')}</div>")
    html_parts.append("</body></html>")

    out_path = cfg.OUT_DIR / "dashboard.html"
    with open(out_path, "w") as f:
        f.write("\n".join(html_parts))
    return out_path


def main():
    print("Loading model comparison data and recomputing per-model test predictions...")
    cmp_df, per_model, y_test = load_everything()
    print(f"Loaded {len(per_model)} fitted models.")

    print("Building traditional charts...")
    plot_metrics_bar(cmp_df)
    plot_confusion_grid(cmp_df, per_model)
    print(f"  saved: {cfg.OUT_FIGURES / 'metrics_bar_comparison.png'}")
    print(f"  saved: {cfg.OUT_FIGURES / 'confusion_matrix_grid.png'}")

    print("Building modern/creative charts...")
    plot_metrics_heatmap(cmp_df)
    plot_radar(cmp_df)
    print(f"  saved: {cfg.OUT_FIGURES / 'metrics_heatmap.png'}")
    print(f"  saved: {cfg.OUT_FIGURES / 'radar_chart.png'}")

    print("Building interactive dashboard...")
    dash_path = build_dashboard(cmp_df, per_model, y_test)
    print(f"  saved: {dash_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
