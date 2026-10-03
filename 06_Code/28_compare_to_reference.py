"""
Final check -- does my rerun reproduce the published numbers?

Compares the tables a fresh run writes to outputs/metrics/ against the
reference copies shipped in 04_Results_Tables/ (and the split files against
05_Data_Processed/). It prints one line per check, PASS or DIFF, and writes
the same report to outputs/reproduction_report.txt.

Tolerances: discrimination and calibration numbers may move in the third
decimal place when library versions or the operating system differ (tree
libraries use floating-point arithmetic that is not bit-identical across
CPUs). A difference of 0.005 or less in AUROC is treated as a successful
reproduction; anything larger is flagged for a closer look.

Run:
    python 06_Code/28_compare_to_reference.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

REF = cfg.PROJECT_ROOT / "04_Results_Tables"
REF_DATA = cfg.PROJECT_ROOT / "05_Data_Processed"
NEW = cfg.OUT_METRICS
TOL_AUC = 0.005
lines = []


def say(status, what, detail=""):
    line = f"[{status}] {what}" + (f"  ({detail})" if detail else "")
    print(line)
    lines.append(line)


def compare_numeric(name, key, cols, tol):
    a, b = REF / name, NEW / name
    if not a.exists() or not b.exists():
        say("SKIP", name, "file missing in reference or new run")
        return
    ra, rb = pd.read_csv(a).set_index(key), pd.read_csv(b).set_index(key)
    worst, where = 0.0, ""
    for c in cols:
        if c not in ra.columns or c not in rb.columns:
            continue
        common = ra.index.intersection(rb.index)
        d = (ra.loc[common, c].astype(float) - rb.loc[common, c].astype(float)).abs()
        if len(d) and d.max() > worst:
            worst, where = float(d.max()), f"{d.idxmax()} / {c}"
    say("PASS" if worst <= tol else "DIFF", f"{name}", f"largest difference {worst:.4f} at {where}")


def main():
    # 1. identical cohort and split
    for f in ["train_seqn.csv", "test_seqn.csv"]:
        a, b = REF_DATA / f, cfg.DATA_PROC_DIR / f
        if a.exists() and b.exists():
            same = set(pd.read_csv(a).SEQN) == set(pd.read_csv(b).SEQN)
            say("PASS" if same else "DIFF", f"{f}: same participants as the published split")
    a, b = REF / "exclusion_flow.csv", NEW / "exclusion_flow.csv"
    if a.exists() and b.exists():
        same = pd.read_csv(a).n_remaining.tolist() == pd.read_csv(b).n_remaining.tolist()
        say("PASS" if same else "DIFF", "exclusion_flow.csv: same cohort counts")

    # 2. model performance and calibration
    compare_numeric("model_comparison.csv", "Model", ["AUC", "AUPRC", "Brier"], TOL_AUC)
    compare_numeric("calibration_metrics.csv", "model", ["calibration_slope"], 0.05)
    compare_numeric("tuning_summary.csv", "model", ["cv_auc"], TOL_AUC)

    # 3. sensitivity analyses
    for f, key in [("sensitivity_diq010.csv", "model"), ("sensitivity_metals_scope.csv", "model"),
                   ("sensitivity_smote_vs_classweight.csv", "model"),
                   ("sensitivity_outcome_definition.csv", "model")]:
        cols = [c for c in pd.read_csv(REF / f).columns if c.startswith("cv_auc")] if (REF / f).exists() else []
        compare_numeric(f, key, cols, TOL_AUC)

    # 4. top of the importance ranking
    a, b = REF / "shap_importance.csv", NEW / "shap_importance.csv"
    if a.exists() and b.exists():
        top_a = pd.read_csv(a).feature_code.head(5).tolist()
        top_b = pd.read_csv(b).feature_code.head(5).tolist()
        say("PASS" if top_a == top_b else "DIFF", "shap_importance.csv: same top-5 predictors",
            f"published {top_a} / rerun {top_b}")

    import os
    if os.environ.get("RUN_ME_MODE") == "quick":
        say("NOTE", "QUICK mode uses 3-fold searches, so performance tables are expected to differ "
            "slightly from the published 10-fold results; run PACKAGED or FULL to reproduce them.")
    out = cfg.OUT_DIR / "reproduction_report.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n_diff = sum(l.startswith("[DIFF]") for l in lines)
    n_checks = sum(l.startswith(("[PASS]", "[DIFF]")) for l in lines)
    print(f"\n{n_checks} checks, {n_diff} flagged. Report saved to {out}")


if __name__ == "__main__":
    main()
