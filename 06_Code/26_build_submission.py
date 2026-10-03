"""
Refresh the publication folders of this package from a finished pipeline run.

2026-10 audit (C9, C11, C24): the package IS the repository, so this script now works in
place. It copies the outputs of the latest run (outputs/, data_processed/, manuscript/)
into the published folders (01_Manuscript, 03_Figures, 04_Results_Tables,
05_Data_Processed, 07_Novelty_Additions, 08_Q1_Rigor_Additions), renders each Word
document to PDF with LibreOffice when it is installed, writes README.md, and regenerates
00_MANIFEST.csv over every published file. The earlier version built a separate
SUBMISSION/ folder from a scripts/ + docx_final/ working layout that is not part of this
package, so it could not run here. The published folders are emptied before copying, so
no stale file from an earlier run survives.

Run (after RUN_ME.py, or after the steps listed in README.md):
    python 06_Code/26_build_submission.py

Two conventions are worth knowing before reading the code.

Figures are renamed on the way in. In the working tree they carry the name of the routine
that draws them (fig6_shap_beeswarm); in the package they carry the number the manuscript
cites (Figure_3_SHAP_Summary). FIGURE_MAP is the single place that mapping is written
down, and the script fails if any entry points at a file that does not exist, so the
package cannot silently omit a figure the text refers to.

The manifest is generated, not maintained. Every file that lands in the package is listed
with its size and an MD5 checksum, alongside a one-line description resolved from
DESCRIPTIONS by exact name first and by pattern second. A file that matches no description
is reported rather than given a vague placeholder.
"""
from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT  # the package is the repository itself
TODAY = date.today().isoformat()

FIG_SRC = ROOT / "outputs" / "figures_pub"
MET = ROOT / "outputs" / "metrics"

# Folders that are regenerated on every run and never published.
NOT_PUBLISHED = {".git", ".venv", "venv", "outputs", "data_processed", "NHANES_data",
                 "manuscript", "diagrams", "__pycache__", ".pytest_cache", "catboost_info"}
NOT_PUBLISHED_FILES = {".DS_Store", "00_MANIFEST.csv"}

ADDITION_FOLDERS = ["07_Novelty_Additions", "08_Q1_Rigor_Additions"]


# ----------------------------------------------------------------------
# Figure numbering: working-tree name -> the number the manuscript cites
# ----------------------------------------------------------------------
FIGURE_MAP: list[tuple[str, str, str]] = [
    ("fig1_participant_flow",        "Figure_1_Participant_Flow",
     "Participant flow through the five exclusion steps, with cycle contributions (Methods)"),
    ("fig2_roc_curves",              "Figure_2_ROC_Curves",
     "Receiver-operating-characteristic curves for all models on the held-out test set"),
    ("fig6_shap_beeswarm",           "Figure_3_SHAP_Summary",
     "SHAP summary (beeswarm) for the leading model, 18 highest-ranked predictors"),
    ("fig7_importance_concordance",  "Figure_4_Importance_Concordance",
     "Agreement between SHAP and permutation importance, and across the four leading models"),
    ("fig8_metals_shap_dependence",  "Figure_5_Metals_SHAP_Dependence",
     "SHAP dependence for the five blood metals in the leading model"),
    ("figM1_analysis_pipeline",      "Figure_S1_Analysis_Pipeline",
     "Schematic of the leakage-safe modelling pipeline"),
    ("fig3_auc_forest",              "Figure_S2_AUROC_Forest",
     "AUROC forest plot with paired differences (DeLong) against the leading model"),
    ("fig4_calibration",             "Figure_S3_Calibration",
     "Calibration curves with slope, intercept and observed-to-expected ratio"),
    ("fig5_decision_curve",          "Figure_S4_Decision_Curve",
     "Decision-curve analysis: net benefit against treat-all and treat-none"),
    ("fig9_subgroup_auc",            "Figure_S5_Subgroup_AUROC",
     "Discrimination within age, sex and race/ethnicity subgroups"),
]


# ----------------------------------------------------------------------
# What goes in, and where
# ----------------------------------------------------------------------
DOCS: list[tuple[Path, str, str]] = [
    (ROOT / "manuscript" / "Manuscript_Methods_AI.docx",
     "01_Manuscript/01_Methods_AI.docx",
     "Methods section, AI/ML component (TRIPOD+AI structured, 21 verified references)"),
    (ROOT / "manuscript" / "Manuscript_Results_AI.docx",
     "01_Manuscript/02_Results_AI.docx",
     "Results section, AI/ML component, with Tables 1-2 and Figures 2-5 embedded"),
    (ROOT / "manuscript" / "Manuscript_Supplement_AI.docx",
     "01_Manuscript/03_Supplementary_Material.docx",
     "Supplementary material: Tables S1-S7 and Figures S1-S5"),
]
# The proposal is edited in place in 02_Proposal/ (its PDF is re-rendered below).
PROPOSAL = ROOT / "02_Proposal" / "Research_Proposal.docx"

DATA: list[tuple[Path, str, str]] = [
    (ROOT / "data_processed" / "model_ready.csv",
     "05_Data_Processed/model_ready.csv",
     "Analytic cohort with engineered predictors and the MetS outcome; input to modelling"),
    (ROOT / "data_processed" / "nhanes_with_outcome.csv",
     "05_Data_Processed/nhanes_with_outcome.csv",
     "Cleaned pooled NHANES with the harmonized MetS outcome derived, before feature engineering"),
    (ROOT / "data_processed" / "train_seqn.csv",
     "05_Data_Processed/train_seqn.csv",
     "SEQN identifiers of the training partition, so the split is exactly reproducible"),
    (ROOT / "data_processed" / "test_seqn.csv",
     "05_Data_Processed/test_seqn.csv",
     "SEQN identifiers of the held-out test partition"),
]

DESCRIPTIONS: dict[str, str] = {
    # metrics
    "model_comparison.csv": "Discrimination, calibration and classification metrics for all 11 models",
    "model_comparison_significance.csv": "AUROC differences against the leading model: paired-bootstrap 95% CI and DeLong p-value",
    "calibration_metrics.csv": "Calibration slope, intercept, observed-to-expected ratio and Brier score",
    "decision_curve_data.csv": "Net benefit across threshold probabilities for the decision-curve analysis",
    "subgroup_auc.csv": "Discrimination and calibration within age, sex and race/ethnicity subgroups",
    "shap_importance.csv": "Mean absolute SHAP value per transformed column, leading model",
    "permutation_importance.csv": "Permutation importance per predictor, leading model",
    "permutation_importance_by_model.csv": "Permutation importance per predictor for each of the four leading models",
    "table1_by_mets_status.csv": "Table 1: cohort characteristics by metabolic-syndrome status",
    "exclusion_flow.csv": "Counts excluded and remaining at each of the five exclusion steps",
    "strobe_flow.csv": "STROBE participant-flow counts, including per-cycle contributions",
    "missingness_report.csv": "Missingness per predictor in the analytic cohort",
    "missingness_report_age_filtered_precohort.csv": "Missingness before the metals-availability exclusion",
    "collinearity_flags.csv": "Predictor pairs flagged by the training-partition-only collinearity screen",
    "predictors_used.csv": "The predictor columns carried into the models",
    "missing_flag_columns.csv": "Continuous predictors that receive a missing-value flag in the models",
    "incremental_value.csv": "Test-set comparison of the full model with no-metals, lead-only, no-BMI and age+BMI-only models",
    "repeated_cv.csv": "Repeated stratified 10-fold cross-validation (5 repeats) for all models: mean and spread of AUROC",
    "outcome_definition_comparison.csv": "Harmonized versus measured-value-only MetS outcome definitions",
    "sensitivity_diq010.csv": "Sensitivity arm: adding physician-diagnosed diabetes as a predictor",
    "sensitivity_smote_vs_classweight.csv": "Sensitivity arm: SMOTENC versus class weighting",
    "sensitivity_outcome_definition.csv": "Sensitivity arm: alternative outcome definitions (lab-only, no lipid-medication clause, first BP reading)",
    "sensitivity_metals_scope.csv": "Sensitivity arm: restricting the metal panel to blood lead alone",
    "tuning_summary.csv": "Cross-validated tuning result per model, with the selected hyperparameters",
    "best_params.json": "Selected hyperparameters per model",
    "thresholds.json": "Out-of-fold decision thresholds selected per model",
    "publication_numbers.json": "Every number quoted in the manuscripts, written at the moment each figure was drawn",
    "reference_verification.json": "PubMed audit record for the 21 Methods references",
    "proposal_reference_verification.json": "PubMed audit record for the proposal reference bank",
    "test_predictions.csv": "Predicted probabilities on the test set; every figure derives from this one file",
    # code
    "config.py": "Paths, variable definitions, clinical labels, MetS thresholds, random seed",
    "pipeline_utils.py": "Shared pipeline helpers: split loading, preprocessing, SMOTENC, feature names",
    "stats_utils.py": "Shared statistics helpers: DeLong test, bootstrap intervals, p-value formatting",
    "additions_common.py": "Shared setup for the 07/08 exploratory analyses (paths, fitted model, covariates)",
    "readme_template.md": "Text of README.md, filled in by 26_build_submission.py",
    "RUN_ME.py": "One-click launcher: installs packages in a private environment and runs the whole pipeline",
    "RUN_ME_Windows.bat": "Double-click launcher for Windows (calls RUN_ME.py)",
    "RUN_ME_macOS.command": "Double-click launcher for macOS (calls RUN_ME.py)",
    "RUN_ME_Linux.sh": "Launcher for Linux (calls RUN_ME.py)",
    "FIX_CHECKLIST.md": "Checklist of every issue found in the October 2026 audit and how each was fixed",
    ".gitignore": "Files and folders git does not track (regenerated outputs, OS clutter)",
    "Research_Proposal.docx": "Research proposal for the two-phase study (ML phase done, MR phase planned)",
    "Humanized Proposal.docx": "Partial plain-language rewrite of the proposal's first sections (draft)",
    "pubstyle.py": "Figure style: Okabe-Ito palette, sizes, fonts, dual PNG/PDF export",
    "docx_style.py": "Manuscript docx style: headings, three-line tables, captions, page-break handling",
    "feature_labels.py": "Short figure labels and the caption glossaries that define them",
    "references.py": "The 21 verified references cited in the Methods",
    "requirements.txt": "Pinned package versions for the analysis environment",
    "README.md": "What this package contains, how it is organised, and how to reproduce it",
    "00_MANIFEST.csv": "Every file in this package with its size, checksum and description",
}

PATTERNS: list[tuple[str, str]] = [
    ("Figure_", "Publication figure (600 dpi PNG and vector PDF)"),
    ("07_Novelty_Additions/code/", "Exploratory analysis script (07): see 07_Novelty_Additions/README.md"),
    ("08_Q1_Rigor_Additions/code/", "Exploratory analysis script (08): see 08_Q1_Rigor_Additions/README.md"),
    ("07_Novelty_Additions/tables/", "Exploratory result table (07): see 07_Novelty_Additions/README.md"),
    ("08_Q1_Rigor_Additions/tables/", "Exploratory result table (08): see 08_Q1_Rigor_Additions/README.md"),
    ("07_Novelty_Additions/figures/", "Exploratory figure (07): see 07_Novelty_Additions/README.md"),
    ("08_Q1_Rigor_Additions/figures/", "Exploratory figure (08): see 08_Q1_Rigor_Additions/README.md"),
    ("README.md", "Guide to the files in this folder"),
    (".pdf", "PDF rendering of the corresponding document"),
]

CODE_ORDER_NOTE = {
    "00": "download the raw NHANES files from the CDC",
    "01": "merge raw NHANES modules into one pooled table",
    "02": "clean and derive the harmonized MetS outcome",
    "03": "feature engineering",
    "04": "train/test split",
    "06": "cross-validated hyperparameter tuning",
    "07": "model training",
    "08": "test-set evaluation with DeLong comparisons and bootstrap intervals",
    "09": "SHAP explanation",
    "10": "sensitivity analyses",
    "11": "survey-weighted reporting",
    "12": "exploratory visualisation and interactive dashboard",
    "13": "incremental value of the metals, simple baselines and repeated cross-validation",
    "14": "diagram generation",
    "15": "publication figure set and the numbers JSON",
    "16": "build the Methods document",
    "17": "build the Results document",
    "18": "verify every number in the manuscripts against pipeline output",
    "20": "verify the Methods citations",
    "21": "build the Supplementary Material document",
    "22": "the proposal's verified reference bank",
    "23": "renumber the proposal's citations and append its reference list",
    "24": "audit the proposal reference bank against PubMed",
    "25": "verify the proposal's citations",
    "26": "refresh the published folders, README and manifest from a finished run",
    "27": "verify the published package",
    "28": "compare a fresh run with the published results",
    "29": "update the research proposal text, tables and figures from the final results",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(rel: str) -> str:
    name = Path(rel).name
    if rel.startswith("06_Code/") or "/" not in rel:
        if name in DESCRIPTIONS:
            return DESCRIPTIONS[name]
        stem = name.split("_")[0]
        if stem.isdigit() and stem in CODE_ORDER_NOTE and name.endswith(".py"):
            return f"Pipeline step {stem}: {CODE_ORDER_NOTE[stem]}"
    for token, text in PATTERNS:
        if token in rel and token != ".pdf":
            return text
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    if name.endswith(".pdf"):
        return "PDF rendering of the corresponding document"
    return ""


def copy(src: Path, rel: str) -> None:
    dst = PKG / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def to_pdf(src: Path, out_dir: Path) -> Path | None:
    """Render a .docx to PDF with LibreOffice. Returns None if soffice is unavailable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                        "--outdir", str(out_dir), str(src)],
                       check=True, capture_output=True, timeout=180)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    produced = out_dir / (src.stem + ".pdf")
    return produced if produced.exists() else None


def published_files():
    for f in sorted(PKG.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(PKG)
        if rel.parts[0] in NOT_PUBLISHED or any(p in NOT_PUBLISHED for p in rel.parts):
            continue
        if f.name in NOT_PUBLISHED_FILES or f.name.startswith("~$"):
            continue
        yield f, str(rel).replace("\\", "/")


def refresh_folder(folder: Path) -> None:
    """Empty a published output folder before copying the new run into it."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def main() -> int:
    missing = [p for p, _, _ in DOCS + DATA if not p.exists()]
    for stem, _, _ in FIGURE_MAP:
        for ext in ("png", "pdf"):
            f = FIG_SRC / f"{stem}.{ext}"
            if not f.exists():
                missing.append(f)
    if missing:
        print("FAIL: these inputs are missing; run the pipeline before packaging:")
        for m in missing:
            print("   " + str(m.relative_to(ROOT)))
        return 1

    # --- documents, and a PDF of each so a reader does not need Word
    refresh_folder(PKG / "01_Manuscript")
    for src, rel, desc in DOCS:
        copy(src, rel)
        DESCRIPTIONS[Path(rel).name] = desc
        pdf = to_pdf(PKG / rel, PKG / Path(rel).parent)
        if pdf is None:
            print(f"NOTE: LibreOffice not found; {rel} has no PDF rendering")
    if PROPOSAL.exists():
        to_pdf(PROPOSAL, PROPOSAL.parent)

    # --- figures, renamed to the numbering the manuscripts use
    refresh_folder(PKG / "03_Figures" / "Main")
    refresh_folder(PKG / "03_Figures" / "Supplementary")
    for stem, target, desc in FIGURE_MAP:
        sub = "03_Figures/Main" if "Figure_S" not in target else "03_Figures/Supplementary"
        for ext in ("png", "pdf"):
            rel = f"{sub}/{target}.{ext}"
            copy(FIG_SRC / f"{stem}.{ext}", rel)
            DESCRIPTIONS[f"{target}.{ext}"] = (
                f"{desc} ({'600 dpi raster' if ext == 'png' else 'vector, for typesetting'})")

    # --- results tables and audit records (the two citation audit records are
    #     produced by the citation scripts, not by a pipeline run, so they are kept)
    keep = {"reference_verification.json", "proposal_reference_verification.json"}
    res = PKG / "04_Results_Tables"
    saved = {k: (res / k).read_bytes() for k in keep if (res / k).exists()}
    refresh_folder(res)
    for k, b in saved.items():
        (res / k).write_bytes(b)
    for f in sorted(MET.glob("*")):
        if f.suffix in (".csv", ".json"):
            copy(f, f"04_Results_Tables/{f.name}")

    # --- data
    refresh_folder(PKG / "05_Data_Processed")
    for src, rel, desc in DATA:
        copy(src, rel)
        DESCRIPTIONS[Path(rel).name] = desc

    # --- exploratory additions: tables and figures from outputs/<folder>/
    for folder in ADDITION_FOLDERS:
        src = ROOT / "outputs" / folder
        for kind in ("tables", "figures"):
            if (src / kind).exists():
                refresh_folder(PKG / folder / kind)
                for f in sorted((src / kind).glob("*")):
                    if f.suffix in (".csv", ".png", ".pdf"):
                        copy(f, f"{folder}/{kind}/{f.name}")

    # --- proposal diagrams (14_generate_diagrams.py)
    for f in sorted((ROOT / "diagrams").glob("*.png")):
        DESCRIPTIONS[f.name] = "Proposal diagram generated by 14_generate_diagrams.py"

    write_readme()
    write_manifest()

    files = list(published_files())
    total = sum(f.stat().st_size for f, _ in files)
    print(f"package: {PKG}")
    print(f"files  : {len(files) + 1} (including the manifest)")
    print(f"size   : {total / 1e6:.1f} MB")
    return 0


def verification_counts() -> str:
    """Run the numeric audit and report what it actually found, measured at
    packaging time rather than hard-coded."""
    try:
        out = subprocess.run([sys.executable, str(ROOT / "06_Code" / "18_verify_manuscript_numbers.py")],
                             capture_output=True, text=True, timeout=300).stdout
    except Exception:
        return "see the script's output"
    counted = sum(int(ln.split(":")[1]) for ln in out.splitlines()
                  if ln.startswith("numeric tokens checked:"))
    bad = next((ln.split(":")[1].strip() for ln in out.splitlines()
                if ln.startswith("TOTAL needing adjudication:")), "?")
    return f"{counted:,} tokens checked, {bad} requiring adjudication"


def write_readme() -> None:
    counts = verification_counts()
    (PKG / "README.md").write_text(README_TEMPLATE.format(today=TODAY, counts=counts), encoding="utf-8")


def write_manifest() -> None:
    rows, undescribed = [], []
    for f, rel in published_files():
        desc = describe(rel)
        if not desc:
            undescribed.append(rel)
        rows.append({"path": rel, "size_bytes": f.stat().st_size, "md5": md5(f), "description": desc})
    with (PKG / "00_MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "size_bytes", "md5", "description"])
        w.writeheader()
        w.writerows(rows)
    if undescribed:
        print(f"NOTE: {len(undescribed)} file(s) have no description in the manifest:")
        for u in undescribed:
            print("   " + u)


README_TEMPLATE = open(Path(__file__).with_name("readme_template.md"), encoding="utf-8").read()


if __name__ == "__main__":
    sys.exit(main())
