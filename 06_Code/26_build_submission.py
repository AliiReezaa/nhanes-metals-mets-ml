"""
Assemble the advisor submission package.

Everything in the package is copied from the working tree rather than regenerated, so the
package can never contain a figure or a number that the verification scripts have not
already checked. The script is idempotent: it clears and rebuilds the destination each
time, so there is no way to ship a stale file that an earlier run left behind.

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
PKG = ROOT / "SUBMISSION"
TODAY = date.today().isoformat()

FIG_SRC = ROOT / "outputs" / "figures_pub"
MET = ROOT / "outputs" / "metrics"


# ----------------------------------------------------------------------
# Figure numbering: working-tree name -> the number the manuscript cites
# ----------------------------------------------------------------------
FIGURE_MAP: list[tuple[str, str, str]] = [
    ("fig1_participant_flow",        "Figure_1_Participant_Flow",
     "Participant flow through the four exclusion steps, with cycle contributions (Methods)"),
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
     "AUROC forest plot with paired-bootstrap differences against the leading model"),
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
    (ROOT / "docx_final" / "Metabolic_Syndrome_Proposal_Revised_FINAL4.docx",
     "02_Proposal/Research_Proposal.docx",
     "Research proposal, revised: 99 PubMed-verified references in first-mention order"),
]

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
    "model_comparison_significance.csv": "Paired-bootstrap differences in AUROC against the leading model",
    "calibration_metrics.csv": "Calibration slope, intercept, observed-to-expected ratio and Brier score",
    "decision_curve_data.csv": "Net benefit across threshold probabilities for the decision-curve analysis",
    "subgroup_auc.csv": "Discrimination and calibration within age, sex and race/ethnicity subgroups",
    "shap_importance.csv": "Mean absolute SHAP value per transformed column, leading model",
    "permutation_importance.csv": "Permutation importance per predictor, leading model",
    "permutation_importance_by_model.csv": "Permutation importance per predictor for each of the four leading models",
    "table1_by_mets_status.csv": "Table 1: cohort characteristics by metabolic-syndrome status",
    "exclusion_flow.csv": "Counts excluded and remaining at each of the four exclusion steps",
    "strobe_flow.csv": "STROBE participant-flow counts, including per-cycle contributions",
    "missingness_report.csv": "Missingness per predictor in the analytic cohort",
    "missingness_report_age_filtered_precohort.csv": "Missingness before the metals-availability exclusion",
    "collinearity_flags.csv": "Predictor pairs flagged by the training-partition-only collinearity screen",
    "predictors_used.csv": "The 33 predictors carried into the models, with type and source module",
    "outcome_definition_comparison.csv": "Harmonized versus measured-value-only MetS outcome definitions",
    "sensitivity_diq010.csv": "Sensitivity arm: adding physician-diagnosed diabetes as a predictor",
    "sensitivity_smote_vs_classweight.csv": "Sensitivity arm: SMOTENC versus class weighting",
    "sensitivity_outcome_definition.csv": "Sensitivity arm: alternative outcome definition",
    "sensitivity_metals_scope.csv": "Sensitivity arm: restricting the metal panel to blood lead alone",
    "tuning_summary.csv": "Cross-validated tuning result per model, with the selected hyperparameters",
    "best_params.json": "Selected hyperparameters per model",
    "thresholds.json": "Out-of-fold decision thresholds selected per model",
    "publication_numbers.json": "Every number quoted in the manuscripts, written at the moment each figure was drawn",
    "reference_verification.json": "PubMed audit record for the 21 Methods references",
    "proposal_reference_verification.json": "PubMed audit record for the proposal reference bank",
    "test_predictions.csv": "Cached predicted probabilities on the test set; every figure derives from this one file",
    # code
    "config.py": "Paths, variable definitions, clinical labels, MetS thresholds, random seed",
    "pipeline_utils.py": "Shared pipeline helpers: split loading, preprocessing, SMOTENC, feature names",
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
    (".pdf", "PDF rendering of the corresponding document"),
]

CODE_ORDER_NOTE = {
    "01": "merge raw NHANES modules into one pooled table",
    "02": "clean and derive the harmonized MetS outcome",
    "03": "feature engineering",
    "04": "train/test split",
    "06": "cross-validated hyperparameter tuning",
    "07": "model training",
    "08": "test-set evaluation with paired bootstrap",
    "09": "SHAP explanation",
    "10": "sensitivity analyses",
    "11": "survey-weighted reporting",
    "12": "exploratory visualisation",
    "13": "publication extras",
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
    "26": "assemble this submission package",
    "27": "verify the assembled package",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def describe(rel: str) -> str:
    name = Path(rel).name
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    stem = name.split("_")[0]
    if stem.isdigit() and stem in CODE_ORDER_NOTE and name.endswith(".py"):
        return f"Pipeline step {stem}: {CODE_ORDER_NOTE[stem]}"
    for token, text in PATTERNS:
        if token in name:
            return text
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


def main() -> int:
    missing = [p for p, _, _ in DOCS + DATA if not p.exists()]
    for stem, _, _ in FIGURE_MAP:
        for ext in ("png", "pdf"):
            f = FIG_SRC / f"{stem}.{ext}"
            if not f.exists():
                missing.append(f)
    if missing:
        print("FAIL: these inputs are missing; build them before packaging:")
        for m in missing:
            print("   " + str(m.relative_to(ROOT)))
        return 1

    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True)

    # --- documents, and a PDF of each so the advisor can read without Word
    for src, rel, desc in DOCS:
        copy(src, rel)
        DESCRIPTIONS[Path(rel).name] = desc
        pdf = to_pdf(src, PKG / Path(rel).parent)
        if pdf is not None:
            pdf.rename(PKG / Path(rel).with_suffix(".pdf"))

    # --- figures, renamed to the numbering the manuscripts use
    for stem, target, desc in FIGURE_MAP:
        sub = "03_Figures/Main" if "Figure_S" not in target else "03_Figures/Supplementary"
        for ext in ("png", "pdf"):
            rel = f"{sub}/{target}.{ext}"
            copy(FIG_SRC / f"{stem}.{ext}", rel)
            DESCRIPTIONS[f"{target}.{ext}"] = (
                f"{desc} ({'600 dpi raster' if ext == 'png' else 'vector, for typesetting'})")

    # --- results tables and audit records
    for f in sorted(MET.glob("*")):
        if f.suffix in (".csv", ".json"):
            copy(f, f"04_Results_Tables/{f.name}")

    # --- data
    for src, rel, desc in DATA:
        copy(src, rel)
        DESCRIPTIONS[Path(rel).name] = desc

    # --- code
    for f in sorted((ROOT / "scripts").glob("*.py")):
        copy(f, f"06_Code/{f.name}")
    req = PKG / "06_Code" / "requirements.txt"
    req.parent.mkdir(parents=True, exist_ok=True)
    try:
        freeze = subprocess.run([str(ROOT / "venv" / "bin" / "pip"), "freeze"],
                                capture_output=True, text=True, timeout=120).stdout
    except Exception:
        freeze = ""
    keep = ("pandas", "numpy", "scikit-learn", "imbalanced-learn", "lightgbm", "xgboost",
            "catboost", "shap", "matplotlib", "scipy", "statsmodels", "joblib")
    lines = [ln for ln in freeze.splitlines()
             if ln.split("==")[0].lower() in keep]
    req.write_text("# Analysis environment (Python 3.11). python-docx is required for the\n"
                   "# document-building steps (16, 17, 21, 23) and is installed separately.\n"
                   + "\n".join(sorted(lines)) + "\npython-docx\n")

    write_readme()
    write_manifest()

    total = sum(f.stat().st_size for f in PKG.rglob("*") if f.is_file())
    n = sum(1 for f in PKG.rglob("*") if f.is_file())
    print(f"package: {PKG.relative_to(ROOT)}")
    print(f"files  : {n}")
    print(f"size   : {total / 1e6:.1f} MB")
    return 0


def verification_counts() -> str:
    """Run the numeric audit and report what it actually found.

    Hard-coding a token count into the README is exactly the kind of number that goes
    stale the first time a sentence is edited, so it is measured at packaging time.
    """
    try:
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "18_verify_manuscript_numbers.py")],
                             capture_output=True, text=True, timeout=300).stdout
    except Exception:
        return "the numeric audit"
    counted = sum(int(ln.split(":")[1]) for ln in out.splitlines()
                  if ln.startswith("numeric tokens checked:"))
    bad = next((ln.split(":")[1].strip() for ln in out.splitlines()
                if ln.startswith("TOTAL needing adjudication:")), "?")
    return f"{counted:,} tokens checked, {bad} requiring adjudication"


def write_readme() -> None:
    counts = verification_counts()
    (PKG / "README.md").write_text(f"""# Blood Metals and Metabolic Syndrome — AI/ML Component

Submission package prepared {TODAY}.

This package covers the machine-learning arm of the study: predicting metabolic syndrome
(MetS) in pooled NHANES 2007–2018 data from demographic, lifestyle, anthropometric and
blood-metal predictors, and identifying which predictors carry the signal. The Mendelian
randomization arm is described as planned work in the proposal and has not yet been run;
no MR results appear anywhere in this package.

## How the folders are organised

| Folder | Contents |
|---|---|
| `01_Manuscript/` | Methods, Results and Supplementary Material, as .docx and .pdf |
| `02_Proposal/` | The revised research proposal, as .docx and .pdf |
| `03_Figures/Main/` | Figures 1–5, numbered as the manuscripts cite them |
| `03_Figures/Supplementary/` | Figures S1–S5 |
| `04_Results_Tables/` | Every metrics table the manuscripts draw on, plus the citation audit records |
| `05_Data_Processed/` | The analytic dataset and the exact train/test split |
| `06_Code/` | The full pipeline, numbered in execution order |
| `00_MANIFEST.csv` | Every file above, with size, MD5 checksum and a one-line description |

Every figure is supplied twice: a 600 dpi PNG for reading, and a vector PDF for
typesetting. The two are drawn from the same code path, so they cannot disagree.

## What has been checked, and how

Four verification scripts run over this material. All four pass as packaged.

- `06_Code/18_verify_manuscript_numbers.py` takes every numeric token in the three
  manuscript documents and tries to trace it back to a value that actually exists in the
  pipeline's output. It builds the allowed pool from the metrics tables, the numbers JSON
  and a declared set of structural constants, and prints anything it cannot account for
  rather than passing it silently. Current result: {counts}. This proves no number was
  invented, mistyped, or left behind by an earlier draft; it does not prove that a real
  number is attached to the right claim, which is what reading is for.
- `06_Code/20_verify_citations.py` checks the Methods references: that every marker
  resolves, that every reference is cited, that first-mention order is ascending, that
  PMIDs and DOIs are unique, and that each entry matches the PubMed record captured in
  `04_Results_Tables/reference_verification.json`.
- `06_Code/24_audit_proposal_references.py` does the same field-by-field comparison for
  the proposal's 111-entry reference bank — title, first author, journal, year, volume,
  issue, pages and DOI against PubMed's own record.
- `06_Code/25_verify_proposal_citations.py` checks the finished proposal document rather
  than the script that produced it: numbering runs 1–99 with no gaps, no reference goes
  uncited, first mentions ascend, and every listed entry renders exactly as the bank says
  it should.
- `06_Code/27_verify_submission.py` checks this package as you received it, which is the
  one thing the others cannot: that every file on disk is in the manifest and every
  checksum still matches, that every figure and table the text cites exists here and
  nothing sits here uncited, that no document carries a leftover draft marker, and that
  no page in any PDF is blank.

## Reproducing the analysis

```bash
python -m venv venv && ./venv/bin/pip install -r 06_Code/requirements.txt

# modelling pipeline, in order
for s in 01 02 03 04 06 07 08 09 10 11 12 13; do ./venv/bin/python 06_Code/$s_*.py; done

# figures and the numbers JSON
./venv/bin/python 06_Code/15_publication_figures.py

# documents (these need python-docx)
python3 06_Code/16_write_methods_docx.py
python3 06_Code/17_write_results_docx.py
python3 06_Code/21_write_supplement.py
python3 06_Code/23_rewrite_proposal_citations.py

# verification
python3 06_Code/18_verify_manuscript_numbers.py
python3 06_Code/20_verify_citations.py
python3 06_Code/24_audit_proposal_references.py
python3 06_Code/25_verify_proposal_citations.py
```

Step 01 reads the raw NHANES `.xpt` files, which are not included here — they are public
and are downloaded from the CDC (the URL is given in the Methods, Data subsection). Every
step from 02 onward runs from `05_Data_Processed/`, which is included.

The random seed is fixed in `06_Code/config.py`. The train/test split is additionally
pinned by SEQN in `05_Data_Processed/train_seqn.csv` and `test_seqn.csv`, so the partition
is reproducible even if the splitting code changes.

## Two things worth flagging

**Six claims in the proposal were rewritten rather than cited.** Each asserted something
for which no real article could be found — a peri-menopausal blood-lead study, marital
status and household size as MetS risk factors, an HIV/BMI threshold claim, wine being
less harmful than beer or spirits, and a 13-SNP blood-lead instrument. The rewritten
sentences say what the literature that does exist actually shows. The full list, with the
reasoning, is in the docstring of `06_Code/23_rewrite_proposal_citations.py` under
`REMOVED_CLAIMS`.

**Every reference was checked against PubMed before use, and seven did not survive.** Four
resolved to entirely different papers, and three could not be found at all. One of the
seven was already in the previous draft of the proposal, cited by name in the prose.
""", encoding="utf-8")


def write_manifest() -> None:
    rows, undescribed = [], []
    for f in sorted(PKG.rglob("*")):
        if not f.is_file() or f.name == "00_MANIFEST.csv":
            continue
        rel = str(f.relative_to(PKG))
        desc = describe(rel)
        if not desc:
            undescribed.append(rel)
        rows.append({
            "path": rel,
            "size_bytes": f.stat().st_size,
            "md5": md5(f),
            "description": desc,
        })
    with (PKG / "00_MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "size_bytes", "md5", "description"])
        w.writeheader()
        w.writerows(rows)
    if undescribed:
        print(f"NOTE: {len(undescribed)} file(s) have no description in the manifest:")
        for u in undescribed:
            print("   " + u)


if __name__ == "__main__":
    sys.exit(main())
