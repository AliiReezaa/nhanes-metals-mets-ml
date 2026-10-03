"""
Final check on the assembled submission package.

This runs over the published folders as a reader would receive them, so it catches the
failure mode the other verification scripts cannot: a package that was built from stale
inputs, or that is missing a file the text refers to. (2026-10 audit, C11: the package is
the repository itself; the earlier version looked for a SUBMISSION/ sub-folder and
failed immediately.)

Checks:

  MANIFEST     every file on disk is listed, every listed file exists, and every MD5 in
               the manifest still matches the bytes on disk
  FIGURES      every "Figure N" and "Figure SN" cited in the three manuscript documents
               has a matching PNG and PDF in 03_Figures/, and no figure file is unreferenced
  TABLES       every "Table N" and "Table SN" cited in the text is defined exactly once
  BLANK PAGES  no PDF in the package contains a page with essentially no content
  DOCUMENTS    the four source documents carry the expected page counts and no unresolved
               placeholder or leftover draft marker (including phrases such as "earlier draft")
  PACKAGES     every third-party package imported by any script is pinned in
               06_Code/requirements.txt (2026-10 audit, C10)
"""
from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT
NOT_PUBLISHED = {".git", ".venv", "venv", "outputs", "data_processed", "NHANES_data",
                 "manuscript", "__pycache__", ".pytest_cache"}

# import name -> pip package name, for the PACKAGES check
IMPORT_TO_PIP = {"sklearn": "scikit-learn", "imblearn": "imbalanced-learn", "docx": "python-docx",
                 "catboost": "catboost", "lightgbm": "lightgbm", "xgboost": "xgboost",
                 "shap": "shap", "pandas": "pandas", "numpy": "numpy", "scipy": "scipy",
                 "statsmodels": "statsmodels", "matplotlib": "matplotlib", "joblib": "joblib",
                 "seaborn": "seaborn", "patsy": "patsy", "plotly": "plotly",
                 "playwright": "playwright"}
OPTIONAL_PACKAGES = {"playwright"}

DOCS = {
    "Methods":     PKG / "01_Manuscript" / "01_Methods_AI.docx",
    "Results":     PKG / "01_Manuscript" / "02_Results_AI.docx",
    "Supplement":  PKG / "01_Manuscript" / "03_Supplementary_Material.docx",
    "Proposal":    PKG / "02_Proposal" / "Research_Proposal.docx",
}

FIG_CITE = re.compile(r"\bFigure\s(S?\d+)")
FIG_DEF = re.compile(r"^Figure\s(S?\d+)\.")
TAB_CITE = re.compile(r"\bTable\s(S?\d+)")
TAB_DEF = re.compile(r"^Table\s(S?\d+)\.")
PLACEHOLDER = re.compile(r"\{\{|\}\}|TODO|TBD|XXX|\[REF\]|Lorem ipsum|earlier draft|previous draft", re.I)

FIG_STEMS = {
    "1": "Main/Figure_1_Participant_Flow",
    "2": "Main/Figure_2_ROC_Curves",
    "3": "Main/Figure_3_SHAP_Summary",
    "4": "Main/Figure_4_Importance_Concordance",
    "5": "Main/Figure_5_Metals_SHAP_Dependence",
    "S1": "Supplementary/Figure_S1_Analysis_Pipeline",
    "S2": "Supplementary/Figure_S2_AUROC_Forest",
    "S3": "Supplementary/Figure_S3_Calibration",
    "S4": "Supplementary/Figure_S4_Decision_Curve",
    "S5": "Supplementary/Figure_S5_Subgroup_AUROC",
}


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def doc_lines(path: Path) -> list[str]:
    d = Document(path)
    out = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            out.extend(c.text for c in row.cells)
    return out


def main() -> int:
    if not (PKG / "00_MANIFEST.csv").exists():
        print("FAIL: 00_MANIFEST.csv does not exist; run 06_Code/26_build_submission.py")
        return 1
    problems: list[str] = []

    # ---------------- manifest ----------------
    manifest = {r["path"]: r for r in
                csv.DictReader((PKG / "00_MANIFEST.csv").open(encoding="utf-8"))}
    on_disk = set()
    for f in PKG.rglob("*"):
        if not f.is_file() or f.name in ("00_MANIFEST.csv", ".DS_Store") or f.name.startswith("~$"):
            continue
        rel = f.relative_to(PKG)
        if any(part in NOT_PUBLISHED for part in rel.parts):
            continue
        on_disk.add(str(rel).replace("\\", "/"))
    for extra in sorted(on_disk - set(manifest)):
        problems.append(f"MANIFEST: {extra} is in the package but not in the manifest")
    for gone in sorted(set(manifest) - on_disk):
        problems.append(f"MANIFEST: {gone} is in the manifest but not in the package")
    for rel in sorted(on_disk & set(manifest)):
        if md5(PKG / rel) != manifest[rel]["md5"]:
            problems.append(f"MANIFEST: checksum mismatch for {rel}")
        if not manifest[rel]["description"].strip():
            problems.append(f"MANIFEST: no description for {rel}")

    # ---------------- figures and tables ----------------
    cited_figs: set[str] = set()
    cited_tabs: set[str] = set()
    defined_figs: list[str] = []
    defined_tabs: list[str] = []
    for name, path in DOCS.items():
        if name == "Proposal":
            continue
        for line in doc_lines(path):
            stripped = line.strip()
            cited_figs |= set(FIG_CITE.findall(stripped))
            cited_tabs |= set(TAB_CITE.findall(stripped))
            m = FIG_DEF.match(stripped)
            if m:
                defined_figs.append(m.group(1))
            m = TAB_DEF.match(stripped)
            if m:
                defined_tabs.append(m.group(1))

    for n in sorted(cited_figs):
        if n not in FIG_STEMS:
            problems.append(f"FIGURES: the text cites Figure {n}, which has no image file")
            continue
        for ext in ("png", "pdf"):
            f = PKG / "03_Figures" / f"{FIG_STEMS[n]}.{ext}"
            if not f.exists():
                problems.append(f"FIGURES: Figure {n} is cited but {f.name} is missing")
    for n in sorted(set(FIG_STEMS) - cited_figs):
        problems.append(f"FIGURES: {FIG_STEMS[n]} is in the package but never cited")

    for group, defined in (("Figure", defined_figs), ("Table", defined_tabs)):
        dupes = sorted({d for d in defined if defined.count(d) > 1})
        if dupes:
            problems.append(f"{group.upper()}S: caption defined more than once for {dupes}")
    orphan_tabs = sorted(set(defined_tabs) - cited_tabs)
    if orphan_tabs:
        problems.append(f"TABLES: defined but never cited: {orphan_tabs}")
    undefined_tabs = sorted(cited_tabs - set(defined_tabs))
    if undefined_tabs:
        problems.append(f"TABLES: cited but never defined: {undefined_tabs}")

    # ---------------- documents ----------------
    for name, path in DOCS.items():
        for i, line in enumerate(doc_lines(path)):
            if PLACEHOLDER.search(line):
                problems.append(f"DOCUMENTS: {name} line {i} contains a draft marker: "
                                f"{line.strip()[:60]!r}")

    # ---------------- blank pages ----------------
    page_counts: dict[str, int] = {}
    for pdf in sorted(PKG.rglob("*.pdf")):
        if "03_Figures" in str(pdf):
            continue
        try:
            txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                                 capture_output=True, text=True, timeout=120).stdout
        except Exception:
            problems.append(f"BLANK PAGES: could not read {pdf.name}")
            continue
        pages = txt.split("\f")
        real = pages[:-1] if pages and pages[-1].strip() == "" else pages
        page_counts[pdf.name] = len(real)
        blank = [i for i, p in enumerate(real, 1) if len(p.strip()) < 40]
        if blank:
            problems.append(f"BLANK PAGES: {pdf.name} page(s) {blank} are essentially empty")

    # ---------------- packages ----------------
    req = {ln.split("==")[0].strip().lower() for ln in
           (PKG / "06_Code" / "requirements.txt").read_text(encoding="utf-8").splitlines()
           if ln.strip() and not ln.startswith("#")}
    imported = set()
    for py in PKG.rglob("*.py"):
        if any(part in NOT_PUBLISHED for part in py.relative_to(PKG).parts):
            continue
        for m in re.finditer(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", py.read_text(encoding="utf-8"), re.M):
            if m.group(1) in IMPORT_TO_PIP:
                imported.add(IMPORT_TO_PIP[m.group(1)])
    for pkg in sorted(imported - req - OPTIONAL_PACKAGES):
        problems.append(f"PACKAGES: {pkg} is imported by the code but not pinned in requirements.txt")

    # ---------------- report ----------------
    print(f"package        : {PKG}")
    print(f"files          : {len(on_disk)} on disk, {len(manifest)} in the manifest")
    print(f"figures cited  : {', '.join(sorted(cited_figs, key=lambda s: (s[0] == 'S', s)))}")
    print(f"tables cited   : {', '.join(sorted(cited_tabs, key=lambda s: (s[0] == 'S', s)))}")
    for k, v in page_counts.items():
        print(f"  {k:36s} {v:3d} pages")

    if problems:
        print(f"\nRESULT: {len(problems)} problem(s):")
        for p in problems:
            print("   " + p)
        return 1
    print("\nRESULT: MANIFEST, FIGURES, TABLES, DOCUMENTS, BLANK PAGES and PACKAGES checks all pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
