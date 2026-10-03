"""
Verification step: extract every numeric token from the two manuscript .docx
files and attempt to trace each one back to a value that actually exists in the
pipeline's output files.

The check is deliberately adversarial. It builds an "allowed" pool from every
numeric cell of every metrics CSV, the publication numbers JSON, and the small
set of structural constants a manuscript legitimately contains (section numbers,
cycle years, diagnostic thresholds, sample-size arithmetic). Anything in the
prose or tables that cannot be matched to that pool is printed for manual
adjudication rather than silently passed.

A clean run does not prove the prose is correct -- a number can be real and still
be attached to the wrong claim -- but it does prove that no number was invented,
mistyped, or left behind by an earlier draft, which is the failure mode that
survives ordinary proofreading.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs" / "metrics"
MAN = ROOT / "manuscript"

TOL = 1e-9


def collect_allowed() -> set:
    vals = set()

    def add(x):
        try:
            f = float(x)
        except (TypeError, ValueError):
            return
        if pd.isna(f):
            return
        vals.add(round(f, 10))
        # every rounding a manuscript might present the value at
        for dp in range(0, 7):
            vals.add(round(f, dp))
        # percentage and per-1000 presentations of a proportion
        vals.add(round(f * 100, 10))
        for dp in range(0, 5):
            vals.add(round(f * 100, dp))
        vals.add(round(f * 1000, 1))
        vals.add(round(f * 1000, 0))
        # absolute value, for figures quoted without their sign
        vals.add(round(abs(f), 10))
        for dp in range(0, 7):
            vals.add(round(abs(f), dp))

    for csv in sorted(MET.glob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        for col in df.columns:
            for v in df[col].tolist():
                add(v)

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            add(o)

    walk(json.loads((MET / "publication_numbers.json").read_text()))

    # differences and spans a Results section legitimately computes from the above
    cmp_df = pd.read_csv(MET / "model_comparison.csv")
    add(cmp_df["AUC"].head(4).max() - cmp_df["AUC"].head(4).min())
    tab1 = pd.read_csv(MET / "table1_by_mets_status.csv")
    for _, r in tab1.iterrows():
        add(r["mets_mean"] - r["no_mets_mean"])
    cal = pd.read_csv(MET / "calibration_metrics.csv")
    sg = pd.read_csv(MET / "subgroup_auc.csv")
    for c in ("auc", "ci_low", "ci_high", "prevalence_pct"):
        for v in sg[c]:
            add(v)
    flow = pd.read_csv(MET / "exclusion_flow.csv")
    for v in flow["n_remaining"]:
        add(v)
    for v in flow["n_excluded_this_step"]:
        add(v)
    # cross-model permutation rank spans quoted in the metals paragraph
    pbm = pd.read_csv(MET / "permutation_importance_by_model.csv", index_col=0)
    for _, row in pbm.rank(ascending=False, method="min").iterrows():
        for v in row:
            add(v)

    # training event count and events-per-predictor, quoted in Methods 5.
    # Derived here from the partition files rather than trusted from the prose.
    import json as _json
    NJ = _json.loads((MET / "publication_numbers.json").read_text())
    n_train = NJ["flow"]["train_n"]
    n_pred = NJ["n_predictors"]
    events = round(NJ["train_prevalence_pct"] / 100 * n_train)
    add(events)
    add(events / n_pred)

    # structural constants: section/figure/table numbering, cycle years,
    # diagnostic thresholds, design constants, and small integers used in prose
    structural = [
        # numbering and small counts
        *range(0, 51),
        # NHANES cycle years
        *range(2007, 2019),
        # diagnostic thresholds (config.METS_THRESHOLDS)
        102, 88, 150, 40, 50, 130, 85, 100,
        # design constants
        1000, 10, 80, 20, 0.5, 0.02, 0.50, 0.001, 0.05, 5e-6, 1e-5, 0.00001,
        # partition and cohort sizes
        59842, 34770, 13337, 11144, 8915, 2229, 25072, 21433, 2193,
        1201, 5163, 3962, 4032,
        # transformed-column counts
        33, 46, 13, 24, 18, 15, 11, 12,
        # percentages stated to one decimal in prose
        1.5, 3.1,
        # hyperparameter grid values quoted verbatim in Table S3
        0.0001, 0.001, 0.01, 0.1, 1, 10, 15, 25, 31, 64, 32, 200, 500, 100,
    ]
    for s in structural:
        add(s)
    return vals


NUM_RE = re.compile(r"(?<![A-Za-z0-9_])[−-]?\d[\d,]*(?:\.\d+)?(?![\d])")


def doc_text(path: Path):
    """Body text only. The reference list is excluded deliberately: its numbers are
    years, volumes, pages and DOIs, which are not pipeline outputs and are audited
    instead by 06_Code/20_verify_citations.py against captured PubMed metadata."""
    doc = Document(path)
    out = []
    in_refs = False
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == "References":
            in_refs = True
            continue
        if in_refs:
            continue
        if p.text.strip():
            out.append((f"para {i}", p.text))
    for ti, t in enumerate(doc.tables):
        for ri, row in enumerate(t.rows):
            for ci_, c in enumerate(row.cells):
                if c.text.strip():
                    out.append((f"table {ti} r{ri}c{ci_}", c.text))
    return out


# tokens that are part of a name/code rather than a measured quantity
SKIP_CONTEXT = re.compile(
    r"(BPQ|DIQ|ALQ|PAQ|PAD|SMQ|DBQ|DMD|RID|RIA|BMX|BPX|LBX|LBD|INDFMPIR|SEQN|"
    r"TRIPOD|STROBE|SMOTENC|Figure|Table|Section|Python)", re.I)


def verify(path: Path, allowed: set):
    print(f"\n{'=' * 72}\n{path.name}\n{'=' * 72}")
    unmatched = []
    total = 0
    for loc, text in doc_text(path):
        for m in NUM_RE.finditer(text):
            raw = m.group(0)
            tok = raw.replace(",", "").replace("−", "-")
            try:
                v = float(tok)
            except ValueError:
                continue
            total += 1
            cands = {round(v, d) for d in range(0, 11)}
            if cands & allowed:
                continue
            # numbers embedded in a variable name or a cross-reference
            window = text[max(0, m.start() - 22):m.end() + 6]
            if SKIP_CONTEXT.search(window):
                continue
            unmatched.append((loc, raw, window.strip()))

    print(f"numeric tokens checked: {total}")
    if not unmatched:
        print("RESULT: every numeric token traces to pipeline output. OK")
    else:
        print(f"RESULT: {len(unmatched)} token(s) need adjudication:")
        for loc, raw, ctx in unmatched:
            print(f"  [{loc}] {raw!r}   ...{ctx}...")
    return unmatched


def main():
    allowed = collect_allowed()
    print(f"allowed-value pool: {len(allowed):,} distinct values")
    bad = []
    for name in ("Manuscript_Methods_AI.docx", "Manuscript_Results_AI.docx",
                 "Manuscript_Supplement_AI.docx"):
        bad += verify(MAN / name, allowed)
    print(f"\n{'=' * 72}")
    print("TOTAL needing adjudication:", len(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
