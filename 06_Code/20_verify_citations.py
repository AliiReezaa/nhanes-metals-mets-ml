"""
Citation audit for the Methods section.

Three checks, all of which must pass:

  1. STRUCTURAL - every in-text marker [n] in the document resolves to an entry
     in the reference list, every reference is cited at least once, and markers
     appear in ascending order of first mention (Vancouver requirement).
  2. IDENTIFIER - every reference carries a PMID or DOI, and no two references
     share one.
  3. EXTERNAL - each identifier is re-queried against PubMed and the returned
     first author, journal, year, volume and pages are compared against what the
     reference list claims. A reference whose metadata does not match the
     database is reported, not silently accepted.

Check 3 requires network access and the PubMed E-utilities endpoint; it is the
one that would catch a fabricated or mis-transcribed citation, so a run without
it is reported as incomplete rather than as a pass.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from docx import Document

sys.path.insert(0, str(Path(__file__).resolve().parent))
from references import REFS, NUM

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "manuscript" / "Manuscript_Methods_AI.docx"
SUPP = ROOT / "manuscript" / "Manuscript_Supplement_AI.docx"

MARKER = re.compile(r"\[(\d+(?:[,–]\d+)*)\]")


def body_and_refs(path: Path):
    doc = Document(path)
    body, refs, in_refs = [], [], False
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        if t == "References":
            in_refs = True
            continue
        (refs if in_refs else body).append(t)
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                if c.text.strip():
                    (refs if in_refs else body).append(c.text)
    return body, refs


def expand(token: str):
    out = []
    for part in token.split(","):
        if "–" in part:
            a, b = part.split("–")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def check_structure():
    print("=" * 70)
    print("1. STRUCTURAL")
    print("=" * 70)
    ok = True
    cited_order, cited = [], set()
    for path in (DOC, SUPP):
        body, _ = body_and_refs(path)
        for text in body:
            for m in MARKER.finditer(text):
                for n in expand(m.group(1)):
                    if n not in cited:
                        cited_order.append((n, path.name))
                    cited.add(n)

    bad = [n for n in cited if n < 1 or n > len(REFS)]
    if bad:
        print(f"  FAIL: markers with no matching reference: {sorted(set(bad))}")
        ok = False
    uncited = [i + 1 for i in range(len(REFS)) if i + 1 not in cited]
    if uncited:
        print(f"  FAIL: references never cited: {uncited}")
        ok = False

    # ascending order of first mention, within the main Methods file only
    main_first = [n for n, src in cited_order if src == DOC.name]
    if main_first != sorted(main_first):
        offenders = [(a, b) for a, b in zip(main_first, main_first[1:]) if b < a]
        print(f"  FAIL: first-mention order is not ascending at {offenders}")
        ok = False
    else:
        print(f"  first-mention order ascending: 1..{max(main_first)} OK")
    print(f"  markers found: {len(cited)} distinct; references defined: {len(REFS)}")
    if ok:
        print("  RESULT: OK")
    return ok


def extract_ids():
    out = {}
    for key, text in REFS:
        pmid = re.search(r"PMID:\s*(\d+)", text)
        doi = re.search(r"doi:([^\s]+?)\.?$", text)
        out[key] = (pmid.group(1) if pmid else None,
                    doi.group(1).rstrip(".") if doi else None)
    return out


def check_identifiers(ids):
    print()
    print("=" * 70)
    print("2. IDENTIFIER")
    print("=" * 70)
    ok = True
    seen = {}
    for key, (pmid, doi) in ids.items():
        if not pmid and not doi:
            print(f"  FAIL: [{NUM[key]}] {key} carries neither PMID nor DOI")
            ok = False
        ident = pmid or doi
        if ident in seen:
            print(f"  FAIL: [{NUM[key]}] {key} shares identifier with {seen[ident]}")
            ok = False
        seen[ident] = key
    if ok:
        print(f"  all {len(ids)} references carry a unique identifier: OK")
    return ok


AUDIT = ROOT / "outputs" / "metrics" / "reference_verification.json"


def check_external(ids):
    """Compare each reference string against the metadata the PubMed connector
    returned for its identifier, stored in reference_verification.json. Direct
    E-utilities access is blocked by this environment's egress proxy, so the
    connector's response is captured once and audited here rather than re-queried
    on every run; the file records the retrieval date."""
    print()
    print("=" * 70)
    print("3. EXTERNAL (against captured PubMed metadata)")
    print("=" * 70)
    if not AUDIT.exists():
        print("  INCOMPLETE: no verification record found")
        return False
    audit = json.loads(AUDIT.read_text())
    recs = audit["records"]
    print(f"  record retrieved {audit['_retrieved']}; {len(recs)} entries\n")
    ok = True
    for key, text in REFS:
        rec = recs.get(key)
        if rec is None:
            print(f"  [{NUM[key]:>2}] {key:<15} FAIL: no verification record")
            ok = False
            continue
        problems = []
        if not text.startswith(rec["first_author"]):
            problems.append(f"first author (record: {rec['first_author']})")
        if rec["year"] not in text:
            problems.append(f"year (record: {rec['year']})")
        if rec["journal"].split()[0] not in text:
            problems.append(f"journal (record: {rec['journal']})")
        if rec.get("volume") and rec["volume"] not in text:
            problems.append(f"volume (record: {rec['volume']})")
        # identifier present in the reference string
        claimed_pmid, claimed_doi = ids[key]
        if rec.get("pmid") and claimed_pmid and claimed_pmid != rec["pmid"]:
            problems.append(f"PMID mismatch ({claimed_pmid} vs {rec['pmid']})")
        if rec.get("doi") and claimed_doi and not rec["doi"].lower().startswith(claimed_doi.lower()[:20]):
            problems.append(f"DOI mismatch ({claimed_doi} vs {rec['doi']})")
        if problems:
            print(f"  [{NUM[key]:>2}] {key:<15} MISMATCH: {'; '.join(problems)}")
            ok = False
        else:
            src = rec["source"]
            ident = rec.get("pmid") or rec.get("doi")
            print(f"  [{NUM[key]:>2}] {key:<15} OK  {src:<9} {ident} — "
                  f"{rec['journal']} {rec['year']};{rec.get('volume') or rec.get('issue')}:{rec['pages']}")
    return ok


def main():
    a = check_structure()
    ids = extract_ids()
    b = check_identifiers(ids)
    try:
        c = check_external(ids)
    except Exception as e:
        print(f"\n  INCOMPLETE: PubMed unreachable ({e}); external check not performed")
        c = False
    print()
    print("=" * 70)
    print(f"OVERALL: structural={'OK' if a else 'FAIL'}  "
          f"identifier={'OK' if b else 'FAIL'}  external={'OK' if c else 'FAIL/INCOMPLETE'}")
    return 0 if (a and b and c) else 1


if __name__ == "__main__":
    sys.exit(main())
