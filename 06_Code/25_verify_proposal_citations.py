"""
Verify the rewritten proposal's citation system.

This is the counterpart to scripts/20_verify_citations.py, which does the same job for the
Methods manuscript. It reads the finished .docx rather than the script that produced it,
so it checks what a reader will actually see.

Six checks, each of which the previous draft would have failed:

  STRUCTURAL  every in-text number resolves to an entry; the list runs 1..N with no gaps;
              no entry goes uncited; no {{placeholder}} survives the build
  ORDER       numbers appear in strictly ascending first-mention order, which is what makes
              a collision like the old [44] -- serving an income/ethnicity claim in one
              section and the IVW estimator in another -- impossible by construction
  UNIQUENESS  no PMID and no DOI appears against two different entries
  RENDERING   every entry is byte-identical to what format_reference() produces for the
              reference the numbering implies, so the list cannot drift from the bank
  EXTERNAL    every cited entry's PMID is present in the captured PubMed audit record
  COVERAGE    reports which bank entries went uncited (informational, not a failure)
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docx_final" / "Metabolic_Syndrome_Proposal_Revised_FINAL4.docx"
AUDIT = ROOT / "outputs" / "metrics" / "proposal_reference_verification.json"

spec = importlib.util.spec_from_file_location(
    "propref", ROOT / "scripts" / "22_proposal_references.py")
propref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(propref)
REFS, format_reference = propref.REFS, propref.format_reference

MARKER = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")
ENTRY = re.compile(r"^(\d+)\.\s")
PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")


def expand(marker_body: str) -> list[int]:
    out: list[int] = []
    for part in marker_body.split(","):
        part = part.strip()
        if "-" in part:
            a, b = (int(x) for x in part.split("-"))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    doc = Document(DOC)
    paras = [p.text for p in doc.paragraphs]

    try:
        head = next(i for i, t in enumerate(paras) if t.strip() == "References")
    except StopIteration:
        print("FAIL: the document has no References heading")
        return 1

    body_paras, list_paras = paras[:head], paras[head + 1:]
    failures: list[str] = []

    # ---------------- reference list ----------------
    entries: dict[int, str] = {}
    for t in list_paras:
        t = t.strip()
        if not t:
            continue
        m = ENTRY.match(t)
        if not m:
            failures.append(f"STRUCTURAL: unnumbered line in the reference list: {t[:70]!r}")
            continue
        n = int(m.group(1))
        if n in entries:
            failures.append(f"STRUCTURAL: reference number {n} appears twice in the list")
        entries[n] = t

    n_refs = len(entries)
    if entries and sorted(entries) != list(range(1, n_refs + 1)):
        missing = sorted(set(range(1, max(entries) + 1)) - set(entries))
        failures.append(f"STRUCTURAL: reference list is not 1..N; gaps at {missing}")

    # ---------------- in-text markers ----------------
    cited: set[int] = set()
    first_mention: list[int] = []
    for i, t in enumerate(body_paras):
        if PLACEHOLDER.search(t):
            failures.append(f"STRUCTURAL: unresolved placeholder in paragraph {i}")
        for m in MARKER.finditer(t):
            for n in expand(m.group(1)):
                if n < 1 or n > n_refs:
                    failures.append(
                        f"STRUCTURAL: paragraph {i} cites [{n}] but the list has "
                        f"{n_refs} entries")
                    continue
                if n not in cited:
                    cited.add(n)
                    first_mention.append(n)

    uncited = sorted(set(range(1, n_refs + 1)) - cited)
    if uncited:
        failures.append(f"STRUCTURAL: reference(s) never cited in the text: {uncited}")

    if first_mention != sorted(first_mention):
        bad = [(a, b) for a, b in zip(first_mention, first_mention[1:]) if b < a]
        failures.append(f"ORDER: first mentions are not ascending; breaks at {bad[:5]}")

    # ---------------- rendering matches the bank ----------------
    by_number = {n: k for n, k in zip(first_mention, [None] * len(first_mention))}
    order_keys: list[str] = []
    for n in range(1, n_refs + 1):
        want = entries.get(n, "")
        match = next((k for k in REFS if format_reference(k, n) == want), None)
        if match is None:
            failures.append(
                f"RENDERING: entry {n} matches no reference in the bank: {want[:80]!r}")
        else:
            order_keys.append(match)
    by_number.clear()

    if len(set(order_keys)) != len(order_keys):
        dupes = {k for k in order_keys if order_keys.count(k) > 1}
        failures.append(f"UNIQUENESS: reference(s) listed more than once: {sorted(dupes)}")

    seen: dict[str, str] = {}
    for k in order_keys:
        for field in ("pmid", "doi"):
            v = REFS[k].get(field)
            if v and v in seen and seen[v] != k:
                failures.append(f"UNIQUENESS: {field} {v} shared by {k} and {seen[v]}")
            if v:
                seen[v] = k

    # ---------------- external audit record ----------------
    if not AUDIT.is_file():
        failures.append("EXTERNAL: the PubMed audit record is missing; run "
                        "scripts/24_audit_proposal_references.py")
    else:
        records = json.loads(AUDIT.read_text()).get("records", {})
        for k in order_keys:
            pmid = REFS[k].get("pmid")
            if pmid and str(pmid) not in records:
                failures.append(f"EXTERNAL: {k} (PMID {pmid}) has no captured PubMed record")

    # ---------------- report ----------------
    print(f"document        : {DOC.name}")
    print(f"body paragraphs : {len(body_paras)}")
    print(f"references      : {n_refs} listed, {len(cited)} cited")
    print(f"bank coverage   : {len(order_keys)} of {len(REFS)} bank entries used")
    unused = [k for k in REFS if k not in order_keys]
    if unused:
        print(f"uncited in bank : {len(unused)} -> {', '.join(sorted(unused))}")

    if failures:
        print(f"\nRESULT: {len(failures)} problem(s):")
        for f in failures:
            print("   " + f)
        return 1
    print("\nRESULT: STRUCTURAL, ORDER, UNIQUENESS, RENDERING and EXTERNAL checks all pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
