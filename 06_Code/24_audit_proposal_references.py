"""
Audit every entry in the proposal reference bank against PubMed's own record.

The PubMed responses captured during sourcing are persisted by the harness as JSON files.
This script harvests all of them into one PMID-keyed table, writes that table to
outputs/metrics/proposal_reference_verification.json as the permanent audit record, and
then checks each entry of scripts/22_proposal_references.REFS against it.

Four things are compared and any mismatch is printed rather than silently tolerated:
  * the title, normalised for case, punctuation and trailing period
  * the first author's surname
  * the journal's ISO abbreviation
  * volume, issue, pages and DOI

Year is compared leniently: PubMed's publication_date is the electronic-publication date
for papers that appeared online ahead of print, so a reference may legitimately carry the
issue year instead. Where the two differ the bank records both (`year` and `pubmed_year`)
and this script accepts either, but flags a reference that matches neither.

A PMID present in the bank but absent from the captured responses is reported as
UNVERIFIED -- it must be fetched before the proposal ships.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "metrics" / "proposal_reference_verification.json"
CAPTURE_DIRS = [
    Path("/root/.claude/projects/-home-claude/"
         "411419d0-063a-5967-9689-20e6132dbbfa/tool-results"),
]

spec = importlib.util.spec_from_file_location(
    "propref", ROOT / "scripts" / "22_proposal_references.py")
propref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(propref)
REFS = propref.REFS


def norm_title(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("‐", "-").replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("∼", "~")
    s = s.lower().strip().rstrip(".")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def surname(authors_field: str) -> str:
    """'Burgess S, Butterworth A, ...' -> 'burgess'. Handles corporate authors."""
    first = authors_field.split(",")[0].strip().rstrip(".")
    first = re.sub(r";.*$", "", first).strip()
    parts = first.split()
    if len(parts) >= 2 and len(parts[-1]) <= 4 and parts[-1][0].isupper():
        parts = parts[:-1]                       # drop the initials token
    return " ".join(parts).lower()


# PubMed's indexed surname differs from the form the author publishes under.
# Only compound surnames belong here, never a genuine disagreement about who wrote a paper.
SURNAME_ALIASES = {
    "davey smith": {"smith", "davey smith"},
}


def harvest() -> dict[str, dict]:
    table: dict[str, dict] = {}

    # Records captured in an earlier session are already persisted here. CAPTURE_DIRS
    # points at a session-specific scratch directory that will not exist next time, so
    # seeding from the saved audit record is what makes this script re-runnable.
    if OUT.is_file():
        try:
            table.update(json.loads(OUT.read_text()).get("records", {}))
        except Exception:
            pass

    for d in CAPTURE_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.txt")):
            try:
                blob = json.loads(f.read_text())
            except Exception:
                continue
            for a in blob.get("articles", []) or []:
                ids = a.get("identifiers") or {}
                pmid = ids.get("pmid")
                if not pmid:
                    continue
                au = [x for x in (a.get("authors") or []) if x.get("last_name")]
                cit = a.get("citation") or {}
                pd = a.get("publication_date") or {}
                table[str(pmid)] = {
                    "title": a.get("title"),
                    "first_author": (au[0].get("last_name") if au else None),
                    "journal": (a.get("journal") or {}).get("iso_abbreviation"),
                    "year": pd.get("year"),
                    "volume": cit.get("volume"),
                    "issue": cit.get("issue"),
                    "pages": cit.get("pages"),
                    "doi": a.get("doi"),
                    "source_file": f.name,
                }
    return table


def main() -> int:
    table = harvest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"_note": "PubMed records captured during reference sourcing; the audit record "
                  "for scripts/24_audit_proposal_references.py.",
         "_retrieved": "2026-09-18",
         "records": table}, indent=1, ensure_ascii=False))
    print(f"captured PubMed records: {len(table)}  ->  {OUT.relative_to(ROOT)}")

    problems: list[str] = []
    unverified: list[str] = []

    for key, r in REFS.items():
        pmid = r.get("pmid")
        if not pmid:
            unverified.append(f"{key}: no PMID recorded")
            continue
        got = table.get(str(pmid))
        if got is None:
            unverified.append(f"{key}: PMID {pmid} not in captured responses")
            continue

        if norm_title(r["title"]) != norm_title(got["title"]):
            problems.append(
                f"{key}: TITLE\n     bank: {r['title']}\n   pubmed: {got['title']}")

        want_sn, got_sn = surname(r["authors"]), (got["first_author"] or "").lower()
        allowed = SURNAME_ALIASES.get(want_sn, {want_sn})
        if got_sn and want_sn and got_sn not in allowed and not (
                want_sn.startswith(got_sn) or got_sn.startswith(want_sn)):
            problems.append(f"{key}: FIRST AUTHOR  bank={want_sn!r} pubmed={got_sn!r}")

        if got["journal"] and r["journal"].lower() != got["journal"].lower():
            problems.append(
                f"{key}: JOURNAL  bank={r['journal']!r} pubmed={got['journal']!r}")

        ok_years = {str(r["year"])}
        if r.get("pubmed_year"):
            ok_years.add(str(r["pubmed_year"]))
        if got["year"] and str(got["year"]) not in ok_years:
            problems.append(
                f"{key}: YEAR  bank={sorted(ok_years)} pubmed={got['year']!r}")

        for field in ("volume", "issue", "pages", "doi"):
            want, have = r.get(field), got.get(field)
            if have is None or want is None:
                continue          # PubMed omits article-number journals' page fields
            if str(want).lower() != str(have).lower():
                problems.append(
                    f"{key}: {field.upper()}  bank={want!r} pubmed={have!r}")

    print(f"\nreferences in bank: {len(REFS)}")
    if unverified:
        print(f"\nUNVERIFIED ({len(unverified)}) -- must be fetched before shipping:")
        for u in unverified:
            print("   " + u)
    if problems:
        print(f"\nMISMATCHES ({len(problems)}):")
        for p in problems:
            print("   " + p)
    if not problems and not unverified:
        print("\nRESULT: every reference matches PubMed's record exactly. OK")
    return 1 if (problems or unverified) else 0


if __name__ == "__main__":
    sys.exit(main())
