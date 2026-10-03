"""
Short technical labels for figure axes, and the glossary text that defines them.

Figures and tables need different things from a variable name. A table has a whole column
to spend and should say "Current smoking intensity (0 = none, 1 = some days, 2 = every
day)"; a beeswarm axis has about 25 characters before the label either collides with the
plot or forces the font down to something unreadable in print. So this module keeps a
short form for every predictor, and -- crucially -- a matching glossary so that no
abbreviation ever appears in a figure without being expanded in that figure's caption.

`abbrev(code)` returns the short form; `glossary(codes)` returns the caption sentence
defining exactly the abbreviations that figure uses, in the order they appear, expanding
each to the full clinical name from config.CLINICAL_NAMES. Nothing here invents a name:
every expansion is the label the pipeline already carries for that variable.

Abbreviations are chosen to be standard where a standard exists (BMI; the IUPAC element
symbols Pb, Cd, Hg, Se, Mn; PIR for the NHANES income-to-poverty ratio) and otherwise to
be self-evident in context.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "cfg_for_labels", Path(__file__).resolve().parent / "config.py")
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)


# code -> short label used on figure axes
ABBREV: dict[str, str] = {
    # anthropometry and demography
    "BMXBMI":       "BMI",
    "RIDAGEYR":     "Age",
    "sex_female":   "Sex (female)",
    "BPXPLS":       "Resting pulse",
    "INDFMPIR":     "PIR",
    "DMDEDUC2":     "Education",
    "DMDHHSIZ":     "Household size",
    "exam_summer":  "Exam season",

    # blood metals
    "LBXBPB":       "Blood Pb",
    "LBXBCD":       "Blood Cd",
    "LBXTHG":       "Blood Hg",
    "LBXBSE":       "Blood Se",
    "LBXBMN":       "Blood Mn",

    # physical activity
    "PAQ605":       "Vigorous work PA",
    "PAQ650":       "Vigorous rec. PA",
    "PAQ665":       "Moderate rec. PA",
    "PAD645":       "Active transport",

    # smoking, alcohol, diet
    "SMQ020":       "Ever smoked",
    "SMQ040":       "Smoking intensity",
    "ALQ_PAST12":   "Drank past 12 mo",
    "ALQ130":       "Drinks/drinking day",
    "DBQ700":       "Diet quality",

    # one-hot categories
    "RIDRETH1_1.0":  "Eth: Mex. American",
    "RIDRETH1_2.0":  "Eth: Other Hispanic",
    "RIDRETH1_3.0":  "Eth: NH White",
    "RIDRETH1_4.0":  "Eth: NH Black",
    "RIDRETH1_5.0":  "Eth: Other/Multi",
    "RIDRETH1_nan":  "Eth: missing",
    "DMDMARTL_1.0":  "Marital: Married",
    "DMDMARTL_2.0":  "Marital: Widowed",
    "DMDMARTL_3.0":  "Marital: Divorced",
    "DMDMARTL_4.0":  "Marital: Separated",
    "DMDMARTL_5.0":  "Marital: Never married",
    "DMDMARTL_6.0":  "Marital: Partnered",
    "DMDMARTL_nan":  "Marital: missing",
}

# Codes whose short label a reader cannot reconstruct from the label alone, and which
# therefore have to be expanded in the caption. Labels left out of this set are ones that
# already read as ordinary English -- "Age", "Education", "Household size",
# "Marital: Divorced" -- and glossing those would pad the caption without informing anyone.
GLOSS_REQUIRED: set[str] = {
    "BMXBMI", "INDFMPIR", "BPXPLS", "exam_summer",
    "LBXBPB", "LBXBCD", "LBXTHG", "LBXBSE", "LBXBMN",
    "PAQ605", "PAQ650", "PAQ665", "PAD645",
    "SMQ020", "SMQ040", "ALQ_PAST12", "ALQ130", "DBQ700",
    "RIDRETH1_1.0", "RIDRETH1_2.0", "RIDRETH1_3.0", "RIDRETH1_4.0",
    "RIDRETH1_5.0", "RIDRETH1_nan",
}

MISSING_PREFIX = "missingindicator_"


def abbrev(code: str) -> str:
    """Short figure label for a feature-matrix column name.

    Falls through to config.label_feature() for anything not in ABBREV -- including
    missing-value indicator columns -- so this never raises and never silently drops a
    variable from a plot.
    """
    if code in ABBREV:
        return ABBREV[code]
    # FIX (2026-10 audit, C4): MissingIndicator = 1 means MISSING, so the
    # short label says "(missing)", not "(obs.)".
    if code == _cfg.SE_MN_FLAG_NAME:
        return "Se/Mn not measured"
    if code.startswith(MISSING_PREFIX):
        base = code[len(MISSING_PREFIX):]
        return f"{abbrev(base)} (missing)"
    return _cfg.label_feature(code)


def abbrevs(codes) -> list[str]:
    return [abbrev(c) for c in codes]


def full_name(code: str) -> str:
    """The long clinical label, for tables and for the caption glossary."""
    return _cfg.label_feature(code)


def glossary(codes, *, include_full: bool = True) -> str:
    """Caption sentence defining every abbreviation the given figure uses.

    Returns '' when the figure uses no abbreviated labels, so a caption can concatenate
    it unconditionally.
    """
    seen: list[str] = []
    for c in codes:
        if c in ABBREV and c not in seen:
            seen.append(c)
    if not seen:
        return ""

    parts = []
    for c in seen:
        if not include_full or c not in GLOSS_REQUIRED:
            continue
        short, long = ABBREV[c], full_name(c)
        if short.rstrip(".") == long.rstrip("."):
            continue
        parts.append(f"{short}, {long[0].lower() + long[1:]}")

    if not parts:
        return ""
    return "Abbreviations: " + "; ".join(parts) + "."


GLOSSARY_PREFIX = "Abbreviations: "


def parse_glossary(text: str) -> list[tuple[str, str]]:
    """Inverse of glossary(): recover the (short, expansion) pairs from a rendered string.

    Lets a later caption work out which abbreviations an earlier caption already defined,
    so the document can follow the usual journal convention of defining each abbreviation
    once and pointing back to it thereafter.
    """
    if not text:
        return []
    body = text[len(GLOSSARY_PREFIX):] if text.startswith(GLOSSARY_PREFIX) else text
    pairs = []
    for chunk in body.rstrip(".").split("; "):
        short, _, expansion = chunk.partition(", ")
        if short and expansion:
            pairs.append((short.strip(), expansion.strip()))
    return pairs


def incremental_glossary(text: str, seen: dict[str, str], back_ref: str) -> str:
    """Glossary for a caption whose abbreviations were mostly defined in an earlier one.

    `seen` is updated in place with every pair this caption covers, so callers can thread
    one dict through a document in reading order. `back_ref` names the figure that carried
    the first definitions, e.g. "Figure 3".
    """
    pairs = parse_glossary(text)
    if not pairs:
        return ""
    new = [(s, e) for s, e in pairs if s not in seen]
    reused = [s for s, _ in pairs if s in seen]
    seen.update(dict(pairs))

    if not reused:
        return GLOSSARY_PREFIX + "; ".join(f"{s}, {e}" for s, e in new) + "."
    if not new:
        return f"Abbreviations as in {back_ref}."
    return (f"Abbreviations as in {back_ref}, plus "
            + "; ".join(f"{s}, {e}" for s, e in new) + ".")


if __name__ == "__main__":
    longest = max(ABBREV.values(), key=len)
    print(f"{len(ABBREV)} abbreviations; longest is {longest!r} at {len(longest)} chars")
    over = {c: s for c, s in ABBREV.items() if len(s) > 24}
    print("over 24 chars:", over or "none")
    import itertools
    dupes = [s for s, g in itertools.groupby(sorted(ABBREV.values()))
             if len(list(g)) > 1]
    print("duplicate short labels:", dupes or "none")
    print()
    print(glossary(["BMXBMI", "RIDAGEYR", "LBXBPB", "PAQ650", "INDFMPIR"]))
