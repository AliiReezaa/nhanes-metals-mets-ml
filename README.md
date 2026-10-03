# Blood Metals and Metabolic Syndrome — AI/ML Component

Submission package prepared 2026-09-18.

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
  rather than passing it silently. Current result: 997 tokens checked, 0 requiring adjudication. This proves no number was
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
