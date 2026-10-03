# Blood Metals and Metabolic Syndrome: Machine-Learning Component

Package refreshed {today}.

This package covers the machine-learning arm of the study: predicting metabolic syndrome
(MetS) in pooled NHANES 2007-2018 adults from demographic, lifestyle, body-size and
blood-metal predictors, and identifying which predictors carry the signal. The Mendelian
randomization (MR) arm is described as planned work in the proposal and has not been run;
no MR results appear anywhere in this package.

## Run everything with one click

| System | Do this |
|---|---|
| Windows | Double-click `RUN_ME_Windows.bat` |
| macOS | Double-click `RUN_ME_macOS.command` (first time: right-click, Open) |
| Linux | Run `bash RUN_ME_Linux.sh` |
| Any | Run `python RUN_ME.py` (or `python3 RUN_ME.py`) |

You need Python 3.11, 3.12 or 3.13 from python.org. The launcher installs the pinned
packages of `06_Code/requirements.txt` into a private `.venv` folder, asks one question
(FULL from the raw NHANES files, PACKAGED from the cleaned data in `05_Data_Processed/`,
or a QUICK test drive), runs every step in order, and ends with
`outputs/reproduction_report.txt`, which compares your numbers with the published ones.
FULL and PACKAGED take a few hours (the 10-fold tuning fits several thousand models), so
keep the computer awake. Everything the run creates goes into `outputs/`,
`data_processed/` and `manuscript/`; the published folders are only changed by
`06_Code/26_build_submission.py`.

## How the folders are organised

| Folder | Contents |
|---|---|
| `01_Manuscript/` | Methods, Results and Supplementary Material, as .docx and .pdf |
| `02_Proposal/` | The research proposal (and a partial plain-language draft), as .docx and .pdf |
| `03_Figures/Main/`, `03_Figures/Supplementary/` | Figures 1-5 and S1-S5, numbered as the manuscripts cite them |
| `04_Results_Tables/` | Every metrics table the manuscripts draw on, plus the citation audit records |
| `05_Data_Processed/` | The analytic dataset and the exact train/test split |
| `06_Code/` | The pipeline, numbered in execution order, plus shared helper modules |
| `07_Novelty_Additions/`, `08_Q1_Rigor_Additions/` | Exploratory analyses, each with its own README |
| `00_MANIFEST.csv` | Every published file with size, MD5 checksum and a one-line description |
| `FIX_CHECKLIST.md` | Every issue found in the October 2026 audit and the commit that fixed it |

Every figure is supplied as a 600 dpi PNG and a vector PDF drawn from the same code.

## Running the steps by hand

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r 06_Code/requirements.txt

python 06_Code/00_download_nhanes.py          # FULL only: raw files into NHANES_data/
for s in 01 02 03 04 06 07 08 09 10 11 13; do python 06_Code/"$s"_*.py; done
python 06_Code/12_visualize.py                # optional charts and dashboard
for f in 07_Novelty_Additions/code/*.py 08_Q1_Rigor_Additions/code/*.py; do python "$f"; done
python 06_Code/14_generate_diagrams.py        # proposal diagrams
python 06_Code/15_publication_figures.py      # figures and the numbers JSON
python 06_Code/16_write_methods_docx.py
python 06_Code/17_write_results_docx.py
python 06_Code/21_write_supplement.py
python 06_Code/29_update_proposal.py          # proposal text, tables and figures
python 06_Code/26_build_submission.py         # refresh the published folders
python 06_Code/18_verify_manuscript_numbers.py
python 06_Code/20_verify_citations.py
python 06_Code/27_verify_submission.py
```

To start from the shipped cleaned data instead of the raw files (PACKAGED), copy
`05_Data_Processed/nhanes_with_outcome.csv` into a `data_processed/` folder and start the
loop at 03. The random seed is fixed in `06_Code/config.py`, and the split is pinned by
SEQN in `05_Data_Processed/train_seqn.csv` and `test_seqn.csv`.

## What has been checked, and how

- `06_Code/18_verify_manuscript_numbers.py` traces every numeric token in the three
  manuscript documents back to a value in the pipeline's output. Current result: {counts}.
- `06_Code/20_verify_citations.py` checks the Methods references against the PubMed
  records captured in `04_Results_Tables/reference_verification.json`.
- `06_Code/24_audit_proposal_references.py` and `25_verify_proposal_citations.py` audit
  the proposal's reference bank and its citation numbering.
- `06_Code/27_verify_submission.py` checks the package as received: every published file
  is in the manifest with a matching checksum, every figure and table the text cites
  exists, no document carries a draft marker, no PDF page is blank, and every
  third-party package the code imports is pinned in `requirements.txt`.
- `06_Code/28_compare_to_reference.py` compares a fresh run with the published tables.

## Data

The raw NHANES `.xpt` files are public and are not included; `00_download_nhanes.py`
fetches them from the CDC website (https://wwwn.cdc.gov/nchs/nhanes/), or they can be
downloaded by hand into `NHANES_data/NHANES_<years>/`.
