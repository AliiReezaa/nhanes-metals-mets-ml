# Fix checklist (manual chapter 28 plus the October 2026 full re-audit)

Status keys: `[ ]` open, `[x]` fixed and committed, `[~]` addressed by disclosure (cannot be fixed with the available data), `[>]` code committed, output pending the final run.
Each item names the commit that closes it. Items marked **(re-run)** change the cohort, the label or the features, so every downstream number is regenerated in the final re-run (item R1).

Decisions confirmed with the study owner on 2026-10-03:
- Lipid-medication clause: keep it in the primary outcome, state it as a choice, add a sensitivity arm without it.
- Blood pressure: primary outcome uses the mean of all available readings; first-reading-only becomes a sensitivity arm.
- MR proposal: lead and selenium become primary exposures, cadmium secondary; the relaxed P < 5e-6 rule applies to every metal with an F > 10 check, and mercury is kept as exploratory.

## 28.1 Coding and data-handling issues

| ID | Issue | Fix | Status |
|---|---|---|---|
| C1 | Refusal codes 7/9 erased real ALQ130 answers of 7 and 9 drinks/day | Per-variable sentinel maps (ALQ130: 777/999; 1/2 items: 7/9; DMDMARTL: 77/99; PAD645: 7777/9999) **(re-run)** || [x] 33f8237 |
| C2 | Non-drinkers' ALQ130 median-imputed (3,280 structural zeros) | ALQ130 = 0 for anyone who did not drink in the past 12 months **(re-run)** || [x] 33f8237 |
| C3 | Never-smokers' SMQ040 left to mode imputation | SMQ040 = 0 ("not at all") when SMQ020 = No **(re-run)** || [x] 33f8237 |
| C4 | Missingness flags labelled "was measured/observed" (1 actually means missing) | Relabel everywhere as "missing (imputed)"; fix docstrings || [x] f301b07 |
| C5 | No fasting check (487 fasted < 8 h; 610 rows have zero fasting weight) | Exclude participants who did not meet the NCHS fasting-subsample criteria (WTSAF2YR = 0, or PHAFSTHR outside 8 to 24 h) **(re-run)** || [x] 33f8237 |
| C6 | Table 1 uses the examination weight; Methods claims division by cycles | Use the fasting-subsample weight WTSAF2YR / 6 for every row; describe what is done || [x] eca7e75 |
| C7 | Metals summarised as arithmetic means | Weighted geometric means (tests on the log scale) || [x] eca7e75 |
| C8 | Bootstrap p-values printed as "p < 0.001" from 1,000 resamples | Paired DeLong test for AUROC differences (the method already cited), bootstrap kept for CIs || [x] af27205 |
| C9 | README loop `06_Code/$s_*.py` runs the wrong files; other README errors | Regenerate README from 26_build_submission.py with correct commands and the launcher || [x] fe713b6 |
| C10 | requirements.txt misses seaborn, patsy, plotly; python-docx unpinned | Pin every imported package; build the list from the real imports || [x] fe713b6 |
| C11 | Manifest out of date; 27_verify_submission.py fails; stray Word lock file | Make 26/27 work in this layout, rebuild manifest, lock file excluded || [x] fe713b6 |
| C12 | 08 code and tables say "07_Q1_Rigor_Additions" | Rename to 08 and regenerate the checklist tables || [x] 688a2fa |
| C13 | Pregnant participants (103) not excluded (new) | Exclude RIDEXPRG = 1 **(re-run)** || [x] 33f8237 |
| C14 | PAD645 code 9999 kept as 9,999 minutes (new) | Covered by C1 sentinel map **(re-run)** || [x] 33f8237 |
| C15 | ALQ_EVER mixes "12+ drinks in a year" (2007-2016) with "ever drank" (2017-2018) (new) | Replace with a construct asked the same way in all cycles: drank alcohol in the past 12 months **(re-run)** || [x] 33f8237 |
| C16 | BP label uses the first reading only (new; decision above) | Mean of all available readings; zero diastolic readings treated as invalid **(re-run)** || [x] 33f8237 |
| C17 | Stale caches reused across runs (test predictions, permutation tables, best params, thresholds, old .joblib models) (new) | Clear caches at the start of the steps that own them; launcher starts each run from a clean outputs folder || [x] 7250dcf, e28f8cb |
| C18 | Constant and duplicate missing-indicator columns; RIDRETH1_nan all zeros (new) | Indicators only for features with missing values; no NaN dummy for complete nominal variables; Se/Mn flag labelled as a 2007-2010 period marker || [x] f301b07 |
| C19 | ALQ_EVER binary variable sent through the continuous path (new) | New binary alcohol variable goes through the binary path || [x] f301b07 |
| C20 | Table 1 design-based tests ignore covariance and use z (new) | Linearise the difference directly; t with df = PSUs - strata || [x] eca7e75 |
| C21 | Sensitivity analyses use untuned defaults; GB class-weight arm NaN (new) | Use the tuned hyperparameters; balanced sample weights for Gradient Boosting || [x] d9c51cd |
| C22 | 07/08 scripts depend on the working directory and hard-code predictors and hyperparameters; duplicated setup files (new) | Resolve paths from the file location, import config and best_params.json, share one setup module || [x] 688a2fa |
| C23 | Small code defects: fig6 hard-codes CatBoost; 13 uses bootstrap mean as subgroup estimate; dead code; stale docstrings; DCA stops at 0.50; 28_compare checks tautological in PACKAGED mode (new) | Fix each || [x] 7250dcf, bea4271 |
| C24 | Proposal-citation scripts (22 to 25) and 26 expect a `scripts/` and `docx_final/` layout that is not shipped (new) | Point them at this layout; document which inputs they need || [x] fe713b6 |
| C25 | Temporal-validation intercept computed differently from the main calibration intercept (new) | Use the same offset-model intercept || [x] 1fc3e70 |

## 28.2 Analytical gaps

| ID | Gap | Fix | Status |
|---|---|---|---|
| A1 | No model without metals (proposal Q3 unanswered) | Incremental-value analysis: full vs no-metals vs lead-only, paired DeLong and bootstrap || [x] ab8ee74 |
| A2 | No age + BMI baseline | Logistic regression on age and BMI alone, same comparison || [x] ab8ee74 |
| A3 | BMI kept while waist defines the outcome | No-BMI model in the same analysis; limitation stated || [x] ab8ee74 |
| A4 | One random split, one seed | Repeated stratified cross-validation (5 x 10-fold) for all 11 models || [x] ab8ee74 |
| A5 | Interaction p-values from SHAP correlations on imputed rows | Explicit logistic interaction models on observed rows with FDR control; SHAP correlations kept as descriptive only, on observed rows || [x] 7b2c37f |
| A6 | Dose-response and WQS omit smoking and cycle (and race, education, alcohol) | Full adjustment set; WQS weights estimated and tested on separate halves || [x] 55ce1a5 |
| A7 | Temporal validation reuses all-cycle hyperparameters | Re-tune with 10-fold CV on 2007-2014 only || [x] 1fc3e70 |
| A8 | Outcome-definition sensitivity arms for the two decisions above | No-lipid-clause arm and first-BP-reading arm || [x] d9c51cd |

## 28.3 Proposal and document inconsistencies

| ID | Issue | Fix | Status |
|---|---|---|---|
| P1 | Proposal Table 1 citation numbers wrong (9 of 9) and wrong citations in 3.2.4 | Correct every number against the reference list || [x] f6cf853 (code), 8c07187 (applied to the final results) |
| P2 | Removed 13-SNP lead claim still in Table 1, H6, H8 and Figure 2 | Remove everywhere; redraw Figure 2 || [x] 6706bd4 (figure); text in 29_update_proposal.py |
| P3 | Proposal Figure 1 shows old numbers and the old definition | Redraw from the final numbers || [x] 8c07187 |
| P4 | Table 1 contradictions (cadmium "lowest", selenium ~230, draft wording) | Correct to the final numbers; remove draft wording || [x] f6cf853 (code), 8c07187 (applied to the final results) |
| P5 | Exposure selection logic (decision above) | Lead and selenium primary, cadmium secondary, manganese and mercury exploratory || [x] f6cf853 (code), 8c07187 (applied to the final results) |
| P6 | Instrument rule contradicts the mercury exclusion (decision above) | One rule for every exposure || [x] f6cf853 (code), 8c07187 (applied to the final results) |
| P7 | MR sensitivity methods infeasible with few SNPs; "odds ratios" for continuous outcomes; composite-MetS promises | Tiered methods by SNP count; betas; component-level wording || [x] f6cf853 (code), 8c07187 (applied to the final results) |
| P8 | H4 (and H3) written after the results; Q3 had no analysis | Reframe as ML-phase findings; Q3 answered by A1 || [x] f6cf853 (code), 8c07187 (applied to the final results) |
| P9 | Humanized proposal: listed errors plus further typos and causal overstatements | Correct every error; mark it as a partial draft || [x] d972124 |
| P10 | Outcome rationale wrong: the 2005 AHA/NHLBI criteria already include treatment (Methods and proposal) | Correct wording in Methods, Supplement and proposal || [x] f6cf853 |
| P11 | Decision-curve result misinterpreted (47 "additional true cases") | Express as net reduction in unnecessary interventions; fix publication_numbers || [x] 7250dcf, d781cff |
| P12 | Methods ref [8] (Lo et al., manganese only) cited for the five-metal panel; ref [21] is a different permutation method | Cite Bulka et al. 2019 for the panel; Breiman 2001 / Fisher et al. 2019 for permutation importance || [x] 6e9f296 |
| P13 | Methods says no temporal validation, but 08 did one | Report it (re-tuned, A7) || [x] f6cf853 |
| P14 | Methods claims bootstrap intervals for threshold metrics that were not computed | Compute them || [x] af27205 |
| P15 | Methods missingness sentence miscounts; Figure 4B cited for metals it does not show; Table S3 caption; Figure 4 caption | Correct each || [x] f6cf853 |
| P16 | 07 dossier says selenium "protective"; READMEs overstate 07/08 findings; Claude outputs folder stale | Correct after re-run; remove the stale folder || [x] d781cff, 8c07187 |
| P17 | Proposal internal contradictions (literature review on which metal is most consistent; ref [46] description; PhenoScanner; window units; sample overlap; Table 1 variables vs Results Table 1) | Correct each || [x] f6cf853 (code), 8c07187 (applied to the final results) |
| P18 | Limitations to add: waist cut-offs for Asian participants, single-measure metals, BMI and waist overlap, lipid clause | Add to Methods/Results limitations || [x] 6e9f296 |

## Final steps

| ID | Step | Status |
|---|---|---|
| R1 | Full re-run from the raw NHANES files with all fixes; regenerate tables, figures, manuscript and supplement | [x] 8c07187 (full run from the raw files, 28 steps, all passed) |
| R2 | Verification scripts (18, 20, 27, 28) all pass | [x] 8c07187 (18: 0 unexplained numbers; 20, 24, 25, 27 pass; 28: 13 checks, 0 flagged) |
| R3 | Independent review of the final package | [ ] |
| R4 | Update the beginner's manual to the final version | [ ] |
| R5 | Copy the final package back to the study folder | [ ] |
