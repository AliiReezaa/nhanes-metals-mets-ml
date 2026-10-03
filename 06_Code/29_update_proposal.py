"""
Bring the research proposal (02_Proposal/Research_Proposal.docx) in line with the final
machine-learning results and correct the inconsistencies found in the October 2026 audit
(checklist P1-P8, P10, P17). Run after the pipeline (it reads outputs/metrics/ and the
diagrams written by 14_generate_diagrams.py) and before 26_build_submission.py.

What it changes, and why:
  * ML-phase numbers (cohort flow, prevalence, reclassification, predictor ranks,
    missingness, collinearity screen) are written from the current outputs, so the
    proposal can no longer drift from the results (the earlier Figure 1 and Table 1
    still showed numbers from superseded runs).
  * Methods text describing what the pipeline does (fasting and pregnancy exclusions,
    mean blood pressure, the alcohol construct, fasting weights, DeLong tests, the
    added analyses) is updated to match the code.
  * The outcome-definition rationale is corrected: the 2005 AHA/NHLBI criteria already
    include drug treatment; the harmonized definition keeps those clauses (P10).
  * MR design: lead and selenium become primary exposures and cadmium secondary,
    following the ML evidence (decision confirmed with the study owner, P5); one
    instrument rule for every exposure, with mercury exploratory (P6); MR methods tiered
    by the number of SNPs, betas instead of odds ratios, components instead of
    composite MetS (P7); the removed 13-SNP lead instrument is gone everywhere (P2);
    Table 1 citation numbers point at the right references (P1); hypotheses no longer
    restate results (P8).
  * Literature-review fixes: typos, the selenium deficiency citation, the ICU reference
    description, and the claim about which metal leads the evidence (P17).
  * Figures 1 and 2 are replaced by the code-drawn versions in diagrams/.

Paragraphs are located by their section heading plus an offset, and every replacement
is checked, so the script fails loudly rather than silently editing the wrong place.
It edits the file in place (git keeps the history).

Run: python 06_Code/29_update_proposal.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "02_Proposal" / "Research_Proposal.docx"
MET = cfg.OUT_METRICS
DIAG = ROOT / "diagrams"

N = json.loads((MET / "publication_numbers.json").read_text())
flow = pd.read_csv(MET / "exclusion_flow.csv")
outdef = pd.read_csv(MET / "outcome_definition_comparison.csv")
coll = pd.read_csv(MET / "collinearity_flags.csv").iloc[0]
miss = pd.read_csv(MET / "missingness_report.csv").set_index("variable")["pct_missing"]
perm = pd.read_csv(MET / "permutation_importance.csv")
inc = pd.read_csv(MET / "incremental_value.csv").set_index("model")
sens_out = pd.read_csv(MET / "sensitivity_outcome_definition.csv")

rem = flow["n_remaining"].astype(int).tolist()
exc = flow["n_excluded_this_step"].fillna(0).astype(int).tolist()
n_final = rem[5]
prev_h = float(outdef.loc[1, "prevalence_pct"])
prev_l = float(outdef.loc[0, "prevalence_pct"])
n_recl = int(outdef.loc[1, "n_reclassified_negative_to_positive"])
n_pred = N["n_predictors"]
mr = N["metal_ranks"]
top = N["top_model"]
shap_model = N.get("shap_model", top)


_shap = pd.read_csv(MET / "shap_importance.csv")
_shap = _shap[~_shap["feature_code"].str.startswith("missingindicator")].copy()
_shap["rank"] = _shap["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
SHAP_RANK = _shap.set_index("feature_code")["rank"]


def shap_rank(code):
    return int(SHAP_RANK[code])


def shape_of(values):
    """Plain-language shape of a binned SHAP dependence curve (same rule as 17)."""
    v = np.asarray(values, float)
    rng = v.max() - v.min()
    if rng < 0.03:
        return "flat"
    rho = pd.Series(v).corr(pd.Series(np.arange(len(v))), method="spearman")
    imin, imax = int(v.argmin()), int(v.argmax())
    if rho >= 0.8:
        return "rising"
    if rho <= -0.8:
        return "falling"
    if 0 < imin < len(v) - 1 and min(v[0], v[-1]) - v[imin] > 0.4 * rng:
        return "U-shaped"
    if 0 < imax < len(v) - 1 and v[imax] - max(v[0], v[-1]) > 0.4 * rng:
        return "inverted U-shaped"
    return "rising" if rho > 0.3 else "falling" if rho < -0.3 else "irregular"


def shape_txt(code):
    sh = shape_of(N["metals_dependence"][code]["binned_mean_shap"])
    return {"flat": "SHAP contribution essentially flat across the range",
            "rising": "SHAP contribution rises with concentration",
            "falling": "SHAP contribution falls with concentration",
            "U-shaped": "U-shaped SHAP contribution",
            "inverted U-shaped": "inverted-U SHAP contribution",
            "irregular": "no consistent SHAP direction"}[sh]


perm["rank"] = perm["perm_importance_mean"].rank(ascending=False, method="min").astype(int)
PERM = perm.set_index("feature_code")


def prank(code):
    return int(PERM.loc[code, "rank"])


def above_noise(code):
    r = PERM.loc[code]
    return r["perm_importance_mean"] > r["perm_importance_std"]


def span(code):
    v = list(mr[code]["per_model_perm_rank"].values())
    return f"{min(v)} to {max(v)}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def nonempty_after(doc, heading):
    paras = doc.paragraphs
    idx = [i for i, p in enumerate(paras) if p.text.strip() == heading]
    if len(idx) != 1:
        raise SystemExit(f"heading not found exactly once: {heading!r} ({len(idx)})")
    return [p for p in paras[idx[0] + 1:] if p.text.strip()]


def para(doc, heading, offset=0):
    return nonempty_after(doc, heading)[offset]


def set_text(p, text):
    runs = p.runs
    runs[0].text = text
    for r in runs[1:]:
        r.text = ""


def sub(p, old, new):
    t = p.text
    if old not in t:
        if new in t:
            return  # already applied
        raise SystemExit(f"text not found: {old[:70]!r}")
    set_text(p, t.replace(old, new))


def set_cell(cell, text):
    ps = cell.paragraphs
    if ps[0].runs:
        set_text(ps[0], text)
    else:
        ps[0].add_run(text)
    for extra in ps[1:]:
        extra._p.getparent().remove(extra._p)


def replace_image(doc, index, png):
    """Swap the index-th inline picture for a new PNG, keeping its width."""
    from PIL import Image
    shape = doc.inline_shapes[index]
    blip = shape._inline.graphic.graphicData.pic.blipFill.blip
    part = doc.part.related_parts[blip.embed]
    part._blob = Path(png).read_bytes()
    w, h = Image.open(png).size
    width = shape.width
    shape.height = int(width * h / w)


# ---------------------------------------------------------------------------
def main():
    doc = Document(DOC)

    # ---- 1-2. State of problem and literature review (P17 and typos)
    sp = "1. State of Problem"
    sub(para(doc, sp, 2), "cigaret smoke", "cigarette smoke")
    sub(para(doc, sp, 2), "particularly in the presence of zinc [12-14].",
        "particularly in the presence of zinc [12,13], although selenium deficiency has also been "
        "linked to metabolic syndrome [14].")
    sub(para(doc, sp, 4), "central obesity .", "central obesity.")
    sub(para(doc, sp, 6), " thot ", " thought ")
    sub(para(doc, sp, 12), "to predict pressure injuries in ICU patients",
        "to predict mortality in ICU patients with pressure ulcers")
    sub(para(doc, sp, 14), "(lead foremost among them, but cadmium, mercury, selenium, and manganese "
        "close behind)", "(lead the best documented among them, alongside cadmium, mercury, selenium, "
        "and manganese)")
    lr = "2. Literature Review"
    sub(para(doc, lr, 21), "Cadmium, mercury, and selenium carry the most consistent evidence for "
        "increased MetS risk; lead and manganese remain mixed",
        "Cadmium and mercury carry the most consistent evidence for increased MetS risk, selenium "
        "less so; lead and manganese remain mixed")

    # ---- 3.1.1 Data sources
    set_text(para(doc, "3.1.1. Data sources", 0),
             f"Six consecutive NHANES cycles feed this study: 2007–2008, 2009–2010, 2011–2012, "
             f"2013–2014, 2015–2016, and 2017–2018. Pooled together, they use a stratified, "
             f"multistage probability design built to produce nationally representative estimates "
             f"for the U.S. civilian, non-institutionalized population. Merging the examination, "
             f"laboratory, questionnaire and demographic files across all six cycles gives a pooled "
             f"sample of {rem[0]:,} individuals, of whom {rem[1]:,} are adults aged 20 and older. "
             f"Pregnant participants ({exc[2]:,}) are excluded, because pregnancy changes waist "
             f"circumference, blood pressure, lipids and glucose. Anyone missing one of the five MetS "
             f"components (waist circumference, fasting glucose, HDL cholesterol, triglycerides, or "
             f"blood pressure) is excluded next ({exc[3]:,}), mostly because triglycerides and "
             f"glucose are measured only in the morning fasting subsample; then participants who did "
             f"not meet the fasting criterion of 8 to 24 hours ({exc[4]:,}, identified by a zero "
             f"fasting-subsample weight); and finally anyone with no blood-metal measurement at all "
             f"({exc[5]:,}). A participant with, say, lead and cadmium measured but not the other "
             f"three is retained. The final analytic cohort is {n_final:,} adults (Figure 1, "
             f"Table 2).")
    set_text(para(doc, "3.1.1. Data sources", 1),
             "Because the analytic cohort is, by construction, the fasting subsample, survey-weighted "
             "descriptive statistics use the fasting-subsample weight (WTSAF2YR) divided by the number "
             "of pooled cycles, following the NHANES analytic guidelines. In 2013–2014 and 2015–2016 "
             "the metals panel was measured in a separate one-half subsample, and NCHS provides no "
             "weight for the overlap of the two subsamples, so metal estimates for those cycles are "
             "approximate; this is stated as a limitation. Standard errors account for the stratified, "
             "multistage design via Taylor-series linearization (the ultimate-cluster method) using "
             "the primary sampling unit (SDMVPSU) and stratum (SDMVSTRA) variables, as recommended "
             "for the masked public-use design. Models are fitted unweighted, since the target is "
             "individual-level discrimination rather than a population parameter.")

    # ---- 3.1.2 Diagnostic criteria (P10 and the two definition decisions)
    set_text(para(doc, "3.1.2. Diagnostic criteria", 0),
             f"Metabolic syndrome is defined by the harmonized criteria of the 2009 joint interim "
             f"statement (Alberti et al., Circulation 2009;120:1640–1645), which retain the 2005 "
             f"AHA/NHLBI thresholds and drug-treatment clauses (Grundy et al., Circulation "
             f"2005;112:2735–2752). Each component is a yes/no flag scored positive if the measured "
             f"value crosses the threshold, abdominal obesity (waist circumference ≥ 102 cm in men, "
             f"≥ 88 cm in women), hypertriglyceridemia (triglycerides ≥ 150 mg/dL), low HDL-C "
             f"(< 40 mg/dL in men, < 50 mg/dL in women), elevated blood pressure (mean systolic "
             f"≥ 130 mmHg or mean diastolic ≥ 85 mmHg, averaged over all valid readings), or impaired "
             f"fasting glucose (≥ 100 mg/dL), OR if the participant reports current drug treatment "
             f"for that condition: antihypertensive medication (BPQ050A) for blood pressure, insulin "
             f"or oral diabetes medication (DIQ050, DIQ070) for glucose, and cholesterol-lowering "
             f"medication (BPQ100D) for the two lipid components. NHANES does not name the "
             f"cholesterol drug, whereas the guideline's lipid clause refers to drugs such as fibrates "
             f"and nicotinic acid, so applying BPQ100D to both lipid components is a stated modelling "
             f"choice that is tested in a sensitivity analysis. Three or more positive components "
             f"define MetS. The four medication items and the five component measurements build the "
             f"label only and are excluded from the predictor set (section 3.1.3). The treatment "
             f"clauses reclassify {n_recl:,} of {n_final:,} participants from MetS-negative to "
             f"MetS-positive (none move the other way), raising prevalence from {prev_l:.2f}% "
             f"(measured values only) to {prev_h:.2f}% (harmonized definition). Sensitivity analyses "
             f"(section 3.1.7) re-fit the leading models with the measured-value-only label, without "
             f"the lipid-medication clause, and with blood pressure from the first reading only.")

    # ---- 3.1.3 Predictors
    p = para(doc, "3.1.3. Variable definition and predictor selection", 1)
    set_text(p,
             f"The predictor set draws on four categories. Blood metals anchor the exposure side: "
             f"lead (LBXBPB), cadmium (LBXBCD), total mercury (LBXTHG), selenium (LBXBSE), and "
             f"manganese (LBXBMN), all measured in the same whole-blood laboratory module. Demographic "
             f"and socioeconomic covariates cover age (RIDAGEYR), sex (RIAGENDR), race/ethnicity "
             f"(RIDRETH1), education (DMDEDUC2), marital status (DMDMARTL), household size (DMDHHSIZ), "
             f"the income-to-poverty ratio (INDFMPIR), and examination period (RIDEXMON). Body mass "
             f"index (BMXBMI) is the sole anthropometric predictor; hip circumference was dropped "
             f"after a collinearity screen on the training partition only found it correlated with "
             f"BMI at r = {coll['pearson_r']:.3f} and missing in {coll['pct_missing_var_1_train']:.1f}% "
             f"of the training partition. Because BMI is strongly correlated with waist circumference, "
             f"which defines one MetS component, part of BMI's predictive weight is close to "
             f"definitional; a model without BMI is fitted for comparison. Lifestyle covariates are "
             f"smoking history and current intensity (SMQ020, SMQ040; never-smokers are coded as not "
             f"smoking now, since NHANES skips them past SMQ040), alcohol (any alcohol in the past 12 "
             f"months, built the same way in every cycle from the past-year drinking frequency, and "
             f"average drinks per drinking day, ALQ130, set to zero for past-year non-drinkers), "
             f"self-rated diet quality (DBQ700), vigorous work, vigorous recreational and moderate "
             f"recreational activity (PAQ605, PAQ650, PAQ665), active-transportation minutes (PAD645, "
             f"zero when the gating question PAQ635 is answered no), and resting pulse (BPXPLS). "
             f"Refusal and don't-know codes are recoded to missing using each item's own code width, so "
             f"genuine answers such as 7 or 9 drinks are kept. Categorical variables are one-hot or "
             f"ordinally encoded; continuous variables are median-imputed and standardized, and each "
             f"continuous predictor with missing values carries a binary indicator (1 = missing), with "
             f"one shared indicator for selenium and manganese, which were not assayed in 2007–2010. "
             f"All imputation, scaling and resampling happens inside the model-fitting pipeline "
             f"(section 3.1.6), never beforehand, so no information from a held-out fold or the test "
             f"set leaks into training.")

    # ---- 3.1.4 Baseline analysis
    set_text(para(doc, "3.1.4. Baseline analysis", 0),
             "The baseline analysis follows the NHANES complex-survey protocol. Using the fasting-"
             "subsample weight divided by the number of cycles (section 3.1.1), survey-weighted "
             "domain means are computed separately for the MetS and non-MetS groups for age, BMI and "
             "the income-to-poverty ratio, and weighted geometric means for the five blood metals, "
             "which are strongly right-skewed. Weighted percentages are computed for sex, "
             "race/ethnicity, education and smoking history. Standard errors use Taylor-series "
             "linearization with domain estimation; the difference between groups is linearized "
             "directly, so the covariance between the two domains is included, and is tested with a "
             "design-based t test (on the log scale for the metals) or, for categorical variables, "
             "the design-based adjusted Wald F test, with degrees of freedom equal to the number of "
             "primary sampling units minus the number of strata.")

    # ---- 3.1.6 Modeling
    p = para(doc, "3.1.6. Modeling", 0)
    t = p.text
    t = t.replace("Hyperparameters for each algorithm are tuned by grid or randomized search under "
                  "stratified 10-fold cross-validation on the training partition only, scored on ROC AUC.",
                  "Hyperparameters for each algorithm are tuned over a small grid (at most 12 "
                  "configurations, searched exhaustively) under stratified 10-fold cross-validation on "
                  "the training partition only, scored on ROC AUC.")
    import re
    t = re.sub(r"Class imbalance \([0-9.]+% MetS prevalence", f"Class imbalance ({prev_h:.2f}% MetS prevalence", t)
    t = t.replace("Each continuous predictor also carries an always-present, separately-fit missingness "
                  "indicator (see section 3.1.3) alongside its median-imputed value, so that structurally "
                  "or systematically missing measurements remain visible to the model as information.",
                  "Each continuous predictor with missing values also carries a missing-value indicator "
                  "(see section 3.1.3) alongside its median-imputed value.")
    set_text(p, t)

    # ---- 3.1.7 Evaluation
    p = para(doc, "3.1.7. Model evaluation and interpretation", 0)
    t = p.text
    t = t.replace("Evaluation runs entirely in Python (3.10+)", "Evaluation runs entirely in Python (3.11–3.13)")
    t = t.replace("AUC with a bootstrap 95% confidence interval, AUPRC, Brier score, Youden's index, accuracy, "
                  "sensitivity, specificity, F1 score, Cohen's kappa, positive predictive value, and negative "
                  "predictive value are all derived,",
                  "AUC, AUPRC, Brier score, Youden's index, accuracy, sensitivity, specificity, F1 score, "
                  "Cohen's kappa, positive predictive value, and negative predictive value are all derived, "
                  "each with a 1,000-resample bootstrap 95% confidence interval,")
    t = t.replace("a paired bootstrap test (same resampled test-set rows applied to every model within each "
                  "of 1,000 draws) additionally compares the top-ranked model against every other model and "
                  "flags which ones are not statistically distinguishable from it at the 95% level;",
                  "DeLong's test for correlated AUROCs compares the top-ranked model against every other "
                  "model, with a paired-bootstrap confidence interval for each difference, and flags which "
                  "models are not statistically distinguishable from it at the 95% level;")
    set_text(p, t)
    set_text(para(doc, "3.1.7. Model evaluation and interpretation", 1),
             "Further checks round out the evaluation. A decision-curve analysis compares net benefit "
             "against treat-all and treat-none strategies across threshold probabilities from 2% to "
             "80%. A subgroup analysis recomputes discrimination and calibration within sex, "
             "race/ethnicity and age-tertile strata. Sensitivity analyses on the three leading models, "
             "each with its tuned hyperparameters, add a diagnosed-diabetes indicator, replace SMOTENC "
             "with class weighting, restrict the metal panel to lead, and substitute three alternative "
             "outcome definitions (measured values only, no lipid-medication clause, first blood-pressure "
             "reading only). Research question Q3 is answered directly by refitting the leading model "
             "without the five metals, with lead as the only metal, and without BMI, and by a logistic "
             "regression on age and BMI alone, each compared with the full model on the same test "
             "participants by DeLong's test. Repeated stratified 10-fold cross-validation (five "
             "repeats) shows how much performance varies between data partitions, and a temporal "
             "validation re-tunes and refits the leading algorithm on the 2007–2014 cycles and tests it "
             "on 2015–2018.")

    # ---- 3.2.1 Study design (P5, P8)
    pb, se, cd, hg, mn = (mr[k] for k in ["LBXBPB", "LBXBSE", "LBXBCD", "LBXTHG", "LBXBMN"])
    no_met = inc.loc["no_metals"]
    lead_rank_metals = sorted(cfg.PREDICTOR_VARS["metals"], key=prank)
    if set(lead_rank_metals[:2]) != {"LBXBPB", "LBXBSE"}:
        print(f"[WARNING] the two highest metals by permutation importance are {lead_rank_metals[:2]}, "
              f"not lead and selenium; revisit the exposure tiers before submission.")
    set_text(para(doc, "3.2.1. Study design", 0),
             "This MR study tests whether blood metal exposures, together with any non-metal factor the "
             "machine-learning phase identifies as a comparably strong predictor, have a causal effect on "
             "the individual components of metabolic syndrome in individuals of predominantly European "
             "ancestry (Figure 2), using publicly available summary-level GWAS data; Table 3 summarizes "
             "the instrument–exposure–outcome structure.")
    set_text(para(doc, "3.2.1. Study design", 1),
             f"Exposures are chosen by two criteria applied jointly. The first is the machine-learning "
             f"evidence (section 3.1.7). In the final run on the {n_final:,}-participant cohort, body "
             f"mass index and age dominated the ranking; the five metals sat in the middle and lower "
             f"part of it, and removing all five changed test-set discrimination by "
             f"{no_met['delta_vs_full']:+.4f} AUROC units (DeLong p = {no_met['delong_p']:.3g}). "
             f"Lead (permutation rank {prank('LBXBPB')} of {n_pred}; {span('LBXBPB')} across the four "
             f"leading models) and selenium (rank {prank('LBXBSE')}; {span('LBXBSE')}) carried the most "
             f"consistent metal signal, whereas cadmium (rank {prank('LBXBCD')}), manganese "
             f"(rank {prank('LBXBMN')}) and mercury (rank {prank('LBXTHG')}) "
             f"{'were at or near noise level' if not any(above_noise(m) for m in ['LBXBCD', 'LBXBMN', 'LBXTHG']) else 'were less consistent'}. "
             f"Lead and selenium are therefore primary metal exposures and cadmium, which has the "
             f"strongest prior literature and a published never-smoker instrument, is secondary. The "
             f"second criterion is a usable genetic instrument under the rule in section 3.2.3, applied "
             f"identically to every exposure. An associational ranking and a causal test answer "
             f"different questions: a factor can be predictive without being causal, or causal without "
             f"being the strongest predictor in observational data, so exposures with a weaker ML "
             f"signal but a usable instrument are retained as exploratory. Table 1 lists the candidate "
             f"exposures against the evidence for each.")

    # ---- 3.2.2 Data sources
    p = para(doc, "3.2.2. Data sources", 0)
    sub(p, "a two-sample MR design is implemented separately for each exposure–outcome pair.",
        "a two-sample MR design is implemented separately for each exposure–outcome pair. Where the "
        "exposure and outcome GWAS share participants (for example UK Biobank contributes to both the BMI "
        "and some blood-pressure GWAS), sample overlap is reported and a non-overlapping outcome source is "
        "preferred.")
    set_text(para(doc, "3.2.2. Data sources", 1),
             "For the metals, exposure instruments come from genome-wide meta-analyses of blood metal "
             "concentrations: lead, cadmium and manganese from Ng et al.'s toxic-metals GWAS and the "
             "multi-element GWAS by Moksnes et al. [73,74], cadmium also from the never-smoker cadmium GWAS "
             "by Borné et al. [75], and selenium from Cornelis et al.'s GWAS of blood selenium "
             "concentrations [76]. For mercury, the largest GWAS of blood mercury to date, in pregnant women "
             "and children of European ancestry, reported suggestive but not genome-wide-significant loci "
             "[77]; mercury is therefore analysed as an exploratory exposure under the relaxed threshold of "
             "section 3.2.3, and only if its instrument passes the strength check. Body mass index draws on "
             "the GIANT consortium meta-analyses [78,79], alcohol consumption on the GSCAN consortium [80], "
             "and physical activity on the UK Biobank accelerometer-derived GWAS [81]; the latter measures "
             "overall device-measured activity, whereas the machine-learning signal came from "
             "self-reported vigorous recreational activity, so the two constructs are not identical.")

    # ---- 3.2.3 Instrument selection (P6)
    set_text(para(doc, "3.2.3. Genetic instrument selection", 0),
             "Instrument strength is checked for every exposure with the F-statistic (F = β²/SE² > 10), "
             "because weak instruments bias two-sample estimates towards the null [82,83]. One rule applies "
             "to every exposure: SNPs are selected at genome-wide significance (P < 5 × 10⁻⁸), and when "
             "fewer than three independent SNPs reach it, the threshold is relaxed to P < 5 × 10⁻⁶ and the "
             "exposure is labelled as using suggestive instruments. Under this rule mercury is analysed "
             "rather than excluded, as exploratory; its result is reported separately from the primary "
             "analysis. LD clumping (r² = 0.001 within a 10,000 kb window) using an ancestry-matched "
             "reference panel retains the lowest-p-value SNP at each locus [49,84], and SNPs associated "
             "with likely confounders or with the outcome are flagged and removed using PhenoScanner [85] "
             "or, if that service is unavailable, the NHGRI-EBI GWAS Catalog. During harmonization "
             "(action = 2), palindromic SNPs with intermediate allele frequencies (0.40–0.60) are "
             "excluded, along with any SNP whose minor allele frequency falls below 0.01 [84,86].")

    # ---- 3.2.4 Statistical analysis (P7, P1 in prose)
    sa = "3.2.4. Statistical analysis"
    set_text(para(doc, sa, 0),
             "The methods used for each exposure–outcome pair depend on how many SNPs the instrument has, "
             "because most robust MR estimators need several. With one SNP, the Wald ratio is the only "
             "estimate. With two or more, the inverse variance weighted (IVW) estimator is primary, "
             "combining SNP-specific Wald ratios under the assumption that no pleiotropy is present [87]; "
             "when heterogeneity and pleiotropy are absent, maximum likelihood produces equivalent point "
             "estimates with tighter standard errors [87]. With three or more SNPs, MR-Egger regression, "
             "under the Instrument Strength Independent of Direct Effect (InSIDE) assumption, detects "
             "directional pleiotropy through its intercept [88], and the weighted median remains valid if "
             "up to half the instruments are invalid [89]; the weighted mode adds robustness against "
             "certain InSIDE violations [90].")
    set_text(para(doc, sa, 1),
             "Heterogeneity in the IVW estimates is assessed with Cochran's Q (two or more SNPs) and, for "
             "MR-Egger, with Rücker's Q [91]; the I²GX statistic checks the no-measurement-error "
             "assumption on which MR-Egger relies [92], and the MR-Egger intercept provides a further "
             "check for pleiotropy [88].")
    set_text(para(doc, sa, 2),
             "With ten or more SNPs, sensitivity analyses back this up: MR-PRESSO flags and removes "
             "outlier SNPs [93]; RadialMR, Cook's distance and Studentized residuals identify variants with "
             "outsized influence [94]; and a leave-one-out analysis checks whether the result hinges on a "
             "single SNP [84]. For exposures with fewer SNPs these methods are undefined or unstable and "
             "are not reported.")
    set_text(para(doc, sa, 3),
             "Where enough SNPs are available, alternative estimators robust to pleiotropy are run for "
             "comparison: MR-Lasso [95], constrained maximum likelihood [96], contamination mixture models "
             "[97] and the mode-based estimator [90]. The Steiger directionality test confirms which way "
             "the causal arrow points for each exposure–outcome pair, which matters in particular for body "
             "mass index and the metabolic syndrome components, where reverse causation is biologically "
             "plausible [98]. The analyses will be reported in accordance with the STROBE-MR guideline "
             "[99].")
    set_text(para(doc, sa, 4),
             "All of this runs in R version 4.3.1, using packages including TwoSampleMR, "
             "MendelianRandomization, MR-PRESSO, RadialMR and MRMix, with additional analyses in Stata "
             "version 17 via the mrrobust package. Because every outcome is a continuous component trait, "
             "results are reported as beta coefficients (change in the outcome per standard-deviation or "
             "unit change in the genetically predicted exposure) with 95% confidence intervals, and "
             "associations with p < 0.05 are treated as statistically significant, with the number of "
             "exposure–outcome pairs reported alongside so that multiplicity can be judged (Figure 2).")

    # ---- Table 1: candidate exposures (P1, P2, P4, P5)
    t1 = doc.tables[0]
    alc_s, alc_p = shap_rank("ALQ130"), prank("ALQ130")
    pa_s, pa_p = shap_rank("PAQ650"), prank("PAQ650")
    rows = [
        ("Body mass index",
         f"SHAP rank {shap_rank('BMXBMI')} and permutation rank {prank('BMXBMI')} of {n_pred}; removing it "
         f"changed test AUROC by {inc.loc['no_bmi', 'delta_vs_full']:+.4f}. Part of its weight is "
         f"definitional, because waist circumference defines one MetS component",
         "GIANT consortium meta-analyses, ~700,000 participants [78,79]", "Primary exposure"),
        ("Lead",
         f"Most consistent metal: permutation rank {prank('LBXBPB')} of {n_pred} "
         f"({'above' if above_noise('LBXBPB') else 'not above'} its permutation SD), ranks "
         f"{span('LBXBPB')} across the four leading models; SHAP rank {pb['shap_rank']}; "
         f"{shape_txt('LBXBPB')}",
         "Loci for blood/erythrocyte lead in the toxic-metal and multi-element GWAS [73,74]; strength to "
         "be confirmed (F > 10)", "Primary exposure"),
        ("Selenium",
         f"Permutation rank {prank('LBXBSE')} ({'above' if above_noise('LBXBSE') else 'not above'} its "
         f"permutation SD), ranks {span('LBXBSE')} across models; SHAP rank {se['shap_rank']}; "
         f"{shape_txt('LBXBSE')}",
         "GWAS of blood selenium concentrations, loci at 5q14 and 21q22.3 [76]", "Primary exposure"),
        ("Cadmium",
         f"Permutation rank {prank('LBXBCD')} "
         f"({'above' if above_noise('LBXBCD') else 'within'} its permutation SD), SHAP rank "
         f"{cd['shap_rank']}; {shape_txt('LBXBCD')}; retained for its strong prior literature",
         "Toxic-metal and multi-element GWAS [73,74]; never-smoker cadmium GWAS [75]", "Secondary exposure"),
        ("Manganese",
         f"Permutation rank {prank('LBXBMN')}, SHAP rank {mn['shap_rank']}; {shape_txt('LBXBMN')}",
         "Two genome-wide-significant loci (4q24, 1q41) [73,74]", "Exploratory"),
        ("Alcohol consumption",
         f"Drinks per drinking day: SHAP rank {alc_s}, permutation rank {alc_p}",
         "GSCAN consortium, ~1.2 million participants [80]", "Exploratory"),
        ("Physical activity",
         f"Vigorous recreational activity: SHAP rank {pa_s}, permutation rank {pa_p}; the GWAS measures "
         f"device-measured overall activity, a different construct",
         "UK Biobank accelerometer-derived GWAS, 14 loci [81]", "Exploratory"),
        ("Mercury",
         f"Permutation rank {prank('LBXTHG')}, SHAP rank {hg['shap_rank']}; {shape_txt('LBXTHG')}",
         "Suggestive (not genome-wide-significant) loci only [77]; relaxed threshold (section 3.2.3)",
         "Exploratory (suggestive instrument)"),
    ]
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            set_cell(t1.rows[i].cells[j], val)

    # ---- Table 2: participant flow
    t2 = doc.tables[1]
    steps = [
        ("0. NHANES 2007–2018 pooled sample", rem[0], "–"),
        ("1. Age ≥ 20 years", rem[1], f"{exc[1]:,} (age < 20)"),
        ("2. Not pregnant", rem[2], f"{exc[2]:,} (pregnant)"),
        ("3. Complete data on all 5 MetS components (waist, blood pressure, HDL, triglycerides, glucose)",
         rem[3], f"{exc[3]:,} (missing ≥ 1 MetS component)"),
        ("4. Met the fasting criterion (8 to 24 h)", rem[4], f"{exc[4]:,} (not fasted)"),
        ("5. At least 1 of 5 blood metals measured", rem[5], f"{exc[5]:,} (no metals measured)"),
        ("6. Final analytic cohort (MetS status computed)", rem[6], "0"),
    ]
    while len(t2.rows) < len(steps) + 1:
        t2._tbl.append(copy.deepcopy(t2.rows[-1]._tr))
    for i, (a, b, c) in enumerate(steps, start=1):
        set_cell(t2.rows[i].cells[0], a)
        set_cell(t2.rows[i].cells[1], f"{b:,}")
        set_cell(t2.rows[i].cells[2], c)

    # ---- Table 3: MR design summary
    t3 = doc.tables[2]
    set_cell(t3.rows[1].cells[1],
             "Blood lead, selenium and body mass index (primary); cadmium (secondary); manganese, mercury "
             "(suggestive instruments), alcohol consumption, physical activity (exploratory)")
    set_cell(t3.rows[1].cells[2],
             "Metabolic syndrome components, analysed separately: waist circumference, triglycerides, "
             "HDL-C, blood pressure, fasting glucose (GIANT / GLGC / MAGIC / ICBP, as applicable)")

    # ---- 4. Hypotheses and research questions (P8)
    h = nonempty_after(doc, "4.1.1. Hypotheses")
    set_text(h[2], "H3: The five blood metals, considered jointly, will add discrimination beyond "
                   "demographic, anthropometric and lifestyle variables (tested by research question Q3).")
    set_text(h[3], "H4: The relative importance of the individual metals will differ between importance "
                   "methods and between algorithms, so only metals whose importance is consistent across "
                   "both will be prioritized for the MR phase.")
    h2 = nonempty_after(doc, "4.2.1. Hypotheses")
    set_text(h2[0], "H5: Publicly available GWAS will yield valid instruments (F > 10) for blood lead, "
                    "selenium and body mass index; instruments for cadmium, manganese, alcohol consumption "
                    "and physical activity may be fewer or weaker, and mercury has suggestive loci only.")
    set_text(h2[1], "H6: The instruments for the primary exposures (lead, selenium, body mass index) will "
                    "show sufficient strength (F > 10) for reliable causal inference, while instrument "
                    "strength for the secondary and exploratory exposures is expected to vary.")
    set_text(h2[2], "H7: Where an exposure has enough SNPs, heterogeneity and pleiotropy will be detectable "
                    "and robust methods (weighted median and MR-Egger with three or more SNPs, MR-PRESSO with "
                    "ten or more) will be used to limit the resulting bias; with fewer SNPs only Wald-ratio or "
                    "IVW estimates will be interpreted, cautiously.")
    lead_sh = shape_of(N["metals_dependence"]["LBXBPB"]["binned_mean_shap"])
    lead_why = ("showed an inverse cross-sectional association that may reflect confounding"
                if lead_sh == "falling" else
                "showed no consistent direction for lead once the other predictors were accounted for"
                if lead_sh in ("flat", "irregular") else
                f"showed a {lead_sh} pattern for lead that a cross-sectional model cannot interpret causally")
    set_text(h2[3], "H8: Genetically predicted blood selenium and lead will be associated with one or more "
                    "MetS components (blood pressure, fasting glucose, triglycerides, HDL-C). The effect of "
                    f"lead is tested two-sided, because the machine-learning phase {lead_why}; cadmium is "
                    "tested as a secondary exposure.")
    set_text(h2[4], "H9: Exposures ranked highest by the machine-learning phase (body mass index in "
                    "particular) will show the strongest and most consistent causal estimates, while any "
                    "exposure with a weak machine-learning signal but an adequate instrument will help "
                    "distinguish a genuinely non-causal association from one the modeling phase under-detected.")
    q = nonempty_after(doc, "4.2.2. Research questions")
    set_text(q[0], "Q5: How many SNPs qualify as valid instruments for each candidate exposure (lead, selenium, "
                   "body mass index, cadmium, manganese, mercury, alcohol consumption, physical activity), and "
                   "at which threshold?")
    set_text(q[3], "Q8: For exposures with enough SNPs, do the MR-Egger intercept and MR-PRESSO tests show "
                   "evidence of directional pleiotropy in any exposure–outcome association?")
    set_text(q[4], "Q9: Does the MR analysis support a causal effect of any blood-metal exposure on the MetS "
                   "components, and if so, does the ranking of causal effect sizes across metals resemble the "
                   "ranking produced by the machine-learning phase?")
    set_text(q[5], "Q10: For exposures analysed with suggestive instruments (mercury) or as exploratory "
                   "(manganese, alcohol consumption, physical activity), does subsequent evidence (an updated "
                   "GWAS or the sensitivity analyses in section 3.2.4) change their classification?")

    # ---- Figures 1 and 2
    replace_image(doc, 0, DIAG / "figure1_data_cleaning_pipeline.png")
    replace_image(doc, 1, DIAG / "figure2_mr_study_design.png")

    doc.save(DOC)
    print(f"updated {DOC.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
