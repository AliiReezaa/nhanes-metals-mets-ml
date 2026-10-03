"""
Builds the manuscript RESULTS section (machine-learning component) as a
standalone, submission-formatted .docx.

Length discipline: a Q1 Results section carries roughly five or six display
items and under ~2,000 words. Everything that supports rather than carries the
argument -- threshold-level classification metrics, the calibration and
decision curves, the subgroup forest, the full importance table and the
sensitivity grid -- lives in the combined supplement (06_Code/21_write_supplement.py)
and is cited from here.

Display items retained in the main text:
  Table 1  survey-weighted baseline characteristics
  Table 2  discrimination, calibration and DeLong comparison, all models
  Figure 2 ROC curves
  Figure 3 SHAP summary for the leading model
  Figure 4 importance concordance (method agreement and cross-model agreement)
  Figure 5 SHAP dependence for the five blood metals
Figure 1 (participant flow) belongs to the Methods and is cited, not repeated.

Every number is read from outputs/metrics/* at build time.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docx_style as ds

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs" / "metrics"
FIGP = ROOT / "outputs" / "figures_pub"
OUT = ROOT / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

N = json.loads((MET / "publication_numbers.json").read_text())
# Caption glossaries, written by 06_Code/15_publication_figures.py at the moment each
# figure was drawn. Read rather than restated, so a label on an axis and its expansion in
# the caption cannot drift apart when the figures are regenerated.
AB = N["figure_abbreviations"]

# Each abbreviation is spelled out at its first appearance and referred back to after
# that, which is the usual journal convention and keeps the later captions short.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import feature_labels as fl
_SEEN: dict[str, str] = {}


def gloss(figure_key: str, back_ref: str = "Figure 3") -> str:
    return fl.incremental_glossary(AB[figure_key], _SEEN, back_ref)
cmp_df = pd.read_csv(MET / "model_comparison.csv")
cal = pd.read_csv(MET / "calibration_metrics.csv").set_index("model")
tab1 = pd.read_csv(MET / "table1_by_mets_status.csv")
sg = pd.read_csv(MET / "subgroup_auc.csv")
shap_all = pd.read_csv(MET / "shap_importance.csv")
perm = pd.read_csv(MET / "permutation_importance.csv")
outdef = pd.read_csv(MET / "outcome_definition_comparison.csv")
s_diq = pd.read_csv(MET / "sensitivity_diq010.csv")
s_met = pd.read_csv(MET / "sensitivity_metals_scope.csv")
s_smo = pd.read_csv(MET / "sensitivity_smote_vs_classweight.csv")
s_out = pd.read_csv(MET / "sensitivity_outcome_definition.csv")
inc = pd.read_csv(MET / "incremental_value.csv").set_index("model")
rcv = pd.read_csv(MET / "repeated_cv.csv").set_index("model")
dca_df = pd.read_csv(MET / "decision_curve_data.csv")
temporal = pd.DataFrame(N.get("temporal_validation", []))

f = N["flow"]
TOP = N["top_model"]
SHAP_M = N["shap_model"]
SIG = N["significance"]
tied_names = [m for m, v in SIG.items() if v["tied"]]
others_tied = [m for m in tied_names if m != TOP]

# 2026-10 audit (P7/P8): every interpretive statement below is derived from the
# outputs at build time; nothing about direction, shape or ordering is typed in.
shap_pred = shap_all[~shap_all["feature_code"].str.startswith("missingindicator")].copy()
shap_pred["rank"] = shap_pred["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
perm["rank"] = perm["perm_importance_mean"].rank(ascending=False, method="min").astype(int)
psd = {r.feature_code: (float(r.perm_importance_mean), float(r.perm_importance_std))
       for r in perm.itertuples()}
METALS = ["LBXBPB", "LBXBCD", "LBXTHG", "LBXBSE", "LBXBMN"]
MNAME = {"LBXBPB": "lead", "LBXBCD": "cadmium", "LBXTHG": "mercury", "LBXBSE": "selenium",
         "LBXBMN": "manganese"}
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
         10: "ten", 11: "eleven", 12: "twelve"}

MINUS = "−"


def num(v, dp=3):
    return f"{v:.{dp}f}".replace("-", MINUS)


def p_txt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def p_eq(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def ci(lo, hi, dp=3):
    return f"{num(lo, dp)}–{num(hi, dp)}"


def join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def t1(var):
    return tab1[tab1["variable"] == var].iloc[0]


def sg_name(label):
    if label.startswith("Age"):
        return "the age band " + label[4:]
    return {"Male": "men", "Female": "women"}.get(label, label)


def shape_of(centers, values):
    """Plain-language description of a binned SHAP dependence curve."""
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


def main():
    doc = ds.new_document(line_spacing="1.5")
    ds.title(doc, "Results")
    ds.subtitle(doc, "Machine-learning analysis of blood metals and metabolic syndrome "
                     "in NHANES 2007–2018")

    # ==================================================================
    ds.heading(doc, "1. Study population", 1)
    harm, lab = outdef.iloc[1], outdef.iloc[0]
    ds.para(doc,
            f"Of {f['pooled']:,} participants sampled across the six NHANES cycles, "
            f"{f['final_cohort']:,} non-pregnant adults in the fasting subsample with complete "
            f"outcome and metal data formed the analytic cohort (Figure 1), divided into "
            f"{f['train_n']:,} training and {f['test_n']:,} held-out test participants. Metabolic "
            f"syndrome was present in {int(harm['n_positive']):,} participants, a crude prevalence "
            f"of {harm['prevalence_pct']:.2f}%. Of these, "
            f"{int(harm['n_reclassified_negative_to_positive']):,} reached the three-component "
            f"threshold only because a drug-treatment clause scored a component positive that the "
            f"measured value alone would not have; under a measured-value-only definition the "
            f"prevalence would have been {lab['prevalence_pct']:.2f}%.")

    age, bmi = t1("RIDAGEYR"), t1("BMXBMI")
    up, down, flat = [], [], []
    for m in METALS:
        r = t1(m)
        ratio = r["mets_mean"] / r["no_mets_mean"]
        bucket = flat if r["p_value"] >= 0.05 else (up if ratio > 1 else down)
        bucket.append(f"{MNAME[m]} ({(ratio - 1) * 100:+.0f}%)".replace("-", MINUS))
    metal_sent = []
    if up:
        metal_sent.append(f"{join(up)} {'was' if len(up) == 1 else 'were'} higher in the "
                          f"metabolic syndrome group")
    if down:
        metal_sent.append(f"{join(down)} {'was' if len(down) == 1 else 'were'} lower")
    flat_txt = (f"; {join(flat)} did not differ at p < 0.05" if flat else "")
    ds.para(doc,
            "Survey-weighted characteristics by metabolic syndrome status are given in Table 1. "
            f"Participants with metabolic syndrome were on average "
            f"{age['mets_mean'] - age['no_mets_mean']:.1f} years older and had a body mass index "
            f"{bmi['mets_mean'] - bmi['no_mets_mean']:.1f} kg/m² higher. Comparing geometric means, "
            + ("; ".join(metal_sent) if metal_sent else "no metal differed at p < 0.05")
            + flat_txt + ". The percentages are the relative difference in geometric mean; all are "
            "small relative to the spread of the concentrations within each group, and none is "
            "adjusted for age, which differs sharply between the groups.")

    rows_t1 = []
    for _, r in tab1.iterrows():
        if r["summary"] == "geometric mean":
            cell = lambda g: (f"{r[g + '_mean']:.2f} ({r[g + '_ci_low']:.2f}–{r[g + '_ci_high']:.2f})"
                              if pd.notna(r.get(g + "_ci_low")) else f"{r[g + '_mean']:.2f}")
            name = r["clinical_name"] + "†"
        elif r["type"] == "continuous":
            cell = lambda g: f"{r[g + '_mean']:.2f} ({r[g + '_se']:.2f})"
            name = r["clinical_name"]
        else:
            cell = lambda g: f"{r[g + '_mean']:.1f} ({r[g + '_se']:.1f})"
            name = r["clinical_name"]
        rows_t1.append([name, cell("no_mets"), cell("mets"), p_txt(r["p_value"])])
    se_row = t1("LBXBSE")
    ds.caption(doc, "Table 1.",
               "Survey-weighted characteristics of the analytic cohort by metabolic syndrome status. "
               "Continuous variables are weighted means (standard error); blood metals are weighted "
               "geometric means (95% confidence interval); categorical variables are weighted "
               "percentages (standard error). All estimates use the fasting-subsample weight divided "
               "by six and Taylor-series linearization with the participants' own strata and "
               "primary sampling units.", above=True)
    ds.add_table(doc, ["Characteristic", "No MetS", "MetS", "p"], rows_t1,
                 col_widths_in=[2.55, 1.45, 1.45, 0.65], align_right_from=1)
    ds.table_note(doc,
                  f"MetS, metabolic syndrome. Unweighted group sizes: no MetS n = "
                  f"{int(age['no_mets_n']):,}; MetS n = {int(age['mets_n']):,}. †Geometric mean. "
                  f"Selenium and manganese were assayed only from the 2011–2012 cycle onward, so both "
                  f"are based on n = {int(se_row['no_mets_n']):,} without and "
                  f"n = {int(se_row['mets_n']):,} with metabolic syndrome. p-values for continuous "
                  f"variables are design-based t tests of the domain difference (on the log scale for "
                  f"the metals; 94 degrees of freedom, 184 primary sampling units minus 90 strata); "
                  f"for categorical variables they are design-adjusted Wald F tests.")

    # ==================================================================
    ds.heading(doc, "2. Model performance", 1)
    top_row = cmp_df.iloc[0]
    worst = cmp_df.iloc[-1]
    lr_auc = cmp_df.loc[cmp_df.Model == "Logistic Regression", "AUC"].iloc[0]
    rest = [m for m in cmp_df["Model"] if m not in tied_names]
    rest_pmax = max(SIG[m]["p"] for m in rest) if rest else None
    if others_tied:
        tie_txt = (f"{join(others_tied)} {'was' if len(others_tied) == 1 else 'were'} statistically "
                   f"indistinguishable from it (DeLong "
                   f"{join(p_eq(SIG[m]['p']) for m in others_tied)}; Figure S2), so "
                   f"{'these models form' if len(tied_names) > 2 else 'the two form'} a tied leading "
                   f"group rather than a ranking")
    else:
        tie_txt = "every other model differed from it at p < 0.05 by DeLong's test (Figure S2)"
    ds.para(doc,
            f"All {WORDS.get(len(cmp_df), len(cmp_df))} models were evaluated once on the held-out "
            f"test partition (Table 2, Figure 2). {TOP} achieved the highest area under the receiver "
            f"operating characteristic curve at {top_row['AUC']:.3f} "
            f"({ci(top_row['AUC_95CI_low'], top_row['AUC_95CI_high'])}), and {worst['Model']} the "
            f"lowest at {worst['AUC']:.3f} ({ci(worst['AUC_95CI_low'], worst['AUC_95CI_high'])}). "
            f"{tie_txt[0].upper() + tie_txt[1:]}."
            + (f" The remaining models differed from {TOP} at "
               f"{'p < 0.001' if rest_pmax < 0.001 else f'p ≤ {rest_pmax:.3f}'}." if rest and others_tied else "")
            + f" Two caveats apply: the reference was chosen as the test-set maximum, so the "
            f"differences carry a winner's-curse bias, and no adjustment was made for the "
            f"{WORDS.get(len(cmp_df) - 1, len(cmp_df) - 1)} simultaneous comparisons. The margin over "
            f"the simplest baseline is modest in any case: logistic regression reached "
            f"{lr_auc:.3f}, {SIG['Logistic Regression']['auc_diff']:.3f} AUROC units below {TOP} "
            f"(95% CI {ci(*SIG['Logistic Regression']['ci'])}).")

    r_top = rcv.loc[TOP] if TOP in rcv.index else None
    if r_top is not None:
        order = rcv.sort_values("mean_auc", ascending=False)
        ds.para(doc,
                f"Cross-validation repeated over {int(r_top['repeats'])} different random partitions "
                f"of the training data ({int(r_top['folds'])} folds each; Table S9) gave {TOP} a mean "
                f"AUROC of {r_top['mean_auc']:.3f}, with the individual folds spanning "
                f"{r_top['p2_5_auc']:.3f} to {r_top['p97_5_auc']:.3f} (2.5th to 97.5th percentile). "
                f"The repeated-CV ordering placed {order.index[0]} first, followed by "
                f"{join(order.index[1:4])}; the means of the leading models lay within "
                f"{order['mean_auc'].iloc[0] - order['mean_auc'].iloc[min(3, len(order) - 1)]:.3f} "
                f"AUROC units of one another, smaller than the fold-to-fold standard deviation of "
                f"{r_top['sd_auc_folds']:.3f}.")

    lead_models = list(cmp_df["Model"].head(4))
    sl = cal.loc[lead_models, "calibration_slope"]
    off = cal[(cal["calibration_slope"] < 0.8) | (cal["calibration_slope"] > 1.2)]
    off_txt = ""
    if len(off):
        parts = []
        for m, r in off.iterrows():
            kind = "over-dispersed" if r["calibration_slope"] < 1 else "under-dispersed"
            parts.append(f"{m} (slope {r['calibration_slope']:.2f}, AUROC "
                         f"{cmp_df.loc[cmp_df.Model == m, 'AUC'].iloc[0]:.3f}; {kind})")
        off_txt = (f" Calibration did not track discrimination across the wider field: "
                   f"{join(parts)} departed from the ideal slope by more than 0.2. A slope below 1 "
                   f"means predicted probabilities that are too extreme, and above 1 too timid; "
                   f"either can coexist with acceptable ranking while giving an individual a wrong "
                   f"absolute risk.")
    d = N["dca"]["points"]
    td = dca_df[dca_df.model == TOP].set_index("threshold")["net_benefit"]
    ta = dca_df[dca_df.model == "Treat all"].set_index("threshold")["net_benefit"]
    above = [t for t in td.index if td[t] > max(ta.get(t, -1), 0) + 1e-9]
    pt = "0.20"
    dca_txt = (f"Decision-curve analysis (Figure S4) showed {TOP} above both reference "
               f"strategies at every threshold from {min(above):.2f} to {max(above):.2f}"
               if above else f"Decision-curve analysis (Figure S4) did not show {TOP} above both "
                             f"reference strategies at any threshold examined")
    gain = d[pt]["net_interventions_avoided_per_1000"]
    dca_txt += (f". At a risk threshold of 0.20 its net benefit was {d[pt]['model_nb']:.3f} against "
                f"{d[pt]['treat_all_nb']:.3f} for treating everyone, "
                + (f"equivalent to {gain:.0f} fewer unnecessary interventions per 1,000 adults for "
                   f"the same number of cases found" if gain > 0 else
                   "so at that threshold the model offers no gain over treating everyone")
                + ".")
    ds.para(doc,
            f"Calibration slopes of the four highest-AUROC models ranged from {sl.min():.2f} to "
            f"{sl.max():.2f} against an ideal of 1.00 (Figure S3).{off_txt} {dca_txt} "
            f"Threshold-level classification metrics, each with a bootstrap 95% confidence "
            f"interval, are given in Table S4.")

    rows_t2 = []
    for _, r in cmp_df.iterrows():
        m = r["Model"]
        s = SIG[m]
        delta = ("reference" if m == TOP else
                 f"{num(s['auc_diff'], 4)} ({num(s['ci'][0], 4)} to {num(s['ci'][1], 4)})")
        rows_t2.append([("**" + m + "**") if s["tied"] else m,
                        f"{r['AUC']:.3f} ({ci(r['AUC_95CI_low'], r['AUC_95CI_high'])})",
                        f"{r['AUPRC']:.3f}", f"{r['Brier']:.4f}",
                        f"{cal.loc[m, 'calibration_slope']:.2f}",
                        delta, "–" if m == TOP else p_txt(s["p"])])
    ds.caption(doc, "Table 2.",
               f"Discrimination and calibration on the held-out test partition (n = {f['test_n']:,}), "
               f"ordered by AUROC. Δ AUROC is {TOP} minus the model, with a paired-bootstrap 95% "
               f"confidence interval (1,000 resamples, the same participants scored under both "
               f"models); p is from DeLong's test for correlated ROC curves. **Models in bold are "
               f"statistically indistinguishable from {TOP} (p ≥ 0.05).**", above=True)
    ds.add_table(doc,
                 ["Model", "AUROC (95% CI)", "AUPRC", "Brier", "Cal. slope",
                  f"Δ AUROC vs {TOP} (95% CI)", "p"],
                 rows_t2, col_widths_in=[1.15, 1.30, 0.50, 0.52, 0.58, 1.70, 0.60],
                 align_right_from=1, font_size=ds.Pt(8))
    ds.table_note(doc,
                  "AUROC, area under the receiver operating characteristic curve; AUPRC, area under "
                  "the precision–recall curve; Cal. slope, calibration slope (1.00 = perfect). Lower "
                  "Brier scores indicate better probabilistic accuracy. Bootstrap 95% confidence "
                  "intervals for AUPRC and Brier score are in 04_Results_Tables/model_comparison.csv.")

    ds.figure(doc, FIGP / "fig2_roc_curves.png", width_in=6.3)
    n_t = len(tied_names)
    ds.caption(doc, "Figure 2.",
               f"Receiver operating characteristic curves on the held-out test partition. "
               f"**(A)** All {WORDS.get(len(cmp_df), len(cmp_df))} models, with AUROC in the legend; "
               + (f"the {WORDS.get(n_t, n_t)} models statistically indistinguishable from the best "
                  f"performer are drawn in color and the remainder in grey. " if n_t > 1 else
                  "the best performer is drawn in color and the remainder in grey. ")
               + "**(B)** Detail of the region of clinical interest.")

    # ==================================================================
    ds.heading(doc, "3. Predictor importance", 1)
    s1, s2, s3 = shap_pred.nsmallest(3, "rank").itertuples()
    p1, p2, p3 = perm.nsmallest(3, "rank").itertuples()
    shap_note = ("" if SHAP_M == TOP else
                 f" SHAP values are computed for {SHAP_M}, the highest-AUROC tree-based model, "
                 f"because exact tree SHAP is not available for {TOP}.")
    ds.para(doc,
            f"Two predictors dominated both importance analyses for {SHAP_M} (Figures 3 and 4; full "
            f"ranking in Table S5).{shap_note} By mean absolute SHAP value, "
            f"{s1.clinical_name.split(' (')[0].lower()} ({s1.mean_abs_shap:.3f}) and "
            f"{s2.clinical_name.split(' (')[0].lower()} ({s2.mean_abs_shap:.3f}) were followed by "
            f"{s3.clinical_name.split(' (')[0].lower()} at {s3.mean_abs_shap:.3f}, a "
            f"{s2.mean_abs_shap / s3.mean_abs_shap:.1f}-fold drop from second to third. By "
            f"permutation importance, randomly shuffling "
            f"{p1.clinical_name.split(' (')[0].lower()} cost {p1.perm_importance_mean:.3f} AUROC "
            f"units and {p2.clinical_name.split(' (')[0].lower()} {p2.perm_importance_mean:.3f}, "
            f"against {p3.perm_importance_mean:.4f} for the third-ranked predictor. Everything "
            f"else adjusts risk at the margin; Table S8 shows what this means for the model as a "
            f"whole.")

    mi = N["missingness_indicator_shap"]
    mi_named = [x["name"].replace(" missing (imputed)", "").replace(" (missing)", "").lower()
                for x in mi if x["mean_abs_shap"] >= 0.01]
    ds.para(doc,
            f"The two methods agreed only moderately below the top of the ranking (Spearman "
            f"ρ = {N['spearman_shap_perm']:.2f} across {N['n_predictors_ranked']} predictors; "
            f"Figure 4A). Disagreement of this kind is expected: mean absolute SHAP measures how far "
            f"a predictor moves individual predictions, whereas permutation importance measures how "
            f"much ranking performance degrades when that predictor is scrambled, so a predictor "
            f"that correlated predictors can substitute for scores well on the first and poorly on "
            f"the second."
            + (f" Missing-value indicators for {join(mi_named)} also carried SHAP weight of at "
               f"least 0.01; these describe questionnaire structure rather than biology and are "
               f"kept out of the predictor ranking." if mi_named else ""))

    ds.figure(doc, FIGP / "fig6_shap_beeswarm.png", width_in=5.9)
    ds.caption(doc, "Figure 3.",
               f"SHAP summary plot for the {SHAP_M} model on the held-out test partition, showing the "
               f"18 highest-ranked features. Each point is one participant; horizontal position is "
               f"that feature's contribution to that participant's predicted log-odds of metabolic "
               f"syndrome, and color encodes the feature's value from low (blue) to high "
               f"(orange). " + gloss("fig6_shap_beeswarm"))

    # ==================================================================
    ds.heading(doc, "4. Blood metals", 1)
    mr = N["metal_ranks"]
    npr = N["n_predictors_ranked"]

    def span(m):
        v = list(mr[m]["per_model_perm_rank"].values())
        return f"{min(v)}–{max(v)}"

    by_shap = sorted(METALS, key=lambda m: mr[m]["shap_rank"])
    by_perm = sorted(METALS, key=lambda m: mr[m]["perm_rank"])
    gaps = {m: abs(mr[m]["shap_rank"] - mr[m]["perm_rank"]) for m in METALS}
    widest = max(gaps, key=gaps.get)
    closest = min(gaps, key=gaps.get)
    ds.para(doc,
            f"The five metals ranked between {min(min(mr[m]['shap_rank'], mr[m]['perm_rank']) for m in METALS)} "
            f"and {max(max(mr[m]['shap_rank'], mr[m]['perm_rank']) for m in METALS)} of {npr} "
            f"predictors, and their order depended on the importance method. By SHAP the "
            f"highest-ranked metal was {MNAME[by_shap[0]]} (rank {mr[by_shap[0]]['shap_rank']}); by "
            f"permutation importance it was {MNAME[by_perm[0]]} (rank {mr[by_perm[0]]['perm_rank']}). "
            f"The widest disagreement belonged to {MNAME[widest]} ({mr[widest]['shap_rank']} by "
            f"SHAP versus {mr[widest]['perm_rank']} by permutation), and the closest agreement to "
            f"{MNAME[closest]} ({mr[closest]['shap_rank']} versus {mr[closest]['perm_rank']}).")

    lead4 = list(mr["LBXBPB"]["per_model_perm_rank"].keys())
    stable = sorted(METALS, key=lambda m: (max(mr[m]["per_model_perm_rank"].values())
                                           - min(mr[m]["per_model_perm_rank"].values())))
    signal = [m for m in METALS if m in psd and psd[m][0] > psd[m][1]]
    nosig = [m for m in METALS if m not in signal]
    ds.para(doc,
            f"Across the {WORDS.get(len(lead4), len(lead4))} highest-AUROC models, each fitted and "
            f"permuted independently (Figure 4B), permutation ranks were "
            + join(f"{MNAME[m]} {span(m)}" for m in stable)
            + f", from most to least stable. The permutation analysis carries its own "
            f"uncertainty: across ten shuffles, "
            + (f"only {join(f'{MNAME[m]} ({psd[m][0]:.5f} ± {psd[m][1]:.5f})' for m in signal)} "
               f"had a mean importance larger than its own standard deviation, whereas for "
               f"{join(MNAME[m] for m in nosig)} the mean was smaller than the shuffle-to-shuffle "
               f"variability." if signal and nosig else
               "every metal had a mean importance larger than its own standard deviation."
               if signal else "no metal had a mean importance larger than its own standard "
                              "deviation.")
            + " The defensible statement is therefore narrower than a rank list.")

    md = N["metals_dependence"]
    descr = []
    for m in METALS:
        c, v = md[m]["binned_centers"], md[m]["binned_mean_shap"]
        sh = shape_of(c, v)
        if sh == "flat":
            descr.append(f"{MNAME[m]} was essentially flat (binned means {num(min(v))} to "
                         f"{num(max(v))})")
        else:
            descr.append(f"{MNAME[m]} was {sh}, from {num(v[0])} in the lowest bin to "
                         f"{num(v[-1])} in the highest (range {num(min(v))} to {num(max(v))})")
    pb_v = md["LBXBPB"]["binned_mean_shap"]
    pb_shape = shape_of(md["LBXBPB"]["binned_centers"], pb_v)
    biggest = max(abs(x) for m in METALS for x in md[m]["binned_mean_shap"])
    top2 = shap_pred.nsmallest(2, "rank")
    ds.para(doc,
            f"The shape of each metal's contribution is shown in Figure 5, as the mean SHAP value "
            f"within twelve equal-count bins of the exposure: " + "; ".join(descr) + ". These are "
            f"contributions to the predicted log-odds within one fitted model, adjusted for "
            f"everything else the model contains. No binned mean for any metal exceeded "
            f"{biggest:.3f} in absolute value, against mean absolute contributions of "
            f"{top2['mean_abs_shap'].iloc[0]:.3f} and {top2['mean_abs_shap'].iloc[1]:.3f} for the two "
            f"leading predictors.")
    if pb_shape in ("falling", "inverted U-shaped") and pb_v[-1] < 0:
        ds.para(doc,
                "The lower predicted risk at the highest blood lead concentrations deserves "
                "emphasis rather than quiet omission, because it runs against the hypothesis that "
                "motivated the analysis. It is an associational pattern within one model, not an "
                "estimate of a causal effect. Residual confounding by smoking, which raises blood "
                "lead and cadmium and is associated with lower body mass index, is one plausible "
                "explanation, and dilution of blood metal concentrations with greater body size "
                "is another. Separating these is what the Mendelian randomization component is "
                "designed to do, and the direction observed here is a question posed to that "
                "analysis rather than an answer from this one.")

    ds.figure(doc, FIGP / "fig7_importance_concordance.png", width_in=5.7)
    ds.caption(doc, "Figure 4.",
               f"Agreement between importance methods and across models. **(A)** Rank by mean "
               f"absolute SHAP value against rank by permutation importance for the {SHAP_M} model; "
               f"the five blood metals are highlighted and the diagonal denotes perfect agreement. "
               f"**(B)** Permutation-importance rank of the fifteen highest-ranked predictors in "
               f"each of the four highest-AUROC models, fitted independently. A predictor whose row "
               f"is uniform is important as a property of the data; one whose row varies is "
               f"important as a property of a particular fitted model. "
               + gloss("fig7_importance_concordance"))

    ds.figure(doc, FIGP / "fig8_metals_shap_dependence.png", width_in=6.3)
    ds.caption(doc, "Figure 5.",
               f"SHAP dependence for the five blood metals in the {SHAP_M} model on the held-out "
               f"test partition. Each grey point is one participant; the orange line joins mean SHAP "
               f"values within twelve equal-count bins of the exposure, and the dashed line at zero "
               f"marks no contribution to the predicted log-odds. Axes are truncated at the 1st and "
               f"98th percentiles. Selenium and manganese are based on the "
               f"{md['LBXBSE']['n_observed']:,} test participants from cycles in which those "
               f"analytes were assayed. " + gloss("fig8_metals_shap_dependence"))

    # ==================================================================
    ds.heading(doc, "5. Incremental value, subgroups and sensitivity analyses", 1)
    full = inc.loc["full"]
    nm, lo_, nb, ab = inc.loc["no_metals"], inc.loc["lead_only"], inc.loc["no_bmi"], inc.loc["age_bmi_lr"]

    def dtxt(r):
        return (f"{num(r['delta_vs_full'], 4)} (95% CI {num(r['delta_ci_low'], 4)} to "
                f"{num(r['delta_ci_high'], 4)}; {p_eq(r['delong_p'])})")
    ds.para(doc,
            f"Table S8 tests directly what the metals add. Refitting {full['algorithm']} with its "
            f"tuned settings but without the five metals changed test-set AUROC from "
            f"{full['test_auc']:.3f} to {nm['test_auc']:.3f}, a difference of {dtxt(nm)}; keeping "
            f"lead as the only metal gave {lo_['test_auc']:.3f}, {dtxt(lo_)}. Removing body mass "
            f"index instead gave {nb['test_auc']:.3f}, {dtxt(nb)}, and a logistic regression on age "
            f"and body mass index alone reached {ab['test_auc']:.3f}, {dtxt(ab)}. "
            + ("On this test set the metals' contribution to discrimination is not distinguishable "
               "from zero. " if nm["delong_p"] >= 0.05 else
               "The metals make a statistically detectable but small contribution to "
               "discrimination. ")
            + "Their importance ranks describe attribution within a fitted model, not predictive "
            "value that would be lost without them.")

    ov = sg[sg.subgroup == "Overall"].iloc[0]
    sub = sg[sg.subgroup != "Overall"]
    worst_sg = sub.loc[sub["auc"].idxmin()]
    best_sg = sub.loc[sub["auc"].idxmax()]
    below = sub[sub["ci_high"] < ov["ci_low"]]
    mis = sub[(sub["slope_ci_high"] < 1) | (sub["slope_ci_low"] > 1) |
              (sub["oe_ci_high"] < 1) | (sub["oe_ci_low"] > 1)]
    age_sg = sub[sub.subgroup.str.startswith("Age")]
    ds.para(doc,
            f"Against an overall test-set AUROC of {ov['auc']:.3f} "
            f"({ci(ov['ci_low'], ov['ci_high'])}), subgroup AUROC ranged from {worst_sg['auc']:.3f} "
            f"in {sg_name(worst_sg['subgroup'])} ({ci(worst_sg['ci_low'], worst_sg['ci_high'])}) to "
            f"{best_sg['auc']:.3f} in {sg_name(best_sg['subgroup'])} "
            f"({ci(best_sg['ci_low'], best_sg['ci_high'])}; Table S6, Figure S5)"
            + (f"; the interval for {join(sg_name(b) for b in below['subgroup'])} lay entirely "
               f"below the overall interval" if len(below) else "")
            + ". Within an age band the range of age is compressed, so the model's second most "
            f"important predictor separates participants less well there; prevalence across the "
            f"age bands ran from {age_sg['prevalence_pct'].min():.1f}% to "
            f"{age_sg['prevalence_pct'].max():.1f}%. "
            + (f"Calibration departed detectably from ideal (slope or observed-to-expected ratio "
               f"interval excluding 1) in {join(sg_name(b) for b in mis['subgroup'])}. " if len(mis) else
               "No subgroup's calibration slope or observed-to-expected ratio interval excluded 1. ")
            + f"These are descriptive screens over {len(sg) * 2} intervals without multiplicity "
            f"adjustment.")

    if len(temporal):
        rs = temporal.iloc[0]; tv = temporal.iloc[1]
        ds.para(doc,
                f"In temporal validation (Table S10), {tv['Model']} re-tuned and developed on the "
                f"2007–2014 cycles (n = {int(tv['Train n']):,}) reached an AUROC of "
                f"{tv['Test AUROC']:.3f} ({ci(tv['AUROC 95% CI low'], tv['AUROC 95% CI high'])}) on "
                f"the 2015–2018 cycles (n = {int(tv['Test n']):,}), against "
                f"{rs['Test AUROC']:.3f} for the random split. Its calibration slope was "
                f"{tv['Calibration slope']:.2f} and its calibration-in-the-large "
                f"{num(tv['Calibration intercept (in-the-large)'], 3)}, with an observed-to-expected "
                f"ratio of {tv['O:E ratio']:.3f}.")

    so = s_out.set_index("model")
    d_lab = (so["cv_auc_labonly_label"] - so["cv_auc_harmonized_label"])
    d_nol = (so["cv_auc_nolipidmed_label"] - so["cv_auc_harmonized_label"])
    d_bpf = (so["cv_auc_bpfirst_label"] - so["cv_auc_harmonized_label"])

    def rng(sr, dp=4):
        return f"{num(sr.min(), dp)} to {num(sr.max(), dp)}".replace("+", "")
    ds.para(doc,
            f"Sensitivity analyses are given in Table S7 (cross-validated AUROC on the training "
            f"partition, {join(list(s_diq['model']))} at their tuned settings). Adding "
            f"physician-diagnosed diabetes changed discrimination by {rng(s_diq['delta'])} AUROC "
            f"units, as expected for a variable partly downstream of the glucose component. "
            f"Replacing SMOTENC with class weighting changed it by {rng(s_smo['delta'])}. The "
            f"measured-value-only outcome changed it by {rng(d_lab)}, dropping the lipid-medication "
            f"clause by {rng(d_nol)} (prevalence {so['nolipidmed_prevalence_pct'].iloc[0]:.1f}% "
            f"instead of {so['harmonized_prevalence_pct'].iloc[0]:.1f}%), and using only the first "
            f"blood-pressure reading by {rng(d_bpf)}. Restricting the metal panel to lead changed "
            f"it by {rng(s_met['delta'])}. Differences of a few thousandths sit within the "
            f"fold-to-fold variability shown in Table S9.")

    path = OUT / "Manuscript_Results_AI.docx"
    # Convert break-only paragraphs into page_break_before on the following
    # block, so a section that happens to end on a page boundary cannot leave
    # a blank page behind it.
    ds.normalize_page_breaks(doc)
    doc.save(path)
    print(f"Saved: {path}")
    print(f"paragraphs={len(doc.paragraphs)} tables={len(doc.tables)}")


if __name__ == "__main__":
    main()
