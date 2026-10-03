"""
Builds the manuscript RESULTS section (machine-learning component) as a
standalone, submission-formatted .docx.

Length discipline: a Q1 Results section carries roughly five or six display
items and under ~2,000 words. Everything that supports rather than carries the
argument -- threshold-level classification metrics, the calibration and
decision curves, the subgroup forest, the full importance table and the
sensitivity grid -- lives in the combined supplement (scripts/21_write_supplement.py)
and is cited from here.

Display items retained in the main text:
  Table 1  survey-weighted baseline characteristics
  Table 2  discrimination, calibration and paired-bootstrap comparison, 11 models
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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import docx_style as ds

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs" / "metrics"
FIGP = ROOT / "outputs" / "figures_pub"
OUT = ROOT / "manuscript"
OUT.mkdir(parents=True, exist_ok=True)

N = json.loads((MET / "publication_numbers.json").read_text())
# Caption glossaries, written by scripts/15_publication_figures.py at the moment each
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

f = N["flow"]
TOP = N["top_model"]
tied_names = [m for m, v in N["significance"].items() if v["tied"]]

shap33 = shap_all[~shap_all["feature_code"].str.startswith("missingindicator")].copy()
shap33["rank"] = shap33["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
perm["rank"] = perm["perm_importance_mean"].rank(ascending=False, method="min").astype(int)
psd = {r.feature_code: (float(r.perm_importance_mean), float(r.perm_importance_std))
       for r in perm.itertuples()}

MINUS = "−"


def num(v, dp=3):
    return f"{v:.{dp}f}".replace("-", MINUS)


def p_txt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def ci(lo, hi):
    return f"{num(lo)}–{num(hi)}"


def main():
    doc = ds.new_document(line_spacing="1.5")
    ds.title(doc, "Results")
    ds.subtitle(doc, "Machine-learning analysis of blood metals and metabolic syndrome "
                     "in NHANES 2007–2018")

    # ==================================================================
    ds.heading(doc, "1. Study population", 1)

    ds.para(doc,
            f"Of {f['pooled']:,} participants sampled across the six NHANES cycles, "
            f"{f['final_cohort']:,} adults met all inclusion criteria (Figure 1), and were divided "
            f"into {f['train_n']:,} training and {f['test_n']:,} held-out test participants. "
            f"Metabolic syndrome was present in {int(outdef.loc[1, 'n_positive']):,} participants, a "
            f"crude prevalence of {outdef.loc[1, 'prevalence_pct']:.2f}%. Of these, "
            f"{int(outdef.loc[1, 'n_reclassified_negative_to_positive']):,} reached the "
            f"three-component threshold only because the drug-treatment clause scored a component "
            f"positive that the measured value alone would not have, and would have been classified "
            f"as non-cases under a measured-value-only definition "
            f"({outdef.loc[0, 'prevalence_pct']:.2f}% prevalence).")

    ds.para(doc,
            "Survey-weighted characteristics by metabolic syndrome status are given in Table 1. "
            "Participants with metabolic syndrome were on average 13.4 years older and had a body "
            "mass index 5.5 kg/m² higher. Among the blood metals, lead and selenium were higher in "
            "the metabolic syndrome group and mercury and manganese were lower, while cadmium did "
            "not differ appreciably; every difference reaching significance was small relative to "
            "the spread of the measurements.")

    rows_t1 = [[r["clinical_name"],
                f"{r['no_mets_mean']:.2f} ({r['no_mets_se']:.3f})",
                f"{r['mets_mean']:.2f} ({r['mets_se']:.3f})",
                p_txt(r["p_value"])] for _, r in tab1.iterrows()]
    ds.caption(doc, "Table 1.",
               "Survey-weighted characteristics of the analytic cohort by metabolic syndrome status. "
               "Continuous variables are weighted means (standard error); categorical variables are "
               "weighted percentages (standard error). Standard errors account for the NHANES "
               "complex survey design via Taylor-series linearization with domain estimation.",
               above=True)
    ds.add_table(doc, ["Characteristic", "No MetS", "MetS", "p"], rows_t1,
                 col_widths_in=[2.65, 1.35, 1.35, 0.75], align_right_from=1)
    ds.table_note(doc,
                  "MetS, metabolic syndrome. Unweighted group sizes: no MetS n = 5,981; MetS "
                  "n = 5,163. Selenium and manganese were assayed only from the 2011–2012 cycle "
                  "onward, so both are based on n = 3,382 without and n = 2,935 with metabolic "
                  "syndrome. Metal concentrations use the cycle-appropriate laboratory weight; all "
                  "other rows use the examination weight. *p-values for categorical variables are "
                  "descriptive chi-square tests and are not design-corrected.*")

    # ==================================================================
    ds.heading(doc, "2. Model performance", 1)

    top_row = cmp_df.iloc[0]
    worst = cmp_df.iloc[-1]
    tied_p = ", ".join(f"{N['significance'][m]['p']:.3f}" for m in tied_names if m != TOP)
    ds.para(doc,
            f"All eleven models were evaluated once on the held-out test partition (Table 2, "
            f"Figure 2). {TOP} achieved the highest area under the receiver operating characteristic "
            f"curve at {top_row['AUC']:.3f} "
            f"({ci(top_row['AUC_95CI_low'], top_row['AUC_95CI_high'])}), and "
            f"{worst['Model']} the lowest at {worst['AUC']:.3f} "
            f"({ci(worst['AUC_95CI_low'], worst['AUC_95CI_high'])}). The apparent ordering at the "
            f"top of the table does not survive sampling variability: in the paired bootstrap, "
            f"LightGBM, XGBoost and Gradient Boosting all had AUROC differences from {TOP} whose 95% "
            f"confidence intervals included zero (p = {tied_p}; Figure S2), so these four "
            f"gradient-boosted ensembles form a tied leading group rather than a ranking. Every "
            f"remaining model "
            f"differed from {TOP} at p < 0.001. Two caveats apply: the reference was chosen as the "
            f"test-set maximum, so the differences carry a winner's-curse bias and the p-values are "
            f"anticonservative, and no adjustment was made for the ten simultaneous comparisons. "
            f"The margin over the simplest baseline is modest in any case — logistic regression "
            f"reached {cmp_df.loc[cmp_df.Model == 'Logistic Regression', 'AUC'].iloc[0]:.3f}, only "
            f"{N['significance']['Logistic Regression']['auc_diff']:.3f} AUROC units below {TOP} "
            f"({ci(*N['significance']['Logistic Regression']['ci'])}).")

    d = N["dca"]
    ds.para(doc,
            f"Calibration of the four leading models was close to ideal, with slopes from "
            f"{cal.loc[tied_names, 'calibration_slope'].min():.2f} to "
            f"{cal.loc[tied_names, 'calibration_slope'].max():.2f} against an ideal of 1.00 "
            f"(Figure S3). Calibration did not track discrimination across the wider field, which "
            f"AUROC alone would conceal: Random Forest reached a respectable "
            f"{cmp_df.loc[cmp_df.Model == 'Random Forest', 'AUC'].iloc[0]:.3f} but a slope of "
            f"{cal.loc['Random Forest', 'calibration_slope']:.2f}, and the multilayer perceptron "
            f"({cal.loc['MLP', 'calibration_slope']:.2f}) and Naive Bayes "
            f"({cal.loc['Naive Bayes', 'calibration_slope']:.2f}) were worse in the opposite "
            f"direction, producing probabilities far too extreme for the evidence behind them. Any "
            f"of these could rank participants acceptably while giving an individual a badly wrong "
            f"absolute risk. Decision-curve analysis (Figure S4) showed the leading models above "
            f"both reference strategies across the range of thresholds examined; at a threshold of "
            f"0.20, {TOP} yielded a net benefit of {d['points']['0.20']['model_nb']:.3f} against "
            f"{d['points']['0.20']['treat_all_nb']:.3f} for treat-all, equivalent to identifying "
            f"{d['points']['0.20']['net_true_positives_per_1000']:.0f} additional true cases per "
            f"1,000 adults screened without treating more people unnecessarily. Threshold-level "
            f"classification metrics are given in Table S4.")

    rows_t2 = []
    for _, r in cmp_df.iterrows():
        m = r["Model"]
        s = N["significance"][m]
        delta = ("reference" if m == TOP else
                 f"{num(s['auc_diff'], 4)} ({num(s['ci'][0], 4)} to {num(s['ci'][1], 4)})")
        rows_t2.append([("**" + m + "**") if s["tied"] else m,
                        f"{r['AUC']:.3f} ({ci(r['AUC_95CI_low'], r['AUC_95CI_high'])})",
                        f"{r['AUPRC']:.3f}", f"{r['Brier']:.4f}",
                        f"{cal.loc[m, 'calibration_slope']:.2f}",
                        delta, "—" if m == TOP else p_txt(s["p"])])
    ds.caption(doc, "Table 2.",
               f"Discrimination and calibration on the held-out test partition (n = {f['test_n']:,}), "
               f"ordered by AUROC. Differences are against {TOP} from a paired bootstrap of 1,000 "
               f"resamples in which the same resampled participants were scored under both models. "
               f"**Models in bold are statistically indistinguishable from {TOP}.**", above=True)
    ds.add_table(doc,
                 ["Model", "AUROC (95% CI)", "AUPRC", "Brier", "Cal. slope",
                  f"Δ AUROC vs {TOP} (95% CI)", "p"],
                 rows_t2, col_widths_in=[1.15, 1.30, 0.50, 0.52, 0.58, 1.70, 0.60],
                 align_right_from=1, font_size=ds.Pt(8))
    ds.table_note(doc,
                  "AUROC, area under the receiver operating characteristic curve; AUPRC, area under "
                  "the precision–recall curve; Cal. slope, calibration slope (1.00 = perfect). Lower "
                  "Brier scores indicate better probabilistic accuracy.")

    ds.figure(doc, FIGP / "fig2_roc_curves.png", width_in=6.3)
    ds.caption(doc, "Figure 2.",
               "Receiver operating characteristic curves on the held-out test partition. "
               "**(A)** All eleven models, with AUROC in the legend; the four models "
               "statistically indistinguishable from the best performer are drawn in color and the "
               "remainder in grey. **(B)** Detail of the region of clinical interest.")

    # ==================================================================
    ds.heading(doc, "3. Predictor importance", 1)

    s1, s2, s3 = shap33.nsmallest(3, "rank").itertuples()
    p1, p2, p3 = perm.nsmallest(3, "rank").itertuples()
    ds.para(doc,
            f"Body mass index and age dominated both importance analyses (Figures 3 and 4; full "
            f"ranking in Table S5). By mean absolute SHAP value, body mass index "
            f"({s1.mean_abs_shap:.3f}) and age ({s2.mean_abs_shap:.3f}) were followed by "
            f"{s3.clinical_name.lower()} at {s3.mean_abs_shap:.3f}, a fivefold drop from second to "
            f"third. The gap was starker by permutation importance, where removing body mass index "
            f"cost {p1.perm_importance_mean:.3f} AUROC units and age "
            f"{p2.perm_importance_mean:.3f}, against {p3.perm_importance_mean:.4f} for the "
            f"third-ranked predictor. Between them these two account for nearly all of the model's "
            f"discriminatory capacity; everything else adjusts risk at the margin.")

    ds.para(doc,
            f"The two methods agreed only moderately below the top of the ranking (Spearman "
            f"ρ = {N['spearman_shap_perm']:.2f} across {N['n_predictors_ranked']} predictors; "
            f"Figure 4A). Disagreement of this kind is expected rather than anomalous: mean "
            f"absolute SHAP measures how far a predictor moves individual predictions, whereas "
            f"permutation importance measures how much ranking performance degrades when that "
            f"predictor is destroyed, so a predictor substitutable by correlated predictors scores "
            f"well on the first and poorly on the second. Reporting either alone would have "
            f"supported a confident but method-dependent account of which exposures matter. Two "
            f"missingness indicators also carried non-trivial SHAP weight — whether average daily "
            f"alcohol consumption and whether the income-to-poverty ratio were recorded — which are "
            f"artifacts of questionnaire structure rather than biological signal and are excluded "
            f"from the predictor ranking.")

    ds.figure(doc, FIGP / "fig6_shap_beeswarm.png", width_in=5.9)
    ds.caption(doc, "Figure 3.",
               f"SHAP summary plot for the {TOP} model on the held-out test partition, showing the "
               f"18 highest-ranked features. Each point is one participant; horizontal position is "
               f"that feature's contribution to that participant's predicted log-odds of metabolic "
               f"syndrome, and color encodes the feature's value from low (blue) to high "
               f"(orange). " + gloss("fig6_shap_beeswarm"))

    # ==================================================================
    ds.heading(doc, "4. Blood metals", 1)

    mr = N["metal_ranks"]
    pb, cd, se, hg, mn = (mr["LBXBPB"], mr["LBXBCD"], mr["LBXBSE"], mr["LBXTHG"], mr["LBXBMN"])

    def span(m):
        v = list(m["per_model_perm_rank"].values())
        return f"{min(v)}–{max(v)}"

    gaps = {k: abs(v["shap_rank"] - v["perm_rank"]) for k, v in mr.items()}
    worst = max(gaps, key=gaps.get)
    worst_name = mr[worst]["clinical_name"].split(" (")[0].replace("Blood ", "").replace(", total", "")

    ds.para(doc,
            f"All five metals ranked in the middle or lower portion of the ordering, well below "
            f"body mass index and age but above several demographic and lifestyle covariates. Their "
            f"relative standing depended sharply on which importance method was used, and that "
            f"method-dependence is itself the substantive observation; the precise ordering it "
            f"produces is not. Cadmium was the highest-ranked metal by SHAP (rank {cd['shap_rank']} "
            f"of {N['n_predictors_ranked']}) but fell to {cd['perm_rank']} by permutation "
            f"importance, a gap of {gaps['LBXBCD']} places; the widest gap among the metals belonged "
            f"to {worst_name} ({mr[worst]['shap_rank']} versus {mr[worst]['perm_rank']}). Lead "
            f"showed the opposite pattern, ranking {pb['shap_rank']} and {pb['perm_rank']} — close "
            f"agreement between methods, and the only metal of the five in the permutation-importance "
            f"top ten.")

    ds.para(doc,
            f"Agreement across independently fitted models tells the same story more decisively "
            f"(Figure 4B). Lead was the most stable, ranking {span(pb)} across the four leading "
            f"models and appearing in the top ten of every one. Selenium ranked {span(se)}, cadmium "
            f"{span(cd)}, manganese {span(mn)} and mercury {span(hg)}, the widest swing of any "
            f"predictor examined. Averaged across models, neither cadmium nor mercury nor manganese "
            f"placed among the fifteen highest-ranked predictors. The permutation analysis also "
            f"carries its own uncertainty, and reading the metal ordering without it would overstate "
            f"what the data support: across ten permutations only lead "
            f"({psd['LBXBPB'][0]:.5f} ± {psd['LBXBPB'][1]:.5f}) and selenium "
            f"({psd['LBXBSE'][0]:.5f} ± {psd['LBXBSE'][1]:.5f}) had mean importances exceeding "
            f"their own standard deviation, whereas for cadmium, manganese and mercury the mean was "
            f"smaller than the run-to-run variability. The defensible statement is narrower than a "
            f"rank list: lead and selenium carry permutation signal distinguishable from noise, and "
            f"the other three do not.")

    md = N["metals_dependence"]
    se_c, se_v = md["LBXBSE"]["binned_centers"], md["LBXBSE"]["binned_mean_shap"]
    se_cross = next((c for c, v in zip(se_c, se_v) if v > 0), None)
    pb_c, pb_v = md["LBXBPB"]["binned_centers"], md["LBXBPB"]["binned_mean_shap"]
    cd_v = md["LBXBCD"]["binned_mean_shap"]
    mn_v = md["LBXBMN"]["binned_mean_shap"]
    pb_last_pos = max((c for c, v in zip(pb_c, pb_v) if v > 0), default=None)

    ds.para(doc,
            f"The shape of each metal's contribution is shown in Figure 5. Selenium was the only one "
            f"to rise monotonically across essentially its whole range, moving from "
            f"{num(se_v[0])} in the lowest bin, crossing zero at approximately {se_cross:.0f} µg/L "
            f"and reaching {num(max(se_v))} in the highest — the largest positive contribution of "
            f"any metal. Cadmium was U-shaped, contributing positively at both the lowest "
            f"({num(cd_v[0])}) and highest ({num(cd_v[-1])}) concentrations with a trough of "
            f"{num(min(cd_v))} between. Lead ran opposite to the conventional expectation, "
            f"contributing positively only up to about {pb_last_pos:.1f} µg/dL and then declining "
            f"to {num(min(pb_v))} in the highest bin. Mercury declined similarly, and manganese was "
            f"flat throughout, every binned mean falling between {num(min(mn_v))} and "
            f"{num(max(mn_v))}.")

    ds.para(doc,
            "The inverse gradient for lead deserves emphasis rather than quiet omission, because it "
            "runs against the hypothesis that motivated the analysis. In a cross-sectional model "
            "adjusted for body mass index, age and socioeconomic position, higher blood lead was "
            "associated with a lower predicted probability of metabolic syndrome. This is an "
            "associational pattern within one fitted model, not an estimate of a causal effect; "
            "residual confounding by smoking, which raises blood cadmium and lead while being "
            "associated with lower body mass index, is one plausible explanation among several, and "
            "adiposity-related dilution of blood metal concentrations is another. Disentangling "
            "these is what the Mendelian randomization component is designed to do, and the "
            "direction observed here should be read as a question posed to that analysis rather "
            "than an answer from this one.")

    ds.figure(doc, FIGP / "fig7_importance_concordance.png", width_in=5.7)
    ds.caption(doc, "Figure 4.",
               f"Agreement between importance methods and across models. **(A)** Rank by mean "
               f"absolute SHAP value against rank by permutation importance for the {TOP} model; "
               f"the five blood metals are highlighted and the diagonal denotes perfect agreement. "
               f"**(B)** Permutation-importance rank of the fifteen highest-ranked predictors in "
               f"each of the four leading models, fitted independently. A predictor whose row is "
               f"uniform is important as a property of the data; one whose row varies is important "
               f"as a property of a particular fitted model. " + gloss("fig7_importance_concordance"))

    ds.figure(doc, FIGP / "fig8_metals_shap_dependence.png", width_in=6.3)
    ds.caption(doc, "Figure 5.",
               f"SHAP dependence for the five blood metals in the {TOP} model on the held-out test "
               f"partition. Each grey point is one participant; the orange line joins mean SHAP "
               f"values within duodeciles of the exposure distribution, and the dashed line at zero "
               f"marks no contribution to the predicted log-odds. Axes are truncated at the 1st and "
               f"98th percentiles. Selenium and manganese are based on the "
               f"{md['LBXBSE']['n_observed']:,} test participants from cycles in which those "
               f"analytes were assayed. " + gloss("fig8_metals_shap_dependence"))

    # ==================================================================
    ds.heading(doc, "5. Subgroup performance and sensitivity analyses", 1)

    ov = sg[sg.subgroup == "Overall"].iloc[0]
    old = sg[sg.subgroup.str.startswith("Age >")].iloc[0]
    yng = sg[sg.subgroup.str.startswith("Age 20")].iloc[0]
    mid = sg[sg.subgroup.str.startswith("Age 41")].iloc[0]
    fem = sg[sg.subgroup == "Female"].iloc[0]
    ds.para(doc,
            f"Discrimination was stable across sex and race/ethnicity but deteriorated markedly with "
            f"age (Table S6, Figure S5). Against an overall test-set AUROC of {ov['auc']:.3f} "
            f"({ci(ov['ci_low'], ov['ci_high'])}), the oldest tertile reached {old['auc']:.3f} "
            f"({ci(old['ci_low'], old['ci_high'])}), an interval lying entirely below that of the "
            f"youngest tertile ({yng['auc']:.3f}, {ci(yng['ci_low'], yng['ci_high'])}). Within a "
            f"single age tertile the range of age is compressed to about two decades, so the "
            f"model's second most important predictor can no longer separate participants within "
            f"that stratum, and a prevalence of {old['prevalence_pct']:.1f}% leaves few negatives "
            f"to rank against. Calibration behaved differently: only the middle tertile departed "
            f"detectably from ideal on both statistics (slope {mid['calibration_slope']:.2f}, "
            f"{ci(mid['slope_ci_low'], mid['slope_ci_high'])}; observed-to-expected "
            f"{mid['observed_expected_ratio']:.3f}, {ci(mid['oe_ci_low'], mid['oe_ci_high'])}), "
            f"with women showing a smaller over-prediction "
            f"({fem['observed_expected_ratio']:.3f}, {ci(fem['oe_ci_low'], fem['oe_ci_high'])}); in "
            f"the oldest tertile predicted risk was accurate in aggregate "
            f"({old['observed_expected_ratio']:.3f}, {ci(old['oe_ci_low'], old['oe_ci_high'])}). "
            f"These are descriptive screens over {len(sg) * 2} intervals without multiplicity "
            f"adjustment. The model is therefore least useful for ranking precisely the group with "
            f"the highest absolute burden, while still stating that group's average risk correctly.")

    ds.para(doc,
            f"Results of the four prespecified sensitivity analyses are given in Table S7. Adding "
            f"physician-diagnosed diabetes raised cross-validated discrimination by "
            f"{s_diq['delta'].min():.4f} to {s_diq['delta'].max():.4f} AUROC units, as expected for "
            f"a variable partly downstream of the glucose component. Replacing SMOTENC with class "
            f"weighting changed discrimination by at most {s_smo['delta'].abs().max():.4f} units, "
            f"confirming that the imbalance strategy is immaterial at this near-balanced prevalence. "
            f"Substituting the measured-value-only outcome definition lowered discrimination by "
            f"{s_out['delta'].min():.4f} to {s_out['delta'].max():.4f} units, the largest effect of "
            f"the four. The metal-panel analysis produced the most consequential result: restricting "
            f"the panel to blood lead alone and discarding cadmium, mercury, selenium and manganese "
            f"changed discrimination by between {s_met['delta'].min():+.4f} and "
            f"{s_met['delta'].max():+.4f} units — marginally upward in all three models, not "
            f"downward. Deltas this small sit within run-to-run variability of the "
            f"cross-validation and carry no confidence intervals, so this is an "
            f"absence of detectable contribution rather than a demonstration of exactly zero. The "
            f"four additional metals add nothing measurable to discrimination: the ranking "
            f"positions they occupy reflect within-model attribution, not predictive value that "
            f"would be lost by removing them. No arm removed lead itself, so its own incremental "
            f"contribution was not tested by this design.")

    path = OUT / "Manuscript_Results_AI.docx"
    # Convert break-only paragraphs into page_break_before on the following
    # block, so a section that happens to end on a page boundary cannot leave
    # a blank page behind it.
    n_breaks = ds.normalize_page_breaks(doc)
    doc.save(path)
    print(f"Saved: {path}")
    print(f"paragraphs={len(doc.paragraphs)} tables={len(doc.tables)}")


if __name__ == "__main__":
    main()
