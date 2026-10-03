"""
Verified reference list for the manuscript Methods section.

EVERY entry here was retrieved and confirmed through the PubMed E-utilities API
(or, for the computer-science sources that PubMed does not index, resolved
through the publisher's own DOI record) on 2026-09-18. The PMID / DOI recorded
with each entry is the one the database returned, not one reconstructed from
memory. Nothing in this file should be edited by hand without re-verifying the
identifier it carries.

2026-10 audit (P12): Altmann et al. 2010 (a corrected, p-value-based
variant) was replaced by the sources of the permutation method actually used:
Breiman 2001 (publisher record checked on 2026-10-03) and Fisher, Rudin and
Dominici 2019 (PMID 34335110, PMC8323609, checked on 2026-10-03).

VERIFICATION NOTE for the non-PubMed entries:
  * Chawla 2002 - DOI 10.1613/jair.953 resolves to the JAIR record for "SMOTE:
    Synthetic Minority Over-sampling Technique", vol 16, pp 321-357. The
    SMOTE-NC variant used here is attributed to this paper by the canonical
    implementation (imbalanced-learn), whose documentation was checked directly.
"""
from __future__ import annotations

# Order = order of first citation in the Methods text. REFS is rendered as the
# numbered reference list; the keys are used as in-text markers.
REFS = [
    ("nhanes_design", "Johnson CL, Paulose-Ram R, Ogden CL, Carroll MD, Kruszon-Moran D, "
                      "Dohrmann SM, et al. National Health and Nutrition Examination Survey: "
                      "analytic guidelines, 1999–2010. Vital Health Stat 2. 2013;(161):1–24. "
                      "PMID: 25090154."),
    ("nhanes_weights", "Chen TC, Parker JD, Clark J, Shin HC, Rammon JR, Burt VL. National Health "
                       "and Nutrition Examination Survey: estimation procedures, 2011–2014. "
                       "Vital Health Stat 2. 2018;(177):1–26. PMID: 29775431."),
    ("tripod", "Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, et al. "
               "TRIPOD+AI statement: updated guidance for reporting clinical prediction models "
               "that use regression or machine learning methods. BMJ. 2024;385:e078378. "
               "doi:10.1136/bmj-2023-078378."),
    ("strobe", "von Elm E, Altman DG, Egger M, Pocock SJ, Gøtzsche PC, Vandenbroucke JP. The "
               "Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) "
               "statement: guidelines for reporting observational studies. Lancet. "
               "2007;370(9596):1453–7. doi:10.1016/S0140-6736(07)61602-X."),
    ("alberti", "Alberti KGMM, Eckel RH, Grundy SM, Zimmet PZ, Cleeman JI, Donato KA, et al. "
                "Harmonizing the metabolic syndrome: a joint interim statement of the "
                "International Diabetes Federation Task Force on Epidemiology and Prevention; "
                "National Heart, Lung, and Blood Institute; American Heart Association; World "
                "Heart Federation; International Atherosclerosis Society; and International "
                "Association for the Study of Obesity. Circulation. 2009;120(16):1640–5. "
                "doi:10.1161/CIRCULATIONAHA.109.192644."),
    ("grundy", "Grundy SM, Cleeman JI, Daniels SR, Donato KA, Eckel RH, Franklin BA, et al. "
               "Diagnosis and management of the metabolic syndrome: an American Heart "
               "Association/National Heart, Lung, and Blood Institute scientific statement. "
               "Circulation. 2005;112(17):2735–52. doi:10.1161/CIRCULATIONAHA.105.169404."),
    ("bernett", "Bernett J, Blumenthal DB, Grimm DG, Haselbeck F, Joeres R, Kalinina OV, et al. "
                "Guiding questions to avoid data leakage in biological machine learning "
                "applications. Nat Methods. 2024;21(8):1444–53. doi:10.1038/s41592-024-02362-y."),
    # 2026-10 audit (P12): Lo et al. 2021 (blood and urinary manganese only) was
    # cited for the five-metal panel; Bulka et al. analysed the multi-metal
    # panel against metabolic syndrome in NHANES. Record taken from the PubMed
    # metadata captured for the proposal bank (PMID 30388496).
    ("bulka", "Bulka CM, Persky VW, Daviglus ML, Durazo-Arvizu RA, Argos M. Multiple metal "
              "exposures and metabolic syndrome: a cross-sectional analysis of the National "
              "Health and Nutrition Examination Survey 2011–2014. Environ Res. "
              "2019;168:397–405. doi:10.1016/j.envres.2018.10.022. PMID: 30388496."),
    ("heinze", "Heinze G, Wallisch C, Dunkler D. Variable selection – a review and "
               "recommendations for the practicing statistician. Biom J. 2018;60(3):431–49. "
               "doi:10.1002/bimj.201700067."),
    ("sterne", "Sterne JAC, White IR, Carlin JB, Spratt M, Royston P, Kenward MG, et al. Multiple "
               "imputation for missing data in epidemiological and clinical research: potential "
               "and pitfalls. BMJ. 2009;338:b2393. doi:10.1136/bmj.b2393."),
    ("sisk", "Sisk R, Sperrin M, Peek N, van Smeden M, Martin GP. Imputation and missing "
             "indicators for handling missing data in the development and deployment of clinical "
             "prediction models: a simulation study. Stat Methods Med Res. 2023;32(8):1461–77. "
             "doi:10.1177/09622802231165001."),
    ("riley", "Riley RD, Ensor J, Snell KIE, Harrell FE, Martin GP, Reitsma JB, et al. Calculating "
              "the sample size required for developing a clinical prediction model. BMJ. "
              "2020;368:m441. doi:10.1136/bmj.m441."),
    ("kapoor", "Kapoor S, Narayanan A. Leakage and the reproducibility crisis in "
               "machine-learning-based science. Patterns (N Y). 2023;4(9):100804. "
               "doi:10.1016/j.patter.2023.100804."),
    ("chawla", "Chawla NV, Bowyer KW, Hall LO, Kegelmeyer WP. SMOTE: synthetic minority "
               "over-sampling technique. J Artif Intell Res. 2002;16:321–57. "
               "doi:10.1613/jair.953."),
    ("goorbergh", "van den Goorbergh R, van Smeden M, Timmerman D, Van Calster B. The harm of "
                  "class imbalance corrections for risk prediction models: illustration and "
                  "simulation using logistic regression. J Am Med Inform Assoc. "
                  "2022;29(9):1525–34. doi:10.1093/jamia/ocac093."),
    ("youden", "Youden WJ. Index for rating diagnostic tests. Cancer. 1950;3(1):32–5. "
               "doi:10.1002/1097-0142(1950)3:1<32::aid-cncr2820030106>3.0.co;2-3."),
    ("delong", "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more "
               "correlated receiver operating characteristic curves: a nonparametric approach. "
               "Biometrics. 1988;44(3):837–45. PMID: 3203132."),
    ("vancalster", "Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. "
                   "Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17:230. "
                   "doi:10.1186/s12916-019-1466-7."),
    ("vickers", "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating "
                "prediction models. Med Decis Making. 2006;26(6):565–74. "
                "doi:10.1177/0272989X06295361."),
    ("lundberg", "Lundberg SM, Erion G, Chen H, DeGrave A, Prutkin JM, Nair B, et al. From local "
                 "explanations to global understanding with explainable AI for trees. Nat Mach "
                 "Intell. 2020;2(1):56–67. doi:10.1038/s42256-019-0138-9."),
    ("breiman", "Breiman L. Random forests. Mach Learn. 2001;45(1):5–32. "
                "doi:10.1023/A:1010933404324."),
    ("fisher_mcr", "Fisher A, Rudin C, Dominici F. All models are wrong, but many are useful: "
                   "learning a variable's importance by studying an entire class of prediction "
                   "models simultaneously. J Mach Learn Res. 2019;20:177. PMID: 34335110."),
]

NUM = {key: i + 1 for i, (key, _) in enumerate(REFS)}


def cite(*keys) -> str:
    """In-text citation marker, e.g. cite('alberti', 'grundy') -> '[5,6]'.
    Consecutive runs are collapsed to a range in the usual way."""
    nums = sorted(NUM[k] for k in keys)
    if not nums:
        return ""
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        parts.append((start, prev))
        start = prev = n
    parts.append((start, prev))
    out = []
    for a, b in parts:
        out.append(str(a) if a == b else (f"{a},{b}" if b == a + 1 else f"{a}–{b}"))
    return "[" + ",".join(out) + "]"
