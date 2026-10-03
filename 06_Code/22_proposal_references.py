"""
Reference bank for the research proposal.

Every entry in REFS was retrieved from PubMed on 2026-09-18 and its title read back
before being admitted here; four plausible-looking guesses were discarded at that step
because the PMID resolved to an entirely different paper, which is precisely why the
check exists. Entries carry the PMID and DOI so that scripts/20_verify_citations.py can
re-confirm them against the captured metadata in
outputs/metrics/proposal_reference_verification.json.

Numbering is NOT fixed here. The proposal is renumbered in first-mention order by
scripts/23_rewrite_proposal_citations.py, which walks the document body and assigns
[1], [2], ... in the order the markers actually appear. That is what removes the defect
in the previous draft, where the same number served two unrelated literatures in two
different sections (for example [44] was both an income/ethnicity claim and the
inverse-variance-weighted estimator).

Where PubMed's publication_date disagrees with the issue date because the paper appeared
online ahead of print, `year` records the ISSUE year -- the form a journal expects in a
reference list -- and `pubmed_year` records what PubMed returned, so the audit can
reconcile the two without either being treated as an error.
"""
from __future__ import annotations

# key -> bibliographic record
# authors are given in the journal-standard short form (up to three, then "et al.")
REFS: dict[str, dict] = {

    # ------------------------------------------------------------------
    # Lead: exposure pathways, toxicology, and health effects
    # ------------------------------------------------------------------
    "pb_whelan": dict(
        authors="Whelan EA, Piacitelli GM, Gerwel B, et al.",
        title="Elevated blood lead levels in children of construction workers",
        journal="Am J Public Health", year="1997", volume="87", issue="8",
        pages="1352-5", pmid="9279275", doi="10.2105/ajph.87.8.1352"),
    "pb_rinsky": dict(
        authors="Rinsky JL, Higgins S, Angelon-Gaetz K, et al.",
        title="Occupational and take-home lead exposure among lead oxide manufacturing "
              "employees, North Carolina, 2016",
        journal="Public Health Rep", year="2018", volume="133", issue="6",
        pages="700-706", pmid="30231234", doi="10.1177/0033354918795442"),
    "pb_needleman": dict(
        authors="Needleman H.",
        title="Lead poisoning",
        journal="Annu Rev Med", year="2004", volume="55", issue=None,
        pages="209-22", pmid="14746518",
        doi="10.1146/annurev.med.55.091902.103653"),
    "pb_lanphear_iq": dict(
        authors="Lanphear BP, Hornung R, Khoury J, et al.",
        title="Low-level environmental lead exposure and children's intellectual "
              "function: an international pooled analysis",
        journal="Environ Health Perspect", year="2005", volume="113", issue="7",
        pages="894-9", pmid="16002379", doi="10.1289/ehp.7688"),
    "pb_reuben": dict(
        authors="Reuben A, Caspi A, Belsky DW, et al.",
        title="Association of childhood blood lead levels with cognitive function and "
              "socioeconomic status at age 38 years and with IQ change and socioeconomic "
              "mobility between childhood and adulthood",
        journal="JAMA", year="2017", volume="317", issue="12",
        pages="1244-1251", pmid="28350927", doi="10.1001/jama.2017.1712"),
    "pb_navas": dict(
        authors="Navas-Acien A, Guallar E, Silbergeld EK, Rothenberg SJ.",
        title="Lead exposure and cardiovascular disease: a systematic review",
        journal="Environ Health Perspect", year="2007", pubmed_year="2006",
        volume="115", issue="3", pages="472-82", pmid="17431501",
        doi="10.1289/ehp.9785"),
    "pb_lanphear_mort": dict(
        authors="Lanphear BP, Rauch S, Auinger P, et al.",
        title="Low-level lead exposure and mortality in US adults: a population-based "
              "cohort study",
        journal="Lancet Public Health", year="2018", volume="3", issue="4",
        pages="e177-e184", pmid="29544878", doi="10.1016/S2468-2667(18)30025-2"),
    "pb_perlstein": dict(
        authors="Perlstein T, Weuve J, Schwartz J, et al.",
        title="Cumulative community-level lead exposure and pulse pressure: the Normative "
              "Aging Study",
        journal="Environ Health Perspect", year="2007", volume="115", issue="12",
        pages="1696-700", pmid="18087585", doi="10.1289/ehp.10350"),
    "pb_prehtn": dict(
        authors="Lee W, Yoon JH, Roh J, et al.",
        title="The association between low blood lead levels and the prevalence of "
              "prehypertension among nonhypertensive adults in Korea",
        journal="Am J Hum Biol", year="2016", volume="28", issue="5",
        pages="729-35", pmid="27121332", doi="10.1002/ajhb.22857"),

    # ------------------------------------------------------------------
    # Metabolic syndrome: definition, burden, consequences
    # ------------------------------------------------------------------
    "mets_atp3": dict(
        authors="Expert Panel on Detection, Evaluation, and Treatment of High Blood "
                "Cholesterol in Adults.",
        title="Executive summary of the Third Report of the National Cholesterol "
              "Education Program (NCEP) Expert Panel on Detection, Evaluation, and "
              "Treatment of High Blood Cholesterol in Adults (Adult Treatment Panel III)",
        journal="JAMA", year="2001", volume="285", issue="19",
        pages="2486-97", pmid="11368702", doi="10.1001/jama.285.19.2486"),
    "mets_idf": dict(
        authors="Alberti KG, Zimmet P, Shaw J.",
        title="Metabolic syndrome: a new world-wide definition. A consensus statement "
              "from the International Diabetes Federation",
        journal="Diabet Med", year="2006", volume="23", issue="5",
        pages="469-80", pmid="16681555", doi="10.1111/j.1464-5491.2006.01858.x"),
    "mets_grundy2005": dict(
        authors="Grundy SM, Cleeman JI, Daniels SR, et al.",
        title="Diagnosis and management of the metabolic syndrome: an American Heart "
              "Association/National Heart, Lung, and Blood Institute scientific statement",
        journal="Circulation", year="2005", volume="112", issue="17",
        pages="2735-52", pmid="16157765",
        doi="10.1161/CIRCULATIONAHA.105.169404"),
    "mets_alberti2009": dict(
        authors="Alberti KG, Eckel RH, Grundy SM, et al.",
        title="Harmonizing the metabolic syndrome: a joint interim statement of the "
              "International Diabetes Federation Task Force on Epidemiology and "
              "Prevention; National Heart, Lung, and Blood Institute; American Heart "
              "Association; World Heart Federation; International Atherosclerosis "
              "Society; and International Association for the Study of Obesity",
        journal="Circulation", year="2009", volume="120", issue="16",
        pages="1640-5", pmid="19805654",
        doi="10.1161/CIRCULATIONAHA.109.192644"),
    "mets_eckel": dict(
        authors="Eckel RH, Grundy SM, Zimmet PZ.",
        title="The metabolic syndrome",
        journal="Lancet", year="2005", volume="365", issue="9468",
        pages="1415-28", pmid="15836891", doi="10.1016/S0140-6736(05)66378-7"),
    "mets_isomaa": dict(
        authors="Isomaa B, Almgren P, Tuomi T, et al.",
        title="Cardiovascular morbidity and mortality associated with the metabolic "
              "syndrome",
        journal="Diabetes Care", year="2001", volume="24", issue="4",
        pages="683-9", pmid="11315831", doi="10.2337/diacare.24.4.683"),
    "mets_gami": dict(
        authors="Gami AS, Witt BJ, Howard DE, et al.",
        title="Metabolic syndrome and risk of incident cardiovascular events and death: "
              "a systematic review and meta-analysis of longitudinal studies",
        journal="J Am Coll Cardiol", year="2007", volume="49", issue="4",
        pages="403-14", pmid="17258085", doi="10.1016/j.jacc.2006.09.032"),
    "mets_mottillo": dict(
        authors="Mottillo S, Filion KB, Genest J, et al.",
        title="The metabolic syndrome and cardiovascular risk: a systematic review and "
              "meta-analysis",
        journal="J Am Coll Cardiol", year="2010", volume="56", issue="14",
        pages="1113-32", pmid="20863953", doi="10.1016/j.jacc.2010.05.034"),
    "mets_esposito_cancer": dict(
        authors="Esposito K, Chiodini P, Colao A, et al.",
        title="Metabolic syndrome and risk of cancer: a systematic review and "
              "meta-analysis",
        journal="Diabetes Care", year="2012", volume="35", issue="11",
        pages="2402-11", pmid="23093685", doi="10.2337/dc12-0336"),
    "mets_saklayen": dict(
        authors="Saklayen MG.",
        title="The global epidemic of the metabolic syndrome",
        journal="Curr Hypertens Rep", year="2018", volume="20", issue="2",
        pages="12", pmid="29480368", doi="10.1007/s11906-018-0812-z"),
    "mets_ford2002": dict(
        authors="Ford ES, Giles WH, Dietz WH.",
        title="Prevalence of the metabolic syndrome among US adults: findings from the "
              "Third National Health and Nutrition Examination Survey",
        journal="JAMA", year="2002", volume="287", issue="3",
        pages="356-9", pmid="11790215", doi="10.1001/jama.287.3.356"),
    "mets_beltran": dict(
        authors="Beltrán-Sánchez H, Harhay MO, Harhay MM, McElligott S.",
        title="Prevalence and trends of metabolic syndrome in the adult U.S. population, "
              "1999-2010",
        journal="J Am Coll Cardiol", year="2013", volume="62", issue="8",
        pages="697-703", pmid="23810877", doi="10.1016/j.jacc.2013.05.064"),
    "mets_aguilar": dict(
        authors="Aguilar M, Bhuket T, Torres S, et al.",
        title="Prevalence of the metabolic syndrome in the United States, 2003-2012",
        journal="JAMA", year="2015", volume="313", issue="19",
        pages="1973-4", pmid="25988468", doi="10.1001/jama.2015.4260"),
    "mets_hirode": dict(
        authors="Hirode G, Wong RJ.",
        title="Trends in the prevalence of metabolic syndrome in the United States, "
              "2011-2016",
        journal="JAMA", year="2020", volume="323", issue="24",
        pages="2526-2528", pmid="32573660", doi="10.1001/jama.2020.4501"),

    # ------------------------------------------------------------------
    # Age and sex patterning
    # ------------------------------------------------------------------
    "as_pradhan": dict(
        authors="Pradhan AD.",
        title="Sex differences in the metabolic syndrome: implications for "
              "cardiovascular health in women",
        journal="Clin Chem", year="2014", pubmed_year="2013", volume="60", issue="1",
        pages="44-52", pmid="24255079", doi="10.1373/clinchem.2013.202549"),
    "as_carr": dict(
        authors="Carr MC.",
        title="The emergence of the metabolic syndrome with menopause",
        journal="J Clin Endocrinol Metab", year="2003", volume="88", issue="6",
        pages="2404-11", pmid="12788835", doi="10.1210/jc.2003-030242"),
    "as_regitz": dict(
        authors="Regitz-Zagrosek V, Lehmkuhl E, Weickert MO.",
        title="Gender differences in the metabolic syndrome and their role for "
              "cardiovascular disease",
        journal="Clin Res Cardiol", year="2006", volume="95", issue="3",
        pages="136-47", pmid="16598526", doi="10.1007/s00392-006-0351-5"),
    "as_kuk": dict(
        authors="Kuk JL, Ardern CI.",
        title="Age and sex differences in the clustering of metabolic syndrome factors: "
              "association with mortality risk",
        journal="Diabetes Care", year="2010", volume="33", issue="11",
        pages="2457-61", pmid="20699434", doi="10.2337/dc10-0942"),
    "as_hildrum": dict(
        authors="Hildrum B, Mykletun A, Hole T, et al.",
        title="Age-specific prevalence of the metabolic syndrome defined by the "
              "International Diabetes Federation and the National Cholesterol Education "
              "Program: the Norwegian HUNT 2 study",
        journal="BMC Public Health", year="2007", volume="7", issue=None,
        pages="220", pmid="17727697", doi="10.1186/1471-2458-7-220"),
    "as_moore": dict(
        authors="Moore JX, Chaudhary N, Akinyemiju T.",
        title="Metabolic syndrome prevalence by race/ethnicity and sex in the United "
              "States, National Health and Nutrition Examination Survey, 1988-2012",
        journal="Prev Chronic Dis", year="2017", volume="14", issue=None,
        pages="E24", pmid="28301314", doi="10.5888/pcd14.160287"),

    # ------------------------------------------------------------------
    # Socioeconomic position
    # ------------------------------------------------------------------
    "ses_loucks": dict(
        authors="Loucks EB, Magnusson KT, Cook S, et al.",
        title="Socioeconomic position and the metabolic syndrome in early, middle, and "
              "late life: evidence from NHANES 1999-2002",
        journal="Ann Epidemiol", year="2007", volume="17", issue="10",
        pages="782-90", pmid="17697786", doi="10.1016/j.annepidem.2007.05.003"),
    "ses_adler": dict(
        authors="Adler NE, Epel ES, Castellazzo G, Ickovics JR.",
        title="Relationship of subjective and objective social status with psychological "
              "and physiological functioning: preliminary data in healthy white women",
        journal="Health Psychol", year="2000", volume="19", issue="6",
        pages="586-92", pmid="11129362", doi="10.1037//0278-6133.19.6.586"),
    "ses_manuck": dict(
        authors="Manuck SB, Phillips JE, Gianaros PJ, et al.",
        title="Subjective socioeconomic status and presence of the metabolic syndrome in "
              "midlife community volunteers",
        journal="Psychosom Med", year="2010", pubmed_year="2009", volume="72", issue="1",
        pages="35-45", pmid="19933505", doi="10.1097/PSY.0b013e3181c484dc"),

    # ------------------------------------------------------------------
    # Anthropometry
    # ------------------------------------------------------------------
    "an_ashwell": dict(
        authors="Ashwell M, Gunn P, Gibson S.",
        title="Waist-to-height ratio is a better screening tool than waist circumference "
              "and BMI for adult cardiometabolic risk factors: systematic review and "
              "meta-analysis",
        journal="Obes Rev", year="2012", pubmed_year="2011", volume="13", issue="3",
        pages="275-86", pmid="22106927", doi="10.1111/j.1467-789X.2011.00952.x"),
    "an_bergman": dict(
        authors="Bergman RN, Stefanovski D, Buchanan TA, et al.",
        title="A better index of body adiposity",
        journal="Obesity (Silver Spring)", year="2011", volume="19", issue="5",
        pages="1083-9", pmid="21372804", doi="10.1038/oby.2011.38"),
    "an_snijder": dict(
        authors="Snijder MB, Dekker JM, Visser M, et al.",
        title="Associations of hip and thigh circumferences independent of waist "
              "circumference with the incidence of type 2 diabetes: the Hoorn Study",
        journal="Am J Clin Nutr", year="2003", volume="77", issue="5",
        pages="1192-7", pmid="12716671", doi="10.1093/ajcn/77.5.1192"),
    "an_manolopoulos": dict(
        authors="Manolopoulos KN, Karpe F, Frayn KN.",
        title="Gluteofemoral body fat as a determinant of metabolic health",
        journal="Int J Obes (Lond)", year="2010", volume="34", issue="6",
        pages="949-59", pmid="20065965", doi="10.1038/ijo.2009.286"),
    "an_who_asian": dict(
        authors="WHO Expert Consultation.",
        title="Appropriate body-mass index for Asian populations and its implications "
              "for policy and intervention strategies",
        journal="Lancet", year="2004", volume="363", issue="9403",
        pages="157-63", pmid="14726171", doi="10.1016/S0140-6736(03)15268-3"),

    # ------------------------------------------------------------------
    # Lifestyle: physical activity, smoking, alcohol
    # ------------------------------------------------------------------
    "ls_zhang_pa": dict(
        authors="Zhang D, Liu X, Liu Y, et al.",
        title="Leisure-time physical activity and incident metabolic syndrome: a "
              "systematic review and dose-response meta-analysis of cohort studies",
        journal="Metabolism", year="2017", volume="75", issue=None,
        pages="36-44", pmid="28927737", doi="10.1016/j.metabol.2017.08.001"),
    "ls_edwardson": dict(
        authors="Edwardson CL, Gorely T, Davies MJ, et al.",
        title="Association of sedentary behaviour with metabolic syndrome: a "
              "meta-analysis",
        journal="PLoS One", year="2012", volume="7", issue="4",
        pages="e34916", pmid="22514690", doi="10.1371/journal.pone.0034916"),
    "ls_ford_sed": dict(
        authors="Ford ES, Kohl HW, Mokdad AH, Ajani UA.",
        title="Sedentary behavior, physical activity, and the metabolic syndrome among "
              "U.S. adults",
        journal="Obes Res", year="2005", volume="13", issue="3",
        pages="608-14", pmid="15833947", doi="10.1038/oby.2005.65"),
    "ls_sun_smoke": dict(
        authors="Sun K, Liu J, Ning G.",
        title="Active smoking and risk of metabolic syndrome: a meta-analysis of "
              "prospective studies",
        journal="PLoS One", year="2012", volume="7", issue="10",
        pages="e47791", pmid="23082217", doi="10.1371/journal.pone.0047791"),
    "ls_slagter": dict(
        authors="Slagter SN, van Vliet-Ostaptchouk JV, Vonk JM, et al.",
        title="Associations between smoking, components of metabolic syndrome and "
              "lipoprotein particle size",
        journal="BMC Med", year="2013", volume="11", issue=None,
        pages="195", pmid="24228807", doi="10.1186/1741-7015-11-195"),
    "ls_sun_alcohol": dict(
        authors="Sun K, Ren M, Liu D, et al.",
        title="Alcohol consumption and risk of metabolic syndrome: a meta-analysis of "
              "prospective studies",
        journal="Clin Nutr", year="2014", pubmed_year="2013", volume="33", issue="4",
        pages="596-602", pmid="24315622", doi="10.1016/j.clnu.2013.10.003"),

    # ------------------------------------------------------------------
    # Metals and metabolic syndrome
    # ------------------------------------------------------------------
    "me_hasani": dict(
        authors="Hasani M, Khazdouz M, Sobhani S, et al.",
        title="Association of heavy metals and bio-elements blood level with metabolic "
              "syndrome: a systematic review and meta-analysis of observational studies",
        journal="J Diabetes Metab Disord", year="2024", volume="23", issue="2",
        pages="1719-1752", pmid="39610503", doi="10.1007/s40200-024-01500-9"),
    "me_xu_four": dict(
        authors="Xu P, Liu A, Li F, et al.",
        title="Associations between metabolic syndrome and four heavy metals: a "
              "systematic review and meta-analysis",
        journal="Environ Pollut", year="2021", volume="273", issue=None,
        pages="116480", pmid="33486246", doi="10.1016/j.envpol.2021.116480"),
    "me_lu_cd": dict(
        authors="Lu L, Li Y, Chen C, et al.",
        title="Associations of cadmium exposure with risk of metabolic syndrome and its "
              "individual components: a meta-analysis",
        journal="J Expo Sci Environ Epidemiol", year="2023", pubmed_year="2022",
        volume="33", issue="6", pages="846-854", pmid="35585250",
        doi="10.1038/s41370-022-00444-7"),
    "me_tinkov_cd": dict(
        authors="Tinkov AA, Filippini T, Ajsuvakova OP, et al.",
        title="The role of cadmium in obesity and diabetes",
        journal="Sci Total Environ", year="2017", volume="601-602", issue=None,
        pages="741-755", pmid="28577409", doi="10.1016/j.scitotenv.2017.05.224"),
    "me_wong_mn": dict(
        authors="Wong MMH, Chan KY, Lo K.",
        title="Manganese exposure and metabolic syndrome: a systematic review and "
              "meta-analysis",
        journal="Nutrients", year="2022", volume="14", issue="4",
        pages="825", pmid="35215474", doi="10.3390/nu14040825"),
    "me_lo_mn": dict(
        authors="Lo K, Yang JL, Chen CL, et al.",
        title="Associations between blood and urinary manganese with metabolic syndrome "
              "and its components: cross-sectional analysis of National Health and "
              "Nutrition Examination Survey 2011-2016",
        journal="Sci Total Environ", year="2021", volume="780", issue=None,
        pages="146527", pmid="33774283", doi="10.1016/j.scitotenv.2021.146527"),
    "me_yuan_se": dict(
        authors="Yuan R, Zhang Y, Han J.",
        title="The association of selenium exposure with the odds of metabolic syndrome: "
              "a dose-response meta-analysis",
        journal="BMC Endocr Disord", year="2025", volume="25", issue="1",
        pages="49", pmid="39994710", doi="10.1186/s12902-025-01858-8"),
    "me_steinbrenner_se": dict(
        authors="Steinbrenner H, Duntas LH, Rayman MP.",
        title="The role of selenium in type-2 diabetes mellitus and its metabolic "
              "comorbidities",
        journal="Redox Biol", year="2022", volume="50", issue=None,
        pages="102236", pmid="35144052", doi="10.1016/j.redox.2022.102236"),
    "me_weening_se": dict(
        authors="Weening EH, Al-Mubarak AA, Dokter MM, et al.",
        title="Sexual dimorphism in selenium deficiency is associated with metabolic "
              "syndrome and prevalence of heart disease",
        journal="Cardiovasc Diabetol", year="2023", volume="22", issue="1",
        pages="8", pmid="36635707", doi="10.1186/s12933-022-01730-2"),
    "me_bulka": dict(
        authors="Bulka CM, Persky VW, Daviglus ML, et al.",
        title="Multiple metal exposures and metabolic syndrome: a cross-sectional "
              "analysis of the National Health and Nutrition Examination Survey 2011-2014",
        journal="Environ Res", year="2019", pubmed_year="2018", volume="168", issue=None,
        pages="397-405", pmid="30388496", doi="10.1016/j.envres.2018.10.022"),
    "me_wang_mix": dict(
        authors="Wang X, Mukherjee B, Park SK.",
        title="Associations of cumulative exposure to heavy metal mixtures with obesity "
              "and its comorbidities among U.S. adults in NHANES 2003-2014",
        journal="Environ Int", year="2018", volume="121", issue="Pt 1",
        pages="683-694", pmid="30316184", doi="10.1016/j.envint.2018.09.035"),
    "me_moon": dict(
        authors="Moon SS.",
        title="Additive effect of heavy metals on metabolic syndrome in the Korean "
              "population: the Korea National Health and Nutrition Examination Survey "
              "(KNHANES) 2009-2010",
        journal="Endocrine", year="2014", pubmed_year="2013", volume="46", issue="2",
        pages="263-71", pmid="24065312", doi="10.1007/s12020-013-0061-5"),
    "me_rhee": dict(
        authors="Rhee SY, Hwang YC, Woo JT, et al.",
        title="Blood lead is significantly associated with metabolic syndrome in Korean "
              "adults: an analysis based on the Korea National Health and Nutrition "
              "Examination Survey (KNHANES), 2008",
        journal="Cardiovasc Diabetol", year="2013", volume="12", issue=None,
        pages="9", pmid="23302150", doi="10.1186/1475-2840-12-9"),
    "me_choi_tsh": dict(
        authors="Choi JY, Huh DA, Moon KW.",
        title="Association between blood lead levels and metabolic syndrome considering "
              "the effect of the thyroid-stimulating hormone based on the 2013 Korea "
              "National Health and Nutrition Examination Survey",
        journal="PLoS One", year="2020", volume="15", issue="12",
        pages="e0244821", pmid="33382832", doi="10.1371/journal.pone.0244821"),
    "me_park_pb": dict(
        authors="Park YJ, Jung Y, Oh CU.",
        title="Relations between the blood lead level and metabolic syndrome risk factors",
        journal="Public Health Nurs", year="2019", pubmed_year="2018", volume="36",
        issue="2", pages="118-125", pmid="30345540", doi="10.1111/phn.12548"),
    "me_leekim_2012": dict(
        authors="Lee BK, Kim Y.",
        title="Blood cadmium, mercury, and lead and metabolic syndrome in South Korea: "
              "2005-2010 Korean National Health and Nutrition Examination Survey",
        journal="Am J Ind Med", year="2013", pubmed_year="2012", volume="56", issue="6",
        pages="682-92", pmid="22911659", doi="10.1002/ajim.22107"),
    "me_leekim_cd": dict(
        authors="Lee BK, Kim Y.",
        title="Association of blood cadmium level with metabolic syndrome after "
              "adjustment for confounding by serum ferritin and other factors: 2008-2012 "
              "Korean National Health and Nutrition Examination Survey",
        journal="Biol Trace Elem Res", year="2016", pubmed_year="2015", volume="171",
        issue="1", pages="6-16", pmid="26343361", doi="10.1007/s12011-015-0499-9"),
    "me_parkoh": dict(
        authors="Park Y, Oh CU.",
        title="Association of lead, mercury, and cadmium with metabolic syndrome of "
              "young adults in South Korea: the Korea National Health and Nutrition "
              "Examination Survey (KNHANES) 2016",
        journal="Public Health Nurs", year="2021", pubmed_year="2020", volume="38",
        issue="2", pages="232-238", pmid="33368607", doi="10.1111/phn.12855"),
    "me_eom_hg": dict(
        authors="Eom SY, Choi SH, Ahn SJ, et al.",
        title="Reference levels of blood mercury and association with metabolic syndrome "
              "in Korean adults",
        journal="Int Arch Occup Environ Health", year="2014", pubmed_year="2013",
        volume="87", issue="5", pages="501-13", pmid="23824410",
        doi="10.1007/s00420-013-0891-8"),
    "me_lee_hg": dict(
        authors="Lee K.",
        title="Blood mercury concentration in relation to metabolic and weight "
              "phenotypes using the KNHANES 2011-2013 data",
        journal="Int Arch Occup Environ Health", year="2018", pubmed_year="2017",
        volume="91", issue="2", pages="185-193", pmid="29030686",
        doi="10.1007/s00420-017-1269-0"),
    "me_chung_hg": dict(
        authors="Chung JY, Seo MS, Shim JY, Lee YJ.",
        title="Sex differences in the relationship between blood mercury concentration "
              "and metabolic syndrome risk",
        journal="J Endocrinol Invest", year="2015", pubmed_year="2014", volume="38",
        issue="1", pages="65-71", pmid="25053396", doi="10.1007/s40618-014-0132-3"),
    "me_pang": dict(
        authors="Pang Y, Wang Y, Hao H, et al.",
        title="Associations of multiple serum metals with the risk of metabolic syndrome "
              "among the older population in China based on a community study: a "
              "mediation role of peripheral blood cells",
        journal="Ecotoxicol Environ Saf", year="2024", volume="284", issue=None,
        pages="116981", pmid="39232297", doi="10.1016/j.ecoenv.2024.116981"),
    "me_ngu": dict(
        authors="Ngu YJ, Skalny AV, Tinkov AA, et al.",
        title="Association between essential and non-essential metals, body composition, "
              "and metabolic syndrome in adults",
        journal="Biol Trace Elem Res", year="2022", volume="200", issue="12",
        pages="4903-4915", pmid="34993913", doi="10.1007/s12011-021-03077-3"),
    "me_yao_ml": dict(
        authors="Yao J, Du Z, Yang F, et al.",
        title="The relationship between heavy metals and metabolic syndrome using "
              "machine learning",
        journal="Front Public Health", year="2024", volume="12", issue=None,
        pages="1378041", pmid="38686033", doi="10.3389/fpubh.2024.1378041"),

    # ------------------------------------------------------------------
    # Machine learning with SHAP in clinical prediction
    # ------------------------------------------------------------------
    "ml_lundberg": dict(
        authors="Lundberg SM, Erion G, Chen H, et al.",
        title="From local explanations to global understanding with explainable AI for "
              "trees",
        journal="Nat Mach Intell", year="2020", volume="2", issue="1",
        pages="56-67", pmid="32607472", doi="10.1038/s42256-019-0138-9"),
    "ml_wang_htn": dict(
        authors="Wang M.",
        title="Explainable machine-learning-based cardiovascular disease prediction in "
              "patients with hypertension: algorithm comparison and SHapley Additive "
              "exPlanations (SHAP) analysis",
        journal="Arch Cardiovasc Dis", year="2025", volume="119", issue="4",
        pages="273-282", pmid="41535161", doi="10.1016/j.acvd.2025.09.005"),
    "ml_huang_htn": dict(
        authors="Huang C, Xu J, Qiu H, et al.",
        title="Developing nurse-accessible hypertension prediction tools for low-income "
              "populations: a comparative analysis of machine learning algorithms with "
              "SHAP interpretation",
        journal="Int J Nurs Pract", year="2025", volume="31", issue="5",
        pages="e70060", pmid="40984786", doi="10.1111/ijn.70060"),
    "ml_he_htn": dict(
        authors="He S, Zhong X, Chen G, et al.",
        title="U-shaped association between serum chloride and hypertension risk with "
              "nadir around 103 mmol/L: insights from regression and interpretable "
              "machine learning (XGBoost/SHAP) using NHANES 2017-2018",
        journal="Front Physiol", year="2025", volume="16", issue=None,
        pages="1612895", pmid="40630395", doi="10.3389/fphys.2025.1612895"),
    "ml_yan_ad": dict(
        authors="Yan G, Wu S, Qian Q.",
        title="Early diagnosis of Alzheimer's disease using machine learning and blood "
              "biomarkers",
        journal="BMC Med Inform Decis Mak", year="2025", volume="26", issue="1",
        pages="10", pmid="41361420", doi="10.1186/s12911-025-03296-x"),
    "ml_zhang_cog": dict(
        authors="Zhang J, Li W, Wen S, et al.",
        title="An efficient non-invasive model for predicting cognitive impairment based "
              "on comprehensive geriatric assessment: machine learning and SHAP analysis",
        journal="J Alzheimers Dis Rep", year="2026", volume="10", issue=None,
        pages="25424823261415843", pmid="41929978",
        doi="10.1177/25424823261415843"),
    "ml_liu_dr": dict(
        authors="Liu X, Wu S, Yang Y, et al.",
        title="Development of machine learning predictive model for type 2 diabetic "
              "retinopathy using the triglyceride-glucose index explained by SHAP method",
        journal="Front Endocrinol (Lausanne)", year="2025", volume="16", issue=None,
        pages="1631647", pmid="41293734", doi="10.3389/fendo.2025.1631647"),
    "ml_wan_dr": dict(
        authors="Wan X, Zhang R, Wang Y, et al.",
        title="Predicting diabetic retinopathy based on routine laboratory tests by "
              "machine learning algorithms",
        journal="Eur J Med Res", year="2025", volume="30", issue="1",
        pages="183", pmid="40102923", doi="10.1186/s40001-025-02442-5"),
    "ml_chen_pi": dict(
        authors="Chen B, Zhou J, Chen S, et al.",
        title="Interpretable machine learning prediction model for predicting mortality "
              "risk of ICU patients with pressure ulcers based on the Braden Scale: a "
              "clinical study based on MIMIC-IV",
        journal="J Clin Nurs", year="2025", volume="34", issue="8",
        pages="3353-3369", pmid="40506399", doi="10.1111/jocn.17860"),
    "ml_liang_covid": dict(
        authors="Liang Y, Xie S, Zheng X, et al.",
        title="Predicting higher risk factors for COVID-19 short-term reinfection in "
              "patients with rheumatic diseases: a modeling study based on XGBoost "
              "algorithm",
        journal="J Transl Med", year="2024", volume="22", issue="1",
        pages="1144", pmid="39719617", doi="10.1186/s12967-024-05982-2"),
    "ml_xue_ckm": dict(
        authors="Xue M, Chang H, Wang BC, et al.",
        title="Development of an explainable machine learning model for "
              "cardiovascular-kidney-metabolic syndrome prediction based on dietary "
              "antioxidants in a national population",
        journal="J Vasc Res", year="2025", volume="63", issue="2",
        pages="76-102", pmid="41196834", doi="10.1159/000549326"),
    "ml_gu_circ": dict(
        authors="Gu Y, Ye X, Zhao W, et al.",
        title="The circadian syndrome is a better predictor for psoriasis than the "
              "metabolic syndrome via an explainable machine learning method: the NHANES "
              "survey during 2005-2006 and 2009-2014",
        journal="Front Endocrinol (Lausanne)", year="2024", volume="15", issue=None,
        pages="1379130", pmid="38988999", doi="10.3389/fendo.2024.1379130"),

    # ------------------------------------------------------------------
    # Mendelian randomization: concept and reporting
    # ------------------------------------------------------------------
    "mr_dsmith2003": dict(
        authors="Davey Smith G, Ebrahim S.",
        title="'Mendelian randomization': can genetic epidemiology contribute to "
              "understanding environmental determinants of disease?",
        journal="Int J Epidemiol", year="2003", volume="32", issue="1",
        pages="1-22", pmid="12689998", doi="10.1093/ije/dyg070"),
    "mr_lawlor2008": dict(
        authors="Lawlor DA, Harbord RM, Sterne JA, et al.",
        title="Mendelian randomization: using genes as instruments for making causal "
              "inferences in epidemiology",
        journal="Stat Med", year="2008", volume="27", issue="8",
        pages="1133-63", pmid="17886233", doi="10.1002/sim.3034"),
    "mr_emdin2017": dict(
        authors="Emdin CA, Khera AV, Kathiresan S.",
        title="Mendelian randomization",
        journal="JAMA", year="2017", volume="318", issue="19",
        pages="1925-1926", pmid="29164242", doi="10.1001/jama.2017.17219"),
    "mr_ference2012": dict(
        authors="Ference BA, Yoo W, Alesh I, et al.",
        title="Effect of long-term exposure to lower low-density lipoprotein cholesterol "
              "beginning early in life on the risk of coronary heart disease: a Mendelian "
              "randomization analysis",
        journal="J Am Coll Cardiol", year="2012", volume="60", issue="25",
        pages="2631-9", pmid="23083789", doi="10.1016/j.jacc.2012.09.017"),
    "mr_davies2018": dict(
        authors="Davies NM, Holmes MV, Davey Smith G.",
        title="Reading Mendelian randomisation studies: a guide, glossary, and checklist "
              "for clinicians",
        journal="BMJ", year="2018", volume="362", issue=None,
        pages="k601", pmid="30002074", doi="10.1136/bmj.k601"),
    "mr_strobe": dict(
        authors="Skrivankova VW, Richmond RC, Woolf BAR, et al.",
        title="Strengthening the reporting of observational studies in epidemiology "
              "using Mendelian randomization: the STROBE-MR statement",
        journal="JAMA", year="2021", volume="326", issue="16",
        pages="1614-1621", pmid="34698778", doi="10.1001/jama.2021.18236"),

    # ------------------------------------------------------------------
    # Mendelian randomization: estimators and diagnostics
    # ------------------------------------------------------------------
    "mr_ivw": dict(
        authors="Burgess S, Butterworth A, Thompson SG.",
        title="Mendelian randomization analysis with multiple genetic variants using "
              "summarized data",
        journal="Genet Epidemiol", year="2013", volume="37", issue="7",
        pages="658-65", pmid="24114802", doi="10.1002/gepi.21758"),
    "mr_egger": dict(
        authors="Bowden J, Davey Smith G, Burgess S.",
        title="Mendelian randomization with invalid instruments: effect estimation and "
              "bias detection through Egger regression",
        journal="Int J Epidemiol", year="2015", volume="44", issue="2",
        pages="512-25", pmid="26050253", doi="10.1093/ije/dyv080"),
    "mr_wmedian": dict(
        authors="Bowden J, Davey Smith G, Haycock PC, Burgess S.",
        title="Consistent estimation in Mendelian randomization with some invalid "
              "instruments using a weighted median estimator",
        journal="Genet Epidemiol", year="2016", volume="40", issue="4",
        pages="304-14", pmid="27061298", doi="10.1002/gepi.21965"),
    "mr_wmode": dict(
        authors="Hartwig FP, Davey Smith G, Bowden J.",
        title="Robust inference in summary data Mendelian randomization via the zero "
              "modal pleiotropy assumption",
        journal="Int J Epidemiol", year="2017", volume="46", issue="6",
        pages="1985-1998", pmid="29040600", doi="10.1093/ije/dyx102"),
    "mr_ruckerq": dict(
        authors="Bowden J, Del Greco M F, Minelli C, et al.",
        title="A framework for the investigation of pleiotropy in two-sample summary "
              "data Mendelian randomization",
        journal="Stat Med", year="2017", volume="36", issue="11",
        pages="1783-1802", pmid="28114746", doi="10.1002/sim.7221"),
    "mr_isq": dict(
        authors="Bowden J, Del Greco M F, Minelli C, et al.",
        title="Assessing the suitability of summary data for two-sample Mendelian "
              "randomization analyses using MR-Egger regression: the role of the I2 "
              "statistic",
        journal="Int J Epidemiol", year="2016", volume="45", issue="6",
        pages="1961-1974", pmid="27616674", doi="10.1093/ije/dyw220"),
    "mr_presso": dict(
        authors="Verbanck M, Chen CY, Neale B, Do R.",
        title="Detection of widespread horizontal pleiotropy in causal relationships "
              "inferred from Mendelian randomization between complex traits and diseases",
        journal="Nat Genet", year="2018", volume="50", issue="5",
        pages="693-698", pmid="29686387", doi="10.1038/s41588-018-0099-7"),
    "mr_radial": dict(
        authors="Bowden J, Spiller W, Del Greco M F, et al.",
        title="Improving the visualization, interpretation and analysis of two-sample "
              "summary data Mendelian randomization via the radial plot and radial "
              "regression",
        journal="Int J Epidemiol", year="2018", volume="47", issue="4",
        pages="1264-1278", pmid="29961852", doi="10.1093/ije/dyy101"),
    "mr_lasso": dict(
        authors="Rees JMB, Wood AM, Dudbridge F, Burgess S.",
        title="Robust methods in Mendelian randomization via penalization of "
              "heterogeneous causal estimates",
        journal="PLoS One", year="2019", volume="14", issue="9",
        pages="e0222362", pmid="31545794", doi="10.1371/journal.pone.0222362"),
    "mr_cml": dict(
        authors="Xue H, Shen X, Pan W.",
        title="Constrained maximum likelihood-based Mendelian randomization robust to "
              "both correlated and uncorrelated pleiotropic effects",
        journal="Am J Hum Genet", year="2021", volume="108", issue="7",
        pages="1251-1269", pmid="34214446", doi="10.1016/j.ajhg.2021.05.014"),
    "mr_contam": dict(
        authors="Burgess S, Foley CN, Allara E, et al.",
        title="A robust and efficient method for Mendelian randomization with hundreds "
              "of genetic variants",
        journal="Nat Commun", year="2020", volume="11", issue="1",
        pages="376", pmid="31953392", doi="10.1038/s41467-019-14156-4"),
    "mr_steiger": dict(
        authors="Hemani G, Tilling K, Davey Smith G.",
        title="Orienting the causal relationship between imprecisely measured traits "
              "using GWAS summary data",
        journal="PLoS Genet", year="2017", volume="13", issue="11",
        pages="e1007081", pmid="29149188", doi="10.1371/journal.pgen.1007081"),
    "mr_weakinst": dict(
        authors="Burgess S, Thompson SG; CRP CHD Genetics Collaboration.",
        title="Avoiding bias from weak instruments in Mendelian randomization studies",
        journal="Int J Epidemiol", year="2011", volume="40", issue="3",
        pages="755-64", pmid="21414999", doi="10.1093/ije/dyr036"),
    "mr_twosample": dict(
        authors="Pierce BL, Burgess S.",
        title="Efficient design for Mendelian randomization studies: subsample and "
              "2-sample instrumental variable estimators",
        journal="Am J Epidemiol", year="2013", volume="178", issue="7",
        pages="1177-84", pmid="23863760", doi="10.1093/aje/kwt084"),
    "mr_phenoscanner": dict(
        authors="Kamat MA, Blackshaw JA, Young R, et al.",
        title="PhenoScanner V2: an expanded tool for searching human genotype-phenotype "
              "associations",
        journal="Bioinformatics", year="2019", volume="35", issue="22",
        pages="4851-4853", pmid="31233103", doi="10.1093/bioinformatics/btz469"),
    "mr_mrbase": dict(
        authors="Hemani G, Zheng J, Elsworth B, et al.",
        title="The MR-Base platform supports systematic causal inference across the "
              "human phenome",
        journal="Elife", year="2018", volume="7", issue=None,
        pages="e34408", pmid="29846171", doi="10.7554/eLife.34408"),

    # ------------------------------------------------------------------
    # Genome-wide association studies used as instrument sources
    # ------------------------------------------------------------------
    "gw_giant": dict(
        authors="Locke AE, Kahali B, Berndt SI, et al.",
        title="Genetic studies of body mass index yield new insights for obesity biology",
        journal="Nature", year="2015", volume="518", issue="7538",
        pages="197-206", pmid="25673413", doi="10.1038/nature14177"),
    "gw_yengo": dict(
        authors="Yengo L, Sidorenko J, Kemper KE, et al.",
        title="Meta-analysis of genome-wide association studies for height and body mass "
              "index in ~700000 individuals of European ancestry",
        journal="Hum Mol Genet", year="2018", volume="27", issue="20",
        pages="3641-3649", pmid="30124842", doi="10.1093/hmg/ddy271"),
    "gw_gscan": dict(
        authors="Liu M, Jiang Y, Wedow R, et al.",
        title="Association studies of up to 1.2 million individuals yield new insights "
              "into the genetic etiology of tobacco and alcohol use",
        journal="Nat Genet", year="2019", volume="51", issue="2",
        pages="237-244", pmid="30643251", doi="10.1038/s41588-018-0307-5"),
    "gw_doherty": dict(
        authors="Doherty A, Smith-Byrne K, Ferreira T, et al.",
        title="GWAS identifies 14 loci for device-measured physical activity and sleep "
              "duration",
        journal="Nat Commun", year="2018", volume="9", issue="1",
        pages="5257", pmid="30531941", doi="10.1038/s41467-018-07743-4"),
    "gw_ng_metals": dict(
        authors="Ng E, Lind PM, Lindgren C, et al.",
        title="Genome-wide association study of toxic metals and trace elements reveals "
              "novel associations",
        journal="Hum Mol Genet", year="2015", volume="24", issue="16",
        pages="4739-45", pmid="26025379", doi="10.1093/hmg/ddv190"),
    "gw_moksnes": dict(
        authors="Moksnes MR, Hansen AF, Wolford BN, et al.",
        title="A genome-wide association study provides insights into the genetic "
              "etiology of 57 essential and non-essential trace elements in humans",
        journal="Commun Biol", year="2024", volume="7", issue="1",
        pages="432", pmid="38594418", doi="10.1038/s42003-024-06101-z"),
    "gw_borne_cd": dict(
        authors="Borné Y, Söderholm M, Barregard L, Engström G.",
        title="Genome wide association study identifies two loci associated with cadmium "
              "in erythrocytes among never-smokers",
        journal="Hum Mol Genet", year="2016", volume="25", issue="11",
        pages="2342-2348", pmid="27005419", doi="10.1093/hmg/ddw083"),
    "gw_cornelis_se": dict(
        authors="Cornelis MC, Fornage M, Foy M, et al.",
        title="Genome-wide association study of selenium concentrations",
        journal="Hum Mol Genet", year="2015", pubmed_year="2014", volume="24", issue="5",
        pages="1469-77", pmid="25343990", doi="10.1093/hmg/ddu546"),
    "gw_evans_metals": dict(
        authors="Evans DM, Zhu G, Dy V, et al.",
        title="Genome-wide association study identifies loci affecting blood copper, "
              "selenium and zinc",
        journal="Hum Mol Genet", year="2013", volume="22", issue="19",
        pages="3998-4006", pmid="23720494", doi="10.1093/hmg/ddt239"),
    "gw_dack_hg": dict(
        authors="Dack K, Bustamante M, Taylor CM, et al.",
        title="Genome-wide association study of blood mercury in European pregnant women "
              "and children",
        journal="Genes (Basel)", year="2023", volume="14", issue="12",
        pages="2123", pmid="38136945", doi="10.3390/genes14122123"),
}


def format_reference(key: str, number: int) -> str:
    """Render one numbered reference in a Vancouver-style single line."""
    r = REFS[key]
    bits = [f"{number}. {r['authors']} {r['title']}. {r['journal']}. {r['year']}"]
    vol = r.get("volume")
    if vol:
        bits.append(f";{vol}")
        if r.get("issue"):
            bits.append(f"({r['issue']})")
    if r.get("pages"):
        bits.append(f":{r['pages']}")
    line = "".join(bits) + "."
    if r.get("doi"):
        line += f" doi:{r['doi']}."
    if r.get("pmid"):
        line += f" PMID:{r['pmid']}."
    return line


def collapse(nums: list[int]) -> str:
    """[1,2,3,7] -> '1-3,7'.

    A run of exactly two is written out in full ('4,5', never '4-5'), which is the usual
    house style and also the case an earlier version of this function got wrong: it
    emitted only the first of the pair, silently dropping the second citation.
    """
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        run = nums[i:j + 1]
        if len(run) == 1:
            out.append(str(run[0]))
        elif len(run) == 2:
            out.append(f"{run[0]},{run[1]}")
        else:
            out.append(f"{run[0]}-{run[-1]}")
        i = j + 1
    return ",".join(out)


def _self_test() -> None:
    cases = {
        (1,): "1",
        (1, 2): "1,2",
        (1, 2, 3): "1-3",
        (1, 2, 3, 7): "1-3,7",
        (4, 5, 9, 10, 11): "4,5,9-11",
        (73, 74): "73,74",
        (3, 1, 2, 2): "1-3",
    }
    for nums, want in cases.items():
        got = collapse(list(nums))
        assert got == want, f"collapse({list(nums)}) == {got!r}, expected {want!r}"
    total = sum(len(collapse([n]).split(",")) for n in range(1, 4))
    assert total == 3
    print("collapse() self-test passed")


if __name__ == "__main__":
    _self_test()
    print(f"{len(REFS)} verified references in the proposal bank")
    seen: dict[str, str] = {}
    for k, r in REFS.items():
        for field in ("pmid", "doi"):
            v = r.get(field)
            if v:
                if v in seen:
                    raise SystemExit(f"duplicate {field} {v}: {k} and {seen[v]}")
                seen[v] = k
    print("no duplicate PMIDs or DOIs")
