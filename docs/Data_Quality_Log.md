# DEMO_J

Date:
2026-07-13

Reviewer:
Ifthakhar Rashid

Observations

- All selected variables exist.
- Merge key (SEQN) verified.
- RIDEXPRG contains many missing values because pregnancy is only applicable to eligible female participants.
- Survey design variables retained but excluded from ML features.

Decision

Dataset approved for merging.

------------------------------------------------------------

# BMX_J

Date:
2026-07-13

Reviewer:
Ifthakhar Rashid

Observations

- All selected variables verified.
- Merge key (SEQN) verified.
- NHANES already provides BMI (BMXBMI); no need to recalculate.
- Weight and Height retained because they may be useful for future feature engineering and validation.
- Waist circumference has expected missing values because it was measured only for eligible participants.
- Comment variables (BMIWT, BMIHT, BMIWAIST, etc.) excluded because they describe measurement conditions rather than participant characteristics.
- Body Measures Status Code (BMDSTATS) excluded because it is administrative.

Decision

Dataset approved for merging.

------------------------------------------------------------

# GHB_J

Date:
2026-07-13

Reviewer:
Ifthakhar Rashid

Observations

- All selected variables verified.
- Merge key (SEQN) verified.
- HbA1c (LBXGH) available for examined participants aged 12 years and older.
- Missing values are expected due to laboratory non-response.
- HbA1c selected as the primary laboratory biomarker for diabetes label generation.

Decision

Dataset approved for merging.

------------------------------------------------------------

# GLU_J

Date:
2026-07-13

Reviewer:
Ifthakhar Rashid

Observations

- All selected variables verified successfully.
- SEQN merge key matches DEMO_J.
- Fasting glucose (LBXGLU) is available only for the fasting subsample.
- WTSAF2YR retained for future survey-weighted statistical analyses.
- mmol/L variable (LBDGLUSI) excluded because it is a direct conversion of LBXGLU and adds no new information.

Decision

Dataset approved for merging and diabetes label generation.

------------------------------------------------------------

# BPX_J

Date:
2026-07-13

Reviewer:
Ifthakhar Rashid

Observations

- All selected variables verified.
- Merge key (SEQN) confirmed.
- Multiple blood pressure readings available for most participants.
- Fourth reading available only when additional measurement was required.
- Blood pressure averages will be computed during label generation.

Decision

Dataset approved for merging and hypertension label generation.

## DEMO_J re-verification – 2026-08-01
- Re-verified DEMO_J for expanded 7-disease project.
- All previously selected variables confirmed.
- Added RIDSTATR (exam_status) as administrative filter.
- Added RIDEXMON (exam_season) as optional experimental feature.
- Excluded veteran status, citizenship, language, proxy, interpreter, and detailed household composition variables due to U.S./survey-specificity, aligning with clinical generalisability principle.
- Dataset re-exported: data/interim/DEMO_verified.csv

## BMX_J re-verification – 2026-08-01
- Re-verified BMX_J for expanded 7-disease project.
- All previously selected anthropometric variables confirmed as universal clinical predictors.
- Updated “Diseases” column: BMI, waist circumference, hip circumference now mapped to All (were previously Diabetes, Hypertension, CKD only).
- Added BMDSTATS (bmx_status) as administrative filter variable.
- Excluded all comment variables (BMIWT, BMIHT, etc.) and infant-only measurements (recumbent length, head circumference) – survey-specific or age-irrelevant.
- Dataset re-exported: data/interim/BMX_verified.csv

## GHB_J re-verification – 2026-08-01
- Re-verified GHB_J for 7-disease scope.
- No new variables added. 
- Updated GHB-002 role: HbA1c is now a predictor for all diseases except diabetes (where it remains a label variable only).
- Dataset re-exported: data/interim/GHB_verified.csv

## GLU_J re-verification – 2026-08-01
- Re-verified GLU_J for 7-disease scope.
- Excluded LBDGLUSI (mmol/L) – redundant unit conversion of LBXGLU.
- Updated GLU-002 role: Fasting glucose is now a predictor for all diseases except diabetes (where it remains a label variable only).
- Dataset re-exported: data/interim/GLU_verified.csv

## BPX_J re-verification – 2026-08-01
- Re-verified BPX_J for 7-disease scope.
- Added BPXPLS (pulse_rate) as core predictor – universal vital sign.
- Added BPXPULS (pulse_regularity) as recommended predictor for CVD risk.
- Updated all systolic and diastolic readings (BPXSY1–4, BPXDI1–4): ML role now LABEL (Hypertension) + PRED (All other diseases); Diseases set to All.
- Excluded PEASCCT1, BPXCHR (child HR), BPAARM, BPACSZ, BPXPTY, BPXML1, and all enhancement flags (BPAEN1–4) – survey‑specific artefacts or irrelevant age groups.
- Dataset re-exported: data/interim/BPX_verified.csv

## BIOPRO_J re-verification – 2026-08-01
- Re-verified BIOPRO_J for 7-disease scope.
- Excluded LBXSGL, LBXSCH, LBXSTR (non-reference duplicates; reference methods in GLU_J, TCHOL_J, TRIGLY_J will be used).
- Excluded all SI conversion variables (LBDxxxSI) and laboratory comment codes.
- Updated all 21 analyte features: Diseases column set to “All”; ML Role adjusted for creatinine (LABEL CKD), ALT, AST, GGT (LABEL Liver Disease) to prevent predictor leakage.
- Added ALT, AST, GGT, and BMI to Disease Label Variables for Liver Disease / NAFLD Risk.
- Dataset re-exported: data/interim/BIOPRO_verified.csv

## DIQ_J verification – 2026-08-06
- Dataset loaded: DIQ_J.xpt, n=8897
- Variables verified: SEQN, DIQ010, DIQ160, DID040, DIQ080 – all present.
- DIQ010 distribution: Yes=893, No=7816, Borderline=184. Consistent with expected diabetes prevalence.
- DID040: Age at diagnosis range 1–80; 666 (<1 yr) present. Will derive duration as (current age – DID040), with non‑diabetics = 0.
- DIQ160: Prediabetes awareness (age ≥12) – Yes=597, No=5089. For participants <12, not asked; will be set to 0.
- DIQ080: Retinopathy only asked of diabetics (n=893). Will be set to 0 for non‑diabetics.
- Excluded all subjective risk perception and diabetes care process variables per generalisability filter.
- Dataset saved as data/interim/DIQ_verified.csv

## KIQ_U_J verification – 2026-08-06
- Dataset loaded: KIQ_U_J.xpt, n=5569 (ages 20+)
- Variables verified: SEQN, KIQ022, KIQ025, KIQ026 – all present.
- KIQ022: Yes=223, No=5337, Don't know=9. 
- KIQ025 (asked if KIQ022=1): Yes=19, No=204. For others (n=5346), value set to 0.
- KIQ026: Yes=554, No=5004, Don't know=11. Mode imputation will be applied for missing/refused.
- Excluded all incontinence, nocturia, and stone passage variables (KIQ029, KIQ005–KIQ480) per generalisability filter.
- Dataset saved as data/interim/KIQ_U_verified.csv

## SMQ_J verification – 2026-08-06
- Dataset loaded: SMQ_J.xpt, n=6724 (ages 12+)
- Variables verified: SEQN, SMQ020, SMD030, SMQ040, SMD650, SMD057, SMQ890, SMQ900, SMQ910 – all present.
- SMQ020 (18+): Yes=2359, No=3497. For ages 12‑17, not asked (NaN) – will filter to adults for modeling.
- SMD030: Age range 7‑76; 0 for never smoked regularly (62). Set to 0 for never smokers.
- SMQ040: Every day=805, Some days=216, Not at all=1338. For never smokers, impute as Not at all.
- SMD650: median ~10 cigs/day; SMD057: median ~15 cigs/day. Both variables set to 0 for inapplicable groups.
- SMQ890, SMQ900, SMQ910 asked only 18+; for youth, impute 0.
- Excluded all brand, tar, nicotine, CO, quit attempt details per generalisability filter.
- Dataset saved as data/interim/SMQ_verified.csv

## ALQ_J verification – 2026-08-06
- Dataset loaded: ALQ_J.xpt, n=5533 (ages 18+ only; ages 12‑17 in restricted file)
- Variables verified: SEQN, ALQ111, ALQ121, ALQ130, ALQ142, ALQ151 – all present.
- ALQ111 (Ever drinker): Yes=4545, No=585.
- ALQ121 (Frequency): Mode=10 (1‑2 times/year), 1049 never in last year.
- ALQ130 (Avg drinks/day): Range 1‑15; median ~2. Set to 0 for non‑drinkers.
- ALQ142 (Binge frequency): 2051 never; for those with ALQ121=0, value set to 0.
- ALQ151 (Lifetime heavy): Yes=680, No=3857.
- Excluded ALQ270, ALQ280, ALQ290, ALQ170 (redundant binge/extreme measures) per clinical generalisability.
- Dataset saved as data/interim/ALQ_verified.csv

## PAQ_J verification – 2026-08-06
- Dataset loaded: PAQ_J.xpt, n=5856 (ages 18+)
- Variables verified: SEQN, PAQ605, PAQ620, PAQ635, PAQ650, PAQ665, PAD680 – all present.
- PAQ605 (Vigorous work): Yes=1389, No=4461.
- PAQ620 (Moderate work): Yes=2439, No=3412.
- PAQ635 (Active transport): Yes=1439, No=4417.
- PAQ650 (Vigorous recreation): Yes=1434, No=4422.
- PAQ665 (Moderate recreation): Yes=2308, No=3548.
- PAD680 (Sedentary time): median 360 min (6 hours). 35 DK responses imputed with median.
- Excluded raw days/minutes per domain (10 variables) – to be used only in feature engineering for derived MET-minutes.
- Dataset saved as data/interim/PAQ_verified.csv

## DBQ_J verification – 2026-08-06
- Dataset loaded: DBQ_J.xpt, n=9254 (all ages)
- Variables verified: SEQN, DBQ700, DBQ197, DBD895, DBD900, DBD905, DBD910 – all present.
- DBQ700 (16+): Excellent=471, Very Good=1241, Good=2411, Fair=1619, Poor=417.
- DBQ197 (1+): Never=1364, Rarely=1496, Sometimes=2357, Often=3649, Varied=30. Recoded Varied to mode.
- DBD895 (meals away from home): 0–21 meals; 5 coded 5555 (>21) set to 22; 1 refused excluded.
- DBD900 (fast food meals): 0–21; 2 coded 5555 set to 22.
- DBD905 (ready‑to‑eat foods): 0–90; 1 coded 6666 (>90) set to 91.
- DBD910 (frozen meals/pizza): 0–90; 2 coded 6666 set to 91.
- Excluded all age‑limited and U.S.‑specific variables per generalisability filter.
- Dataset saved as data/interim/DBQ_verified.csv

## MCQ_J verification – 2026-08-06
- Dataset loaded: MCQ_J.xpt, n=8897 (ages 1+)
- Variables verified: SEQN, MCQ160b–f, MCQ160l, MCQ300c, MCQ300a – all present.
- CVD label variables (20+): CHF=201, CHD=265, Angina=161, MI=270, Stroke=273. Missing for <20 set to 0.
- Liver condition (20+): Yes=294; missing for <20 set to 0.
- Family history diabetes (20+): Yes=2624; family early MI (20+): Yes=709. Missing for <20 set to 0.
- All refusal/don't know responses coded to 0 (no condition).
- Excluded all other medical conditions, abdominal pain, gallstones, health advice/behaviors per generalisability filter.
- Dataset saved as data/interim/MCQ_verified.csv

## BPQ_J verification – 2026-08-06
- Dataset loaded: BPQ_J.xpt, n=6161 (ages 16+)
- Variables verified: SEQN, BPQ020 – both present.
- BPQ020: Yes=2137, No=4014. Refused/Don't know (10) excluded from label.
- All other variables (repeat visits, age, medication, cholesterol) excluded per generalisability.
- Dataset saved as data/interim/BPQ_verified.csv

## CBC_J verification – 2026-08-06
- Dataset loaded: CBC_J.xpt, n=8366 (ages 1+)
- Variables verified: SEQN, LBXWBCSI, LBXHGB, LBXRDW, LBXPLTSI, LBXLYPCT, LBXNEPCT – all present.
- WBC: median ~6.8 x10³/µL; Hemoglobin: median ~13.7 g/dL.
- RDW: median ~13.4%; Platelets: median ~242 x10³/µL.
- Lymphocyte% and Neutrophil% will be used to derive NLR during feature engineering.
- All other CBC parameters (RBC, HCT, MCV, MCH, MCHC, MPV, basophil%, eosinophil%, absolute counts, NRBC, LC variables) excluded.
- Dataset saved as data/interim/CBC_verified.csv

## HDL_J verification – 2026-08-06
- Dataset loaded: HDL_J.xpt, n=7435 (ages 6+)
- Variables verified: SEQN, LBDHDD – both present.
- HDL: range 10‑189 mg/dL; median ~51 mg/dL.
- LBDHDDSI (mmol/L) excluded – redundant unit conversion.
- Dataset saved as data/interim/HDL_verified.csv

## TCHOL_J verification – 2026-08-06
- Dataset loaded: TCHOL_J.xpt, n=7435 (ages 6+)
- Variables verified: SEQN, LBXTC – both present.
- Total cholesterol: range 76‑446 mg/dL; median ~186 mg/dL.
- LBDTCSI (mmol/L) excluded – redundant unit conversion.
- Dataset saved as data/interim/TCHOL_verified.csv

## TRIGLY_J verification – 2026-08-06
- Dataset loaded: TRIGLY_J.xpt, n=3036 (ages 12+, morning session)
- Variables verified: SEQN, LBXTR – both present.
- Triglycerides: range 10‑2684 mg/dL; median ~98 mg/dL.
- All LDL‑C derived variables and SI conversion excluded – redundant/collinear.
- Fasting weight (WTSAF2YR) already exists from GLU_J; not duplicated.
- Dataset saved as data/interim/TRIGLY_verified.csv

## INS_J verification – 2026-08-06
- Dataset loaded: INS_J.xpt, n=3036 (ages 12+, morning session)
- Variables verified: SEQN, LBXIN – both present.
- Fasting insulin: range 0.71‑485.1 µU/mL; median ~9.8 µU/mL.
- SI conversion (LBDINSI) and comment code (LBDINLC) excluded.
- Fasting weight (WTSAF2YR) already exists from GLU_J/TRIGLY_J; not duplicated.
- Dataset saved as data/interim/INS_verified.csv

## HSCRP_J verification – 2026-08-06
- Dataset loaded: HSCRP_J.xpt, n=8366 (ages 1+)
- Variables verified: SEQN, LBXHSCRP – both present.
- hs‑CRP: range 0.11‑182.82 mg/L; median ~1.6 mg/L.
- Comment code (LBDHRPLC) excluded.
- Dataset saved as data/interim/HSCRP_verified.csv

## ALB_CR_J verification – 2026-08-06
- Dataset loaded: ALB_CR_J.xpt, n=7936 (ages 3+)
- Variables verified: SEQN, URDACT – both present.
- UACR: range 0.27‑11676.92 mg/g; median ~6.8 mg/g.
- Raw urine albumin (URXUMA) and creatinine (URXUCR) excluded – redundant.
- Comment codes and alternative unit variables excluded.
- Dataset saved as data/interim/ALB_CR_verified.csv

## HSQ_J verification – 2026-08-06
- Dataset loaded: HSQ_J.xpt, n=8366 (ages 1+)
- Variables verified: SEQN, HSD010 – both present.
- HSD010 (12+): Excellent=619, Very good=1544, Good=2454, Fair=1175, Poor=172.
- All other variables excluded per generalisability filter.
- Dataset saved as data/interim/HSQ_verified.csv

## 2015-2016 cycle verification – 2026-08-10
- Downloaded all 22 NHANES 2015-2016 components (suffix `_I`) into `data/raw/NHANES_2015_2016/`.
- Verified all components with the same protocol as 2017-2018.
- All variable names and value encodings matched the 2017-2018 cycle.
- Added cycle marker: `cycle = 0` for all 2015-2016 rows.
- Saved verified CSVs to `data/interim/{component}_2015_verified.csv`.
- Final merged 2015 shape: (2598, 546). Duplicate SEQN: 0.

## 2013-2014 cycle verification – 2026-09-01
- Downloaded 21 publicly available NHANES 2013-2014 components (suffix `_H`).
- **HSCRP_H.xpt was not publicly released** for this cycle (only available via NCHS RDC).
- All other 21 components verified with the same protocol.
- Variable names matched the 2015-2018 cycles (same variable codes).
- Added cycle marker: `cycle = 0` temporarily (reassigned to 0/1/2 in the combined dataset).
- Saved verified CSVs to `data/interim/{component}_2013_verified.csv`.
- HSCRP for 2013-2014 participants was added as NaN in the combined dataset and imputed later with the median from 2015-2018.

## Combined 2013-2018 dataset – 2026-09-05
- Merged three cycles on `[SEQN, cycle]`.
- Assigned cycle codes: 2013-2014 = 0, 2015-2016 = 1, 2017-2018 = 2.
- Final combined master shape: (5139, 118) after curated feature selection.
- Participants per disease label (after exclusions):
  - Diabetes: 5082 rows (1133 positive)
  - Hypertension: 5139 rows (2893 positive)
  - CKD: 5139 rows (473 positive)
  - Dyslipidemia: 4837 rows (2844 positive)
  - CVD: 5139 rows (633 positive)
  - Metabolic Syndrome: 4315 rows (1613 positive)
  - Liver Disease: 5126 rows (730 positive)
- No duplicate SEQN+cycle after merges.

## Leakage audit – 2026-09-08
- **Round 1:** Checked `feature_names.json` against `leakage_map.json`. No explicit leakage found.
- **Round 2:** Discovered uncurated columns were included in the predictor set. Fix: added curated feature filter (`SELECTED_FEATURES`). Model AUROC dropped from ~1.0 to ~0.85–0.95.
- **Round 3:** Found duplicate lab variables still present:
  - `LBXSGL` (serum glucose from BIOPRO) → leaked **diabetes** label
  - `LBXSCH` (serum cholesterol) → leaked **dyslipidemia** label
  - `LBXSTR` (serum triglycerides) → leaked **dyslipidemia** label
  - `met_syndrome_count` → leaked **metabolic syndrome** label
  - `eGFR_stage` → leaked **CKD** label
  - `ASCVD_risk` → leaked **hypertension** and **dyslipidemia** labels
  - `obesity_class` → leaked **liver disease** label
- All the above were added to the leakage map and dropped before model fitting.
- Post-audit high-correlation check (>0.95 with any label): **passed, no features flagged**.
- This leakage removal is documented in `Research_Decision_Log.md` and `Experiment_Log.md`.

## External validation on 2021-2023 – 2026-09-09
- Downloaded all 22 NHANES 2021-2023 components (suffix `_L`) into `data/raw/NHANES_2021_2023/`.
- Variable renames required:
  - `BPXO_L.xpt` used `BPXOSY1..4`, `BPXODI1..4`, `BPXOPLS`, `BPXOPULS` → renamed to BPX equivalents.
  - `TRIGLY_L.xpt` used `LBXTLG` → renamed to `LBXTR`.
- All seven disease labels produced valid prevalence.
- Final external cohort size: 3425 participants with complete data across all seven diseases.
- External metrics saved to `data/external_validation/external_validation_metrics.csv`.

## Final model state – 2026-09-10
- Retrained final models on the combined 2013-2018 dataset using the tuned hyperparameters.
- External validation completed on 2021-2023 without retraining.
- All artifacts saved:
  - `models/{disease}_final_model.pkl`
  - `models/{disease}_scaler.pkl`
  - `models/{disease}_feature_importance.json`
  - `models/tuned_model_metrics.csv`
  - `models/external_validation_metrics.csv`
  - `models/threshold_optimization.csv`
- No data loss occurred during any leakage fix. Only columns (features) were removed, not participants (rows).