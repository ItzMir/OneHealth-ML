# Target Diseases — OneHealth-ML

Seven chronic non-communicable diseases were selected for simultaneous prediction.
Each is defined from routine NHANES variables using internationally accepted clinical guidelines.

---

## 1. Type 2 Diabetes Mellitus

**Guideline:** American Diabetes Association (ADA) Standards of Care 2024.

**Definition:**
- HbA1c ≥ 6.5%, OR
- Fasting plasma glucose ≥ 126 mg/dL, OR
- Self-reported doctor diagnosis (DIQ010 = 1).
- Pregnant participants (RIDEXPRG = 1) are excluded.

**Variables used for the label:**
- `LBXGH` (HbA1c, GHB_J)
- `LBXGLU` (fasting glucose, GLU_J)
- `DIQ010` (diagnosis, DIQ_J)
- `RIDEXPRG` (pregnancy, DEMO_J)

**Clinical justification:**
- ~537 million adults affected globally.
- Frequently asymptomatic until complications appear.
- Strong evidence that early detection reduces morbidity.

**NHANES prevalence (2013-2018 combined):** 1133/5082 = 22.3%.

---

## 2. Hypertension

**Guideline:** American College of Cardiology / American Heart Association (ACC/AHA) 2017.

**Definition:**
- Mean systolic BP ≥ 130 mmHg, OR
- Mean diastolic BP ≥ 80 mmHg, OR
- Self-reported doctor diagnosis (BPQ020 = 1).

**Variables used for the label:**
- `BPXSY1-4`, `BPXDI1-4` (BPX_J)
- `BPQ020` (BPQ_J)

**Clinical justification:**
- Leading modifiable risk factor for CVD, stroke, and CKD.
- Simple, non-invasive measurement.
- Silent onset in most patients.

**NHANES prevalence:** 2893/5139 = 56.3%.

---

## 3. Chronic Kidney Disease (CKD)

**Guideline:** Kidney Disease: Improving Global Outcomes (KDIGO) 2024.

**Definition:**
- eGFR < 60 mL/min/1.73 m² (CKD-EPI 2021, race-free), OR
- Self-reported dialysis (KIQ025 = 1), OR
- Self-reported weak/failing kidneys (KIQ022 = 1).

**Variables used for the label:**
- `LBXSCR` (creatinine, BIOPRO_J)
- `RIDAGEYR`, `RIAGENDR` (for eGFR)
- `KIQ022`, `KIQ025` (KIQ_U_J)

**Clinical justification:**
- Often diagnosed at late stages when dialysis or transplant is required.
- Strongly associated with diabetes and hypertension.
- Early detection allows slowing of progression.

**NHANES prevalence:** 473/5139 = 9.2%.

---

## 4. Dyslipidemia / Hypercholesterolemia

**Guideline:** National Cholesterol Education Program (NCEP) Adult Treatment Panel III.

**Definition:**
- Total cholesterol ≥ 200 mg/dL, OR
- HDL < 40 mg/dL (men) / < 50 mg/dL (women), OR
- Triglycerides ≥ 150 mg/dL.

**Variables used for the label:**
- `LBXTC` (TCHOL_J)
- `LBDHDD` (HDL_J)
- `LBXTR` (TRIGLY_J)

**Clinical justification:**
- Core modifiable risk factor for CVD.
- Frequently co-occurs with diabetes and metabolic syndrome.
- Standard component of lipid panels.

**NHANES prevalence:** 2844/4837 = 58.8%.

---

## 5. Cardiovascular Disease (CVD)

**Guideline:** NHANES Medical Conditions Questionnaire (MCQ).

**Definition:**
Any self-reported history of:
- Congestive heart failure (MCQ160B)
- Coronary heart disease (MCQ160C)
- Angina (MCQ160D)
- Myocardial infarction / heart attack (MCQ160E)
- Stroke (MCQ160F)

**Variables used for the label:** `MCQ160B-F` (MCQ_J).

**Clinical justification:**
- Leading cause of mortality globally.
- Self-reported history is a valid proxy for established disease.
- Enables risk stratification for secondary prevention.

**NHANES prevalence:** 633/5139 = 12.3%.

---

## 6. Metabolic Syndrome

**Guideline:** International Diabetes Federation / NCEP ATP III (harmonised 2009).

**Definition:**
Presence of ≥3 of the following 5 components:
1. Waist circumference > 102 cm (men) / > 88 cm (women).
2. Triglycerides ≥ 150 mg/dL.
3. HDL < 40 mg/dL (men) / < 50 mg/dL (women).
4. Blood pressure ≥ 130/85 mmHg.
5. Fasting glucose ≥ 100 mg/dL.

**Variables used for the label:**
- `BMXWAIST` (BMX_J)
- `LBXTR`, `LBDHDD`, `LBXGLU`
- Mean systolic and diastolic BP (from BPX_J)

**Clinical justification:**
- Powerful predictor of diabetes and CVD.
- Represents a cluster of related risk factors.
- Early lifestyle intervention can reverse the syndrome.

**NHANES prevalence:** 1613/4315 = 37.4%.

---

## 7. Liver Disease / NAFLD Risk

**Guideline:** American Association for the Study of Liver Diseases (AASLD) 2018.

**Definition (early-risk proxy):**
- Elevated liver enzymes (ALT > 40 U/L men / > 30 U/L women, or AST > 40 U/L, or GGT > 50 U/L), AND
- BMI ≥ 25 kg/m².

**Note:** This is a proxy for non-alcoholic fatty liver disease (NAFLD) risk, not a confirmed diagnosis. It is intended for screening.

**Variables used for the label:**
- `LBXSATSI`, `LBXSASSI`, `LBXSGTSI` (BIOPRO_J)
- `BMXBMI` (BMX_J)

**Clinical justification:**
- NAFLD affects ~25% of adults globally.
- Silent progression to fibrosis and cirrhosis.
- Strongly linked to obesity, diabetes, and metabolic syndrome.

**NHANES prevalence:** 730/5126 = 14.2%.

---

## Design rationale

The seven diseases were selected because:
1. They share metabolic pathways (insulin resistance, inflammation, dyslipidemia).
2. They frequently co-occur in the same patients.
3. They are all detectable from routine health data.
4. Early detection materially improves outcomes.
5. Together, they cover the majority of NCD-related mortality and morbidity.