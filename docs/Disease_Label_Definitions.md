# Disease Label Definitions
**Project:** OneHealth-ML – Early Multi-Disease Prediction using NHANES 2017–2018  
**Version:** 2.1  
**Last updated:** [Date]

## Objective
Define clinical rules for creating supervised learning labels **before model training**.
All rules follow internationally accepted guidelines. Variables used for label creation are **excluded** as predictors for that disease model.

---

## General Exclusion Rules
Applied to all diseases before label generation:
- Missing participant identifier (`SEQN`)
- Duplicate `SEQN`
- Missing **all** required diagnostic variables for a given disease (participant is labelled `NaN` for that disease only)

---

## 1. Diabetes Mellitus (Type 2)
**Guideline:** ADA 2024 Standards of Care

**Diagnostic Variables**
| Variable          | Dataset | Use             |
|-------------------|---------|-----------------|
| LBXGH (HbA1c)     | GHB_J   | Label           |
| LBXGLU (Fasting Glucose) | GLU_J | Label     |
| DIQ010 (Doctor-diagnosed diabetes) | DIQ_J | Label |
| RIDEXPRG (Pregnancy) | DEMO_J | Exclusion |

**Positive Label** (ANY of the following):
- HbA1c ≥ 6.5%
- Fasting Plasma Glucose ≥ 126 mg/dL
- Self-reported doctor-diagnosed diabetes (`DIQ010 = 1`)

**Exclusion:** Pregnant participants (`RIDEXPRG = 1`) are excluded **only** from the diabetes label because pregnancy alters glucose metabolism.

**Predictor Exclusion:** For the diabetes model, **never** use HbA1c, fasting glucose, or DIQ010 as features.

---

## 2. Hypertension
**Guideline:** ACC/AHA 2017

**Diagnostic Variables**
| Variable          | Dataset | Use             |
|-------------------|---------|-----------------|
| BPXSY1–4          | BPX_J   | Label (systolic)|
| BPXDI1–4          | BPX_J   | Label (diastolic)|
| BPQ020 (Doctor-diagnosed hypertension) | BPQ_J | Label (optional) |

**Calculation:**
- **Average SBP** = mean of all available systolic readings (minimum 2; if only 1, use with caution or exclude)
- **Average DBP** = mean of all available diastolic readings

**Positive Label** (ANY of the following):
- Average SBP ≥ 130 mmHg
- Average DBP ≥ 80 mmHg
- Self-reported doctor-diagnosed hypertension (`BPQ020 = 1`) – *optional, increases sensitivity*

**Predictor Exclusion:** For the hypertension model, exclude **all** blood pressure measurements and the hypertension diagnosis variable.

---

## 3. Chronic Kidney Disease (CKD)
**Guideline:** KDIGO 2024

**Diagnostic Variables**
| Variable          | Dataset | Use             |
|-------------------|---------|-----------------|
| LBXSCR (Serum Creatinine) | BIOPRO_J | eGFR calculation |
| Age, Sex, Race    | DEMO_J  | eGFR calculation |
| KIQ022 (Weak/failing kidneys) | KIQ_U_J | Label |
| KIQ025 (Dialysis in past 12 months) | KIQ_U_J | Label |

**Derived Variable:**  
Estimated GFR (eGFR) using **CKD‑EPI 2021 equation** (no race coefficient).

**Positive Label** (ANY of the following):
- eGFR < 60 mL/min/1.73 m²
- Self-reported dialysis (`KIQ025 = 1`)
- Self-reported weak/failing kidneys (`KIQ022 = 1`)

**Predictor Exclusion:** For the CKD model, exclude **creatinine, eGFR, KIQ022, KIQ025**.

---

## 4. Dyslipidemia / Hypercholesterolemia
**Guideline:** AHA / NCEP ATP III

**Diagnostic Variables**
| Variable               | Dataset   | Use   |
|------------------------|-----------|-------|
| LBXTC (Total Cholesterol) | TCHOL_J | Label |
| LBDHDD (Direct HDL)    | HDL_J     | Label |
| LBXTR (Triglycerides)  | TRIGLY_J  | Label |

**Positive Label** (ANY of the following):
- Total Cholesterol ≥ 200 mg/dL
- HDL < 40 mg/dL (males) or < 50 mg/dL (females)
- Triglycerides ≥ 150 mg/dL

*Optional:* Self-reported cholesterol-lowering medication (from BPQ or MCQ) can be added as a positive criterion.

**Predictor Exclusion:** For the dyslipidemia model, exclude **total cholesterol, HDL, triglycerides, and medication variables**.

---

## 5. Cardiovascular Disease (CVD)
**Source:** NHANES Medical Conditions Questionnaire (MCQ_J)

**Diagnostic Variables**
| Variable   | Condition                         |
|------------|-----------------------------------|
| MCQ160B    | Ever told had congestive heart failure |
| MCQ160C    | Ever told had coronary heart disease  |
| MCQ160D    | Ever told had angina/angina pectoris  |
| MCQ160E    | Ever told had heart attack (MI)       |
| MCQ160F    | Ever told had a stroke                |

**Positive Label:** Yes to **any** of the above.

**Predictor Exclusion:** For the CVD model, exclude **all self-reported cardiovascular disease history variables**.

---

## 6. Metabolic Syndrome
**Guideline:** Harmonized IDF/NCEP ATP III (2009)

**Components** (positive if 3 or more present):

| Component               | Threshold                     |
|-------------------------|-------------------------------|
| Waist circumference     | Males > 102 cm, Females > 88 cm (NCEP) |
| Triglycerides           | ≥ 150 mg/dL                   |
| HDL Cholesterol         | Males < 40 mg/dL, Females < 50 mg/dL |
| Blood Pressure          | Systolic ≥ 130 **and/or** Diastolic ≥ 85 mmHg (or on antihypertensive medication) |
| Fasting Glucose         | ≥ 100 mg/dL (or on diabetes medication) |

*Medication use information (from BPQ/DIQ) can be incorporated once questionnaire datasets are verified.*

**Predictor Exclusion:** For the metabolic syndrome model, exclude **waist circumference, triglycerides, HDL, blood pressure, fasting glucose, and any medication variables used to define the syndrome.**

---

## 7. Liver Disease / NAFLD Risk
**Guideline:** AASLD 2018 & clinical biomarker criteria

**Diagnostic Variables**
| Variable       | Dataset   | Use   |
|----------------|-----------|-------|
| ALT (LBXSATSI) | BIOPRO_J  | Label |
| AST (LBXSASSI) | BIOPRO_J  | Label |
| GGT (LBXSGTSI) | BIOPRO_J  | Label |
| BMI (BMXBMI)   | BMX_J     | Label |

**Positive Label (early‑risk proxy):**
- Elevated ALT (> upper limit of normal, ~40 U/L for males, ~30 U/L for females) **OR** AST > upper limit **OR** GGT elevated (> ~50 U/L)
- **AND** BMI ≥ 25 kg/m²

*When triglycerides become available, consider supplementing with Fatty Liver Index (FLI) for a more specific NAFLD label.*

**Predictor Exclusion:** For the liver disease model, exclude **ALT, AST, GGT, and BMI** (since BMI is part of the label definition). Note: If an FLI-based label is used later, exclude its constituent variables.

---

## Predictor Leakage Policy (Unified)
For **every** disease model, variables used **directly or indirectly** to define the positive label must **not** be included as predictor features for that same disease.

## Missing Data Strategy (Labels)
- Participants missing **all** required diagnostic variables → label set to `NaN` (excluded from that disease model).
- Participants with partial data (e.g., only one of two required lab tests) → label determined using available data if clinically acceptable; otherwise `NaN`.
- This strategy is documented in the Experiment Log for full reproducibility.

---

*This document will be updated as new datasets (TCHOL_J, HDL_J, TRIGLY_J, DIQ_J, MCQ_J, etc.) are verified and integrated.*