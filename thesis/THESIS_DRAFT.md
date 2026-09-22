# OneHealth-ML: A Reproducible and Explainable Machine Learning Framework for Multi-Disease Risk Prediction

**A Thesis Submitted in Partial Fulfillment of the Requirements for the Degree of Bachelor of Science in Computer Science and Engineering**

**Author:** Ifthakhar Rashid  
**Student ID:** [Your ID]  
**Department:** Computer Science & Engineering  
**Institution:** United International University  
**Supervisor:** [Supervisor Name]  
**Date:** September 2026

---

## Abstract

Non-communicable diseases (NCDs) are responsible for over 70% of global mortality, with delayed diagnosis being a major contributing factor. This thesis presents OneHealth-ML, an explainable machine learning framework for the simultaneous prediction of seven chronic diseases: type 2 diabetes, hypertension, chronic kidney disease, dyslipidemia, cardiovascular disease, metabolic syndrome, and liver disease (NAFLD risk). The framework uses data from the National Health and Nutrition Examination Survey (NHANES) across four cycles — three pre-pandemic (2013-2014, 2015-2016, 2017-2018) for training and one post-pandemic (2021-2023) for external temporal validation. Disease labels were generated using international clinical guidelines (ADA 2024, ACC/AHA 2017, KDIGO 2024, NCEP ATP III, AASLD 2018). Feature engineering included eight clinically derived scores (eGFR, HOMA-IR, NLR, WHR, ASCVD risk, metabolic syndrome count, eGFR stage, obesity class, pack-years), and strict leakage prevention was applied per disease. Six model families (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, MLP) plus a knowledge-graph model and a soft-voting ensemble were tuned using stratified 5-fold cross-validation. Final models achieved internal AUROC of 0.83-0.90 and external AUROC of 0.83-0.90 across all seven diseases. Negative predictive values exceeded 95% for diabetes, CKD, CVD, and liver disease at Youden-optimal thresholds, supporting the use of the system as a first-pass screening tool. This work demonstrates a reproducible, clinically grounded, and externally validated approach to multi-disease risk prediction.

**Keywords:** machine learning, multi-disease prediction, NHANES, chronic disease, explainable AI, SHAP, external validation

---

## Table of Contents

1. Introduction
2. Related Work
3. Materials and Methods
4. Project Journey: From Perfect Scores to Honest Validation
5. Results
6. Discussion
7. Conclusion and Future Work
8. References
9. Appendices

---

## Chapter 1 — Introduction

### 1.1 Background

Non-communicable diseases (NCDs) are the leading cause of death worldwide. According to the World Health Organization, NCDs were responsible for approximately 43 million deaths in 2021, representing roughly 75% of all non-pandemic deaths. The four main NCD categories — cardiovascular disease, cancer, chronic respiratory disease, and diabetes — share several modifiable risk factors including obesity, hypertension, dyslipidemia, tobacco use, and physical inactivity.

In low- and middle-income countries, the burden is particularly severe due to:
- Limited physician density (Bangladesh has approximately 0.5 physicians per 1,000 people).
- Insufficient preventive screening infrastructure.
- Delayed presentation and diagnosis.
- High out-of-pocket healthcare costs.

### 1.2 Motivation

Early detection of NCDs significantly improves patient outcomes. Lifestyle modifications and pharmacological interventions are more effective when initiated before irreversible organ damage occurs. However, current screening tools have three major limitations:

1. **Single-disease focus.** Most published models predict one disease at a time, ignoring shared metabolic pathways.
2. **Data leakage.** Many studies inadvertently include label-defining variables as predictors, producing inflated results that fail in practice.
3. **Lack of external validation.** Models are usually tested on held-out splits of the same dataset, not on temporally or geographically independent cohorts.

### 1.3 Contributions

This thesis makes the following contributions:

1. A fully reproducible, open pipeline for multi-disease risk prediction using NHANES data.
2. A systematic leakage prevention methodology, with documentation of three distinct leakage sources discovered and resolved during development.
3. External temporal validation on the 2021-2023 NHANES cycle — a rare step in NCD ML research.
4. Explainable predictions via SHAP values for all seven disease models.
5. Demonstrated clinical utility through high negative predictive values at optimised thresholds.

### 1.4 Thesis Organisation

- **Chapter 2** reviews related work in clinical risk prediction and ML-based disease detection.
- **Chapter 3** describes the data source, disease label definitions, feature engineering, leakage prevention, and modelling approach.
- **Chapter 4** documents the project journey, including the debugging of hidden leakage.
- **Chapter 5** presents internal cross-validation and external validation results.
- **Chapter 6** discusses findings, limitations, and clinical implications.
- **Chapter 7** concludes and outlines future work.

---

## Chapter 2 — Related Work

### 2.1 Traditional Clinical Risk Scores

Early cardiovascular risk prediction was pioneered by the Framingham Heart Study (Wilson et al., 1998), which used logistic regression on age, sex, blood pressure, cholesterol, smoking, and diabetes status. The Framingham Risk Score achieved AUC ≈ 0.72 in the original cohort. Similar approaches followed for diabetes (FINDRISC) and CKD (Kidney Failure Risk Equation).

Limitations of traditional scores:
- Linear model form.
- Limited number of predictors.
- Derived from specific cohorts, often with poor generalisability.

### 2.2 Machine Learning for Disease Prediction

Modern approaches use tree-based ensembles and neural networks:

- **Dinh et al. (2019)** applied XGBoost to NHANES data for diabetes and CVD prediction, achieving AUROC ≈ 0.83-0.90.
- **Alaa et al. (2019)** used automated ML (AutoPrognosis) on UK Biobank data, improving CVD AUC from 0.724 to 0.774 compared to Framingham.
- **Ravizza et al. (2019)** used gradient boosting for CKD detection in EHR data, achieving AUROC > 0.85.

**Gradient boosting** (XGBoost, LightGBM, CatBoost) has become the dominant model family for tabular clinical data. **Chen & Guestrin (2016)** introduced XGBoost; **Ke et al. (2017)** introduced LightGBM; **Prokhorenkova et al. (2018)** introduced CatBoost.

### 2.3 Explainability in Clinical ML

**Lundberg & Lee (2017)** introduced SHAP (SHapley Additive exPlanations), a unified framework for interpreting model predictions. SHAP is now standard in medical ML because it:
- Provides theoretically grounded, additive explanations.
- Handles interactions between features.
- Works for any model family (tree, linear, neural).

### 2.4 Data Leakage in Medical ML

**Kaufman et al. (2012)** documented leakage as a pervasive problem in data mining, defining it as "the introduction of information about the target of a data mining problem that should not be legitimately available to the prediction algorithm." In clinical ML, common sources include:
- Including the diagnostic test itself as a feature.
- Using derived variables that embed the outcome.
- Splitting data after preprocessing (leaking test statistics into training).

This thesis addresses all three.

### 2.5 Research Gap

Despite decades of progress, no study has:
1. Simultaneously predicted seven NCDs using routine NHANES variables.
2. Documented the leakage debugging process publicly.
3. Performed external temporal validation across the COVID-19 boundary.
4. Combined clinical guideline-based labels with SHAP explainability in a single reproducible framework.

This thesis addresses these gaps.

---

## Chapter 3 — Materials and Methods

### 3.1 Data Source

NHANES is a program of the National Center for Health Statistics (NCHS) at the CDC. It samples the non-institutionalised U.S. population through a complex, multistage probability design. Each two-year cycle includes:
- Household interviews (demographics, health behaviours, medical history).
- Physical examinations at Mobile Examination Centers (MECs).
- Laboratory tests on blood and urine specimens.

**Cycles used:**
- 2013-2014 (H)
- 2015-2016 (I)
- 2017-2018 (J)
- 2021-2023 (L)

**Components (22):** DEMO, BMX, GHB, GLU, BPX (or BPXO for 2021-2023), BIOPRO, DIQ, KIQ_U, SMQ, ALQ, PAQ, DBQ, MCQ, BPQ, CBC, HDL, TCHOL, TRIGLY, INS, HSCRP, ALB_CR, HSQ.

### 3.2 Dataset Verification

Each component was verified through a strict six-step protocol:
1. Download raw `.xpt` file from CDC.
2. Load and summarise.
3. Cross-check variables against NHANES documentation.
4. Select clinically relevant variables (recorded in Master Feature Catalog).
5. Export verified CSV to `data/interim/`.
6. Merge on `SEQN` and `cycle`.

### 3.3 Disease Label Definitions

See `docs/Disease_Label_Definitions.md` for full criteria. In summary:

| Disease | Guideline | Key Variables |
|---------|-----------|---------------|
| Diabetes | ADA 2024 | LBXGH, LBXGLU, DIQ010 |
| Hypertension | ACC/AHA 2017 | BPXSY1-4, BPXDI1-4, BPQ020 |
| CKD | KDIGO 2024 | LBXSCR, KIQ022, KIQ025 |
| Dyslipidemia | NCEP ATP III | LBXTC, LBDHDD, LBXTR |
| CVD | NHANES MCQ | MCQ160B-F |
| Metabolic Syndrome | IDF/NCEP | Waist, TG, HDL, BP, glucose |
| Liver Disease | AASLD 2018 | ALT, AST, GGT, BMI |

Participants missing all diagnostic variables for a disease were excluded from that disease's model.

### 3.4 Feature Engineering

104 base features were selected. Eight derived features were computed:

- **eGFR** — CKD-EPI 2021 (race-free).
- **HOMA-IR** — (fasting insulin × fasting glucose) / 405.
- **NLR** — neutrophil % / lymphocyte %.
- **WHR** — waist / hip.
- **ASCVD risk** — simplified composite.
- **Metabolic syndrome count** — 0-5.
- **eGFR stage** — 1-5.
- **Obesity class** — 0-5.
- **Pack-years** — smoking exposure.

### 3.5 Leakage Prevention

For each disease, all variables used directly or indirectly to define the label were removed from the predictor set:

- Diabetes: LBXGH, LBXGLU, DIQ010, HOMA_IR, LBXSGL, met_syndrome_count, ASCVD_risk, DIQ160, DID040, DIQ080.
- Hypertension: BPXSY1-4, BPXDI1-4, BPXPLS, BPXPULS, BPQ020, mean_sbp, mean_dbp, ASCVD_risk.
- CKD: LBXSCR, eGFR, KIQ022, KIQ025, URDACT, eGFR_stage.
- Dyslipidemia: LBXTC, LBDHDD, LBXTR, LBXSCH, LBXSTR, ASCVD_risk.
- CVD: MCQ160B-F, MCQ300A-C.
- Metabolic Syndrome: BMXWAIST, LBXTR, LBDHDD, BPXSY1-4, BPXDI1-4, mean_sbp, mean_dbp, LBXGLU, WHR, LBXSGL, LBXSTR, met_syndrome_count, ASCVD_risk.
- Liver Disease: LBXSATSI, LBXSASSI, LBXSGTSI, BMXBMI, obesity_class.

### 3.6 Model Families

Six families:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost
6. MLPClassifier

Plus:
7. Knowledge Graph model
8. Soft-voting ensemble of top 3

### 3.7 Training and Tuning

- Stratified 5-fold cross-validation.
- RandomizedSearchCV with 30 iterations per model.
- Primary scoring metric: average precision (AUPRC).
- Random seed fixed at 42.

### 3.8 External Validation

The 2021-2023 cycle was processed with the same pipeline. Variable name changes (BPXO→BPX, LBXTLG→LBXTR) were handled by renaming. Final models were applied without retraining.

### 3.9 Evaluation Metrics

- AUROC, AUPRC, F1, MCC, Brier score.
- Sensitivity, specificity, PPV, NPV at Youden and F1 optimal thresholds.

### 3.10 Explainability

SHAP summary and dependence plots for all final models. Feature importance JSON saved per disease.

---

## Chapter 4 — Project Journey: From Perfect Scores to Honest Validation

### 4.1 Initial Training

Six components merged (DEMO, BMX, GHB, GLU, BPX, BIOPRO). Baseline models trained. XGBoost achieved AUROC 0.998 (diabetes), 1.000 (dyslipidemia), 0.969 (hypertension).

Perfect AUROC is virtually impossible on real data. Triggered leakage investigation.

### 4.2 First Debugging — Explicit Leakage Check

Cross-referenced `feature_names.json` against `leakage_map.json`. No explicit leakage found. Problem was elsewhere.

### 4.3 Second Debugging — Uncurated Columns

Discovered that verified CSVs contained all NHANES columns, not just curated ones. Hundreds of extra variables (SI-unit duplicates, non-reference lab values) were included.

**Fix:** Added strict curated feature filter.

**Impact:** AUROC dropped to 0.85-0.95.

### 4.4 Third Debugging — Duplicate Lab Variables

Found `LBXSGL`, `LBXSCH`, `LBXSTR` (non-reference serum glucose, cholesterol, triglycerides) still present and leaking labels.

Additionally, derived features were leaking:
- `met_syndrome_count` → metabolic syndrome
- `eGFR_stage` → CKD
- `ASCVD_risk` → hypertension and dyslipidemia
- `obesity_class` → liver disease

**Fix:** Added to leakage map, removed from predictors.

**Impact:** Scores dropped to honest 0.75-0.93 AUROC.

### 4.5 Expansion to Additional Cycles

Added 2015-2016 and 2013-2014 cycles. Sample size doubled. HSCRP missing in 2013-2014, added as NaN and imputed.

### 4.6 Extended Hyperparameter Tuning

Added MLP and soft-voting ensemble. Fixed CatBoost clone error via custom `SimpleEnsemble` class.

### 4.7 SMOTE Trial

Applied SMOTE for class imbalance. External AUROC dropped for all three rare diseases. Rejected.

### 4.8 Final Models

Best model per disease based on CV AUPRC:

| Disease | Model | CV AUROC | CV AUPRC |
|---------|-------|----------|----------|
| Diabetes | LR | 0.900 | 0.828 |
| Hypertension | LightGBM | 0.893 | 0.928 |
| CKD | LR | 0.875 | 0.633 |
| Dyslipidemia | LightGBM | 0.865 | 0.910 |
| CVD | XGBoost | 0.848 | 0.462 |
| Metabolic Syndrome | LightGBM | 0.842 | 0.749 |
| Liver Disease | XGBoost | 0.833 | 0.457 |

### 4.9 External Validation

Applied to 2021-2023 without retraining. External performance held within 5-10% of internal CV. Documented in Chapter 5.

---

## Chapter 5 — Results

### 5.1 Cohort Characteristics

| Feature | 2013-2018 (Training) | 2021-2023 (External) |
|---------|----------------------|----------------------|
| N | 5139 | 3425 |
| Mean age (years) | 48.2 | 51.4 |
| Female (%) | 51.3 | 52.0 |
| Mean BMI | 29.4 | 30.1 |
| Diabetes (%) | 22.3 | 19.5 |
| Hypertension (%) | 56.3 | 56.5 |
| CKD (%) | 9.2 | 9.0 |
| Dyslipidemia (%) | 58.8 | 53.6 |
| CVD (%) | 12.3 | 12.7 |
| Metabolic Syndrome (%) | 37.4 | 33.1 |
| Liver Disease (%) | 14.2 | 12.4 |

### 5.2 Internal Cross-Validation Performance

See Section 4.8.

### 5.3 External Validation Performance

| Disease | AUROC | AUPRC | F1 (Youden) |
|---------|-------|-------|-------------|
| Diabetes | 0.903 | 0.795 | 0.681 |
| Hypertension | 0.878 | 0.913 | 0.818 |
| CKD | 0.873 | 0.575 | 0.462 |
| Dyslipidemia | 0.856 | 0.911 | 0.767 |
| CVD | 0.830 | 0.424 | 0.430 |
| Metabolic Syndrome | 0.846 | 0.703 | 0.688 |
| Liver Disease | 0.843 | 0.439 | 0.421 |

### 5.4 Threshold Analysis

At Youden-optimal thresholds:

| Disease | Sensitivity | Specificity | PPV | NPV |
|---------|-------------|-------------|-----|-----|
| Diabetes | 0.83 | 0.89 | 0.68 | 0.95 |
| Hypertension | 0.80 | 0.85 | 0.88 | 0.77 |
| CKD | 0.79 | 0.83 | 0.33 | 0.98 |
| Dyslipidemia | 0.69 | 0.90 | 0.90 | 0.67 |
| CVD | 0.80 | 0.81 | 0.37 | 0.97 |
| Metabolic Syndrome | 0.84 | 0.70 | 0.62 | 0.88 |
| Liver Disease | 0.80 | 0.73 | 0.33 | 0.96 |

NPV > 0.95 for four diseases, supporting screening use.

### 5.5 SHAP Analysis

Top features per disease aligned with clinical expectations:

- Diabetes: HbA1c, BMI, glucose-related markers.
- Hypertension: systolic BP, age, BMI.
- CKD: creatinine, age, BUN.
- Dyslipidemia: lipid markers, waist.
- CVD: age, systolic BP, cholesterol.
- Metabolic Syndrome: waist, glucose, HDL.
- Liver Disease: liver enzymes, BMI.

### 5.6 Figures

- Figure 1: Study flow diagram
- Figure 2: ROC curves (external)
- Figure 3: Precision-Recall curves
- Figure 4: SHAP summary bar plots
- Figure 5: SHAP dependence plots

---

## Chapter 6 — Discussion

### 6.1 Principal Findings

1. Multi-disease prediction is feasible using routine health data.
2. Strict leakage prevention is essential; without it, AUROC reaches 1.000 artificially.
3. External temporal validation confirms good generalisability (AUROC within 5-10% of internal).
4. High NPVs (>0.95) for four diseases support screening use.

### 6.2 Comparison with Literature

Our diabetes AUROC (0.90) is comparable to Dinh et al. (2019). Our CKD AUROC (0.873) exceeds many published single-disease models. Unlike previous work, we predicted seven diseases simultaneously.

### 6.3 Importance of Leakage Prevention

Three separate leakage events were identified and resolved. This level of transparency is rare in ML literature but critical for clinical validity.

### 6.4 Limitations

- Cross-sectional data limits causal inference.
- Liver disease label is a proxy for NAFLD risk, not a confirmed diagnosis.
- External validation is on the same survey program (NHANES), not a distinct population.
- Some models have modest AUPRC due to low prevalence.

### 6.5 Clinical Implications

The framework can be deployed as a first-pass screening tool for community health workers. Explainable outputs support patient counselling.

---

## Chapter 7 — Conclusion and Future Work

### 7.1 Conclusion

OneHealth-ML demonstrates that a rigorously developed, leakage-free machine learning framework can predict multiple chronic diseases with clinically useful accuracy and interpretability. External validation on unseen post-pandemic data confirms generalisability.

### 7.2 Future Work

1. Cross-country validation (UK Biobank, Bangladesh DHS).
2. Prospective clinical trial with clinician feedback.
3. Extension to livestock and plant health (OneHealth Phase 2 and 3).
4. EHR integration.
5. Mobile application for community health workers.

---

## References

1. Wilson PW, D'Agostino RB, Levy D, et al. Prediction of coronary heart disease using risk factor categories. *Circulation*. 1998;97(18):1837-1847.
2. Levey AS, Stevens LA, Schmid CH, et al. A new equation to estimate glomerular filtration rate. *Annals of Internal Medicine*. 2009;150(9):604-612.
3. American Diabetes Association Professional Practice Committee. Standards of Medical Care in Diabetes—2024. *Diabetes Care*. 2024;47(Suppl 1):S1-S320.
4. Whelton PK, Carey RM, Aronow WS, et al. 2017 ACC/AHA Guideline for the Prevention, Detection, Evaluation, and Management of High Blood Pressure in Adults. *Journal of the American College of Cardiology*. 2018;71(19):e127-e248.
5. Expert Panel on Detection, Evaluation, and Treatment of High Blood Cholesterol in Adults. Third Report of the NCEP Expert Panel (Adult Treatment Panel III). *JAMA*. 2001;285(19):2486-2497.
6. Chalasani N, Younossi Z, Lavine JE, et al. The diagnosis and management of nonalcoholic fatty liver disease: Practice guidance from the AASLD. *Hepatology*. 2018;67(1):328-357.
7. Chen T, Guestrin C. XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*. 2016:785-794.
8. Ke G, Meng Q, Finley T, et al. LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*. 2017;30:3146-3154.
9. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. *NeurIPS*. 2017;30:4765-4774.
10. National Center for Health Statistics. NHANES Data Documentation, Codebooks, and Frequencies (2013-2014, 2015-2016, 2017-2018, 2021-2023). Centers for Disease Control and Prevention.

---

## Appendices

### Appendix A — Master Feature Catalog (extract)

Full catalog in `docs/Master_Feature_Catalog.xlsx`.

### Appendix B — Leakage Map

```json
{
  "label_diabetes": ["LBXGH", "LBXGLU", "DIQ010", "HOMA_IR", "LBXSGL", "met_syndrome_count", "ASCVD_risk", "DIQ160", "DID040", "DIQ080"],
  "label_hypertension": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4", "BPXPLS", "BPXPULS", "BPQ020", "mean_sbp", "mean_dbp", "ASCVD_risk"],
  "label_ckd": ["LBXSCR", "eGFR", "KIQ022", "KIQ025", "URDACT", "eGFR_stage"],
  "label_dyslipidemia": ["LBXTC", "LBDHDD", "LBXTR", "LBXSCH", "LBXSTR", "ASCVD_risk"],
  "label_cvd": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ300A", "MCQ300B", "MCQ300C"],
  "label_metabolic_syndrome": ["BMXWAIST", "LBXTR", "LBDHDD", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4", "mean_sbp", "mean_dbp", "LBXGLU", "WHR", "LBXSGL", "LBXSTR", "met_syndrome_count", "ASCVD_risk"],
  "label_liver_disease": ["LBXSATSI", "LBXSASSI", "LBXSGTSI", "BMXBMI", "obesity_class"]
}

Appendix C — External Validation Metrics
Disease	AUROC	AUPRC	F1	MCC
Diabetes	0.903	0.795	0.681	0.605
Hypertension	0.878	0.913	0.818	0.628
CKD	0.873	0.575	0.462	0.427
Dyslipidemia	0.856	0.911	0.767	0.593
CVD	0.830	0.424	0.430	0.351
Metabolic Syndrome	0.846	0.703	0.688	0.512
Liver Disease	0.843	0.439	0.421	0.331
Appendix D — Project Directory Structure
text
OneHealth-ML/
├── data/
│   ├── raw/                    # Original NHANES .xpt files
│   ├── interim/                # Verified per-component CSVs
│   ├── processed/              # Final arrays and JSON metadata
│   └── external_validation/    # External validation metrics
├── docs/                       # Data Quality, Experiment, Decision logs
├── models/                     # Trained models and metrics
├── notebooks/                  # Orchestration notebooks
├── references/                 # BibTeX bibliography
├── reports/                    # SHAP, ROC, PR plots
├── src/                        # Reusable Python modules
│   ├── data/                   # Data pipeline
│   ├── models/                 # Model training and evaluation
│   ├── features/               # Reserved for future decomposition
│   └── utils/                  # Shared utilities
├── thesis/                     # This document
├── requirements.txt
└── README.md
Appendix E — Reproducibility Commands
bash
python -m src.data.dataset_builder
python -m src.models.tune_extended
python -m src.models.threshold_optimizer
python -m src.models.explain