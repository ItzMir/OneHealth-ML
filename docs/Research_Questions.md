# Research Questions — OneHealth-ML

**Principal Investigator:** Ifthakhar Rashid  
**Institution:** United International University  
**Date:** 2026-09-11

---

## Primary Research Question

**PRQ:** Can a single, reproducible machine learning framework simultaneously predict seven chronic non-communicable diseases using routinely collected health survey data, and does it generalise to an unseen, more recent population cycle?

---

## Secondary Research Questions

**RQ1 — Multi-disease feasibility**  
Can a unified data pipeline produce valid, clinically aligned labels for seven diseases using only routine demographic, anthropometric, laboratory, and lifestyle variables?

**Hypothesis:** Yes — all seven diseases share overlapping biomarkers and can be defined unambiguously from NHANES variables.

**Method:** Generate labels per ADA 2024, ACC/AHA 2017, KDIGO 2024, NCEP ATP III, and AASLD 2018. Report prevalence and positive counts.

---

**RQ2 — Impact of data leakage on reported performance**  
How much does strict leakage prevention change the reported performance of a medical ML model, compared to a naive pipeline that includes label-defining variables?

**Hypothesis:** Naive pipelines will show inflated AUROC (≥0.99). Leakage-free pipelines will show realistic AUROC (0.75–0.93).

**Method:** Reproduce both pipelines, quantify the difference, document each leakage source.

---

**RQ3 — Temporal generalisability**  
Do models trained on pre-pandemic NHANES cycles (2013-2018) generalise to a post-pandemic cycle (2021-2023) with different data collection protocols?

**Hypothesis:** External AUROC will be within 5–10% of internal performance, indicating good generalisability.

**Method:** Apply final models without retraining to 2021-2023 data, harmonise variable names, compute external AUROC/AUPRC.

---

**RQ4 — Model family comparison**  
Which model family (linear, tree-based, boosting, neural) performs best for each of the seven diseases?

**Hypothesis:** Different diseases will favour different model families, with gradient boosting dominating most and logistic regression performing competitively on linear-separable problems.

**Method:** Train and tune six families with 5-fold CV; select final model per disease by highest mean CV AUPRC.

---

**RQ5 — Clinical utility of thresholds**  
Can thresholds be optimised to achieve high negative predictive value, making the models suitable for first-pass screening?

**Hypothesis:** Youden-optimal thresholds will yield NPV > 0.95 for at least four diseases.

**Method:** Compute Youden and F1 thresholds; report sensitivity, specificity, PPV, NPV at each.

---

**RQ6 — Feature alignment with clinical knowledge**  
Do the top-ranked features from SHAP analysis align with established clinical risk factors for each disease?

**Hypothesis:** Yes — age, BMI, systolic BP, HbA1c, creatinine, and lipid measures will dominate in the expected diseases.

**Method:** Generate SHAP summary and dependence plots; compare top 10 features per disease with clinical guidelines.

---

## Exploratory Research Questions

**ERQ1:** Does adding earlier NHANES cycles (2013-2014, 2015-2016) improve model stability and external generalisability for rare diseases (CKD, CVD, liver)?

**ERQ2:** Does synthetic oversampling (SMOTE) improve external performance for rare diseases?

**ERQ3:** Does a knowledge-graph-enhanced model using clinical feature-disease relationships outperform purely data-driven models?

**ERQ4:** Can a soft-voting ensemble of the top three models outperform any single model family?

---

## Deliverables

Each research question maps to a specific deliverable:

| RQ | Deliverable |
|----|-------------|
| PRQ | Full pipeline and external validation metrics |
| RQ1 | `labeled_master_combined.csv` with 7 disease labels |
| RQ2 | `Experiment_Log.md` (leakage documentation) |
| RQ3 | `external_validation_metrics.csv` |
| RQ4 | `tuned_model_metrics.csv` |
| RQ5 | `threshold_optimization.csv` and `clinical_metrics.csv` |
| RQ6 | `reports/figures/*_feature_importance.png` and SHAP plots |