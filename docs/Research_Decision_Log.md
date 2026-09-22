# Research Decision Log — OneHealth-ML

Every significant methodological decision, with rationale, alternatives considered, and references.

---

## D1 — Choice of NHANES as data source

**Decision:** Use NHANES cycles 2013-2014, 2015-2016, 2017-2018 for training and 2021-2023 for external validation.

**Rationale:**
- Nationally representative of the U.S. non-institutionalised population.
- Multimodal: demographics, exams, labs, questionnaires.
- Standardised across cycles with official codebooks.
- Contains all variables needed to define the seven target diseases.
- Publicly available with no licensing restrictions.

**Alternatives considered:**
- UK Biobank (requires application; not publicly downloadable).
- MIMIC-III (hospital ICU data; not suitable for chronic-disease screening).
- Kaggle datasets (heterogeneous, inconsistent variable coding, poor documentation).

**References:** CDC NHANES documentation.

---

## D2 — Choice of seven target diseases

**Decision:** Predict diabetes, hypertension, CKD, dyslipidemia, CVD, metabolic syndrome, and liver disease (NAFLD risk).

**Rationale:**
- Shared metabolic pathways — simultaneous prediction makes clinical sense.
- Frequently co-occur — capturing one improves the others.
- All definable from routine NHANES variables.
- Clinically actionable through lifestyle and pharmacological intervention.

**References:** ADA 2024, ACC/AHA 2017, KDIGO 2024, NCEP ATP III, AASLD 2018.

---

## D3 — Label definitions follow international guidelines

**Decision:** Use ADA, ACC/AHA, KDIGO, NCEP, and AASLD criteria.

**Rationale:**
- Ensures clinical validity of the labels.
- Enables comparison with other published models.
- Provides a defensible legal and ethical basis for the labels.

**Alternatives considered:**
- Data-driven clustering (rejected — lacks clinical interpretability).
- Self-report alone (rejected — under-diagnosis bias).

---

## D4 — Training on pre-pandemic cycles, validating on post-pandemic cycle

**Decision:** Train on 2013-2018; validate on 2021-2023.

**Rationale:**
- True temporal external validation is rare in NCD ML research.
- 2021-2023 is protocol-distinct (e.g., BPXO instead of BPX).
- Tests whether the model generalises across a major population-health disruption (COVID-19).

**Alternatives considered:**
- Random split within all four cycles (rejected — leaks temporal information).
- K-fold across all cycles (rejected — no true holdout).

---

## D5 — Curated feature set instead of all columns

**Decision:** Use only the 104 curated features plus 8 derived features.

**Rationale:**
- NHANES verified CSVs contain hundreds of extra columns never intended for modelling.
- These included SI-unit duplicates and non-reference lab values.
- Verified to cause severe leakage during the initial debugging.

**Impact:** Model performance dropped from implausible ~1.000 AUROC to honest 0.75–0.93 AUROC.

---

## D6 — Strict leakage prevention per disease

**Decision:** For every disease, remove all variables used to define its label.

**Rationale:**
- Prevent the model from trivially reconstructing the label.
- Ensures clinical applicability (a screening tool cannot rely on the diagnostic test itself).
- Documented in `leakage_map.json` and enforced programmatically.

**Reference:** Kaufman et al., *Patterns* (2021) — "Leakage in data mining".

---

## D7 — Six model families + Knowledge Graph + ensemble

**Decision:** Compare LR, RF, XGBoost, LightGBM, CatBoost, MLP, and a custom Knowledge Graph model. Also test a soft-voting ensemble of the top 3.

**Rationale:**
- Different inductive biases suit different diseases.
- LR is interpretable and often best for linearly separable problems.
- Tree-based ensembles dominate tabular clinical data.
- MLP captures non-linear interactions.
- KG integrates clinical domain knowledge.
- Ensemble improves robustness.

**Alternatives considered:**
- TabNet, transformers (rejected — no measurable gain on this sample size, difficult to interpret).

**References:** Chen & Guestrin 2016, Ke et al. 2017, Prokhorenkova et al. 2018.

---

## D8 — Model selection by CV AUPRC

**Decision:** Select the final model per disease based on mean 5-fold CV AUPRC, not AUROC or accuracy.

**Rationale:**
- AUPRC is more sensitive to class imbalance (CKD, CVD, liver disease are rare).
- AUROC can be misleading when positives are rare.
- Accuracy is meaningless in imbalanced settings.

---

## D9 — Threshold optimisation for clinical utility

**Decision:** Report Youden and F1 optimal thresholds, plus sensitivity, specificity, PPV, NPV.

**Rationale:**
- A probabilistic classifier must be converted into a decision.
- For screening, high NPV is critical (rule-out capability).
- Youden balances sensitivity and specificity; F1 balances precision and recall.

---

## D10 — Rejection of SMOTE

**Decision:** Do not use SMOTE in the final models.

**Rationale:**
- Synthetic minority samples did not reflect the temporal distribution shift.
- External AUROC dropped for all three low-prevalence diseases.
- Documented as a negative result in `Experiment_Log.md`.

---

## D11 — Preserve HSCRP for 2013-2014 via imputation

**Decision:** Add `LBXHSCRP` as NaN for 2013-2014 participants and impute with the median from 2015-2018.

**Rationale:**
- HSCRP_H was not publicly released for 2013-2014.
- Dropping the feature entirely would lose information for the majority of the training data.
- Median imputation is a standard, defensible approach and preserves all participants.

---

## D12 — SHAP for explainability

**Decision:** Use SHAP summary and dependence plots for every final model.

**Rationale:**
- SHAP provides a theoretically grounded, additive explanation.
- Global summary plots show overall feature importance.
- Dependence plots reveal interactions and monotonic effects.
- Widely accepted in clinical ML literature.

**Reference:** Lundberg & Lee, NeurIPS 2017.

---

## D13 — Document everything in markdown and Excel

**Decision:** Maintain `Data_Quality_Log.md`, `Experiment_Log.md`, `Research_Decision_Log.md`, and `Master_Feature_Catalog.xlsx`.

**Rationale:**
- Reproducibility and auditability.
- Enables independent verification.
- Required by thesis and journal submission guidelines.