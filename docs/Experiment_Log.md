# Experiment Log — OneHealth-ML

Chronological record of every experiment, metric, and debugging iteration
across the lifetime of the project. New entries are appended in reverse-chronological
order (newest first).

---

## 2026-09-10 — Final tuned models and external validation complete

**Final model selection (by highest mean CV AUPRC):**

| Disease | Best Model | CV AUROC | CV AUPRC |
|---------|------------|----------|----------|
| Diabetes | Logistic Regression | 0.900 | 0.828 |
| Hypertension | LightGBM | 0.893 | 0.928 |
| CKD | Logistic Regression | 0.875 | 0.633 |
| Dyslipidemia | LightGBM | 0.865 | 0.910 |
| CVD | XGBoost | 0.848 | 0.462 |
| Metabolic Syndrome | LightGBM | 0.842 | 0.749 |
| Liver Disease | XGBoost | 0.833 | 0.457 |

**External validation on 2021-2023:**

| Disease | AUROC | AUPRC |
|---------|-------|-------|
| Diabetes | 0.903 | 0.795 |
| Hypertension | 0.878 | 0.913 |
| CKD | 0.873 | 0.575 |
| Dyslipidemia | 0.856 | 0.911 |
| CVD | 0.830 | 0.424 |
| Metabolic Syndrome | 0.846 | 0.703 |
| Liver Disease | 0.843 | 0.439 |

**Threshold optimization:** Youden thresholds give NPV > 0.95 for diabetes, CKD, CVD, and liver disease. All thresholds saved to `models/threshold_optimization.csv`.

**Files:** `models/tuned_model_metrics.csv`, `models/external_validation_metrics.csv`, `models/threshold_optimization.csv`.

---

## 2026-09-08 — Leakage audit round 3 (final)

**Goal:** Remove remaining hidden leakage.

**Findings:**
- `LBXSGL` (serum glucose from BIOPRO) was leaking the diabetes label.
- `LBXSCH` (serum cholesterol) was leaking the dyslipidemia label.
- `LBXSTR` (serum triglycerides) was leaking the dyslipidemia label.
- `met_syndrome_count` was leaking the metabolic syndrome label.
- `eGFR_stage` was leaking the CKD label.
- `ASCVD_risk` was leaking the hypertension and dyslipidemia labels.
- `obesity_class` was leaking the liver disease label.

**Action:** All were added to the leakage map and removed per disease. Post-fix high-correlation check (>0.95 with any label): passed.

**Impact:** Model performance dropped to honest, realistic levels.

---

## 2026-09-05 — Combined 2013-2018 dataset built

**Shape:** (5139, 118) after curated feature selection.

**Participants per disease after label exclusions:**
- Diabetes: 5082 rows (1133 positive)
- Hypertension: 5139 rows (2893 positive)
- CKD: 5139 rows (473 positive)
- Dyslipidemia: 4837 rows (2844 positive)
- CVD: 5139 rows (633 positive)
- Metabolic Syndrome: 4315 rows (1613 positive)
- Liver Disease: 5126 rows (730 positive)

**Cycle codes:** 2013-2014 = 0, 2015-2016 = 1, 2017-2018 = 2.

---

## 2026-09-01 — 2013-2014 cycle verified

- 21 components downloaded (HSCRP_H not publicly released).
- HSCRP added as NaN and imputed later.
- All other variables verified with the same protocol.

---

## 2026-08-30 — SMOTE trial

**Goal:** Improve performance for CKD, CVD, and liver disease via synthetic minority over-sampling.

**Method:** SMOTE applied only to the training set; final tuned hyperparameters reused.

**Results:**
- CKD: AUROC 0.863 → 0.845 (drop)
- CVD: AUROC 0.867 → 0.858 (drop)
- Liver: AUROC 0.822 → 0.819 (drop)

**Decision:** SMOTE rejected. Synthetic samples did not reflect the external distribution.

---

## 2026-08-20 — Extended hyperparameter tuning

**Added:**
- MLPClassifier with grid over hidden layer sizes, alpha, learning rate.
- Soft-voting ensemble of the top 3 models per disease.
- Custom `SimpleEnsemble` class to avoid CatBoost cloning error inside sklearn's `VotingClassifier`.

**Improvements:**
- Hypertension AUPRC: 0.844 → 0.928
- Dyslipidemia AUPRC: 0.740 → 0.910
- Metabolic Syndrome AUPRC: 0.743 → 0.749

---

## 2026-08-15 — Expansion to 2015-2016 and 2013-2014

**Goal:** Increase sample size to reduce variance and improve generalisability.

**Actions:** Downloaded and verified both cycles, combined into a single training set.

**Impact:** Positive counts roughly doubled for rare diseases.

---

## 2026-07-30 — Leakage audit round 2

**Findings:** Hundreds of uncurated columns from NHANES verified CSVs were still included in the predictor set. This included SI-unit duplicates and non-reference lab values.

**Fix:** Introduced strict curated feature filter (`SELECTED_FEATURES`).

**Impact:** AUROC dropped from ~1.000 to ~0.85–0.95.

---

## 2026-07-25 — Leakage audit round 1

**Method:** Cross-referenced `feature_names.json` against `leakage_map.json`.

**Result:** No explicit leakage detected. Problem was elsewhere.

---

## 2026-07-20 — Baseline training (initial)

**Data:** NHANES 2017-2018, six components only.

**Models:** LR, RF, XGBoost, LightGBM, CatBoost.

**Result:** XGBoost achieved AUROC 0.998 (diabetes), 1.000 (dyslipidemia), 0.969 (hypertension).

**Observation:** Perfect scores flagged as suspicious. Triggered the leakage investigation.

---

## 2026-07-13 — Initial dataset verification (DEMO, BMX, GHB, GLU, BPX, BIOPRO)

- All six components verified with `01_dataset_verification.ipynb`.
- Verified CSVs saved to `data/interim/`.
- Documented in `Data_Quality_Log.md`.


## 2026-09-27 — Permutation importance audit and leak fix

Full permutation importance on external 2021–2023 (N=3425, 5 perms/feature).

Findings:
- Diabetes: LBXSOSSI (+0.28), LBXSNASI (+0.23), LBXSBU (+0.08) dominated
- Metabolic syndrome: same three features dominated
- Hypertension, CKD, dyslipidemia, CVD, liver disease: no leakage

Fix: Updated leakage_map. Retrained OneHealth-Net v3.
Results: Mean external AUROC 0.8245 (was 0.8342). Diabetes 0.8683 (was 0.9335).