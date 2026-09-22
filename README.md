# OneHealth-ML

**Early Multi-Disease Prediction System using Machine Learning**

Author: Ifthakhar Rashid  
Affiliation: Department of Computer Science & Engineering, United International University  
Contact: irashid2330945@bscse.uiu.ac.bd

---

## Overview

OneHealth-ML is an explainable machine learning framework that predicts the risk of seven interconnected chronic diseases using routinely collected health data from NHANES (2013–2018 training; 2021–2023 external validation):

1. Type 2 Diabetes
2. Hypertension
3. Chronic Kidney Disease (CKD)
4. Dyslipidemia
5. Cardiovascular Disease (CVD)
6. Metabolic Syndrome
7. Liver Disease / NAFLD Risk

The system is designed for research, thesis, and eventual clinical deployment.

---

## Key Features

- **Reproducible pipeline** — every step is scripted and versioned.
- **Leakage-free** — label-defining variables removed per disease.
- **Multi-model** — Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, MLP, Knowledge Graph, and soft-voting ensembles.
- **External validation** — tested on a post-pandemic NHANES cycle (2021–2023) without retraining.
- **Explainable** — SHAP and feature importance plots for each disease.
- **Documented** — Data Quality Log, Experiment Log, Research Decision Log, Master Feature Catalog.

---

## Project Structure
OneHealth-ML/
├── data/
│ ├── raw/ # NHANES .xpt files (2013–2014, 2015–2016, 2017–2018, 2021–2023)
│ ├── interim/ # Verified per-component CSVs
│ ├── processed/ # Final arrays, scalers, JSON metadata
│ └── external_validation/ # External validation metrics
├── docs/ # Markdown logs + Master Feature Catalog
├── models/ # Trained models, scalers, feature importances
├── notebooks/ # Orchestration notebooks
├── references/ # BibTeX bibliography
├── reports/ # Figures (SHAP, ROC, PR)
├── src/ # Reusable Python modules
├── thesis/ # Thesis draft
├── requirements.txt
└── README.md