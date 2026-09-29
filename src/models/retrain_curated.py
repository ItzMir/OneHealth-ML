"""
Retrain OneHealth-ML models on curated features only.
Exports deployment artifacts for FastAPI backend.

Usage:
    python src/models/retrain_curated.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, brier_score_loss

# -------------------------------
# Configuration
# -------------------------------

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/processed/labeled_master.csv"
OUTPUT_DIR = PROJECT_ROOT / "models"
SCALER_DIR = PROJECT_ROOT / "scalers"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------------------------
# Curated feature list (raw variables)
# -------------------------------
# These are the variables that the web form will collect.
# Derived features (eGFR, HOMA-IR, NLR, WHR, mean BP) will be computed later.

CURATED_RAW_FEATURES = [
    "RIDAGEYR",        # age
    "RIAGENDR",        # sex (1=male, 2=female)
    "BMXBMI",          # BMI
    "BMXWAIST",        # waist circumference
    "BMXHIP",          # hip circumference
    "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
    "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4",
    "BPXPLS",          # pulse rate
    "BPXPULS",         # pulse regularity
    "LBXGH",           # HbA1c
    "LBXGLU",          # fasting glucose
    "LBXIN",           # insulin
    "LBXSCR",          # creatinine
    "LBXSBU",          # BUN
    "LBXSAL",          # albumin
    "LBXSTP",          # total protein
    "LBXSCA",          # calcium
    "LBXSNASI",        # sodium
    "LBXSKSI",         # potassium
    "LBXSCLSI",        # chloride
    "LBXSTB",          # total bilirubin
    "LBXSAPSI",        # ALP
    "LBXSATSI",        # ALT
    "LBXSASSI",        # AST
    "LBXSGTSI",        # GGT
    "LBXSPH",          # phosphorus
    "LBXSC3SI",        # bicarbonate
    "LBXSUA",          # uric acid
    "LBXSIR",          # iron
    "LBXSGB",          # globulin
    "LBXSCK",          # creatine kinase
    "LBXSLDSI",        # LDH
    "LBXSOSSI",        # osmolality
    "LBXTC",           # total cholesterol
    "LBDHDD",          # HDL
    "LBXTR",           # triglycerides
    "LBXHSCRP",        # hsCRP
    "URDACT",          # urine ACR
    "LBXWBCSI",        # WBC
    "LBXHGB",          # hemoglobin
    "LBXPLTSI",        # platelets
    "LBXNEPCT",        # neutrophil %
    "LBXLYPCT",        # lymphocyte %
    "SMQ020",          # smoking status (1=every day, 2=some days, 3=not at all)
    "ALQ111",          # ever had alcohol drink (1=yes,2=no)
    "PAQ605",          # moderate physical activity (1=yes,2=no)
]

# -------------------------------
# Load master dataset
# -------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded master dataset: {df.shape}")

# -------------------------------
# Derived features
# -------------------------------
def egfr_ckdepi(scr, age, sex):
    kappa = np.where(sex == 1, 0.9, 0.7)
    alpha = np.where(sex == 1, -0.302, -0.241)
    ratio = scr / kappa
    return 142 * (np.minimum(ratio, 1) ** alpha) * (np.maximum(ratio, 1) ** -1.200) * (0.9938 ** age) * np.where(sex == 1, 1.0, 1.012)

df["eGFR"] = egfr_ckdepi(df["LBXSCR"], df["RIDAGEYR"], df["RIAGENDR"])

df["mean_sbp"] = df[["BPXSY1","BPXSY2","BPXSY3","BPXSY4"]].mean(axis=1, skipna=True)
df["mean_dbp"] = df[["BPXDI1","BPXDI2","BPXDI3","BPXDI4"]].mean(axis=1, skipna=True)

df["HOMA_IR"] = (df["LBXIN"] * df["LBXGLU"]) / 405
df["NLR"] = df["LBXNEPCT"] / df["LBXLYPCT"].replace(0, np.nan)
df["NLR"] = df["NLR"].replace([np.inf, -np.inf], np.nan)
df["WHR"] = df["BMXWAIST"] / df["BMXHIP"].replace(0, np.nan)

# Replace NaN with median for derived features (will be imputed later too)

# -------------------------------
# Label generation functions (same as before)
# -------------------------------

def label_diabetes(row):
    if row["RIDEXPRG"] == 1:
        return np.nan
    if pd.isna(row["LBXGH"]) and pd.isna(row["LBXGLU"]) and pd.isna(row["DIQ010"]):
        return np.nan
    if (row["LBXGH"] >= 6.5) or (row["LBXGLU"] >= 126) or (row["DIQ010"] == 1):
        return 1
    return 0

def label_hypertension(row):
    if pd.isna(row["mean_sbp"]) and pd.isna(row["mean_dbp"]) and pd.isna(row["BPQ020"]):
        return np.nan
    if (row["mean_sbp"] >= 130) or (row["mean_dbp"] >= 80) or (row["BPQ020"] == 1):
        return 1
    return 0

def label_ckd(row):
    if pd.isna(row["eGFR"]) and pd.isna(row["KIQ022"]) and pd.isna(row["KIQ025"]):
        return np.nan
    if (row["eGFR"] < 60) or (row["KIQ022"] == 1) or (row["KIQ025"] == 1):
        return 1
    return 0

def label_dyslipidemia(row):
    if pd.isna(row["LBXTC"]) and pd.isna(row["LBDHDD"]) and pd.isna(row["LBXTR"]):
        return np.nan
    hdl_low = (row["LBDHDD"] < 40) if row["RIAGENDR"] == 1 else (row["LBDHDD"] < 50)
    if (row["LBXTC"] >= 200) or hdl_low or (row["LBXTR"] >= 150):
        return 1
    return 0

def label_cvd(row):
    cols = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
    if row[cols].isna().all():
        return np.nan
    if (row[cols] == 1).any():
        return 1
    return 0

def label_metabolic_syndrome(row):
    components = 0
    if (row["RIAGENDR"] == 1 and row["BMXWAIST"] > 102) or (row["RIAGENDR"] == 2 and row["BMXWAIST"] > 88):
        components += 1
    if row["LBXTR"] >= 150: components += 1
    hdl_low = (row["LBDHDD"] < 40) if row["RIAGENDR"] == 1 else (row["LBDHDD"] < 50)
    if hdl_low: components += 1
    if not pd.isna(row["mean_sbp"]) and not pd.isna(row["mean_dbp"]):
        if row["mean_sbp"] >= 130 or row["mean_dbp"] >= 85: components += 1
    if row["LBXGLU"] >= 100: components += 1
    if pd.isna(row["BMXWAIST"]) or pd.isna(row["LBXTR"]) or pd.isna(row["LBDHDD"]) or pd.isna(row["mean_sbp"]) or pd.isna(row["LBXGLU"]):
        return np.nan
    return 1 if components >= 3 else 0

def label_liver_disease(row):
    if pd.isna(row["LBXSATSI"]) and pd.isna(row["LBXSASSI"]) and pd.isna(row["LBXSGTSI"]) and pd.isna(row["BMXBMI"]):
        return np.nan
    alt_uln = 40 if row["RIAGENDR"] == 1 else 30
    enzyme_elevated = (row["LBXSATSI"] > alt_uln) or (row["LBXSASSI"] > 40) or (row["LBXSGTSI"] > 50)
    if enzyme_elevated and row["BMXBMI"] >= 25:
        return 1
    return 0

# Apply labels
df["label_diabetes"] = df.apply(label_diabetes, axis=1)
df["label_hypertension"] = df.apply(label_hypertension, axis=1)
df["label_ckd"] = df.apply(label_ckd, axis=1)
df["label_dyslipidemia"] = df.apply(label_dyslipidemia, axis=1)
df["label_cvd"] = df.apply(label_cvd, axis=1)
df["label_metabolic_syndrome"] = df.apply(label_metabolic_syndrome, axis=1)
df["label_liver_disease"] = df.apply(label_liver_disease, axis=1)

# -------------------------------
# Define predictor set (curated + derived)
# -------------------------------
CURATED_DERIVED_FEATURES = ["eGFR", "HOMA_IR", "NLR", "WHR", "mean_sbp", "mean_dbp"]

# Leakage map: label-defining variables per disease
LEAKAGE_MAP = {
    "label_diabetes": ["LBXGH", "LBXGLU", "HOMA_IR", "DIQ010"],
    "label_hypertension": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
                            "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4",
                            "BPXPLS", "BPXPULS", "mean_sbp", "mean_dbp", "BPQ020"],
    "label_ckd": ["LBXSCR", "eGFR", "KIQ022", "KIQ025", "URDACT"],
    "label_dyslipidemia": ["LBXTC", "LBDHDD", "LBXTR"],
    "label_cvd": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
    "label_metabolic_syndrome": ["BMXWAIST", "LBXTR", "LBDHDD",
                                  "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
                                  "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4",
                                  "mean_sbp", "mean_dbp", "LBXGLU", "WHR"],
    "label_liver_disease": ["LBXSATSI", "LBXSASSI", "LBXSGTSI", "BMXBMI"],
}

# -------------------------------
# Feature selection per disease
# -------------------------------
feature_names_per_disease = {}
for label in LEAKAGE_MAP.keys():
    exclude = set(LEAKAGE_MAP[label])
    # Use curated raw + derived, minus excluded
    features = [f for f in CURATED_RAW_FEATURES + CURATED_DERIVED_FEATURES if f not in exclude]
    feature_names_per_disease[label] = features

# Save feature_names.json
with open(os.path.join(ARTIFACT_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_names_per_disease, f, indent=2)

# Save leakage_map.json
with open(os.path.join(ARTIFACT_DIR, "leakage_map.json"), "w") as f:
    json.dump(LEAKAGE_MAP, f, indent=2)

# -------------------------------
# Prepare data per disease
# -------------------------------
# We will retrain all models and choose best via CV AUPRC, then save that model.
models_to_train = {
    "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced', C=0.1),
    "random_forest": RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=5, max_features='sqrt', random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1),
    "xgboost": XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=1.0, min_child_weight=10, random_state=RANDOM_STATE, eval_metric='logloss'),
    "lightgbm": LGBMClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0, reg_alpha=1.0, min_child_samples=20, random_state=RANDOM_STATE, class_weight='balanced', verbose=-1),
    "catboost": CatBoostClassifier(iterations=200, depth=2, learning_rate=0.05, subsample=0.8, l2_leaf_reg=5.0, random_seed=RANDOM_STATE, auto_class_weights='Balanced', verbose=0),
}

results = []
best_models = {}

for label in feature_names_per_disease.keys():
    print(f"\n=== Processing {label} ===")
    features = feature_names_per_disease[label]

    # Drop rows with missing label
    mask = df[label].notna()
    data = df.loc[mask, features].copy()
    y = df.loc[mask, label].astype(int)

    # Impute missing values with median
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(data)

    # Split train/test (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale using training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    disease_key = label.replace("label_", "")
    joblib.dump(scaler, os.path.join(SCALER_DIR, f"{disease_key}_scaler.pkl"))

    # Cross-validation to select best model
    cv_scores = {}
    for model_name, model in models_to_train.items():
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        auc_scores, auprc_scores = [], []
        for train_idx, val_idx in skf.split(X_train_scaled, y_train):
            X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # Clone model
            model_clone = model.__class__(**model.get_params()) if hasattr(model, "get_params") else None
            if model_clone is None:
                # For CatBoost, need special handling
                model_clone = CatBoostClassifier(iterations=200, depth=2, learning_rate=0.05, subsample=0.8, l2_leaf_reg=5.0, random_seed=RANDOM_STATE, auto_class_weights='Balanced', verbose=0)
            model_clone.fit(X_tr, y_tr)
            y_prob = model_clone.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, y_prob))
            auprc_scores.append(average_precision_score(y_val, y_prob))

        cv_scores[model_name] = {
            "cv_auroc_mean": np.mean(auc_scores),
            "cv_auroc_std": np.std(auc_scores),
            "cv_auprc_mean": np.mean(auprc_scores),
            "cv_auprc_std": np.std(auprc_scores),
        }
        print(f"  {model_name}: AUROC={np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}, AUPRC={np.mean(auprc_scores):.3f} ± {np.std(auprc_scores):.3f}")

    # Choose best model by CV AUPRC
    best_model_name = max(cv_scores, key=lambda k: cv_scores[k]['cv_auprc_mean'])
    print(f"  Best: {best_model_name}")

    # Retrain best model on full training data (X_train_scaled) and evaluate on test
    best_model_cls = models_to_train[best_model_name].__class__
    if best_model_name == "catboost":
        best_model = CatBoostClassifier(iterations=200, depth=2, learning_rate=0.05, subsample=0.8, l2_leaf_reg=5.0, random_seed=RANDOM_STATE, auto_class_weights='Balanced', verbose=0)
    else:
        best_model = best_model_cls(**models_to_train[best_model_name].get_params())
    best_model.fit(X_train_scaled, y_train)

    # Test metrics
    y_prob_test = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    test_metrics = {
        "auroc": roc_auc_score(y_test, y_prob_test),
        "auprc": average_precision_score(y_test, y_prob_test),
        "f1": f1_score(y_test, y_pred_test),
        "mcc": matthews_corrcoef(y_test, y_pred_test),
        "brier": brier_score_loss(y_test, y_prob_test),
    }
    print(f"  Test metrics: {test_metrics}")

    # Save model
    joblib.dump(best_model, os.path.join(OUTPUT_DIR, f"{disease_key}_{best_model_name}.pkl"))

    # Save feature importance (global) if available
    try:
        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
        elif hasattr(best_model, "coef_"):
            importances = np.abs(best_model.coef_[0])
        else:
            importances = None
        if importances is not None:
            importance_dict = {feat: float(imp) for feat, imp in zip(features, importances)}
            with open(os.path.join(ARTIFACT_DIR, f"{disease_key}_feature_importance.json"), "w") as f:
                json.dump(importance_dict, f, indent=2)
    except:
        pass

    results.append({
        "disease": disease_key,
        "best_model": best_model_name,
        **cv_scores[best_model_name],
        **test_metrics
    })

# Save overall metrics
with open(os.path.join(ARTIFACT_DIR, "model_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nAll artifacts exported successfully.")
print("Files saved:")
print(f"  Models: {OUTPUT_DIR}")
print(f"  Scalers: {SCALER_DIR}")
print(f"  Artifacts: {ARTIFACT_DIR}")