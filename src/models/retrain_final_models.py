# src/models/retrain_final_models.py

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, brier_score_loss

RANDOM_STATE = 42
PROCESSED_DIR = "../data/processed"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

DISEASE_MAP = {
    "diabetes": "label_diabetes",
    "hypertension": "label_hypertension",
    "ckd": "label_ckd",
    "dyslipidemia": "label_dyslipidemia",
    "cvd": "label_cvd",
    "metabolic_syndrome": "label_metabolic_syndrome",
    "liver_disease": "label_liver_disease",
}

def load_disease_data(disease):
    base = os.path.join(PROCESSED_DIR, disease)
    X_train = np.load(f"{base}_X_train.npy")
    X_test  = np.load(f"{base}_X_test.npy")
    y_train = np.load(f"{base}_y_train.npy")
    y_test  = np.load(f"{base}_y_test.npy")
    return X_train, X_test, y_train, y_test

def load_model_instance(disease):
    """Load existing final model and return a new instance with same class and hyperparameters."""
    model_path = os.path.join(MODEL_DIR, f"{disease}_final_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Final model not found: {model_path}")
    old_model = joblib.load(model_path)
    params = old_model.get_params()
    if isinstance(old_model, LogisticRegression):
        new_model = LogisticRegression(**params)
    elif isinstance(old_model, RandomForestClassifier):
        new_model = RandomForestClassifier(**params)
    elif isinstance(old_model, XGBClassifier):
        new_model = XGBClassifier(**params)
    elif isinstance(old_model, LGBMClassifier):
        new_model = LGBMClassifier(**params)
    elif isinstance(old_model, CatBoostClassifier):
        new_model = CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {type(old_model)}")
    return new_model, type(old_model).__name__

def evaluate_model(model, X, y):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    return {
        'auroc': roc_auc_score(y, y_prob),
        'auprc': average_precision_score(y, y_prob),
        'f1': f1_score(y, y_pred),
        'mcc': matthews_corrcoef(y, y_pred),
        'brier': brier_score_loss(y, y_prob),
    }

def main():
    results = []
    for disease in DISEASE_MAP.keys():
        print(f"\n{'='*60}\nRetraining {disease}\n{'='*60}")
        X_train, X_test, y_train, y_test = load_disease_data(disease)

        # Create model with same hyperparameters
        model, model_name = load_model_instance(disease)
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        metrics['disease'] = disease
        metrics['model'] = model_name
        results.append(metrics)

        # Save updated final model (overwrite)
        model_path = os.path.join(MODEL_DIR, f"{disease}_final_model.pkl")
        joblib.dump(model, model_path)
        print(f"Saved: {model_path}")
        print(f"  AUROC={metrics['auroc']:.3f}, AUPRC={metrics['auprc']:.3f}, F1={metrics['f1']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(MODEL_DIR, "retrained_final_metrics.csv"), index=False)
    print(f"\nSummary saved to {MODEL_DIR}/retrained_final_metrics.csv")

if __name__ == "__main__":
    main()