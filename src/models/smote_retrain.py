# src/models/smote_retrain.py

import os
import numpy as np
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

RANDOM_STATE = 42
PROCESSED_DIR = "../data/processed"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

DISEASES_TO_SMOTE = ["ckd", "cvd", "liver_disease"]

def load_disease_data(disease):
    base = os.path.join(PROCESSED_DIR, disease)
    X_train = np.load(f"{base}_X_train.npy")
    X_test  = np.load(f"{base}_X_test.npy")
    y_train = np.load(f"{base}_y_train.npy")
    y_test  = np.load(f"{base}_y_test.npy")
    return X_train, X_test, y_train, y_test

def get_model_from_file(disease):
    """Load the final tuned model and return a new instance with same class and hyperparameters."""
    model_path = os.path.join(MODEL_DIR, f"{disease}_final_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Final model not found for {disease}: {model_path}")
    model = joblib.load(model_path)
    params = model.get_params()
    # Determine class
    if isinstance(model, LogisticRegression):
        new_model = LogisticRegression(**params)
    elif isinstance(model, RandomForestClassifier):
        new_model = RandomForestClassifier(**params)
    elif isinstance(model, XGBClassifier):
        new_model = XGBClassifier(**params)
    elif isinstance(model, LGBMClassifier):
        new_model = LGBMClassifier(**params)
    elif isinstance(model, CatBoostClassifier):
        new_model = CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    return new_model, model.__class__.__name__

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
    for disease in DISEASES_TO_SMOTE:
        print(f"\n{'='*60}\nSMOTE Retraining: {disease}\n{'='*60}")
        X_train, X_test, y_train, y_test = load_disease_data(disease)

        print(f"Before SMOTE: train positives = {y_train.sum()}, negatives = {len(y_train)-y_train.sum()}")

        # Apply SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: train positives = {y_train_smote.sum()}, negatives = {len(y_train_smote)-y_train_smote.sum()}")

        # Get new model with same hyperparameters
        model, model_name = get_model_from_file(disease)
        model.fit(X_train_smote, y_train_smote)

        # Evaluate on original test set
        metrics = evaluate_model(model, X_test, y_test)
        metrics['disease'] = disease
        metrics['model'] = model_name
        results.append(metrics)

        # Save SMOTE model
        out_path = os.path.join(MODEL_DIR, f"{disease}_smote_model.pkl")
        joblib.dump(model, out_path)
        print(f"Saved: {out_path}")
        print(f"  AUROC={metrics['auroc']:.3f}, AUPRC={metrics['auprc']:.3f}, F1={metrics['f1']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(MODEL_DIR, "smote_retrain_metrics.csv"), index=False)
    print(f"\nSummary saved to {MODEL_DIR}/smote_retrain_metrics.csv")

if __name__ == "__main__":
    main()