# src/models/threshold_analysis.py

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, roc_auc_score, average_precision_score
)

MODEL_DIR = "../models"
PROCESSED_DIR = "../data/processed"
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
    X_test = np.load(f"{base}_X_test.npy")
    y_test = np.load(f"{base}_y_test.npy")
    return X_test, y_test

def evaluate_at_threshold(y_true, y_pred_prob, threshold):
    y_pred = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

def main(target_recall=0.90):
    results = []
    for disease in DISEASE_MAP.keys():
        model_path = os.path.join(MODEL_DIR, f"{disease}_final_model.pkl")
        if not os.path.exists(model_path):
            print(f"Model not found for {disease}, skipping.")
            continue
        model = joblib.load(model_path)
        X_test, y_test = load_disease_data(disease)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Find threshold for target recall
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        # Find threshold closest to target recall
        idx = np.argmin(np.abs(recalls - target_recall))
        best_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

        metrics = evaluate_at_threshold(y_test, y_prob, best_threshold)
        metrics['disease'] = disease
        metrics['auroc'] = roc_auc_score(y_test, y_prob)
        metrics['auprc'] = average_precision_score(y_test, y_prob)
        results.append(metrics)

        print(f"\n{disease}: threshold={best_threshold:.3f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}, Specificity: {metrics['specificity']:.3f}")
        print(f"  PPV: {metrics['ppv']:.3f}, NPV: {metrics['npv']:.3f}")
        print(f"  AUROC: {metrics['auroc']:.3f}, AUPRC: {metrics['auprc']:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(MODEL_DIR, "clinical_metrics.csv"), index=False)
    print(f"\nSaved clinical metrics to {MODEL_DIR}/clinical_metrics.csv")

if __name__ == "__main__":
    main(target_recall=0.90)