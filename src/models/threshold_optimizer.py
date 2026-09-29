# src/models/threshold_optimizer.py

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    confusion_matrix, roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef
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

def compute_metrics(y_true, y_pred_prob, threshold):
    y_pred = (y_pred_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    youden = sensitivity + specificity - 1
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'mcc': mcc,
        'youden': youden,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

def best_threshold_youden(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    idx = np.argmax(youden)
    return thresholds[idx]

def best_threshold_f1(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    idx = np.argmax(f1_scores)
    return thresholds[idx] if idx < len(thresholds) else 0.5

def main():
    results = []
    for disease in DISEASE_MAP.keys():
        model_path = os.path.join(MODEL_DIR, f"{disease}_final_model.pkl")
        if not os.path.exists(model_path):
            print(f"Model not found for {disease}, skipping.")
            continue
        model = joblib.load(model_path)
        X_test, y_test = load_disease_data(disease)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Threshold by Youden's J
        thresh_youden = best_threshold_youden(y_test, y_prob)
        metrics_youden = compute_metrics(y_test, y_prob, thresh_youden)

        # Threshold by F1
        thresh_f1 = best_threshold_f1(y_test, y_prob)
        metrics_f1 = compute_metrics(y_test, y_prob, thresh_f1)

        # AUCs
        auroc = roc_auc_score(y_test, y_prob)
        auprc = average_precision_score(y_test, y_prob)

        print(f"\n{'='*60}\n{disease}\n{'='*60}")
        print(f"AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")
        print(f"\n[Youden Threshold: {thresh_youden:.3f}]")
        for k,v in metrics_youden.items():
            if k not in ['tp','fp','fn','tn']:
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"\n[F1 Threshold: {thresh_f1:.3f}]")
        for k,v in metrics_f1.items():
            if k not in ['tp','fp','fn','tn']:
                print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

        # Save both metric sets with disease name
        metrics_youden['disease'] = disease
        metrics_youden['method'] = 'youden'
        metrics_youden['auroc'] = auroc
        metrics_youden['auprc'] = auprc
        metrics_f1['disease'] = disease
        metrics_f1['method'] = 'f1'
        metrics_f1['auroc'] = auroc
        metrics_f1['auprc'] = auprc
        results.extend([metrics_youden, metrics_f1])

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(MODEL_DIR, "threshold_optimization.csv"), index=False)
    print(f"\nSaved threshold optimization results to {MODEL_DIR}/threshold_optimization.csv")

if __name__ == "__main__":
    main()