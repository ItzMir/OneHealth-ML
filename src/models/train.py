import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, brier_score_loss
)
from src.models.knowledge_graph import KGModel

RANDOM_STATE = 42
N_SPLITS = 5   # 5‑fold CV
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

def load_metadata():
    with open(os.path.join(PROCESSED_DIR, "feature_names.json"), "r") as f:
        feature_names = json.load(f)
    with open(os.path.join(PROCESSED_DIR, "leakage_map.json"), "r") as f:
        leakage_map = json.load(f)
    return feature_names, leakage_map

def get_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced', C=0.1
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=3, min_samples_leaf=5, max_features='sqrt',
            random_state=RANDOM_STATE, class_weight='balanced', n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_weight=10,
            random_state=RANDOM_STATE, eval_metric='logloss'
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=3.0, reg_alpha=1.0, min_child_samples=20,
            random_state=RANDOM_STATE, class_weight='balanced', verbose=-1
        ),
        "catboost": CatBoostClassifier(
            iterations=200, depth=2, learning_rate=0.05,
            subsample=0.8, l2_leaf_reg=5.0,
            random_seed=RANDOM_STATE, auto_class_weights='Balanced', verbose=0
        ),
    }

def evaluate_model(model, X, y, feature_names=None):
    if feature_names is not None and isinstance(model, KGModel):
        y_prob = model.predict_proba(X, feature_names)[:, 1]
        y_pred = model.predict(X, feature_names)
    else:
        y_prob = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
    return {
        "auroc": roc_auc_score(y, y_prob),
        "auprc": average_precision_score(y, y_prob),
        "f1": f1_score(y, y_pred),
        "mcc": matthews_corrcoef(y, y_pred),
        "brier": brier_score_loss(y, y_prob),
    }

def cv_train_and_evaluate(model, X, y, feature_names=None, model_name=""):
    """
    Perform stratified 5-fold CV. Returns list of metric dicts (one per fold).
    Also fits the model on the full X,y for saving.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Clone model (for non-KG models we can just re-instantiate; for KG we need a new instance)
        if isinstance(model, KGModel):
            model_fold = KGModel()
            model_fold.fit(X_tr, y_tr, "", {}, feature_names)  # disease & leakage_map not needed for CV eval? We'll need them
            # Actually KGModel requires leakage_map and disease for graph building; we'll skip CV for KG for now.
            # We'll handle KG separately later.
            continue
        else:
            model_fold = model.__class__(**model.get_params())
            model_fold.fit(X_tr, y_tr)

        fold_metrics.append(evaluate_model(model_fold, X_val, y_val, feature_names))

    # After CV, refit on full training set for saving
    model.fit(X, y)

    # Aggregate metrics
    avg_metrics = {}
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        avg_metrics[f"{key}_mean"] = np.mean(values)
        avg_metrics[f"{key}_std"] = np.std(values)
    return avg_metrics, fold_metrics

def main():
    feature_names_all, leakage_map = load_metadata()
    results = []

    for disease, label_col in DISEASE_MAP.items():
        print(f"\n{'='*60}\nTraining: {disease}\n{'='*60}")
        X_train, X_test, y_train, y_test = load_disease_data(disease)
        feat_names = feature_names_all[label_col]

        models = get_models()

        # Train Knowledge Graph separately (no CV due to graph re-building complexity)
        kg = KGModel()
        kg.fit(X_train, y_train, disease, leakage_map, feat_names)
        # Evaluate KG on test set (no CV)
        kg_test_metrics = evaluate_model(kg, X_test, y_test, feat_names)
        kg_test_metrics["disease"] = disease
        kg_test_metrics["model"] = "knowledge_graph"
        results.append(kg_test_metrics)
        joblib.dump(kg, os.path.join(MODEL_DIR, f"{disease}_knowledge_graph.pkl"))
        print(f"  knowledge_graph (test): AUROC={kg_test_metrics['auroc']:.3f}, "
              f"AUPRC={kg_test_metrics['auprc']:.3f}, F1={kg_test_metrics['f1']:.3f}")

        # For other models, run CV and evaluate on test set
        for name, model in models.items():
            print(f"  {name}:")
            # CV on training data
            cv_avg, cv_folds = cv_train_and_evaluate(model, X_train, y_train, feat_names, model_name=name)

            # Test set evaluation (model is already fit on full training data)
            test_metrics = evaluate_model(model, X_test, y_test, feat_names)
            test_metrics["disease"] = disease
            test_metrics["model"] = name
            # Add CV metrics to the record
            test_metrics.update({f"cv_{k}": v for k, v in cv_avg.items()})
            results.append(test_metrics)

            # Save model
            fname = os.path.join(MODEL_DIR, f"{disease}_{name}.pkl")
            joblib.dump(model, fname)

            print(f"    Test:  AUROC={test_metrics['auroc']:.3f}, "
                  f"AUPRC={test_metrics['auprc']:.3f}, F1={test_metrics['f1']:.3f}")
            print(f"    CV:    AUROC={cv_avg['auroc_mean']:.3f}±{cv_avg['auroc_std']:.3f}, "
                  f"AUPRC={cv_avg['auprc_mean']:.3f}±{cv_avg['auprc_std']:.3f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(MODEL_DIR, "model_metrics.csv"), index=False)
    print(f"\nAll models and metrics saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()