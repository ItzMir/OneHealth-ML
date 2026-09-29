# src/models/tune.py

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, randint, loguniform

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

def load_feature_names(disease):
    with open(os.path.join(PROCESSED_DIR, "feature_names.json")) as f:
        fn = json.load(f)
    # feature_names.json keys are label_disease
    label = DISEASE_MAP[disease]
    return fn[label]

def get_param_grids():
    """Define hyperparameter search spaces for each model type."""
    lr_grid = {
        'C': loguniform(1e-3, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],  # liblinear supports l1/l2, saga supports both
    }
    rf_grid = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 15),
        'min_samples_leaf': randint(1, 10),
        'min_samples_split': randint(2, 10),
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None],
    }
    xgb_grid = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': loguniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_lambda': loguniform(1e-3, 10),
        'reg_alpha': loguniform(1e-3, 10),
        'min_child_weight': randint(1, 10),
    }
    lgbm_grid = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': loguniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_lambda': loguniform(1e-3, 10),
        'reg_alpha': loguniform(1e-3, 10),
        'min_child_samples': randint(5, 50),
    }
    cat_grid = {
        'iterations': randint(100, 500),
        'depth': randint(3, 10),
        'learning_rate': loguniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'l2_leaf_reg': loguniform(1e-3, 10),
        'border_count': randint(32, 255),
    }
    return {
        'logistic_regression': lr_grid,
        'random_forest': rf_grid,
        'xgboost': xgb_grid,
        'lightgbm': lgbm_grid,
        'catboost': cat_grid,
    }

def get_model_constructor(model_name):
    if model_name == 'logistic_regression':
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    elif model_name == 'random_forest':
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == 'xgboost':
        return XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False)
    elif model_name == 'lightgbm':
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    elif model_name == 'catboost':
        return CatBoostClassifier(random_seed=RANDOM_STATE, verbose=0)
    else:
        raise ValueError(f"Unknown model: {model_name}")

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

def tune_disease(disease, model_names, n_iter=30):
    """
    Tune each specified model for a given disease.
    Returns list of dicts with metrics and best models.
    """
    X_train, X_test, y_train, y_test = load_disease_data(disease)
    feature_names = load_feature_names(disease)
    # Trim feature_names to match X columns (if needed)
    feature_names = feature_names[:X_train.shape[1]]

    param_grids = get_param_grids()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for model_name in model_names:
        print(f"  Tuning {model_name}...")
        base_model = get_model_constructor(model_name)
        grid = param_grids[model_name]
        search = RandomizedSearchCV(
            base_model,
            param_distributions=grid,
            n_iter=n_iter,
            scoring='average_precision',
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Evaluate on test set
        test_metrics = evaluate_model(best_model, X_test, y_test)
        test_metrics['disease'] = disease
        test_metrics['model'] = model_name
        test_metrics['best_params'] = search.best_params_
        test_metrics['cv_auprc_mean'] = search.cv_results_['mean_test_score'][search.best_index_]
        test_metrics['cv_auprc_std'] = search.cv_results_['std_test_score'][search.best_index_]
        results.append(test_metrics)

        # Save intermediate model
        fname = os.path.join(MODEL_DIR, f"{disease}_{model_name}_tuned.pkl")
        joblib.dump(best_model, fname)

        print(f"    Test AUROC={test_metrics['auroc']:.3f}, AUPRC={test_metrics['auprc']:.3f}, "
              f"CV AUPRC={test_metrics['cv_auprc_mean']:.3f}±{test_metrics['cv_auprc_std']:.3f}")

    return results, feature_names

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from the final model.
    For linear models, use absolute coefficients.
    For tree models, use feature_importances_.
    """
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = np.zeros(len(feature_names))
    # Sort descending
    indices = np.argsort(importance)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = [float(importance[i]) for i in indices]
    return {name: imp for name, imp in zip(sorted_names, sorted_importances)}

def main():
    # Models to tune (we exclude knowledge_graph because it's custom)
    model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost']

    all_results = []
    final_models = {}

    for disease in DISEASE_MAP.keys():
        print(f"\n{'='*60}\nTuning {disease}\n{'='*60}")
        results, feature_names = tune_disease(disease, model_names, n_iter=25)
        all_results.extend(results)

        # Choose best model based on CV AUPRC
        best_result = max(results, key=lambda x: x['cv_auprc_mean'])
        final_model_name = best_result['model']
        final_model = joblib.load(os.path.join(MODEL_DIR, f"{disease}_{final_model_name}_tuned.pkl"))
        final_models[disease] = final_model

        # Save final feature importance
        importance = get_feature_importance(final_model, feature_names)
        with open(os.path.join(MODEL_DIR, f"{disease}_feature_importance.json"), "w") as f:
            json.dump(importance, f, indent=2)

        # Save final scaler (already saved during feature engineering, copy it)
        scaler_src = os.path.join(PROCESSED_DIR, f"{disease}_scaler.pkl")
        scaler_dst = os.path.join(MODEL_DIR, f"{disease}_scaler.pkl")
        if os.path.exists(scaler_src):
            joblib.dump(joblib.load(scaler_src), scaler_dst)

        # Save final model with a consistent name
        final_model_path = os.path.join(MODEL_DIR, f"{disease}_final_model.pkl")
        joblib.dump(final_model, final_model_path)

        print(f"\n  Best model for {disease}: {final_model_name} (CV AUPRC={best_result['cv_auprc_mean']:.3f})")

    # Save summary CSV
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(MODEL_DIR, "tuned_model_metrics.csv"), index=False)
    print(f"\nTuning complete. Summary saved to {MODEL_DIR}/tuned_model_metrics.csv")

if __name__ == "__main__":
    main()