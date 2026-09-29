# src/models/tune_extended.py

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
)
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


class SimpleEnsemble:
    """
    A simple soft-voting ensemble that averages predict_proba from
    a list of already-fitted models.
    Avoids cloning issues with CatBoost (np.float64 parameters).
    """

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        # Models are already fitted; nothing to do.
        return self

    def predict_proba(self, X):
        probas = [model.predict_proba(X) for model in self.models]
        return np.mean(probas, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def load_disease_data(disease):
    base = os.path.join(PROCESSED_DIR, disease)
    X_train = np.load(f"{base}_X_train.npy")
    X_test = np.load(f"{base}_X_test.npy")
    y_train = np.load(f"{base}_y_train.npy")
    y_test = np.load(f"{base}_y_test.npy")
    return X_train, X_test, y_train, y_test


def load_feature_names(disease):
    with open(os.path.join(PROCESSED_DIR, "feature_names.json")) as f:
        fn = json.load(f)
    return fn[DISEASE_MAP[disease]]


def get_param_grids():
    lr_grid = {
        'C': loguniform(1e-3, 10),
        'l1_ratio': uniform(0, 1),
        'solver': ['saga'],
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
    mlp_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
        'alpha': loguniform(1e-4, 1e-1),
        'learning_rate_init': loguniform(1e-4, 1e-2),
        'max_iter': [1000],
        'early_stopping': [True],
        'random_state': [RANDOM_STATE],
    }
    return {
        'logistic_regression': lr_grid,
        'random_forest': rf_grid,
        'xgboost': xgb_grid,
        'lightgbm': lgbm_grid,
        'catboost': cat_grid,
        'mlp': mlp_grid,
    }


def get_model_constructor(model_name):
    if model_name == 'logistic_regression':
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    elif model_name == 'random_forest':
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == 'xgboost':
        return XGBClassifier(
            random_state=RANDOM_STATE, eval_metric='logloss', use_label_encoder=False
        )
    elif model_name == 'lightgbm':
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)
    elif model_name == 'catboost':
        return CatBoostClassifier(random_seed=RANDOM_STATE, verbose=0)
    elif model_name == 'mlp':
        return MLPClassifier(random_state=RANDOM_STATE)
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


def tune_disease(disease, n_iter=30):
    X_train, X_test, y_train, y_test = load_disease_data(disease)
    feature_names = load_feature_names(disease)
    feature_names = feature_names[:X_train.shape[1]]

    param_grids = get_param_grids()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    best_models = {}
    for model_name, grid in param_grids.items():
        print(f"  Tuning {model_name}...")
        base_model = get_model_constructor(model_name)
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

        test_metrics = evaluate_model(best_model, X_test, y_test)
        test_metrics['disease'] = disease
        test_metrics['model'] = model_name
        test_metrics['best_params'] = search.best_params_
        test_metrics['cv_auprc_mean'] = search.cv_results_['mean_test_score'][search.best_index_]
        test_metrics['cv_auprc_std'] = search.cv_results_['std_test_score'][search.best_index_]
        results.append(test_metrics)
        best_models[model_name] = best_model

        print(
            f"    Test AUROC={test_metrics['auroc']:.3f}, "
            f"AUPRC={test_metrics['auprc']:.3f}, "
            f"CV AUPRC={test_metrics['cv_auprc_mean']:.3f}±{test_metrics['cv_auprc_std']:.3f}"
        )

        # Save individual model
        joblib.dump(best_model, os.path.join(MODEL_DIR, f"{disease}_{model_name}_tuned.pkl"))

    # Build simple ensemble from top 3 models by CV AUPRC
    top3 = sorted(results, key=lambda x: x['cv_auprc_mean'], reverse=True)[:3]
    top3_names = [r['model'] for r in top3]
    print(f"  Building simple ensemble from: {top3_names}")
    ensemble = SimpleEnsemble(models=[best_models[name] for name in top3_names])
    ensemble_metrics = evaluate_model(ensemble, X_test, y_test)
    ensemble_metrics['disease'] = disease
    ensemble_metrics['model'] = 'voting_ensemble'
    ensemble_metrics['best_params'] = {'components': top3_names}
    # Use mean of top3 CV scores as a proxy for ensemble CV (not strictly correct, but acceptable)
    ensemble_metrics['cv_auprc_mean'] = np.mean([r['cv_auprc_mean'] for r in top3])
    ensemble_metrics['cv_auprc_std'] = np.mean([r['cv_auprc_std'] for r in top3])
    results.append(ensemble_metrics)
    best_models['voting_ensemble'] = ensemble
    joblib.dump(ensemble, os.path.join(MODEL_DIR, f"{disease}_voting_ensemble_tuned.pkl"))
    print(
        f"    Ensemble Test AUROC={ensemble_metrics['auroc']:.3f}, "
        f"AUPRC={ensemble_metrics['auprc']:.3f}"
    )

    return results, feature_names


def main():
    all_results = []
    for disease in DISEASE_MAP.keys():
        print(f"\n{'='*60}\nTuning {disease}\n{'='*60}")
        results, feature_names = tune_disease(disease, n_iter=30)
        all_results.extend(results)

        # Choose best model based on CV AUPRC
        best_result = max(results, key=lambda x: x['cv_auprc_mean'])
        final_model_name = best_result['model']
        final_model_path = os.path.join(MODEL_DIR, f"{disease}_{final_model_name}_tuned.pkl")
        final_model = joblib.load(final_model_path)
        joblib.dump(final_model, os.path.join(MODEL_DIR, f"{disease}_final_model.pkl"))

        # Save feature importance or coefficients
        if hasattr(final_model, 'feature_importances_'):
            importance = final_model.feature_importances_
        elif hasattr(final_model, 'coef_'):
            importance = np.abs(final_model.coef_).flatten()
        elif isinstance(final_model, SimpleEnsemble):
            importances = []
            for m in final_model.models:
                if hasattr(m, 'feature_importances_'):
                    importances.append(m.feature_importances_)
                elif hasattr(m, 'coef_'):
                    importances.append(np.abs(m.coef_).flatten())
            importance = np.mean(importances, axis=0) if importances else np.zeros(len(feature_names))
        else:
            importance = np.zeros(len(feature_names))

        indices = np.argsort(importance)[::-1]
        imp_dict = {feature_names[i]: float(importance[i]) for i in indices}
        with open(os.path.join(MODEL_DIR, f"{disease}_feature_importance.json"), "w") as f:
            json.dump(imp_dict, f, indent=2)

        print(
            f"\n  Best model for {disease}: {final_model_name} "
            f"(CV AUPRC={best_result['cv_auprc_mean']:.3f})"
        )

    pd.DataFrame(all_results).to_csv(
        os.path.join(MODEL_DIR, "tuned_model_metrics.csv"), index=False
    )
    print(f"\nTuning complete. Summary saved to {MODEL_DIR}/tuned_model_metrics.csv")


if __name__ == "__main__":
    main()