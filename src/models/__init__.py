# src/models/__init__.py
"""
Model training and evaluation modules for OneHealth-ML.

Modules
-------
- train                    : Baseline training of LR, RF, XGBoost, LightGBM, CatBoost + KG
- tune                     : First-generation RandomizedSearchCV tuning
- tune_extended            : Final tuning (adds MLP, ensemble, CatBoost fix)
- knowledge_graph          : Custom knowledge-graph model using Node2Vec embeddings
- smote_retrain            : SMOTE retraining experiment (kept for record; not used in final)
- retrain_final_models     : Retrain final models on the combined 2013-2018 dataset
- threshold_optimizer      : Find Youden and F1 optimal thresholds per disease
- threshold_analysis       : Evaluate at fixed recall thresholds (for clinical screening)
- explain                  : SHAP and feature importance plots for final models
"""

__all__ = [
    "train",
    "tune",
    "tune_extended",
    "knowledge_graph",
    "smote_retrain",
    "retrain_final_models",
    "threshold_optimizer",
    "threshold_analysis",
    "explain",
]