# src/models/deep_learning/__init__.py
"""
Deep learning models for OneHealth-ML.

Modules:
- onehealth_net: OneHealth-Net architecture (multi-task transformer)
- train_onehealth_net: Training pipeline for OneHealth-Net
- external_validation: External validation on 2021-2023 cohort
- permutation_importance: Post-hoc leakage audit via permutation
"""

from .onehealth_net import OneHealthNet