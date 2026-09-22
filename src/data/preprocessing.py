# src/data/preprocessing.py
"""
Split, impute, scale, and save numpy arrays per disease.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.data.feature_engineering import (
    SELECTED_FEATURES, DERIVED_FEATURES, LEAKAGE_MAP,
)

RANDOM_STATE = 42


def build_and_save(df: pd.DataFrame, processed_dir: str) -> None:
    """Build per-disease train/test arrays, impute, scale, save."""
    os.makedirs(processed_dir, exist_ok=True)
    label_cols = [c for c in df.columns if c.startswith("label_")]

    all_predictors = SELECTED_FEATURES + DERIVED_FEATURES
    feature_names_json = {}

    for label_col in label_cols:
        disease = label_col.replace("label_", "")
        mask = df[label_col].notna()
        y = df.loc[mask, label_col].astype(int)
        exclude = set(LEAKAGE_MAP.get(label_col, []))
        feats = [c for c in all_predictors if c not in exclude and c in df.columns]

        X = df.loc[mask, feats].copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        X = X.dropna(axis=1, how="all")
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        np.save(os.path.join(processed_dir, f"{disease}_X_train.npy"), X_train)
        np.save(os.path.join(processed_dir, f"{disease}_X_test.npy"), X_test)
        np.save(os.path.join(processed_dir, f"{disease}_y_train.npy"), y_train)
        np.save(os.path.join(processed_dir, f"{disease}_y_test.npy"), y_test)
        joblib.dump(scaler, os.path.join(processed_dir, f"{disease}_scaler.pkl"))

        feature_names_json[label_col] = X.columns.tolist()
        print(f"{disease}: train={X_train.shape}, test={X_test.shape}, pos_train={y_train.sum()}")

    with open(os.path.join(processed_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names_json, f, indent=2)
    with open(os.path.join(processed_dir, "leakage_map.json"), "w") as f:
        json.dump(LEAKAGE_MAP, f, indent=2)