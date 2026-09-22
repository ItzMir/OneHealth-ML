# src/data/__init__.py
"""
Data pipeline modules for OneHealth-ML.

Typical usage:
    from src.data.dataset_builder import main as build_dataset
    build_dataset()

Modules
-------
- download            : Download raw NHANES .xpt files from CDC
- merge_components    : Merge verified per-component CSVs into one dataframe per cycle
- merge_cycles        : Concatenate 2013-2014, 2015-2016, 2017-2018 into a single training set
- label_generation    : Apply clinical rules to produce seven disease labels
- feature_engineering : Derive clinical scores and apply curated feature selection
- preprocessing       : Impute, scale, split, and save numpy arrays per disease
- dataset_builder     : End-to-end orchestrator that runs the full pipeline
"""

__all__ = [
    "download",
    "merge_components",
    "merge_cycles",
    "label_generation",
    "feature_engineering",
    "preprocessing",
    "dataset_builder",
]