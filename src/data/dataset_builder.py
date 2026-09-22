# src/data/dataset_builder.py
"""
End-to-end dataset builder: merge cycles, add labels, engineer features, save.
Usage: python -m src.data.dataset_builder
"""

import os
import pandas as pd

from src.data.merge_cycles import merge_training_cycles
from src.data.label_generation import add_all_labels
from src.data.feature_engineering import add_derived_features, select_curated
from src.data.preprocessing import build_and_save

PROCESSED_DIR = "../data/processed"


def main():
    print("Step 1: Merging cycles...")
    df = merge_training_cycles(interim_dir="../data/interim")

    print("\nStep 2: Adding derived means and labels...")
    df = add_all_labels(df)

    print("\nStep 3: Engineering features...")
    df = add_derived_features(df)
    df = select_curated(df)

    labeled_path = os.path.join(PROCESSED_DIR, "labeled_master_combined.csv")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(labeled_path, index=False)
    print(f"Saved {labeled_path}")

    print("\nStep 4: Splitting, scaling, saving arrays...")
    build_and_save(df, PROCESSED_DIR)

    print("\nDataset build complete.")


if __name__ == "__main__":
    main()