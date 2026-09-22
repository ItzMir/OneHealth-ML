# src/data/merge_components.py
"""
Merge the verified per-component CSVs into one dataframe per cycle.
Usage:
    from src.data.merge_components import merge_cycle
    df = merge_cycle("2017-2018", interim_dir="../data/interim")
"""

import os
import pandas as pd

COMPONENTS_ORDER = [
    "DEMO", "BMX", "GHB", "GLU", "BPX", "BIOPRO", "DIQ", "KIQ_U",
    "SMQ", "ALQ", "PAQ", "DBQ", "MCQ", "BPQ", "CBC", "HDL",
    "TCHOL", "TRIGLY", "INS", "HSCRP", "ALB_CR", "HSQ",
]

CYCLE_SUFFIX = {
    "2013-2014": "_2013_verified.csv",
    "2015-2016": "_2015_verified.csv",
    "2017-2018": "_verified.csv",
    "2021-2023": "_2021_verified.csv",
}

def merge_cycle(cycle: str, interim_dir: str = "../data/interim") -> pd.DataFrame:
    """Merge all components for a given cycle on SEQN and cycle."""
    suffix = CYCLE_SUFFIX[cycle]
    dfs = {}

    for comp in COMPONENTS_ORDER:
        path = os.path.join(interim_dir, f"{comp}{suffix}")
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        df = pd.read_csv(path)
        df["cycle"] = cycle  # keep human-readable
        dfs[comp] = df
        print(f"Loaded {comp}: {df.shape}")

    if "DEMO" not in dfs:
        raise RuntimeError("DEMO component is required")

    merged = dfs["DEMO"].copy()
    for comp in COMPONENTS_ORDER[1:]:
        if comp in dfs:
            merged = merged.merge(dfs[comp], on=["SEQN", "cycle"], how="inner")
            print(f"Merged {comp}: {merged.shape}")

    print(f"\nFinal {cycle} shape: {merged.shape}")
    print("Duplicate SEQN+cycle:", merged[["SEQN", "cycle"]].duplicated().sum())
    return merged