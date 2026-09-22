# src/data/merge_cycles.py
"""
Concatenate the three training cycles (2013-14, 2015-16, 2017-18)
into a single training dataframe.
"""

import pandas as pd
from src.data.merge_components import merge_cycle

TRAINING_CYCLES = ["2013-2014", "2015-2016", "2017-2018"]
CYCLE_CODES = {"2013-2014": 0, "2015-2016": 1, "2017-2018": 2, "2021-2023": 3}

def merge_training_cycles(interim_dir: str = "../data/interim") -> pd.DataFrame:
    frames = []
    for cycle in TRAINING_CYCLES:
        df = merge_cycle(cycle, interim_dir=interim_dir)
        df["cycle"] = CYCLE_CODES[cycle]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"\nCombined training shape: {combined.shape}")
    return combined