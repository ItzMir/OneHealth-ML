# src/data/feature_engineering.py
"""
Feature engineering: derived features + curated feature selection.
"""

import numpy as np
import pandas as pd

SELECTED_FEATURES = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR", "DMDEDUC2", "DMDMARTL",
    "DMDBORN4", "DMDYRSUS", "INDHHIN2", "DMDHHSIZ", "RIDEXMON",
    "BMXWT", "BMXHT", "BMXBMI", "BMXWAIST", "BMXHIP", "BMXARMC", "BMXARML", "BMXLEG",
    "LBXGH", "LBXGLU",
    "BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2", "BPXSY3", "BPXDI3", "BPXSY4", "BPXDI4",
    "BPXPLS", "BPXPULS",
    "LBXSCR", "LBXSBU", "LBXSAL", "LBXSTP", "LBXSCA", "LBXSNASI", "LBXSKSI",
    "LBXSCLSI", "LBXSTB", "LBXSAPSI", "LBXSATSI", "LBXSASSI", "LBXSGTSI",
    "LBXSPH", "LBXSC3SI", "LBXSUA", "LBXSIR", "LBXSGB", "LBXSCK", "LBXSLDSI",
    "LBXSOSSI",
    "DIQ160", "DID040", "DIQ080",
    "KIQ022", "KIQ025", "KIQ026",
    "SMQ020", "SMD030", "SMQ040", "SMD650", "SMD057", "SMQ890", "SMQ900", "SMQ910",
    "ALQ111", "ALQ121", "ALQ130", "ALQ142", "ALQ151",
    "PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665", "PAD680",
    "DBQ700", "DBQ197", "DBD895", "DBD900", "DBD905", "DBD910",
    "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F",
    "MCQ160L", "MCQ300C", "MCQ300A",
    "LBXWBCSI", "LBXHGB", "LBXRDW", "LBXPLTSI", "LBXLYPCT", "LBXNEPCT",
    "LBDHDD", "LBXTC", "LBXTR",
    "LBXIN", "LBXHSCRP", "URDACT",
    "HSD010",
]

ADMIN_VARS = ["SEQN", "WTMEC2YR", "SDMVPSU", "SDMVSTRA", "RIDSTATR", "BMDSTATS", "WTSAF2YR"]
TEMP_SOURCES = ["DIQ010", "BPQ020", "mean_sbp", "mean_dbp", "eGFR"]

DERIVED_FEATURES = ["NLR", "HOMA_IR", "WHR", "ASCVD_risk", "met_syndrome_count", "eGFR_stage", "obesity_class", "pack_years"]

LEAKAGE_MAP = {
    "label_diabetes": ["LBXGH", "LBXGLU", "DIQ010", "HOMA_IR", "LBXSGL", "met_syndrome_count", "ASCVD_risk", "DIQ160", "DID040", "DIQ080"],
    "label_hypertension": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4", "BPXPLS", "BPXPULS", "BPQ020", "mean_sbp", "mean_dbp", "ASCVD_risk"],
    "label_ckd": ["LBXSCR", "eGFR", "KIQ022", "KIQ025", "URDACT", "eGFR_stage"],
    "label_dyslipidemia": ["LBXTC", "LBDHDD", "LBXTR", "LBXSCH", "LBXSTR", "ASCVD_risk"],
    "label_cvd": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F", "MCQ300A", "MCQ300B", "MCQ300C"],
    "label_metabolic_syndrome": ["BMXWAIST", "LBXTR", "LBDHDD", "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
                                  "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4", "mean_sbp", "mean_dbp",
                                  "LBXGLU", "WHR", "LBXSGL", "LBXSTR", "met_syndrome_count", "ASCVD_risk"],
    "label_liver_disease": ["LBXSATSI", "LBXSASSI", "LBXSGTSI", "BMXBMI", "obesity_class"],
}


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["NLR"] = df["LBXNEPCT"] / df["LBXLYPCT"].replace(0, np.nan)
    df["HOMA_IR"] = (df["LBXIN"] * df["LBXGLU"]) / 405
    df["WHR"] = df["BMXWAIST"] / df["BMXHIP"].replace(0, np.nan)

    df["ASCVD_risk"] = (
        df["RIDAGEYR"] * 0.05 + df.get("LBXTC", 0) * 0.002
        - df.get("LBDHDD", 0) * 0.003 + df.get("mean_sbp", 0) * 0.004
        + df.get("SMQ040", 0).isin([1, 2]).astype(int) * 0.15
        + (df.get("DIQ010", 2) == 1).astype(int) * 0.2
        + (df.get("BPQ020", 2) == 1).astype(int) * 0.1
    ).clip(0, 1)

    df["met_syndrome_count"] = (
        (((df["RIAGENDR"] == 1) & (df["BMXWAIST"] > 102)) | ((df["RIAGENDR"] == 2) & (df["BMXWAIST"] > 88))).astype(int)
        + (df["LBXTR"] >= 150).astype(int)
        + (((df["RIAGENDR"] == 1) & (df["LBDHDD"] < 40)) | ((df["RIAGENDR"] == 2) & (df["LBDHDD"] < 50))).astype(int)
        + ((df["mean_sbp"] >= 130) | (df["mean_dbp"] >= 85)).astype(int)
        + (df["LBXGLU"] >= 100).astype(int)
    )

    def egfr_stage(e):
        if pd.isna(e): return np.nan
        return 1 if e >= 90 else 2 if e >= 60 else 3 if e >= 45 else 4 if e >= 30 else 5
    df["eGFR_stage"] = df["eGFR"].apply(egfr_stage)

    def obesity_class(b):
        if pd.isna(b): return np.nan
        return 0 if b < 18.5 else 1 if b < 25 else 2 if b < 30 else 3 if b < 35 else 4 if b < 40 else 5
    df["obesity_class"] = df["BMXBMI"].apply(obesity_class)

    def pack_years(r):
        if r.get("SMQ020") != 1: return 0
        start = r.get("SMD030")
        if pd.isna(start) or start <= 0: return 0
        cigs = r.get("SMD650", 0)
        cigs = 0 if pd.isna(cigs) else cigs
        return (cigs * max(0, r["RIDAGEYR"] - start)) / 20
    df["pack_years"] = df.apply(pack_years, axis=1)

    return df


def select_curated(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only curated + temporary columns."""
    cols = SELECTED_FEATURES + TEMP_SOURCES + ADMIN_VARS
    cols += [c for c in df.columns if c.startswith("label_")]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()