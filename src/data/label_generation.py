# src/data/label_generation.py
"""
Generate seven disease labels from verified, merged data.
All rules match docs/Disease_Label_Definitions.md.
"""

import numpy as np
import pandas as pd


def egfr_ckdepi_2021(scr, age, sex):
    """CKD-EPI 2021 race-free eGFR."""
    scr = np.asarray(scr, dtype=float)
    age = np.asarray(age, dtype=float)
    sex = np.asarray(sex, dtype=float)
    kappa = np.where(sex == 1, 0.9, 0.7)
    alpha = np.where(sex == 1, -0.302, -0.241)
    ratio = scr / kappa
    ratio_min = np.minimum(ratio, 1)
    ratio_max = np.maximum(ratio, 1)
    return 142 * (ratio_min ** alpha) * (ratio_max ** -1.200) * (0.9938 ** age) * np.where(sex == 1, 1.0, 1.012)


def add_derived_means(df: pd.DataFrame) -> pd.DataFrame:
    bp_sys = [c for c in ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"] if c in df.columns]
    bp_dia = [c for c in ["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"] if c in df.columns]
    df["mean_sbp"] = df[bp_sys].mean(axis=1, skipna=True) if bp_sys else np.nan
    df["mean_dbp"] = df[bp_dia].mean(axis=1, skipna=True) if bp_dia else np.nan
    df["eGFR"] = egfr_ckdepi_2021(df["LBXSCR"], df["RIDAGEYR"], df["RIAGENDR"])
    return df


def label_diabetes(row):
    if row.get("RIDEXPRG") == 1:
        return np.nan
    if pd.isna(row.get("LBXGH")) and pd.isna(row.get("LBXGLU")) and pd.isna(row.get("DIQ010")):
        return np.nan
    if (row.get("LBXGH", 0) >= 6.5) or (row.get("LBXGLU", 0) >= 126) or (row.get("DIQ010") == 1):
        return 1
    return 0


def label_hypertension(row):
    if pd.isna(row.get("mean_sbp")) and pd.isna(row.get("mean_dbp")) and pd.isna(row.get("BPQ020")):
        return np.nan
    if (row.get("mean_sbp", 0) >= 130) or (row.get("mean_dbp", 0) >= 80) or (row.get("BPQ020") == 1):
        return 1
    return 0


def label_ckd(row):
    if pd.isna(row.get("eGFR")) and pd.isna(row.get("KIQ025")) and pd.isna(row.get("KIQ022")):
        return np.nan
    if (row.get("eGFR", 100) < 60) or (row.get("KIQ025") == 1) or (row.get("KIQ022") == 1):
        return 1
    return 0


def label_dyslipidemia(row):
    if pd.isna(row.get("LBXTC")) and pd.isna(row.get("LBDHDD")) and pd.isna(row.get("LBXTR")):
        return np.nan
    hdl_low = (row.get("LBDHDD", 100) < 40) if row.get("RIAGENDR") == 1 else (row.get("LBDHDD", 100) < 50)
    if (row.get("LBXTC", 0) >= 200) or hdl_low or (row.get("LBXTR", 0) >= 150):
        return 1
    return 0


def label_cvd(row):
    cols = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]
    vals = [row.get(c) for c in cols]
    if all(pd.isna(v) for v in vals):
        return np.nan
    return 1 if any(v == 1 for v in vals) else 0


def label_metabolic_syndrome(row):
    components = 0
    if (row.get("RIAGENDR") == 1 and row.get("BMXWAIST", 0) > 102) or (row.get("RIAGENDR") == 2 and row.get("BMXWAIST", 0) > 88):
        components += 1
    if row.get("LBXTR", 0) >= 150:
        components += 1
    hdl_low = (row.get("LBDHDD", 100) < 40) if row.get("RIAGENDR") == 1 else (row.get("LBDHDD", 100) < 50)
    if hdl_low:
        components += 1
    if pd.notna(row.get("mean_sbp")) and pd.notna(row.get("mean_dbp")):
        if row["mean_sbp"] >= 130 or row["mean_dbp"] >= 85:
            components += 1
    if row.get("LBXGLU", 0) >= 100:
        components += 1

    if any(pd.isna(row.get(c)) for c in ["BMXWAIST", "LBXTR", "LBDHDD", "mean_sbp", "LBXGLU"]):
        return np.nan
    return 1 if components >= 3 else 0


def label_liver_disease(row):
    if pd.isna(row.get("LBXSATSI")) and pd.isna(row.get("LBXSASSI")) and pd.isna(row.get("LBXSGTSI")):
        return np.nan
    alt_uln = 40 if row.get("RIAGENDR") == 1 else 30
    ast_uln = 40
    ggt_uln = 50
    elevated = (row.get("LBXSATSI", 0) > alt_uln) or (row.get("LBXSASSI", 0) > ast_uln) or (row.get("LBXSGTSI", 0) > ggt_uln)
    bmi_high = row.get("BMXBMI", 0) >= 25
    return 1 if (elevated and bmi_high) else 0


def add_all_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = add_derived_means(df)
    df["label_diabetes"] = df.apply(label_diabetes, axis=1)
    df["label_hypertension"] = df.apply(label_hypertension, axis=1)
    df["label_ckd"] = df.apply(label_ckd, axis=1)
    df["label_dyslipidemia"] = df.apply(label_dyslipidemia, axis=1)
    df["label_cvd"] = df.apply(label_cvd, axis=1)
    df["label_metabolic_syndrome"] = df.apply(label_metabolic_syndrome, axis=1)
    df["label_liver_disease"] = df.apply(label_liver_disease, axis=1)
    return df