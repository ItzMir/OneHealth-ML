# src/data/verify_all.py
"""
Full verification of all NHANES cycles in the project.
Checks:
1. Every expected file exists
2. Every file is a valid XPORT file (not HTML)
3. Every file has the required variables
4. Reports a clear action list for anything missing/corrupt
"""

import os
import pandas as pd
import numpy as np

RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))

# ---------- Configuration (mirrors the pipeline) ----------
TRAINING_CYCLES = [
    ("NHANES_2005_2006", "_D", "2005-2006"),
    ("NHANES_2007_2008", "_E", "2007-2008"),
    ("NHANES_2009_2010", "_F", "2009-2010"),
    ("NHANES_2011_2012", "_G", "2011-2012"),
    ("NHANES_2013_2014", "_H", "2013-2014"),
    ("NHANES_2015_2016", "_I", "2015-2016"),
    ("NHANES_2017_2018", "_J", "2017-2018"),
]
EXTERNAL_CYCLE = ("NHANES_2021_2023", "_L", "2021-2023")

ALL_CYCLES = TRAINING_CYCLES + [EXTERNAL_CYCLE]

# Older cycles store HSCRP under CRP_*.xpt; 2021+ uses BPXO instead of BPX
FILE_OVERRIDE = {
    ("HSCRP", "_D"): "CRP_D.xpt",
    ("HSCRP", "_E"): "CRP_E.xpt",
    ("HSCRP", "_F"): "CRP_F.xpt",
    ("HSCRP", "_G"): "CRP_G.xpt",
    ("HSCRP", "_H"): "CRP_H.xpt",
    ("BPX",   "_L"): "BPXO_L.xpt",
}

COMPONENTS = ["DEMO","BMX","GHB","GLU","BPX","BIOPRO","DIQ","KIQ_U",
              "SMQ","ALQ","PAQ","DBQ","MCQ","BPQ","CBC","HDL",
              "TCHOL","TRIGLY","INS","HSCRP","ALB_CR","HSQ"]

# Required variables per component (post-harmonisation)
REQUIRED = {
    "DEMO":   ["SEQN","RIDAGEYR","RIAGENDR","INDFMPIR","RIDEXPRG","WTMEC2YR"],
    "BMX":    ["SEQN","BMXWT","BMXHT","BMXBMI","BMXWAIST","BMXARMC","BMXARML","BMXLEG"],
    "GHB":    ["SEQN","LBXGH"],
    "GLU":    ["SEQN","LBXGLU"],
    "BPX":    ["SEQN","BPXSY1","BPXDI1","BPXSY2","BPXDI2","BPXSY3","BPXDI3"],
    "BIOPRO": ["SEQN","LBXSCR","LBXSBU","LBXSAL","LBXSTP","LBXSCA","LBXSNASI",
               "LBXSKSI","LBXSCLSI","LBXSTB","LBXSAPSI","LBXSATSI","LBXSASSI",
               "LBXSGTSI","LBXSPH","LBXSC3SI","LBXSUA","LBXSIR","LBXSGB",
               "LBXSCK","LBXSLDSI","LBXSOSSI"],
    "DIQ":    ["SEQN","DIQ010"],
    "KIQ_U":  ["SEQN","KIQ022"],
    "SMQ":    ["SEQN","SMQ020","SMQ040","SMD030","SMD650"],
    "ALQ":    ["SEQN","ALQ130","ALQ151"],
    "PAQ":    ["SEQN","PAQ605","PAQ620","PAQ635","PAQ650","PAQ665","PAD680"],
    "DBQ":    ["SEQN","DBQ700"],
    "MCQ":    ["SEQN","MCQ160B","MCQ160C","MCQ160D","MCQ160E","MCQ160F"],
    "BPQ":    ["SEQN","BPQ020"],
    "CBC":    ["SEQN","LBXWBCSI","LBXHGB","LBXRDW","LBXPLTSI"],
    "HDL":    ["SEQN","LBDHDD"],
    "TCHOL":  ["SEQN","LBXTC"],
    "TRIGLY": ["SEQN","LBXTR"],
    "INS":    ["SEQN","LBXIN"],
    "HSCRP":  ["SEQN"],          # variable name varies (LBXCRP vs LBXHSCRP)
    "ALB_CR": ["SEQN","URDACT"],
    "HSQ":    ["SEQN","HSD010"],
}


def check_file(fpath):
    """Return (status, message). status in {'ok','html','small','corrupt','missing'}"""
    if not os.path.exists(fpath):
        return "missing", "file not found"
    size = os.path.getsize(fpath)
    if size < 5000:
        with open(fpath, "rb") as f:
            head = f.read(64)
        if b"<html" in head.lower() or b"<!doctype" in head.lower():
            return "html", f"HTML error page ({size} bytes)"
        return "small", f"only {size} bytes"
    with open(fpath, "rb") as f:
        head = f.read(80)
    if b"HEADER RECORD" not in head:
        return "corrupt", f"bad header ({size} bytes)"
    return "ok", f"{size:,} bytes"


def main():
    print("=" * 78)
    print("OneHealth-ML — Full Verification of All NHANES Cycles")
    print("=" * 78)
    print(f"Raw data folder: {RAW}")
    print()

    summary = []          # rows for final table
    action_items = []     # things the user must fix

    for folder, suffix, label in ALL_CYCLES:
        folder_path = os.path.join(RAW, folder)
        print(f"\n--- {label} ({suffix}) ---")

        if not os.path.isdir(folder_path):
            print(f"  ❌ folder missing: {folder_path}")
            action_items.append(f"Create folder {folder_path}")
            continue

        for comp in COMPONENTS:
            fname = FILE_OVERRIDE.get((comp, suffix), f"{comp}{suffix}.xpt")
            fpath = os.path.join(folder_path, fname)

            status, msg = check_file(fpath)

            if status != "ok":
                print(f"  ❌ {comp:8s} {fname:15s} → {status}: {msg}")
                action_items.append(f"{label} {comp}: {msg} ({fname})")
                summary.append((label, comp, fname, "FAIL", msg))
                continue

            # File is a valid XPORT; check columns
            try:
                df = pd.read_sas(fpath)
            except Exception as e:
                print(f"  ❌ {comp:8s} {fname:15s} → pandas cannot read: {e}")
                action_items.append(f"{label} {comp}: pandas read failed ({e})")
                summary.append((label, comp, fname, "FAIL", str(e)[:50]))
                continue

            required = REQUIRED[comp]
            missing = [c for c in required if c not in df.columns]

            # Special case: HSCRP may appear as LBXCRP or LBXHSCRP
            if comp == "HSCRP":
                if "LBXCRP" in df.columns or "LBXHSCRP" in df.columns:
                    missing = []

            # Special case: ALQ old vs new coding
            if comp == "ALQ":
                has_old = "ALQ101" in df.columns and "ALQ120Q" in df.columns
                has_new = "ALQ111" in df.columns and "ALQ121" in df.columns
                if has_old or has_new:
                    missing = [m for m in missing if m not in ["ALQ111","ALQ121","ALQ142"]]

            if missing:
                print(f"  ⚠ {comp:8s} {fname:15s} → missing vars: {missing}")
                action_items.append(f"{label} {comp}: missing variables {missing}")
                summary.append((label, comp, fname, "WARN", f"missing {missing}"))
            else:
                print(f"  ✅ {comp:8s} {fname:15s} → {msg} ({len(df):,} rows)")
                summary.append((label, comp, fname, "OK", msg))

    # ---------------- Final summary ----------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    import collections
    counts = collections.Counter(s[3] for s in summary)
    print(f"Total checks : {len(summary)}")
    print(f"  OK         : {counts.get('OK', 0)}")
    print(f"  WARN       : {counts.get('WARN', 0)}")
    print(f"  FAIL       : {counts.get('FAIL', 0)}")

    if action_items:
        print("\n--- ACTION ITEMS ---")
        for item in action_items:
            print(f"  • {item}")
    else:
        print("\n🎉 All files verified. No action needed.")

    # ---------------- Save report ----------------
    report_path = os.path.join(os.path.dirname(__file__), "verification_report.csv")
    import csv
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cycle", "component", "file", "status", "detail"])
        writer.writerows(summary)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()