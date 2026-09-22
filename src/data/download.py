# src/data/download.py
"""
Download NHANES .xpt files from the CDC portal.
Usage: python -m src.data.download --cycle 2017-2018
"""

import os
import argparse
import urllib.request

CYCLE_URLS = {
    "2013-2014": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/",
    "2015-2016": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/",
    "2017-2018": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/",
    "2021-2023": "https://wwwn.cdc.gov/Nchs/Nhanes/2021-2023/",
}

CYCLE_SUFFIX = {
    "2013-2014": "_H",
    "2015-2016": "_I",
    "2017-2018": "_J",
    "2021-2023": "_L",
}

# Components used across all cycles. BPXO_L is used instead of BPX_L in 2021-2023.
COMPONENTS = [
    "DEMO", "BMX", "GHB", "GLU", "BPX", "BIOPRO", "DIQ", "KIQ_U",
    "SMQ", "ALQ", "PAQ", "DBQ", "MCQ", "BPQ", "CBC", "HDL",
    "TCHOL", "TRIGLY", "INS", "HSCRP", "ALB_CR", "HSQ",
]

def download_cycle(cycle: str, out_dir: str) -> None:
    """Download all components for one NHANES cycle."""
    base_url = CYCLE_URLS[cycle]
    suffix = CYCLE_SUFFIX[cycle]
    os.makedirs(out_dir, exist_ok=True)

    for comp in COMPONENTS:
        # 2021-2023 uses BPXO instead of BPX
        fname = f"{comp}{suffix}.xpt"
        if cycle == "2021-2023" and comp == "BPX":
            fname = f"BPXO{suffix}.xpt"

        url = base_url + fname
        out_path = os.path.join(out_dir, fname)

        if os.path.exists(out_path):
            print(f"Already present: {fname}")
            continue

        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, out_path)
            print(f"Saved {fname}")
        except Exception as e:
            print(f"Failed to download {fname}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", required=True, choices=list(CYCLE_URLS.keys()))
    parser.add_argument("--out", required=True, help="Output folder")
    args = parser.parse_args()
    download_cycle(args.cycle, args.out)