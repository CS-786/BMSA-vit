"""Flatten the public IDC patch collection (<src>/<patient_id>/<0|1>/*.png)
into <dst>/<0|1>/*.png, keeping original file names so the patient ID stays
recoverable for patient-level splitting (dataset key: idc_patient_split).

    python scripts/prepare_idc.py --src /path/to/IDC_regular_ps50_idx5 --dst data/IDC_flat --symlink
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--symlink", action="store_true", help="Symlink instead of copying")
    a = p.parse_args()

    counts: Counter = Counter()
    patients = set()
    for dirpath, _, files in os.walk(a.src):
        label = os.path.basename(dirpath)
        if label not in ("0", "1"):
            continue
        os.makedirs(os.path.join(a.dst, label), exist_ok=True)
        for fn in files:
            if not fn.lower().endswith(".png"):
                continue
            dst = os.path.join(a.dst, label, fn)
            if os.path.exists(dst):
                continue
            src = os.path.join(dirpath, fn)
            if a.symlink:
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
            counts[label] += 1
            patients.add(fn.split("_")[0])
    print(dict(counts), f"total={sum(counts.values())}", f"patients={len(patients)}")


if __name__ == "__main__":
    main()
