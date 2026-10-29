"""One-time conversion of the original BreaKHis_v1 folder tree into an
ImageFolder layout. Run once; training never copies files.

Original file names are kept, so the patient ID stays recoverable for
patient-level splitting (group_by: breakhis_patient).

Example:
    python scripts/prepare_breakhis.py --src /path/to/BreaKHis_v1 --dst data/breakhis_binary --mode binary
    python scripts/prepare_breakhis.py --src /path/to/BreaKHis_v1 --dst data/breakhis_multi  --mode multi --magnification 40
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import Counter

SUBTYPES = {
    "DC": ("malignant", "ductal_carcinoma"),
    "LC": ("malignant", "lobular_carcinoma"),
    "MC": ("malignant", "mucinous_carcinoma"),
    "PC": ("malignant", "papillary_carcinoma"),
    "A": ("benign", "adenosis"),
    "F": ("benign", "fibroadenoma"),
    "PT": ("benign", "phyllodes_tumor"),
    "TA": ("benign", "tubular_adenoma"),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Folder containing histology_slides/breast")
    p.add_argument("--dst", required=True)
    p.add_argument("--mode", choices=["binary", "multi"], required=True)
    p.add_argument("--magnification", default="all", help="40, 100, 200, 400 or all")
    p.add_argument("--symlink", action="store_true", help="Symlink instead of copying")
    a = p.parse_args()

    root = os.path.join(a.src, "histology_slides", "breast")
    if not os.path.isdir(root):
        root = a.src  # allow passing the 'breast' folder directly
    counts: Counter = Counter()
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(".png"):
                continue
            # SOB_B_A-14-22549AB-40-001.png
            parts = fn.split("-")
            code = parts[0].split("_")[-1]
            mag = parts[-2] if len(parts) >= 2 else ""
            if code not in SUBTYPES:
                continue
            if a.magnification != "all" and mag != str(a.magnification):
                continue
            label = SUBTYPES[code][0] if a.mode == "binary" else SUBTYPES[code][1]
            out_dir = os.path.join(a.dst, label)
            os.makedirs(out_dir, exist_ok=True)
            dst = os.path.join(out_dir, fn)
            if os.path.exists(dst):
                continue
            src = os.path.join(dirpath, fn)
            if a.symlink:
                os.symlink(os.path.abspath(src), dst)
            else:
                shutil.copy2(src, dst)
            counts[label] += 1
    for k, v in sorted(counts.items()):
        print(f"{k}: {v}")
    print(f"Total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
