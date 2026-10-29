"""Aggregate summary.json from several runs into per-seed and mean ± SD tables.

Example:
    python scripts/aggregate_seeds.py --runs results/bracs_seeds/seed*
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "kappa", "macro_auroc", "ece_15bin", "brier")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    rows = []
    for r in a.runs:
        with open(os.path.join(r, "summary.json")) as f:
            s = json.load(f)
        rows.append({"run": os.path.basename(r.rstrip("/")), **{k: s[a.split][k] for k in KEYS}})
    df = pd.DataFrame(rows).set_index("run")
    agg = pd.DataFrame({"mean": df.mean(), "sd": df.std(ddof=1)})
    print(f"Per-run ({a.split}):\n{df.round(4)}\n")
    print(f"Mean ± SD over {len(df)} runs:\n{agg.round(4)}")
    out = a.out or os.path.join(os.path.dirname(a.runs[0].rstrip('/')), f"aggregate_{a.split}.csv")
    pd.concat([df, agg.T]).to_csv(out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
