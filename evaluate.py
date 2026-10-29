"""Re-evaluate a finished run from its config.json, splits.csv and best weights.

    python evaluate.py --run_dir results/idc_p4_seed0_<timestamp> --split test
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from bmsa_vit.data import make_dataset
from bmsa_vit.metrics import compute_metrics, format_summary, save_metrics
from bmsa_vit.model import build_bmsa_vit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--weights", default=None)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    a = p.parse_args()

    with open(os.path.join(a.run_dir, "config.json")) as f:
        c = json.load(f)
    part = pd.read_csv(os.path.join(a.run_dir, "splits.csv"))
    part = part[part["split"] == a.split].reset_index(drop=True)
    classes, size = c["classes"], int(c["image_size"])
    model = build_bmsa_vit(image_size=size, patch_size=c["patch_size"], embed_dim=c["embed_dim"],
                           num_heads=c["num_heads"], mlp_dim=c["mlp_dim"], num_blocks=c["num_blocks"],
                           num_classes=len(classes), attention_dropout=c["attention_dropout"],
                           projection_dropout=c["projection_dropout"], mlp_dropout=c["mlp_dropout"],
                           attention_type=c["attention_type"])
    model.load_weights(a.weights or os.path.join(a.run_dir, "best.weights.h5"))
    y = part["label"].to_numpy()
    ds = make_dataset(part["path"].tolist(), y, len(classes), size, c["batch_size"], training=False, seed=0)
    probs = model.predict(ds, verbose=0).astype(np.float64)
    m = compute_metrics(y, probs, classes, n_boot=a.n_bootstrap, seed=c.get("seed", 0))
    save_metrics(m, probs, y, part["path"].tolist(), classes, a.run_dir, f"eval_{a.split}")
    print(format_summary(m))


if __name__ == "__main__":
    main()
