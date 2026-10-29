"""Data: dataset registry, file listing, predefined or leakage-aware splits,
patient-overlap and duplicate checks, and streaming tf.data pipelines.

Images are decoded from disk on the fly, so the full IDC set (277,524
patches) does not need to fit in RAM.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
SPLIT_FOLDERS = {"train": "training", "val": "validation", "test": "testing"}

GROUP_PATTERNS = {
    # BreaKHis: SOB_B_A-14-22549AB-40-001.png -> "14-22549AB"
    "breakhis_patient": r"^SOB_[BM]_[A-Z]+-(\d+-[A-Za-z0-9]+)-\d+-\d+",
    # IDC (Janowczyk & Madabhushi naming): 10253_idx5_x1001_y1001_class0.png -> "10253"
    "idc_patient": r"^(\d+)_idx\d+_",
}


@dataclass
class DatasetSpec:
    name: str
    root: str
    classes: list[str]
    image_size: int
    split_mode: str = "random"   # "folders" (root/training|validation|testing/<class>) or "random"
    group_by: str = "none"


def get_dataset_spec(name: str, registry_path: str, data_root: str,
                     root_override: Optional[str] = None) -> DatasetSpec:
    with open(registry_path, "r", encoding="utf-8") as f:
        reg = yaml.safe_load(f)
    if name not in reg:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(reg)}")
    e = reg[name]
    root = root_override or e["root"]
    if not os.path.isabs(root):
        root = os.path.join(data_root, root)
    return DatasetSpec(name=name, root=root, classes=[str(c) for c in e["classes"]],
                       image_size=int(e["image_size"]), split_mode=e.get("split_mode", "random"),
                       group_by=str(e.get("group_by", "none")))


def _list_class_dir(d: str) -> list[str]:
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Directory not found: {d}")
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))


def list_files(spec: DatasetSpec) -> pd.DataFrame:
    """DataFrame [path, label, class_name, split]; split is filled for split_mode='folders'."""
    rows = []
    if spec.split_mode == "folders":
        for split, folder in SPLIT_FOLDERS.items():
            for label, cls in enumerate(spec.classes):
                rows += [(p, label, cls, split) for p in _list_class_dir(os.path.join(spec.root, folder, cls))]
    elif spec.split_mode == "random":
        for label, cls in enumerate(spec.classes):
            rows += [(p, label, cls, "") for p in _list_class_dir(os.path.join(spec.root, cls))]
    else:
        raise ValueError(f"Unknown split_mode '{spec.split_mode}'")
    if not rows:
        raise RuntimeError(f"No images found under {spec.root}")
    return pd.DataFrame(rows, columns=["path", "label", "class_name", "split"])


def assign_groups(df: pd.DataFrame, group_by: str) -> Optional[np.ndarray]:
    """Group id (patient) per image, or None. If the pattern does not match every
    file name, grouping is disabled with a warning rather than guessed."""
    if group_by in ("none", "", None):
        return None
    pattern = GROUP_PATTERNS.get(group_by)
    if pattern is None:
        if not group_by.startswith("regex:"):
            raise ValueError(f"Unknown group_by '{group_by}'")
        pattern = group_by[len("regex:"):]
    rx = re.compile(pattern)
    groups = [m.group(1) if (m := rx.search(os.path.basename(p))) else None for p in df["path"]]
    n_miss = sum(g is None for g in groups)
    if n_miss:
        print(f"WARNING: group pattern '{group_by}' did not match {n_miss}/{len(groups)} file names; "
              "patient grouping disabled, independence cannot be verified.")
        return None
    return np.asarray(groups)


def subsample(df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    """Optional per-class (and per-split, for folder mode) cap, for quick experiments."""
    if not max_per_class:
        return df
    keys = ["split", "label"] if df["split"].ne("").any() else ["label"]
    return (df.groupby(keys, group_keys=False)
              .apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed))
              .sort_values("path").reset_index(drop=True))


def make_random_splits(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float,
                       seed: int, groups: Optional[np.ndarray]) -> pd.DataFrame:
    """Stratified train/val/test split; grouped by patient when groups are given
    (StratifiedGroupKFold, so realised ratios are approximate)."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    df = df.copy()
    y = df["label"].to_numpy()
    idx = np.arange(len(df))
    if groups is None:
        tmp, test = train_test_split(idx, test_size=test_ratio, random_state=seed, stratify=y)
        tr, va = train_test_split(tmp, test_size=val_ratio / (train_ratio + val_ratio),
                                  random_state=seed, stratify=y[tmp])
    else:
        df["group"] = groups
        k1 = max(2, round(1.0 / test_ratio))
        tmp, test = next(StratifiedGroupKFold(k1, shuffle=True, random_state=seed).split(idx, y, groups))
        k2 = max(2, round((train_ratio + val_ratio) / val_ratio))
        a, b = next(StratifiedGroupKFold(k2, shuffle=True, random_state=seed).split(tmp, y[tmp], groups[tmp]))
        tr, va = tmp[a], tmp[b]
    df.loc[df.index[tr], "split"] = "train"
    df.loc[df.index[va], "split"] = "val"
    df.loc[df.index[test], "split"] = "test"
    return df


def leakage_report(df: pd.DataFrame, hash_files: bool = True) -> dict:
    """Patient overlap between splits (if groups known) and byte-identical files across splits."""
    rep: dict = {"n": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")}}
    if "group" in df:
        g = {s: set(df.loc[df["split"] == s, "group"]) for s in ("train", "val", "test")}
        rep["patients"] = {s: len(v) for s, v in g.items()}
        rep["patient_overlap"] = {"train_x_val": len(g["train"] & g["val"]),
                                  "train_x_test": len(g["train"] & g["test"]),
                                  "val_x_test": len(g["val"] & g["test"])}
    if hash_files:
        h = {}
        for s in ("train", "val", "test"):
            h[s] = set()
            for p in df.loc[df["split"] == s, "path"]:
                with open(p, "rb") as f:
                    h[s].add(hashlib.md5(f.read()).hexdigest())
        rep["identical_files"] = {"train_x_val": len(h["train"] & h["val"]),
                                  "train_x_test": len(h["train"] & h["test"]),
                                  "val_x_test": len(h["val"] & h["test"])}
    return rep


# ---------------------------------------------------------------- tf.data

def _decode(path: tf.Tensor, size: int) -> tf.Tensor:
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (size, size), method="bilinear")  # no-op for native-size patches
    img.set_shape((size, size, 3))
    return tf.cast(img, tf.float32) / 255.0  # [S, S, 3] in [0, 1]


def build_augmenter(cfg: dict, seed: int) -> Optional[tf.keras.Sequential]:
    """Train-only augmentation.

    Defaults reproduce the notebook's ImageDataGenerator (rotation 20 deg, zoom 0.05,
    shift 0.1, shear 0.05, h/v flips, reflect fill). Brightness/contrast jitter
    (named in the paper, magnitudes not reported) is off unless color_jitter > 0.
    """
    if not cfg.get("augment", False):
        return None
    L = tf.keras.layers
    ops = [
        L.RandomFlip("horizontal_and_vertical", seed=seed),
        L.RandomRotation(cfg.get("rotation_deg", 20) / 360.0, fill_mode="reflect", seed=seed),
        L.RandomZoom(cfg.get("zoom", 0.05), fill_mode="reflect", seed=seed),
        L.RandomTranslation(cfg.get("shift", 0.1), cfg.get("shift", 0.1), fill_mode="reflect", seed=seed),
    ]
    shear = cfg.get("shear", 0.05)
    if shear and hasattr(L, "RandomShear"):
        ops.append(L.RandomShear(x_factor=shear, y_factor=shear, fill_mode="reflect", seed=seed))
    cj = cfg.get("color_jitter", 0.0)
    if cj:
        ops += [L.RandomBrightness(cj, value_range=(0.0, 1.0), seed=seed),
                L.RandomContrast(cj, seed=seed)]
    return tf.keras.Sequential(ops, name="train_augmentation")


def make_dataset(paths: list[str], labels: np.ndarray, num_classes: int, size: int, batch_size: int,
                 training: bool, seed: int, augmenter: Optional[tf.keras.Sequential] = None) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels.astype("int32")))
    if training:
        ds = ds.shuffle(min(len(paths), 50_000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_decode(p, size), tf.one_hot(y, num_classes)),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    if training and augmenter is not None:
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 1.0), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
