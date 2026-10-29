"""Train and evaluate BMSA-ViT.

Every key of the YAML config can be overridden on the command line:
    python train.py --config configs/paper.yaml --dataset idc --data_root /data --seed 0
    python train.py --config configs/paper.yaml --dataset breakhis --patch_size 8
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from bmsa_vit.data import (
    assign_groups, build_augmenter, get_dataset_spec, leakage_report, list_files,
    make_dataset, make_random_splits, subsample,
)
from bmsa_vit.metrics import compute_metrics, format_summary, save_metrics
from bmsa_vit.model import build_bmsa_vit, count_macs, count_params
from bmsa_vit.utils import load_yaml, save_json, set_seed, setup_devices

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=os.path.join(HERE, "configs", "paper.yaml"))
    known, _ = pre.parse_known_args()
    cfg = load_yaml(known.config)

    p = argparse.ArgumentParser(description="Train BMSA-ViT", parents=[pre])
    p.add_argument("--datasets_registry", default=os.path.join(HERE, "configs", "datasets.yaml"))
    p.add_argument("--dataset_root", default=None, help="Override the registry root for this dataset")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--multi_gpu", action="store_true")
    p.add_argument("--run_name", default=None)
    for k, v in cfg.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", action=argparse.BooleanOptionalAction, default=v)
        elif v is None:
            p.add_argument(f"--{k}", default=None)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    return p.parse_args()


def class_weight_dict(labels: np.ndarray, k: int, mode: str) -> dict | None:
    if mode in ("none", None, False):
        return None
    counts = np.bincount(labels, minlength=k).astype(float)
    if mode == "max_ratio":           # notebook: classTotals.max() / classTotals
        w = counts.max() / counts
    elif mode == "balanced":
        w = compute_class_weight("balanced", classes=np.arange(k), y=labels)
    else:
        raise ValueError(f"Unknown class_weights '{mode}'")
    return {i: float(x) for i, x in enumerate(w)}


def main() -> None:
    a = parse_args()
    set_seed(a.seed, a.deterministic)
    strategy = setup_devices(a.gpu_id, a.multi_gpu)
    if a.amp:
        keras.mixed_precision.set_global_policy("mixed_float16")

    # ------------------------------------------------------------------ data
    spec = get_dataset_spec(a.dataset, a.datasets_registry, a.data_root, a.dataset_root)
    size, k = spec.image_size, len(spec.classes)
    run_name = a.run_name or f"{spec.name}_p{a.patch_size}_seed{a.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = os.path.join(a.output_dir, run_name)
    os.makedirs(out, exist_ok=True)

    df = list_files(spec)
    groups = assign_groups(df, spec.group_by)
    if groups is not None:
        df["group"] = groups
    if spec.split_mode == "random":
        df = make_random_splits(df, a.train_ratio, a.val_ratio, a.test_ratio, a.data_seed, groups)
    if groups is None:
        print("WARNING: no patient IDs available; patient independence between splits cannot be verified.")
    df = subsample(df, a.max_per_class, a.data_seed)
    df.to_csv(os.path.join(out, "splits.csv"), index=False)
    print(df.groupby(["split", "class_name"]).size().unstack(fill_value=0))

    leak = leakage_report(df, hash_files=a.check_duplicates)
    save_json(leak, os.path.join(out, "leakage_report.json"))
    print(f"Leakage report: {leak}")
    if any(v for d in (leak.get("patient_overlap", {}), leak.get("identical_files", {})) for v in d.values()):
        print("WARNING: patients or identical files are shared between splits (see leakage_report.json).")

    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    labels = {s: parts[s]["label"].to_numpy() for s in parts}
    aug = build_augmenter(vars(a), a.seed)
    ds = {s: make_dataset(parts[s]["path"].tolist(), labels[s], k, size, a.batch_size,
                          training=(s == "train"), seed=a.seed, augmenter=aug if s == "train" else None)
          for s in parts}

    # ----------------------------------------------------------------- model
    def _build() -> keras.Model:
        m = build_bmsa_vit(image_size=size, patch_size=a.patch_size, embed_dim=a.embed_dim,
                           num_heads=a.num_heads, mlp_dim=a.mlp_dim, num_blocks=a.num_blocks,
                           num_classes=k, attention_dropout=a.attention_dropout,
                           projection_dropout=a.projection_dropout, mlp_dropout=a.mlp_dropout,
                           attention_type=a.attention_type)
        m.compile(loss=keras.losses.CategoricalCrossentropy(label_smoothing=a.label_smoothing),
                  optimizer=keras.optimizers.AdamW(learning_rate=a.learning_rate, weight_decay=a.weight_decay),
                  metrics=[keras.metrics.CategoricalAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")])
        return m

    if strategy is not None:
        with strategy.scope():
            model = _build()
    else:
        model = _build()
    model.summary(print_fn=lambda s, **kw: print(s))
    n_params = int(model.count_params())
    macs = count_macs(size, a.patch_size, a.embed_dim, a.mlp_dim, a.num_blocks, k)
    print(f"Tokens: {(size // a.patch_size) ** 2} | params: {n_params:,} | MACs/image: {macs:,} "
          f"(~{2 * macs / 1e9:.3f} GFLOPs)")
    if a.attention_type == "mha_self_attention":
        assert n_params == count_params(size, a.patch_size, a.embed_dim, a.mlp_dim, a.num_blocks, k)
    save_json(vars(a) | {"image_size": size, "classes": spec.classes, "dataset_root": spec.root,
                         "split_mode": spec.split_mode, "group_by": spec.group_by, "n_params": n_params,
                         "macs_per_image": macs, "tf_version": tf.__version__,
                         "keras_version": keras.__version__},
              os.path.join(out, "config.json"))

    cw = class_weight_dict(labels["train"], k, a.class_weights)
    print(f"Class weights: {cw}")

    # ------------------------------------------------------------- training
    mode = "max" if "accuracy" in a.monitor else "min"
    ckpt = os.path.join(out, "best.weights.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(ckpt, monitor=a.monitor, mode=mode, save_best_only=True,
                                        save_weights_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor=a.monitor, mode=mode, factor=a.lr_factor,
                                          patience=a.lr_patience, cooldown=a.lr_cooldown,
                                          min_delta=a.lr_min_delta, min_lr=a.min_lr, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(out, "history.csv")),
    ]
    if a.early_stop_patience > 0:
        callbacks.append(keras.callbacks.EarlyStopping(monitor=a.monitor, mode=mode,
                                                       patience=a.early_stop_patience, verbose=1))
    try:
        import tensorboard  # noqa: F401
        callbacks.append(keras.callbacks.TensorBoard(log_dir=os.path.join(out, "logs")))
    except ImportError:
        pass

    t0 = time.time()
    model.fit(ds["train"], validation_data=ds["val"], epochs=a.epochs, callbacks=callbacks,
              class_weight=cw, verbose=2)
    train_time = time.time() - t0
    model.load_weights(ckpt)  # fixed selection rule: best `monitor`
    plot_history(os.path.join(out, "history.csv"), out)

    # ----------------------------------------------------------- evaluation
    summary = {"train_time_sec": train_time, "n_params": n_params, "macs_per_image": macs}
    for split in ("val", "test"):
        t1 = time.time()
        probs = model.predict(ds[split], verbose=0).astype("float64")
        dt = time.time() - t1
        m = compute_metrics(labels[split], probs, spec.classes, n_boot=a.n_bootstrap, seed=a.seed)
        m["inference_sec_per_image"] = dt / max(1, len(probs))
        save_metrics(m, probs, labels[split], parts[split]["path"].tolist(), spec.classes, out, split)
        summary[split] = {kk: m[kk] for kk in ("accuracy", "balanced_accuracy", "macro_f1", "kappa",
                                               "macro_auroc", "ece_15bin", "brier")}
        summary[split]["per_class"] = m["per_class"]
        print(f"\n[{split}]\n{format_summary(m)}")
    save_json(summary, os.path.join(out, "summary.json"))
    print(f"\nRun saved to {out}")


def plot_history(csv_path: str, out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.read_csv(csv_path)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in ((a1, "loss", "Loss"), (a2, "accuracy", "Accuracy")):
        ax.plot(h[key], label="train")
        ax.plot(h[f"val_{key}"], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
