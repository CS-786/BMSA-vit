"""Evaluation metrics computed from predicted probabilities.

Reports accuracy, balanced accuracy, macro-F1, Cohen's kappa, per-class
sensitivity/specificity/precision/F1, AUROC from probabilities (one-vs-rest),
expected calibration error, Brier score, and percentile bootstrap 95% CIs.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def macro_auroc(y_true: np.ndarray, probs: np.ndarray) -> float:
    k = probs.shape[1]
    try:
        if k == 2:
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                                   labels=np.arange(k)))
    except ValueError:
        return float("nan")


def _scalar_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_pred = probs.argmax(axis=1)
    k = probs.shape[1]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=np.arange(k),
                                   zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_auroc": macro_auroc(y_true, probs),
    }


def bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict[str, list[float]]:
    """Percentile bootstrap CI over test images. Note: if images are not
    independent (several per patient), these intervals are too narrow."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for key, val in _scalar_metrics(y_true[idx], probs[idx]).items():
            samples.setdefault(key, []).append(val)
    return {
        key: [float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2)))]
        for key, v in samples.items()
    }


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, class_names: Sequence[str],
                    n_boot: int = 1000, seed: int = 0) -> dict:
    k = len(class_names)
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(k))
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - tp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)

    per_class = {}
    for i, c in enumerate(class_names):
        try:
            auc_i = float(roc_auc_score((y_true == i).astype(int), probs[:, i]))
        except ValueError:
            auc_i = float("nan")
        per_class[c] = {
            "support": int(cm[i].sum()),
            "sensitivity": float(sens[i]),
            "specificity": float(spec[i]),
            "auroc_ovr": auc_i,
        }

    out = {
        "n": int(len(y_true)),
        **_scalar_metrics(y_true, probs),
        "ece_15bin": expected_calibration_error(probs, y_true),
        "brier": brier_score(probs, y_true),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, labels=np.arange(k), target_names=list(class_names),
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": cm.tolist(),
    }
    if n_boot > 0:
        out["ci95_bootstrap"] = bootstrap_ci(y_true, probs, n_boot=n_boot, seed=seed)
    return out


def save_metrics(metrics: dict, probs: np.ndarray, y_true: np.ndarray, paths: Sequence[str],
                 class_names: Sequence[str], out_dir: str, prefix: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame(probs, columns=[f"p_{c}" for c in class_names])
    pred_df.insert(0, "y_pred", probs.argmax(axis=1))
    pred_df.insert(0, "y_true", y_true)
    pred_df.insert(0, "path", list(paths))
    pred_df.to_csv(os.path.join(out_dir, f"{prefix}_predictions.csv"), index=False)

    pd.DataFrame(metrics["classification_report"]).T.to_csv(
        os.path.join(out_dir, f"{prefix}_classification_report.csv"))

    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(max(5, len(class_names)), max(4, 0.8 * len(class_names))))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({prefix})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix.png"), dpi=200)
    plt.close(fig)


def format_summary(m: dict) -> str:
    ci = m.get("ci95_bootstrap", {})

    def f(key: str) -> str:
        s = f"{m[key]:.4f}"
        if key in ci:
            s += f" [{ci[key][0]:.4f}, {ci[key][1]:.4f}]"
        return s

    lines = [f"n = {m['n']}"]
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "kappa", "macro_auroc"):
        lines.append(f"{key:>18}: {f(key)}")
    lines.append(f"{'ece_15bin':>18}: {m['ece_15bin']:.4f}")
    lines.append(f"{'brier':>18}: {m['brier']:.4f}")
    return "\n".join(lines)
