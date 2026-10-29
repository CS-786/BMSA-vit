"""Seeding, device setup and config helpers."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Optional

import numpy as np
import tensorflow as tf
import yaml


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and TensorFlow. `deterministic=True` also enables
    TF op determinism (slower; some GPU ops may raise errors)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def setup_devices(gpu_id: int = 0, multi_gpu: bool = False) -> Optional[tf.distribute.Strategy]:
    """Single GPU (visible device = gpu_id) or MirroredStrategy over all GPUs.
    Returns the strategy, or None for single-device training."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU found: running on CPU.")
        return None
    try:
        if multi_gpu:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            strategy = tf.distribute.MirroredStrategy()
            print(f"MirroredStrategy with {strategy.num_replicas_in_sync} replicas")
            return strategy
        tf.config.set_visible_devices(gpus[gpu_id], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        print(f"Using GPU {gpu_id} of {len(gpus)}")
    except RuntimeError as e:  # devices already initialised
        print(e)
    return None


def load_yaml(path: Optional[str]) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
