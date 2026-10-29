"""CPU unit tests. Run: pytest -q"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmsa_vit.data import (  # noqa: E402
    DatasetSpec, assign_groups, build_augmenter, leakage_report, list_files, make_dataset, make_random_splits,
)
from bmsa_vit.layers import MultiHeadSelfAttention, PatchExtract  # noqa: E402
from bmsa_vit.metrics import compute_metrics  # noqa: E402
from bmsa_vit.model import build_bmsa_vit, count_params  # noqa: E402


@pytest.mark.parametrize("size,patch,expected", [(48, 4, 554_882), (224, 4, 937_858), (48, 16, 629_762)])
def test_param_counts_match_paper_and_notebook(size, patch, expected):
    # 554,882 = paper Table 1 (IDC); 937,858 + 7 normalisation weights = 937,865 (paper, BreaKHis);
    # 629,762 + 7 = 629,769 (notebook summary)
    assert build_bmsa_vit(image_size=size, patch_size=patch).count_params() == expected
    assert count_params(size, patch) == expected


def test_output_is_probability():
    m = build_bmsa_vit(image_size=32, patch_size=8, embed_dim=16, mlp_dim=32, num_blocks=1, num_classes=3)
    y = m(np.random.rand(2, 32, 32, 3).astype("float32")).numpy()
    assert y.shape == (2, 3)
    np.testing.assert_allclose(y.sum(1), 1.0, atol=1e-5)


def test_patch_extract_raster_order():
    x = np.arange(4 * 4 * 1, dtype="float32").reshape(1, 4, 4, 1)
    p = PatchExtract(2)(x).numpy()
    assert p.shape == (1, 4, 4)
    np.testing.assert_array_equal(p[0, 0], [0, 1, 4, 5])
    np.testing.assert_array_equal(p[0, 1], [2, 3, 6, 7])


def test_mhsa_matches_reference():
    tf.random.set_seed(0)
    layer = MultiHeadSelfAttention(8, 2)
    x = np.random.rand(1, 5, 8).astype("float32")
    y = layer(x, training=False).numpy()
    W = lambda d: (d.kernel.numpy(), d.bias.numpy())  # noqa: E731
    (wq, bq), (wk, bk), (wv, bv), (wo, bo) = W(layer.q), W(layer.k), W(layer.v), W(layer.out)
    q, k, v = x[0] @ wq + bq, x[0] @ wk + bk, x[0] @ wv + bv
    heads = []
    for h in range(2):
        s = slice(4 * h, 4 * h + 4)
        sc = q[:, s] @ k[:, s].T / 2.0
        a = np.exp(sc - sc.max(1, keepdims=True))
        a /= a.sum(1, keepdims=True)
        heads.append(a @ v[:, s])
    ref = np.concatenate(heads, 1) @ wo + bo
    np.testing.assert_allclose(y[0], ref, atol=1e-5)


def test_weights_roundtrip(tmp_path):
    kw = dict(image_size=16, patch_size=4, embed_dim=16, mlp_dim=32, num_blocks=1)
    m1, m2 = build_bmsa_vit(**kw), build_bmsa_vit(**kw)
    p = str(tmp_path / "w.weights.h5")
    m1.save_weights(p)
    m2.load_weights(p)
    x = np.random.rand(2, 16, 16, 3).astype("float32")
    np.testing.assert_allclose(m1(x).numpy(), m2(x).numpy(), atol=1e-6)


def _write(path, rng):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (48, 48, 3), dtype=np.uint8)).save(path)


def test_folder_mode_and_patient_overlap(tmp_path):
    rng = np.random.default_rng(0)
    for split, folder in (("train", "training"), ("val", "validation"), ("test", "testing")):
        for c in ("0", "1"):
            for j in range(3):
                pid = "111" if split != "test" else "222"
                _write(str(tmp_path / folder / c / f"{pid}_idx5_x{j}_y{j}_class{c}_{split}.png"), rng)
    spec = DatasetSpec("t", str(tmp_path), ["0", "1"], 48, "folders", "idc_patient")
    df = list_files(spec)
    df["group"] = assign_groups(df, "idc_patient")
    rep = leakage_report(df)
    assert rep["n"] == {"train": 6, "val": 6, "test": 6}
    assert rep["patient_overlap"]["train_x_val"] == 1 and rep["patient_overlap"]["train_x_test"] == 0
    ds = make_dataset(df["path"].tolist(), df["label"].to_numpy(), 2, 48, 4, True, 0,
                      build_augmenter({"augment": True, "color_jitter": 0.1}, 0))
    x, y = next(iter(ds))
    assert x.shape == (4, 48, 48, 3) and y.shape == (4, 2)
    assert float(tf.reduce_min(x)) >= 0.0 and float(tf.reduce_max(x)) <= 1.0


def test_grouping_disabled_when_names_do_not_match():
    df = pd.DataFrame({"path": ["/a/foo.png", "/a/10253_idx5_x1_y1_class0.png"]})
    assert assign_groups(df, "idc_patient") is None


def test_patient_split_no_leakage():
    rows = [(f"/x/{pid}_idx5_x{j}_y{j}_class{lab}.png", lab, str(lab))
            for pid in range(30) for lab in (pid % 2,) for j in range(4)]
    df = pd.DataFrame(rows, columns=["path", "label", "class_name"])
    df["split"] = ""
    g = assign_groups(df, "idc_patient")
    out = make_random_splits(df, 0.72, 0.08, 0.20, 0, g)
    s = {k: set(out.loc[out.split == k, "group"]) for k in ("train", "val", "test")}
    assert not (s["train"] & s["test"] or s["train"] & s["val"] or s["val"] & s["test"])


def test_binary_metrics_sensitivity_specificity():
    y = np.array([0, 0, 0, 1, 1])
    probs = np.array([[.9, .1], [.8, .2], [.4, .6], [.3, .7], [.6, .4]])
    m = compute_metrics(y, probs, ["0", "1"], n_boot=0)
    assert m["per_class"]["1"]["sensitivity"] == pytest.approx(0.5)      # TPR for IDC-positive
    assert m["per_class"]["1"]["specificity"] == pytest.approx(2 / 3)    # TNR
