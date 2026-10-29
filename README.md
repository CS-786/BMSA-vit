# BMSA-ViT

Lightweight Vision Transformer with multi-head self-attention for breast cancer histopathology classification (IDC and BreaKHis), implemented in TensorFlow / Keras 3.

Code for the paper *BMSA-ViT: Efficient Vision Transformer for Pathology Breast Cancer Classification in Health Care System* (M. U. Saleem, B. Peng, M. S. Mastoi, M. Hassan, M. Z. Rafique).

> A paper-code consistency audit is in [docs/PAPER_CODE_AUDIT.md](docs/PAPER_CODE_AUDIT.md). It lists where the manuscript and the original notebook disagree and should be read before reusing any number from the paper.

## Architecture

```
Image [B,S,S,3] in [0,1]
  -> PatchExtract (P×P, non-overlapping)            -> [B, N, P·P·3],  N = (S/P)²
  -> PatchEmbedding (Dense + learned 1-D position)  -> [B, N, D]
  -> 4 × TransformerEncoder (pre-norm)              -> [B, N, D]
        x = x + MHSA(LN(x))          h heads, scaled dot-product
        x = x + MLP(LN(x))           Dense(mlp_dim, GELU) -> Dense(D), dropout 0.2
  -> GlobalAveragePooling1D -> Dense(softmax)        -> [B, K]
```

Defaults (`configs/paper.yaml`): patch 4, D = 128, 4 heads, 4 blocks, MLP 256, attention/projection dropout 0.1, AdamW (lr 1e-4, weight decay 1e-4), label smoothing 0.1, batch 16, 100 epochs.

### Size and compute

Computed with `python scripts/complexity_table.py` (FLOPs = 2 × multiply-accumulates of all matrix products; softmax, LayerNorm, GELU and additions excluded):

| Input / patch | Tokens | Parameters | GFLOPs / image |
|---|---|---|---|
| 48 / 16 (notebook) | 9 | 629,762 | 0.011 |
| 48 / 8 | 36 | 559,490 | 0.042 |
| **48 / 4 (IDC)** | 144 | **554,882** | **0.195** |
| 48 / 2 | 576 | 605,570 | 1.285 |
| 224 / 16 | 196 | 653,698 | 0.323 |
| 224 / 8 | 784 | 655,234 | 2.119 |
| **224 / 4 (BreaKHis)** | 3,136 | **937,858** | **23.47** |
| 224 / 2 | 12,544 | 2,137,474 | 335.4 |

Most parameters at 224 / 4 are position embeddings (3,136 × 128). Memory grows with N²: in a 7 GB CPU sandbox, 224 / 4 trained at batch 2 but ran out of memory at batch 8. Use `--amp`, a smaller batch, or a larger patch on small GPUs.

## Repository structure

```
BMSA-ViT/
├── bmsa_vit/
│   ├── layers.py     # PatchExtract, PatchEmbedding, MultiHeadSelfAttention, TransformerEncoder
│   ├── model.py      # build_bmsa_vit(), closed-form parameter and MAC counts
│   ├── data.py       # registry, folder/random splits, patient grouping, leakage report, tf.data, augmentation
│   ├── metrics.py    # acc, balanced acc, macro-F1, kappa, sens/spec, AUROC, ECE, Brier, bootstrap CIs
│   └── utils.py      # seeding, devices, config I/O
├── configs/
│   ├── paper.yaml     # settings reported in the paper
│   ├── notebook.yaml  # settings the original notebook actually ran
│   └── datasets.yaml  # dataset roots, classes, image size, split mode, patient grouping
├── scripts/
│   ├── prepare_idc.py         # flatten IDC patient folders, keep patient IDs
│   ├── prepare_breakhis.py    # BreaKHis -> benign/malignant folders
│   ├── run_seeds.sh           # multi-seed training
│   ├── aggregate_seeds.py     # per-seed table + mean ± SD
│   ├── complexity_table.py    # parameter / FLOP table
│   └── check_paper_numbers.py # recompute paper metrics from its confusion matrices
├── tests/test_core.py
├── train.py
├── evaluate.py
├── docs/PAPER_CODE_AUDIT.md
├── legacy/01-idc_notebook.ipynb   # original notebook, unchanged
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/<your-username>/BMSA-ViT.git
cd BMSA-ViT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q    # 11 CPU tests
```

Requires TensorFlow ≥ 2.16 (Keras 3). Tested on CPU with TensorFlow 2.21.0, Keras 3.15.1, Python 3.13. GPU training was not tested in this packaging step.

## Data

| Key | Layout | Size | Split |
|---|---|---|---|
| `idc` | `IDC/{training,validation,testing}/{0,1}/*.png` (as used in the notebook) | 48 | predefined folders; patient overlap is reported |
| `idc_patient_split` | `IDC_flat/{0,1}/*.png` from `scripts/prepare_idc.py` | 48 | patient-level, stratified |
| `breakhis` | `breakhis_binary/{benign,malignant}/*.png` from `scripts/prepare_breakhis.py` | 224 | patient-level, stratified |

```bash
python scripts/prepare_idc.py --src /path/to/IDC_patient_folders --dst data/IDC_flat --symlink
python scripts/prepare_breakhis.py --src /path/to/BreaKHis_v1 --dst data/breakhis_binary --mode binary
```

Random splits default to 72 / 8 / 20 (train / validation / test): the paper's 80/20 train/test split with 10 % of training data held out for checkpoint selection.

## Training

```bash
# paper settings
python train.py --config configs/paper.yaml --dataset idc --data_root /path/to/data --seed 0
python train.py --config configs/paper.yaml --dataset breakhis --data_root /path/to/data --seed 0

# reproduce the notebook run (patch 16, lr 1e-3, batch 32, 5 epochs, selection on training loss)
python train.py --config configs/notebook.yaml --dataset idc --data_root /path/to/data

# any config key can be overridden
python train.py --config configs/paper.yaml --dataset idc --patch_size 8 --color_jitter 0.1

# 3 seeds, same split, mean ± SD
SEEDS="0 1 2" bash scripts/run_seeds.sh breakhis /path/to/data
```

Protocol in `train.py`:

1. Split (predefined folders or patient-level stratified), saved to `splits.csv`.
2. Leakage report: patient overlap and byte-identical files between splits (`leakage_report.json`).
3. Images scaled to [0, 1]; augmentation on the training stream only.
4. Class weights `max(count)/count` from training labels (as in the notebook).
5. Checkpoint with the best `val_loss` is reloaded before evaluation; the test set is used once.

## Outputs

Each run directory contains `config.json`, `splits.csv`, `leakage_report.json`, `best.weights.h5`, `history.csv`, `training_curves.png`, and for validation and test: `*_metrics.json` (accuracy, balanced accuracy, macro-F1, κ, AUROC, per-class sensitivity / specificity, ECE, Brier, 95 % bootstrap CIs), `*_predictions.csv` (per-image probabilities), `*_confusion_matrix.png`, `*_classification_report.csv`, plus `summary.json`.

```bash
python evaluate.py --run_dir results/<run> --split test
```

## Results

To be filled from `scripts/aggregate_seeds.py` (test set, mean ± SD over ≥ 3 seeds). Numbers from the paper are not repeated here; see the audit for their consistency checks.

| Dataset | Split unit | Accuracy | Balanced acc. | Macro-F1 | AUROC | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| IDC | | _ | _ | _ | _ | _ | _ |
| BreaKHis | patient | _ | _ | _ | _ | _ | _ |

## Citation

```bibtex
@unpublished{saleem_bmsavit,
  title  = {{BMSA-ViT}: Efficient Vision Transformer for Pathology Breast Cancer Classification in Health Care System},
  author = {Saleem, Muhammad Usman and Peng, Bo and Mastoi, Muhammad Shahid and Hassan, Mannan and Rafique, Muhammad Zahid},
  note   = {Manuscript. Update with venue, year and DOI after publication}
}
```

## License

MIT, see [LICENSE](LICENSE).
