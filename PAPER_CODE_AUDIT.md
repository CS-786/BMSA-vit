# Paper-code audit: "BMSA-ViT: Efficient Vision Transformer for Pathology Breast Cancer Classification in Health Care System"

Sources compared: `Final_Revised_Paper.pdf` (the paper) and `legacy/01-idc_notebook.ipynb` (the notebook). Numbers marked **computed** come from `scripts/complexity_table.py` and `scripts/check_paper_numbers.py` in this repository. Fix the Critical and Major items before resubmission.

## 1. Method description vs implementation

| ID | Severity | Paper | Notebook / this repo | Action |
|---|---|---|---|---|
| A1 | Critical | Sec. 2.2 describes "M-key (Memory Key)" and "M-value" memory matrices; Eq. 10 defines `M_k = softmax(W_k · X)`. | Standard multi-head self-attention: Q, K, V are linear projections of the tokens. No memory matrix exists. | Either implement the memory attention and re-run, or rewrite Sec. 2.2 and Eq. 6-7, 10 as standard MHSA. |
| A2 | Critical | Abstract: "adapter-style tuning". | No adapters, no pre-training, all weights trained from scratch. | Remove the claim. |
| A3 | Major | Fig. 2 caption: "class token output classified via an MLP head". Sec. 2.2: Eq. 8-10 MLP head. | No class token. Global average pooling over tokens (Eq. 9) then one Dense+softmax layer; no MLP head after pooling. | Correct the caption and the head description. |
| A4 | Minor | Contribution: "ViT mobile models were studied". | Not in the code. | Remove or add the experiment. |
| A5 | Minor | Model called BMSA-ViT, MSA-ViT, MHSA-ViT, MHSAT, "MAS" mechanism; Table 1 caption "BMEA-ViT". | n/a | Use one name. |

## 2. Hyperparameters and protocol

| ID | Severity | Paper (Sec. 3.3, 3.4) | Notebook (values actually passed to `fit`) | This repo |
|---|---|---|---|---|
| H1 | Major | Patch 4×4 | `patch_size = 16` (9 tokens) | `paper.yaml`: 4; `notebook.yaml`: 16 |
| H2 | Major | lr 1e-4 | `learning_rate = 0.001` | as config |
| H3 | Major | 100 epochs | `epochs=NUM_EPOCHS` = 5 (`num_epochs = 50` unused) | as config |
| H4 | Major | batch 16 | generators use `BS = 32` (`batch_size = 4` unused) | as config |
| H5 | Major | "decaying learning rate" | `ReduceLROnPlateau(min_lr=1e-4)`; at lr 1e-4 this never decays | `paper.yaml` uses `min_lr: 1e-6` |
| H6 | Critical | 80/20 stratified train/test | Predefined folders 199,818 / 22,201 / 55,505 (72 / 8 / 20 %); validation folder not used for selection | Folder mode reproduces this; `idc_patient_split` gives a patient-level split |
| H7 | Critical | n/a | `ModelCheckpoint(monitor='loss')`: the tested checkpoint is chosen by **training** loss | `paper.yaml` selects on `val_loss`; `notebook.yaml` keeps `loss` for reproduction |
| H8 | Major | Augmentation: rotation, zoom, flips, brightness/contrast jitter | `ImageDataGenerator`: rotation 20°, zoom 0.05, shift 0.1, shear 0.05, flips; no brightness/contrast | Notebook values by default; `--color_jitter` adds brightness/contrast |
| H9 | Major | Keras 2.15.0 / TF 2.15.1 | Notebook output shows TF 2.6.4 (Kaggle) | This repo: TF ≥ 2.16 / Keras 3 |
| H10 | Critical | BreaKHis results reported | No BreaKHis code in the notebook | Added (`breakhis`, patient-level split) |

## 3. Reproducibility of the notebook itself

| ID | Severity | Problem |
|---|---|---|
| R1 | Critical | `get_model` calls `data_augmentation(inputs)`, but `data_augmentation` is defined only in a commented-out cell. The saved run therefore used a definition from an earlier kernel state. Its summary shows 7 non-trainable weights, i.e. a `Normalization` layer adapted on `x_train`, which is not defined anywhere in the notebook. The notebook cannot be re-run as saved. |
| R2 | Major | When that `data_augmentation` block is active, images are augmented twice: by `ImageDataGenerator` and by the random layers inside the model. |
| R3 | Major | No seeds; `tensorflow_addons`, `keras.utils.np_utils`, `ImageDataGenerator`, `scipy.interp` are removed in current libraries. |

**Parameter counts (computed)** identify the configurations behind Table 1:

| Setting | Trainable params | Paper |
|---|---|---|
| IDC 48×48, patch 4 | 554,882 | Table 1 IDC: 554,882 (exact match) |
| BreaKHis 224×224, patch 4 | 937,858 (+7 normalisation = 937,865) | Table 1 BreaKHis: 937,865 (match with Normalization layer) |
| IDC 48×48, patch 16 (notebook) | 629,762 (+7 = 629,769) | notebook summary: 629,769 |

So the IDC and BreaKHis models in Table 1 were built differently (with and without the Normalization layer), and neither equals the saved notebook configuration.

## 4. Reported numbers that conflict with each other

| ID | Severity | Statement | Check (computed) |
|---|---|---|---|
| N1 | Critical | Sec. 4.3, Table 2, Conclusion: 0.03 GFLOPs | 224/4 (BreaKHis): **23.47 GFLOPs**; 48/4 (IDC): **0.195 GFLOPs**. No 4×4 setting is near 0.03 (FLOPs = 2 × MACs, matmuls only). |
| N2 | Critical | BreaKHis Sec. 4.2: TP=200, FN=0, TN=194, FP=6 | Gives acc 98.5 % (consistent) but κ = **0.970**, macro recall **98.5**, macro F1 **98.5**. Table 1 reports κ 0.89, recall 97.5, F1 97.0. |
| N3 | Critical | BreaKHis test set has 400 images | 20 % of 7,909 = 1,582. The confusion matrix cannot come from the stated 80/20 split. |
| N4 | Major | IDC Sec. 4.2: TPR 0.84, TNR 0.90 | With the notebook test sizes (39,793 / 15,712): acc **88.3 %**, κ **0.720**. Table 1: acc 91.0 %, κ 0.797. |
| N5 | Minor | IDC: "FPR=0.16, FNR=0.10" | With TPR 0.84, TNR 0.90: FPR = 0.10, FNR = 0.16 (swapped). |
| N6 | Major | Sec. 4.4: 2×2 patches "raise parameters +67% for negligible gain (+0.2%)" | Computed: +128 % (224) and +9.1 % (48). Table 3 shows 2×2 **below** 4×4 (97.8 vs 98.5), contradicting "+0.2 % gain". |
| N7 | Minor | κ written as 89 / 79.7 in Table 1 and 0.89 / 0.80 in text | Use one scale. |
| N8 | Major | Conclusion: "state-of-the-art", "consistently outperformed all baselines" | Only single runs, no variance, no baseline code, image-level splits. Not supported as stated. |
| N9 | Major | Sec. 4.3: suitable for "modest hardware" | 224/4 has 3,136 tokens; one attention map per block is 4 × 3136² floats per image. In this sandbox (7 GB RAM, CPU) batch 8 ran out of memory; batch 2 ran. |

## 5. Metrics code

| ID | Severity | Problem | Fix in this repo |
|---|---|---|---|
| M1 | Major | `sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])` treats class 0 (IDC-negative) as positive, so the printed "sensitivity" is the specificity for IDC-positive and vice versa. | Per-class sensitivity/specificity; class `1` = IDC-positive. |
| M2 | Major | `f1_score(..., labels=np.unique(y_hat))` drops classes the model never predicts, inflating macro-F1. | All labels always included. |
| M3 | Minor | "SD of F1" is computed around the weighted mean over 2 classes; not a variance across runs. | Multi-seed mean ± SD (`scripts/run_seeds.sh`). |
| M4 | n/a | No confidence intervals or calibration. | Bootstrap 95 % CIs, ECE, Brier. |

## 6. Data leakage

- **BreaKHis**: 82 patients and 7,909 images; an image-level 80/20 split puts images of the same patient in train and test. Use `--dataset breakhis` (patient-level split).
- **IDC**: patches come from whole-slide images of a limited number of patients. Whether the Kaggle `training/validation/testing` folders are patient-disjoint is not documented in the notebook. `train.py` reports patient overlap between the folders when the original file names (`<patient>_idx5_x..._y..._class..png`) are kept. If overlap is non-zero, use `--dataset idc_patient_split`.

## 7. References (from the paper's own list; verify each)

- Ref. [13] is cited for the Vision Transformer, but the entry is Spanhol et al., the BreaKHis dataset paper.
- Ref. [5] is cited for a CNN patch-selection WSI method, but the entry is Gheflati & Rivaz, a breast ultrasound ViT paper.
- The BreaKHis and IDC dataset papers should be cited where the datasets are introduced (Sec. 3.1).
