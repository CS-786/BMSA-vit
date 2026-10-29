"""Recompute the paper's headline metrics from its own confusion-matrix statements
(Sec. 4.2) to check internal consistency with Table 1.

    python scripts/check_paper_numbers.py
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score


def report(name: str, y: np.ndarray, p: np.ndarray) -> None:
    print(f"{name}: n={len(y)} acc={accuracy_score(y, p):.4f} kappa={cohen_kappa_score(y, p):.4f} "
          f"macro_recall={recall_score(y, p, average='macro'):.4f} macro_F1={f1_score(y, p, average='macro'):.4f}")


def main() -> None:
    # BreaKHis, Sec. 4.2: malignant TP=200, FN=0; benign 194/200 correct, FP=6
    y = np.r_[np.ones(200), np.zeros(200)]
    p = np.r_[np.ones(200), np.zeros(194), np.ones(6)]
    report("BreaKHis (paper CM)", y, p)
    print("  Table 1 reports kappa 0.89, recall 97.5, F1 97.0; 20% of 7,909 images = 1,582, not 400.\n")

    # IDC, Sec. 4.2: TPR=0.84, TNR=0.90, applied to the notebook test-folder sizes (39,793 / 15,712)
    n0, n1 = 39793, 15712
    tn, tp = round(0.90 * n0), round(0.84 * n1)
    y = np.r_[np.zeros(n0), np.ones(n1)]
    p = np.r_[np.zeros(tn), np.ones(n0 - tn), np.ones(tp), np.zeros(n1 - tp)]
    report("IDC (paper TPR/TNR)", y, p)
    print("  Table 1 reports acc 91.0, kappa 0.797, recall 89.5, F1 89.5.")
    print("  Sec. 4.2 also gives FPR=0.16, FNR=0.10; with TPR=0.84, TNR=0.90 these should be FPR=0.10, FNR=0.16.")


if __name__ == "__main__":
    main()
