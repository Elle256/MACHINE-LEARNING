# -*- coding: utf-8 -*-
"""
utils/metrics.py
-----------------
Các metrics chuẩn cho DTA:
  - MSE  : Mean Squared Error (loss chính)
  - CI   : Concordance Index (Harrell's C-index) – đo ranking
  - rm²  : Modified r² (Pearson r² với penalty cho deviation)
"""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concordance Index (CI): tỉ lệ cặp (i, j) mà i có affinity cao hơn j
    VÀ model predict đúng thứ tự đó.

    CI = 1.0 → ranking hoàn hảo
    CI = 0.5 → random
    """
    n = len(y_true)
    concordant = 0
    total = 0

    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j]):
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5

    return concordant / total if total > 0 else 0.0


def concordance_index_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Vectorized CI – nhanh hơn O(n²) naive cho dataset lớn.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Tất cả cặp (i, j) với i < j
    n = len(y_true)
    i_idx, j_idx = np.triu_indices(n, k=1)

    diff_true = y_true[i_idx] - y_true[j_idx]
    diff_pred = y_pred[i_idx] - y_pred[j_idx]

    # Chỉ xét cặp có y_true khác nhau
    valid = diff_true != 0
    diff_true = diff_true[valid]
    diff_pred = diff_pred[valid]

    concordant = np.sum(np.sign(diff_true) == np.sign(diff_pred))
    ties = np.sum(diff_pred == 0)
    total = len(diff_true)

    return float((concordant + 0.5 * ties) / total) if total > 0 else 0.0


def rm2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Modified r² (rm²) theo Roy et al. (2012).
    rm² = r² × (1 - √|r² - r0²|)
    trong đó r0² là r² khi fit qua gốc toạ độ.

    rm² > 0.5 → model có giá trị dự đoán tốt.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Pearson r²
    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r ** 2

    # r0²: hồi quy tuyến tính qua gốc (không có intercept)
    # y_pred ≈ k * y_true  →  k = (y_true · y_pred) / (y_true · y_true)
    k = np.dot(y_true, y_pred) / (np.dot(y_true, y_true) + 1e-8)
    y_pred_0 = k * y_true
    ss_res = np.sum((y_true - y_pred_0) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r0_2 = 1.0 - ss_res / (ss_tot + 1e-8)

    rm2 = r2 * (1.0 - np.sqrt(np.abs(r2 - r0_2)))
    return float(rm2)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fast_ci: bool = True,
) -> Dict[str, float]:
    """
    Tính toàn bộ metrics chuẩn DTA.

    Parameters
    ----------
    y_true, y_pred : numpy arrays
    fast_ci        : Dùng vectorized CI (True) hay naive O(n²) (False)

    Returns
    -------
    dict với keys: mse, ci, rm2
    """
    ci_fn = concordance_index_fast if fast_ci else concordance_index
    return {
        "mse": mse(y_true, y_pred),
        "ci": ci_fn(y_true, y_pred),
        "rm2": rm2_score(y_true, y_pred),
    }
