from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = len(y_true)
    i_idx, j_idx = np.triu_indices(n, k=1)

    diff_true = y_true[i_idx] - y_true[j_idx]
    diff_pred = y_pred[i_idx] - y_pred[j_idx]

    valid = diff_true != 0
    diff_true = diff_true[valid]
    diff_pred = diff_pred[valid]

    concordant = np.sum(np.sign(diff_true) == np.sign(diff_pred))
    ties = np.sum(diff_pred == 0)
    total = len(diff_true)

    return float((concordant + 0.5 * ties) / total) if total > 0 else 0.0


def rm2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    r = np.corrcoef(y_true, y_pred)[0, 1]
    r2 = r ** 2

    k = np.dot(y_true, y_pred) / (np.dot(y_true, y_true) + 1e-8)
    y_pred_0 = k * y_true
    ss_res = np.sum((y_true - y_pred_0) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r0_2 = 1.0 - ss_res / (ss_tot + 1e-8)

    rm2 = r2 * (1.0 - np.sqrt(np.abs(r2 - r0_2)))
    return float(rm2)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, fast_ci: bool = True,) -> Dict[str, float]:

    ci_fn = concordance_index_fast if fast_ci else concordance_index
    return {
        "mse": mse(y_true, y_pred),
        "ci": ci_fn(y_true, y_pred),
        "rm2": rm2_score(y_true, y_pred),
    }
