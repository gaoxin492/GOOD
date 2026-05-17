"""
OOD detection metrics: FPR@TPR95, AUROC, DTERR, AUIN, AUOUT.
"""

import numpy as np


def cal_metric(known: np.ndarray, novel: np.ndarray, method: str = None) -> dict:
    """Compute a suite of OOD detection metrics.

    Args:
        known: scores for in-distribution (known) samples.
        novel: scores for OOD (novel) samples.
        method: optional; pass 'row' to use a fixed threshold of -0.5.

    Returns:
        dict with keys: FPR, AUROC, DTERR, AUIN, AUOUT.
    """
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = {}

    # FPR at 95% TPR
    results["FPR"] = fpr_at_tpr95

    # AUROC
    tpr = np.concatenate([[1.0], tp / tp[0], [0.0]])
    fpr = np.concatenate([[1.0], fp / fp[0], [0.0]])
    results["AUROC"] = -np.trapz(1.0 - fpr, tpr)

    # Detection error (minimum over thresholds)
    results["DTERR"] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    denom = tp + fp
    denom[denom == 0.0] = -1.0
    pin_ind = np.concatenate([[True], denom > 0.0, [True]])
    pin = np.concatenate([[0.5], tp / denom, [0.0]])
    results["AUIN"] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.0] = -1.0
    pout_ind = np.concatenate([[True], denom > 0.0, [True]])
    pout = np.concatenate([[0.0], (fp[0] - fp) / denom, [0.5]])
    results["AUOUT"] = np.trapz(pout[pout_ind], 1.0 - fpr[pout_ind])

    return results


def get_curve(known: np.ndarray, novel: np.ndarray, method: str = None):
    """Compute TP/FP curves and FPR at 95% TPR."""
    known = np.sort(known)
    novel = np.sort(novel)

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == "row":
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones(num_k + num_n + 1, dtype=int)
    fp = -np.ones(num_k + num_n + 1, dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0

    for idx in range(num_k + num_n):
        if k == num_k:
            tp[idx + 1:] = tp[idx]
            fp[idx + 1:] = np.arange(fp[idx] - 1, -1, -1)
            break
        elif n == num_n:
            tp[idx + 1:] = np.arange(tp[idx] - 1, -1, -1)
            fp[idx + 1:] = fp[idx]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[idx + 1] = tp[idx]
                fp[idx + 1] = fp[idx] - 1
            else:
                k += 1
                tp[idx + 1] = tp[idx] - 1
                fp[idx + 1] = fp[idx]

    # Handle tied values
    all_scores = np.concatenate((known, novel))
    all_scores.sort()
    j = num_k + num_n - 1
    for idx in range(num_k + num_n - 1):
        if all_scores[j] == all_scores[j - 1]:
            tp[j] = tp[j + 1]
            fp[j] = fp[j + 1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95
