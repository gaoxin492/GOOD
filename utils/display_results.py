"""
OOD detection metric utilities: AUROC, AUPR, FPR@95 via sklearn.
"""

import numpy as np
import sklearn.metrics as sk

recall_level_default = 0.95


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """High-precision cumulative sum with a stability check."""
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError("cumsum was found to be unstable: "
                           "its last element does not correspond to sum")
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default,
                          pos_label=None):
    """Compute FPR at a given recall (TPR) level."""
    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1]) or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0]) or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    if pos_label is None:
        pos_label = 1.0

    y_true = (y_true == pos_label)
    desc_idx = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_idx]
    y_true = y_true[desc_idx]

    distinct_idx = np.where(np.diff(y_score))[0]
    threshold_idx = np.r_[distinct_idx, y_true.size - 1]

    tps = stable_cumsum(y_true)[threshold_idx]
    fps = 1 + threshold_idx - tps

    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    recall = np.r_[recall[sl], 1]
    fps = np.r_[fps[sl], 0]

    cutoff = np.argmin(np.abs(recall - recall_level))
    return fps[cutoff] / np.sum(np.logical_not(y_true))


def get_measures(pos, neg, recall_level=recall_level_default):
    """Compute AUROC, AUPR, and FPR@recall_level.

    Args:
        pos: scores for the positive (OOD) class.
        neg: scores for the negative (ID) class.

    Returns:
        (auroc, aupr, fpr)
    """
    pos = np.array(pos).reshape(-1, 1)
    neg = np.array(neg).reshape(-1, 1)
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] = 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def show_performance(pos, neg, method_name="Ours", recall_level=recall_level_default):
    """Print AUROC, AUPR, FPR@recall for a method."""
    auroc, aupr, fpr = get_measures(pos, neg, recall_level)
    print(f"\t\t\t{method_name}")
    print(f"FPR{int(100 * recall_level)}:\t\t\t{100 * fpr:.2f}")
    print(f"AUROC:\t\t\t{100 * auroc:.2f}")
    print(f"AUPR:\t\t\t{100 * aupr:.2f}")


def print_measures(auroc, aupr, fpr, method_name="Ours", recall_level=recall_level_default):
    """Print pre-computed metrics."""
    print(f"\t\t\t\t{method_name}")
    print(f"  FPR{int(100 * recall_level)} AUROC AUPR")
    print(f"& {100 * fpr:.2f} & {100 * auroc:.2f} & {100 * aupr:.2f}")


def print_measures_with_std(aurocs, auprs, fprs, method_name="Ours",
                            recall_level=recall_level_default):
    """Print mean +/- std of metrics across runs."""
    print(f"\t\t\t\t{method_name}")
    print(f"  FPR{int(100 * recall_level)} AUROC AUPR")
    print(f"& {100 * np.mean(fprs):.2f} & {100 * np.mean(aurocs):.2f} & {100 * np.mean(auprs):.2f}")
    print(f"& {100 * np.std(fprs):.2f} & {100 * np.std(aurocs):.2f} & {100 * np.std(auprs):.2f}")
