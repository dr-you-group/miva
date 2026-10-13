"""Category 1 — discrimination (spec §4.5, Steyerberg 2010).

Rank-based measures (AUROC, AUPRC) plus the threshold-dependent contingency
measures. AUROC and AUPRC delegate to scikit-learn, which implements exactly
these definitions; everything else is arithmetic on a 2x2 table and is
computed here so that a single pass over the data yields all seven values.

Positive prediction is ``prob >= threshold``. The inclusive comparison is
stated rather than assumed: with ties at the threshold the two conventions
give different sensitivities, and the models stage refits thresholds onto
observed score values, where ties are common.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from ._common import as_arrays, safe_divide

#: Metric names this module produces, in report order.
METRICS: Tuple[str, ...] = (
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "mcc",
    "balanced_accuracy",
    "discrimination_slope",
)

_SKLEARN_MISSING = (
    "AUROC and AUPRC require scikit-learn. Install it with `pip install 'mival[metrics]'`."
)


def _sklearn():
    try:
        from sklearn import metrics as sk_metrics
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(_SKLEARN_MISSING) from exc
    return sk_metrics


def confusion_counts(
    y_true: Sequence, y_prob: Sequence, threshold: float
) -> Tuple[int, int, int, int]:
    """``(tp, fp, tn, fn)`` at ``prob >= threshold``."""
    y, p = as_arrays(y_true, y_prob)
    predicted = p >= threshold
    positive = y == 1.0
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    tn = int(np.sum(~predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    return tp, fp, tn, fn


def auroc(y_true: Sequence, y_prob: Sequence) -> float:
    """Area under the ROC curve; NaN when only one class is present.

    A one-class slice has no defined AUROC. Returning NaN rather than raising
    matters inside the bootstrap, where a resample of a rare-event subgroup
    can legitimately contain no events.
    """
    y, p = as_arrays(y_true, y_prob)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(_sklearn().roc_auc_score(y, p))


def auprc(y_true: Sequence, y_prob: Sequence) -> float:
    """Average precision — the estimator of AUPRC without interpolation bias."""
    y, p = as_arrays(y_true, y_prob)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(_sklearn().average_precision_score(y, p))


def discrimination_slope(y_true: Sequence, y_prob: Sequence) -> float:
    """Yates' slope: mean predicted risk in events minus in non-events."""
    y, p = as_arrays(y_true, y_prob)
    events = p[y == 1.0]
    non_events = p[y == 0.0]
    if events.size == 0 or non_events.size == 0:
        return float("nan")
    return float(np.mean(events) - np.mean(non_events))


def contingency_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    """The seven threshold-dependent measures derived from a 2x2 table."""
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (
        float("nan")
        if denominator == 0.0
        else (tp * tn - fp * fn) / float(np.sqrt(denominator))
    )
    balanced_accuracy = (
        float("nan")
        if np.isnan(sensitivity) or np.isnan(specificity)
        else (sensitivity + specificity) / 2.0
    )
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": safe_divide(tp, tp + fp),
        "npv": safe_divide(tn, tn + fn),
        "f1": safe_divide(2 * tp, 2 * tp + fp + fn),
        "mcc": mcc,
        "balanced_accuracy": balanced_accuracy,
    }


def discrimination_metrics(
    y_true: Sequence, y_prob: Sequence, threshold: float
) -> Dict[str, float]:
    """All of category 1 for one slice."""
    values: Dict[str, float] = {
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
    }
    values.update(contingency_metrics(*confusion_counts(y_true, y_prob, threshold)))
    values["discrimination_slope"] = discrimination_slope(y_true, y_prob)
    return values


def roc_points(y_true: Sequence, y_prob: Sequence):
    """``(fpr, tpr, thresholds)`` for the ROC figure and threshold refitting."""
    y, p = as_arrays(y_true, y_prob)
    if len(np.unique(y)) < 2:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    fpr, tpr, thresholds = _sklearn().roc_curve(y, p)
    return fpr, tpr, thresholds


def pr_points(y_true: Sequence, y_prob: Sequence):
    """``(recall, precision)`` for the precision-recall figure."""
    y, p = as_arrays(y_true, y_prob)
    if len(np.unique(y)) < 2:
        empty = np.array([], dtype=float)
        return empty, empty
    precision, recall, _ = _sklearn().precision_recall_curve(y, p)
    return recall, precision
