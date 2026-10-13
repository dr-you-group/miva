"""Regression metrics (category 'regression'): error, fit, and the derived classification."""

from __future__ import annotations

from typing import Tuple, Dict, Sequence

import numpy as np

from .discrimination import auroc

METRICS = ("mae", "rmse", "r2", "bias", "loa_lo", "loa_hi")
METRIC_STEMS = ("auroc_below",)


def _arrays(y_true: Sequence, y_pred: Sequence):
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def mae(y_true: Sequence, y_pred: Sequence) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.mean(np.abs(p - y)))


def rmse(y_true: Sequence, y_pred: Sequence) -> float:
    y, p = _arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def r2(y_true: Sequence, y_pred: Sequence) -> float:
    y, p = _arrays(y_true, y_pred)
    total = float(np.sum((y - y.mean()) ** 2))
    return float("nan") if total == 0 else 1.0 - float(np.sum((p - y) ** 2)) / total


def bias(y_true: Sequence, y_pred: Sequence) -> float:
    """Mean of prediction minus truth (Bland-Altman bias)."""
    y, p = _arrays(y_true, y_pred)
    return float(np.mean(p - y))


def limits_of_agreement(y_true: Sequence, y_pred: Sequence) -> Tuple[float, float]:
    """Bland-Altman 95% limits: bias -/+ 1.96 SD of the differences."""
    y, p = _arrays(y_true, y_pred)
    diff = p - y
    spread = 1.96 * float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
    return float(np.mean(diff)) - spread, float(np.mean(diff)) + spread


def auroc_below(y_true: Sequence, y_pred: Sequence, cut: float) -> float:
    """AUROC for 'value <= cut', scoring by how far below the cut the prediction falls."""
    y, p = _arrays(y_true, y_pred)
    # auroc wants a score in [0, 1] because it is written for probabilities,
    # and cut - p is unbounded. A sigmoid would bound it but ties every
    # prediction more than 40 points from the cut (expit clips there), which
    # changes the rank order AUROC is made of. A min-max rescale over this
    # slice is bounded and strictly monotone, so no two distinct predictions
    # become equal. All-equal predictions have no spread to rescale, so they
    # get the constant 0.5 and auroc reports the 0.5 of a score that ranks
    # nothing, exactly as the sigmoid did.
    score = cut - p
    low, high = float(np.min(score)), float(np.max(score))
    span = high - low
    scaled = np.full(score.shape, 0.5) if span == 0.0 else (score - low) / span
    return auroc((y <= cut).astype(int), scaled)


def regression_metrics(y_true: Sequence, y_pred: Sequence, cuts: Sequence[float]) -> Dict[str, float]:
    low, high = limits_of_agreement(y_true, y_pred)
    values = {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred),
              "bias": bias(y_true, y_pred), "loa_lo": low, "loa_hi": high}
    for cut in cuts:
        values[f"auroc_below@{cut:g}"] = auroc_below(y_true, y_pred, float(cut))
    return values
