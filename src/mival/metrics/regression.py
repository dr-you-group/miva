"""Regression metrics (category 'regression'): error, fit, and the derived classification."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from mival.criteria import Cut

from .discrimination import auroc

#: Error and agreement, then calibration of the predicted value (STRATOS):
#: calibration-in-the-large and the ``y ~ a + b * p`` line.
CORE_METRICS = ("mae", "rmse", "r2", "bias", "loa_lo", "loa_hi")
#: The error summaries an age-prediction paper reports (PROPHECG-Age, Cho et
#: al., Eur Heart J 2025): MAE, MAPE and MSE each with the SD of its per-record
#: error, and Pearson's r between prediction and truth.
ERROR_SPREAD_METRICS = ("mae_sd", "mse", "mse_sd", "mape", "mape_sd", "pearson_r")
CALIBRATION_METRICS = ("reg_citl", "reg_calibration_intercept", "reg_calibration_slope")
METRICS = CORE_METRICS + ERROR_SPREAD_METRICS + CALIBRATION_METRICS
#: ``auroc_cut@<cut.name>`` for a cut with a direction; ``auroc_below@<v>`` for
#: a bare-number cut, the name results were published under before cuts had one.
METRIC_STEMS = ("auroc_below", "auroc_cut", "residual_frac")


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


def _sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def error_spread(y_true: Sequence, y_pred: Sequence) -> Dict[str, float]:
    """MAE, MSE and MAPE with the SD of each per-record error, and Pearson's r.

    MAPE is over records whose truth is not zero (a percentage of zero has no
    value) and is in percent. r is NaN when either side is constant.
    """
    y, p = _arrays(y_true, y_pred)
    absolute = np.abs(p - y)
    squared = (p - y) ** 2
    nonzero = y != 0
    percent = 100.0 * absolute[nonzero] / np.abs(y[nonzero])
    spread = np.std(y) * np.std(p)
    r = float(np.mean((y - y.mean()) * (p - p.mean())) / spread) if spread > 0 else float("nan")
    return {
        "mae_sd": _sd(absolute),
        "mse": float(np.mean(squared)),
        "mse_sd": _sd(squared),
        "mape": float(np.mean(percent)) if percent.size else float("nan"),
        "mape_sd": _sd(percent),
        "pearson_r": r,
    }


def residual_frac(y_true: Sequence, y_pred: Sequence, cut: Cut) -> float:
    """Share of records whose residual ``pred - truth`` satisfies the cut (the "ECG-aged" share at >= 7)."""
    y, p = _arrays(y_true, y_pred)
    return float(np.mean(cut.events(p - y))) if y.size else float("nan")


def residual_metric_name(cut: Cut) -> str:
    return "residual_frac@%s" % cut.name


def auroc_cut(y_true: Sequence, y_pred: Sequence, cut: Cut) -> float:
    """AUROC for the event ``y <op> value``, scoring by how far past the cut the prediction falls."""
    y, p = _arrays(y_true, y_pred)
    # auroc wants a score in [0, 1] because it is written for probabilities,
    # and the distance past the cut is unbounded. A sigmoid would bound it but
    # ties every prediction more than 40 points from the cut (expit clips
    # there), which changes the rank order AUROC is made of. A min-max rescale
    # over this slice is bounded and strictly monotone, so no two distinct
    # predictions become equal. All-equal predictions have no spread to
    # rescale, so they get the constant 0.5 and auroc reports the 0.5 of a
    # score that ranks nothing, exactly as the sigmoid did.
    score = p - cut.value if cut.higher_is_event else cut.value - p
    low, high = float(np.min(score)), float(np.max(score))
    span = high - low
    scaled = np.full(score.shape, 0.5) if span == 0.0 else (score - low) / span
    return auroc(cut.events(y).astype(int), scaled)


def auroc_below(y_true: Sequence, y_pred: Sequence, cut: float) -> float:
    """AUROC for 'value <= cut' (the bare-number cut)."""
    return auroc_cut(y_true, y_pred, Cut(float(cut), "<=", legacy=True))


def cut_metric_name(cut: Cut) -> str:
    return "auroc_below@%g" % cut.value if cut.legacy else "auroc_cut@%s" % cut.name


def reg_calibration(y_true: Sequence, y_pred: Sequence) -> Dict[str, float]:
    """Calibration of a predicted value: mean(y) - mean(p) and the OLS line ``y ~ a + b * p``.

    Constant predictions have no slope; both line coefficients are then NaN.
    """
    y, p = _arrays(y_true, y_pred)
    citl = float(np.mean(y) - np.mean(p))
    spread = float(np.sum((p - p.mean()) ** 2))
    if spread == 0.0:
        slope = intercept = float("nan")
    else:
        slope = float(np.sum((p - p.mean()) * (y - y.mean()))) / spread
        intercept = float(y.mean() - slope * p.mean())
    return {
        "reg_citl": citl,
        "reg_calibration_intercept": intercept,
        "reg_calibration_slope": slope,
    }


def fit_linear_recalibration(
    y_dev: Sequence, p_dev: Sequence
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """OLS update ``p -> a + b * p`` fitted on development data, or None.

    None for no records or a constant prediction, which has no slope.
    """
    if len(y_dev) == 0:
        return None
    line = reg_calibration(y_dev, p_dev)
    a, b = line["reg_calibration_intercept"], line["reg_calibration_slope"]
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    return lambda values: a + b * np.asarray(values, dtype=float)


def core_regression_metrics(y_true: Sequence, y_pred: Sequence,
                            residual_cuts: Sequence[Cut] = ()) -> Dict[str, float]:
    """Every regression metric that is not about an event cut of the truth.

    ``residual_cuts`` adds the share of records past each cut of the residual.
    """
    low, high = limits_of_agreement(y_true, y_pred)
    values = {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "r2": r2(y_true, y_pred),
              "bias": bias(y_true, y_pred), "loa_lo": low, "loa_hi": high}
    values.update(error_spread(y_true, y_pred))
    values.update(reg_calibration(y_true, y_pred))
    for cut in residual_cuts:
        values[residual_metric_name(cut)] = residual_frac(y_true, y_pred, cut)
    return values


def regression_metrics(y_true: Sequence, y_pred: Sequence, cuts: Sequence[Cut]) -> Dict[str, float]:
    values = core_regression_metrics(y_true, y_pred)
    for cut in cuts:
        values[cut_metric_name(cut)] = auroc_cut(y_true, y_pred, cut)
    return values
