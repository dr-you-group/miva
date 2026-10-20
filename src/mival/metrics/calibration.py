"""Category 2 — calibration (spec §4.5; Steyerberg 2010, Van Calster 2019).

Van Calster's hierarchy is followed literally, weakest level first:

1. **mean** calibration — calibration-in-the-large, ``mean(y) - mean(p)``
2. **weak** calibration — intercept and slope of ``logit(y) ~ a + b*logit(p)``
3. **moderate** calibration — a flexible (restricted cubic spline) curve, and
   its scalar summary, the Estimated Calibration Index (Van Hoorde 2015)

Brier lives here rather than in an "overall performance" layer of its own
(spec §4.5). It is not an independent property but a composite: by the Murphy
(1973) decomposition ``Brier = reliability - resolution + uncertainty``, where
reliability is a calibration quantity and resolution a discrimination one.
Because uncertainty is fixed by prevalence alone, a low-prevalence outcome
such as STEMI makes the raw Brier score mostly a restatement of how rare the
event is — which is why the scaled Brier (IPA) is always reported beside it.

Nothing here uses scikit-learn: it has neither the Murphy decomposition, nor
unpenalized logistic recalibration (``LogisticRegression`` is ridge-penalized
by default, which shrinks the calibration slope towards zero — precisely the
quantity being measured), nor a spline calibration curve.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ._common import as_arrays, expit, logit

#: Metric names this module produces, in report order.
METRICS: Tuple[str, ...] = (
    "citl",
    "calibration_intercept",
    "calibration_intercept_offset",
    "oe_ratio",
    "calibration_slope",
    "eci",
    "brier",
    "brier_scaled",
    "brier_binned",
    "brier_reliability",
    "brier_resolution",
    "brier_uncertainty",
)

#: Murphy's decomposition is exact only when forecasts are grouped by equal
#: value. Continuous scores are all distinct, which makes every group a single
#: record and collapses resolution onto uncertainty, so the reported
#: decomposition bins the forecast. Ten equal-count bins is the convention of
#: the reliability diagram literature.
DEFAULT_MURPHY_BINS = 10

#: Harrell's restricted-cubic-spline knot placement, by number of knots.
_KNOT_QUANTILES = {
    3: (0.10, 0.50, 0.90),
    4: (0.05, 0.35, 0.65, 0.95),
    5: (0.05, 0.275, 0.50, 0.725, 0.95),
}
DEFAULT_KNOTS = 4


def brier(y_true: Sequence, y_prob: Sequence) -> float:
    y, p = as_arrays(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def brier_scaled(y_true: Sequence, y_prob: Sequence) -> float:
    """Index of Prediction Accuracy: 1 - Brier / Brier of the prevalence model.

    Zero means "no better than predicting the observed prevalence for every
    patient", which is the baseline the raw Brier hides when events are rare.
    """
    y, p = as_arrays(y_true, y_prob)
    prevalence = float(np.mean(y))
    reference = prevalence * (1.0 - prevalence)
    if reference == 0.0:
        return float("nan")
    return float(1.0 - np.mean((p - y) ** 2) / reference)


def calibration_in_the_large(y_true: Sequence, y_prob: Sequence) -> float:
    """Level 1 of Van Calster's hierarchy: observed minus predicted risk."""
    y, p = as_arrays(y_true, y_prob)
    return float(np.mean(y) - np.mean(p))


#: Largest logit move of one Newton step in the offset intercept solve.
_MAX_NEWTON_STEP = 5.0


def oe_ratio(y_true: Sequence, y_prob: Sequence) -> float:
    """Observed over expected events, ``sum(y) / sum(p)``; NaN if nothing is expected."""
    y, p = as_arrays(y_true, y_prob)
    expected = float(np.sum(p))
    if expected == 0.0:
        return float("nan")
    return float(np.sum(y) / expected)


def calibration_intercept_offset(y_true: Sequence, y_prob: Sequence) -> float:
    """Calibration intercept with the slope fixed at 1 (Van Calster 2019).

    The intercept-only logistic MLE with ``logit(p)`` as offset, solved by
    one-dimensional Newton. Unlike the joint intercept it is not tangled with
    the slope, so it reads as calibration-in-the-large on the logit scale.
    NaN when only one class is present: the MLE is then infinite.
    """
    y, p = as_arrays(y_true, y_prob)
    if len(np.unique(y)) < 2:
        return float("nan")
    fitted = _fit_offset_intercept(y, logit(p))
    return float("nan") if fitted is None else fitted


def _fit_offset_intercept(y: np.ndarray, offset: np.ndarray) -> Optional[float]:
    """The intercept ``a`` of ``y ~ a + offset``, or None when Newton does not settle.

    Shared by the reported calibration intercept and the intercept
    recalibration, so the two can never disagree.
    """
    a = 0.0
    for _ in range(200):
        fitted = expit(a + offset)
        score = float(np.sum(y - fitted))
        information = float(np.sum(fitted * (1.0 - fitted)))
        if information <= 0.0:
            return None
        # The score is monotone decreasing in a, so a capped Newton step never
        # overshoots into the saturated region where the information vanishes.
        step = float(np.clip(score / information, -_MAX_NEWTON_STEP, _MAX_NEWTON_STEP))
        a += step
        if abs(step) < 1e-12:
            return float(a)
    return None


def _bin_assignments(p: np.ndarray, bins: Optional[int]) -> np.ndarray:
    """Group index per record: by equal forecast value, or by equal-count bin."""
    if bins is None:
        _, inverse = np.unique(p, return_inverse=True)
        return inverse
    if bins < 1:
        raise ValueError(f"bins must be >= 1 or None, got {bins}")
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return np.zeros(p.shape, dtype=int)
    return np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)


def murphy_decomposition(
    y_true: Sequence, y_prob: Sequence, bins: Optional[int] = None
) -> Dict[str, float]:
    """Murphy (1973): ``Brier = reliability - resolution + uncertainty``.

    With ``bins=None`` forecasts are grouped by identical value and the
    identity holds exactly for the Brier score itself. With ``bins=k`` the
    identity holds exactly for the *binned* forecast, returned as
    ``brier_binned``; the gap to ``brier`` is the within-bin forecast variance
    that binning discards. Both are returned so a reader can see the size of
    that gap instead of taking the decomposition on trust.
    """
    y, p = as_arrays(y_true, y_prob)
    # Quantile edges on a tied or tiny score can leave a bin with no records;
    # compacting the labels keeps every group non-empty so the weights sum to 1.
    _, group = np.unique(_bin_assignments(p, bins), return_inverse=True)
    group = np.asarray(group).ravel()
    n = float(y.size)
    counts = np.bincount(group).astype(float)
    mean_forecast = np.bincount(group, weights=p) / counts
    mean_observed = np.bincount(group, weights=y) / counts
    weights = counts / n
    overall = float(np.mean(y))

    reliability = float(np.sum(weights * (mean_forecast - mean_observed) ** 2))
    resolution = float(np.sum(weights * (mean_observed - overall) ** 2))
    uncertainty = float(overall * (1.0 - overall))
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": float(np.mean((p - y) ** 2)),
        "brier_binned": float(np.mean((mean_forecast[group] - y) ** 2)),
        "n_bins": int(counts.size),
    }


def fit_logistic(
    design: np.ndarray, y: np.ndarray, max_iter: int = 50, tol: float = 1e-10
) -> Optional[np.ndarray]:
    """Unpenalized logistic MLE by Newton-Raphson (IRLS).

    Returns ``None`` when the fit does not converge — which for a calibration
    slope usually means the data are separable and the MLE is infinite. A
    diverged fit must not be reported as a number.
    """
    beta = np.zeros(design.shape[1], dtype=float)
    for _ in range(max_iter):
        mu = expit(design @ beta)
        weights = np.clip(mu * (1.0 - mu), 1e-12, None)
        hessian = design.T @ (weights[:, None] * design)
        gradient = design.T @ (y - mu)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step, *_ = np.linalg.lstsq(hessian, gradient, rcond=None)
        if not np.all(np.isfinite(step)):
            return None
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            return beta if np.max(np.abs(beta)) < 1e6 else None
    return None


def logistic_recalibration(y_true: Sequence, y_prob: Sequence) -> Tuple[float, float]:
    """Level 2 (weak calibration): ``(intercept, slope)`` of ``y ~ logit(p)``.

    Perfect weak calibration is intercept 0, slope 1. A slope below 1 is the
    signature of overfitting — predictions too extreme in both directions —
    and is invisible to both AUROC and calibration-in-the-large.
    """
    y, p = as_arrays(y_true, y_prob)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    x = logit(p)
    if np.allclose(x, x[0]):
        return float("nan"), float("nan")
    design = np.column_stack([np.ones_like(x), x])
    beta = fit_logistic(design, y)
    if beta is None:
        return float("nan"), float("nan")
    return float(beta[0]), float(beta[1])


def fit_recalibration(
    y_dev: Sequence, p_dev: Sequence, kind: str
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """A probability update fitted on development data, or None when it cannot be.

    ``kind="intercept"`` shifts the logit by the offset-intercept MLE (slope
    fixed at 1); ``kind="logistic"`` maps ``logit(p)`` to ``a + b * logit(p)``.
    Both are fitted by the solvers the reported calibration metrics use. None
    for a single-class development set, a constant score (no slope) or a fit
    that does not converge — the caller reports no update rows then.
    """
    if kind not in ("intercept", "logistic"):
        raise ValueError(f"unknown recalibration kind {kind!r}")
    y, p = as_arrays(y_dev, p_dev)
    if y.size == 0 or len(np.unique(y)) < 2:
        return None
    x = logit(p)
    if kind == "intercept":
        a = _fit_offset_intercept(y, x)
        if a is None:
            return None
        return lambda values: expit(a + logit(values))
    if np.allclose(x, x[0]):
        return None
    beta = fit_logistic(np.column_stack([np.ones_like(x), x]), y)
    if beta is None:
        return None
    a, b = float(beta[0]), float(beta[1])
    return lambda values: expit(a + b * logit(values))


def _rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell's restricted cubic spline basis (``len(knots) - 1`` columns)."""
    k = knots.size
    scale = (knots[-1] - knots[0]) ** 2
    columns = [x]
    for j in range(k - 2):
        term = (
            np.clip(x - knots[j], 0.0, None) ** 3
            - np.clip(x - knots[k - 2], 0.0, None) ** 3
            * (knots[k - 1] - knots[j])
            / (knots[k - 1] - knots[k - 2])
            + np.clip(x - knots[k - 1], 0.0, None) ** 3
            * (knots[k - 2] - knots[j])
            / (knots[k - 1] - knots[k - 2])
        )
        columns.append(term / scale)
    return np.column_stack(columns)


def _flexible_fit(y: np.ndarray, p: np.ndarray, n_knots: int):
    """Fit the moderate-calibration model; returns a predictor of ``logit(p)``.

    Falls back to the two-parameter (weak) model when the score has too few
    distinct values to place interior knots — with fewer distinct values than
    knots the basis is rank-deficient, and a rank-deficient spline is a worse
    answer than an honest straight line.
    """
    x = logit(p)
    quantiles = _KNOT_QUANTILES.get(n_knots)
    if quantiles is None:
        raise ValueError(f"n_knots must be one of {sorted(_KNOT_QUANTILES)}, got {n_knots}")
    knots = np.unique(np.quantile(x, quantiles))

    def build(values: np.ndarray) -> np.ndarray:
        if knots.size < n_knots:
            return np.column_stack([np.ones_like(values), values])
        return np.column_stack([np.ones_like(values), _rcs_basis(values, knots)])

    beta = fit_logistic(build(x), y)
    if beta is None:
        beta = fit_logistic(np.column_stack([np.ones_like(x), x]), y)
        if beta is None:
            return None

        def linear(values: np.ndarray) -> np.ndarray:
            return expit(np.column_stack([np.ones_like(values), values]) @ beta)

        return linear

    def predict(values: np.ndarray) -> np.ndarray:
        return expit(build(values) @ beta)

    return predict


def flexible_calibration_curve(
    y_true: Sequence,
    y_prob: Sequence,
    n_knots: int = DEFAULT_KNOTS,
    grid_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Level 3 (moderate calibration): predicted risk vs smoothed observed risk.

    Returns ``(predicted, observed)`` over the observed score range. Empty
    arrays when the model cannot be fitted, so a caller can plot nothing
    rather than plot a straight line that looks like perfect calibration.
    """
    y, p = as_arrays(y_true, y_prob)
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 3:
        empty = np.array([], dtype=float)
        return empty, empty
    predict = _flexible_fit(y, p, n_knots)
    if predict is None:
        empty = np.array([], dtype=float)
        return empty, empty
    grid = np.linspace(float(np.min(p)), float(np.max(p)), grid_size)
    return grid, predict(logit(grid))


def eci(y_true: Sequence, y_prob: Sequence, n_knots: int = DEFAULT_KNOTS) -> float:
    """Estimated Calibration Index (Van Hoorde 2015), scaled x100.

    The mean squared distance between the flexible calibration curve and the
    diagonal, evaluated at the observed predictions. Zero is perfect moderate
    calibration; unlike the calibration slope it does not assume the
    miscalibration is linear on the logit scale.
    """
    y, p = as_arrays(y_true, y_prob)
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 3:
        return float("nan")
    predict = _flexible_fit(y, p, n_knots)
    if predict is None:
        return float("nan")
    return float(np.mean((predict(logit(p)) - p) ** 2) * 100.0)


def calibration_metrics(
    y_true: Sequence,
    y_prob: Sequence,
    murphy_bins: Optional[int] = DEFAULT_MURPHY_BINS,
    n_knots: int = DEFAULT_KNOTS,
) -> Dict[str, float]:
    """All of category 2 for one slice."""
    intercept, slope = logistic_recalibration(y_true, y_prob)
    decomposition = murphy_decomposition(y_true, y_prob, bins=murphy_bins)
    return {
        "citl": calibration_in_the_large(y_true, y_prob),
        "calibration_intercept": intercept,
        "calibration_intercept_offset": calibration_intercept_offset(y_true, y_prob),
        "oe_ratio": oe_ratio(y_true, y_prob),
        "calibration_slope": slope,
        "eci": eci(y_true, y_prob, n_knots=n_knots),
        "brier": decomposition["brier"],
        "brier_scaled": brier_scaled(y_true, y_prob),
        "brier_binned": decomposition["brier_binned"],
        "brier_reliability": decomposition["reliability"],
        "brier_resolution": decomposition["resolution"],
        "brier_uncertainty": decomposition["uncertainty"],
    }
