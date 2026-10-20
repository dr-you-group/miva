"""Curves on fixed grids (site results export), with bootstrap bands.

A curve drawn at the observed scores (the ROC's own thresholds, a calibration
smoother evaluated between the min and max score) cannot be pooled: two sites
give two different x axes, and the extreme points of either are single
records. Every curve here is evaluated on a grid fixed before the data are
seen, so a site's curve is a table of values at known x, and the tables of two
sites line up row by row.

Like the rest of the package these are pure functions of ``(y, p)`` and know
nothing about an axis. The grid itself is a study setting
(``mival.criteria.net_benefit_grid``); the calibration grid and the risk
histogram edges are fixed here because they are display resolutions, not
clinical choices.
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._common import as_arrays, logit
from .bootstrap import DEFAULT_ALPHA, Resampler, event_strata
from .calibration import _flexible_fit

#: Predicted-risk grid of the calibration curve: 0.01, 0.02, ..., 0.99.
CALIBRATION_GRID = np.round(np.arange(1, 100) / 100.0, 10)

#: Edges of the risk-distribution histogram: 20 bins of width 0.05 on [0, 1].
RISK_EDGES = np.linspace(0.0, 1.0, 21)

#: Calibration is read only between these quantiles of the predicted risk.
#: Beyond them the smoother is extrapolating from a handful of records.
CALIBRATION_SUPPORT = (0.01, 0.99)

COUNT_KEYS = ("tp", "fp", "tn", "fn")
RATE_KEYS = (
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "net_benefit",
    "treat_all",
    "delta_vs_treat_all",
)


def _above(scores: np.ndarray, weights: Optional[np.ndarray], grid: np.ndarray) -> np.ndarray:
    """Count (or weight) of ``scores >= t`` for every ``t`` in ``grid``."""
    order = np.argsort(scores, kind="stable")
    ordered = scores[order]
    index = np.searchsorted(ordered, grid, side="left")
    if weights is None:
        return (ordered.size - index).astype(float)
    cumulative = np.concatenate([[0.0], np.cumsum(weights[order])])
    return cumulative[-1] - cumulative[index]


def grid_counts(y, p, grid, weights=None) -> Dict[str, np.ndarray]:
    """``tp``, ``fp``, ``tn``, ``fn`` at ``p >= t`` for every grid threshold.

    One sort per class and a binary search per threshold, so the cost is
    O(n log n) however fine the grid. ``weights`` (one per record) turns each
    count into a weighted sum; the survival curves pass inverse probability of
    censoring weights here.
    """
    y, p = as_arrays(y, p)
    thresholds = np.asarray(grid, dtype=float).ravel()
    positive = y == 1.0
    if weights is None:
        w_pos: Optional[np.ndarray] = None
        w_neg: Optional[np.ndarray] = None
        total_pos = float(np.sum(positive))
        total_neg = float(y.size - total_pos)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != y.shape:
            raise ValueError("weights must have one entry per record")
        w_pos, w_neg = w[positive], w[~positive]
        total_pos, total_neg = float(w_pos.sum()), float(w_neg.sum())
    tp = _above(p[positive], w_pos, thresholds)
    fp = _above(p[~positive], w_neg, thresholds)
    return {"tp": tp, "fp": fp, "tn": total_neg - fp, "fn": total_pos - tp}


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise ratio; NaN where the denominator is zero (an undefined rate)."""
    out = np.full(np.shape(numerator), np.nan)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def grid_metrics(counts: Mapping[str, np.ndarray], n, grid) -> Dict[str, np.ndarray]:
    """Rates and net benefit from :func:`grid_counts` (``n`` is the record total).

    Net benefit is ``TP/N - FP/N * t/(1-t)``; treat-all is the same with every
    record called positive. A rate whose denominator is zero is NaN.
    """
    t = np.asarray(grid, dtype=float).ravel()
    if np.any(t <= 0.0) or np.any(t >= 1.0):
        raise ValueError("grid thresholds must lie strictly in (0, 1)")
    tp, fp, tn, fn = (np.asarray(counts[key], dtype=float) for key in COUNT_KEYS)
    total = float(n)
    odds = t / (1.0 - t)
    if total > 0:
        net = tp / total - fp / total * odds
        prevalence = (tp + fn) / total
        treat_all = prevalence - (1.0 - prevalence) * odds
    else:
        net = np.full(t.shape, np.nan)
        treat_all = np.full(t.shape, np.nan)
    return {
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "net_benefit": net,
        "treat_all": treat_all,
        "delta_vs_treat_all": net - treat_all,
    }


def calibration_grid_curve(y, p, grid, n_knots) -> np.ndarray:
    """Smoothed observed risk at each grid point (the flexible calibration curve).

    The fit is the one ``eci`` and the calibration figure use. A grid point
    outside the p1-p99 range of ``p`` is NaN: the curve is not extended to
    the score's minimum and maximum, where it rests on a few records. All NaN
    when the curve cannot be fitted (one class, or fewer than three distinct
    scores), so a reader sees no curve rather than a straight line.
    """
    y, p = as_arrays(y, p)
    points = np.asarray(grid, dtype=float).ravel()
    observed = np.full(points.shape, np.nan)
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 3:
        return observed
    predict = _flexible_fit(y, p, n_knots)
    if predict is None:
        return observed
    lo, hi = np.quantile(p, CALIBRATION_SUPPORT)
    inside = (points >= lo) & (points <= hi)
    if np.any(inside):
        observed[inside] = predict(logit(points[inside]))
    return observed


def risk_histogram(y, p, edges) -> Dict[str, np.ndarray]:
    """Records per predicted-risk bin, split into ``events`` and ``nonevents``.

    Bins are ``[edge_k, edge_k+1)``, with the last one closed so that a score
    of exactly 1 is counted.
    """
    y, p = as_arrays(y, p)
    bins = np.asarray(edges, dtype=float).ravel()
    events, _ = np.histogram(p[y == 1.0], bins=bins)
    nonevents, _ = np.histogram(p[y == 0.0], bins=bins)
    return {"events": events.astype(float), "nonevents": nonevents.astype(float)}


def curve_bootstrap(
    fn: Callable[[np.ndarray, np.ndarray], Mapping[str, np.ndarray]],
    y,
    p,
    groups: Optional[Sequence],
    n_replicates: int,
    seed: int,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Pointwise percentile band ``(lo, hi)`` for every curve ``fn`` returns.

    Resampling is the metrics' own: whole persons (``groups``), stratified by
    whether a person carries an event, from ``default_rng(seed)``. Each grid
    point's interval uses that point's finite replicates only, and is NaN with
    fewer than two (the same rule as :func:`bootstrap.percentile_ci`).
    """
    return curve_bootstraps({"": fn}, y, p, groups, n_replicates, seed, alpha).get("", {})


def curve_bootstraps(
    fns: Mapping[str, Callable[[np.ndarray, np.ndarray], Mapping[str, np.ndarray]]],
    y,
    p,
    groups: Optional[Sequence],
    n_replicates: int,
    seed: int,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """:func:`curve_bootstrap` for several statistics on one set of draws.

    The replicate indices are drawn once and every statistic is evaluated on
    each, so ``curve_bootstraps({"a": f, "b": g}, ...)["a"]`` is exactly
    ``curve_bootstrap(f, ...)`` with the same seed, at the cost of one
    resampling loop instead of two.
    """
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if int(n_replicates) <= 0 or not fns:
        return {}
    resampler = Resampler(y.size, groups=groups, strata=event_strata(y, groups))
    rng = np.random.default_rng(seed)
    collected: Dict[str, Dict[str, list]] = {name: {} for name in fns}
    for _ in range(int(n_replicates)):
        indices = resampler.draw(rng)
        y_draw, p_draw = y[indices], p[indices]
        for statistic, fn in fns.items():
            into = collected[statistic]
            for name, values in fn(y_draw, p_draw).items():
                into.setdefault(name, []).append(np.asarray(values, dtype=float))
    return {
        statistic: {name: _percentile_band(rows, alpha) for name, rows in by_name.items()}
        for statistic, by_name in collected.items()
    }


def _percentile_band(replicates, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    stack = np.vstack(replicates)
    stack[~np.isfinite(stack)] = np.nan
    valid = np.sum(np.isfinite(stack), axis=0)
    with warnings.catch_warnings():
        # A grid point with no finite replicate is NaN by design.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lo = np.nanpercentile(stack, 100.0 * alpha / 2.0, axis=0)
        hi = np.nanpercentile(stack, 100.0 * (1.0 - alpha / 2.0), axis=0)
    lo[valid < 2] = np.nan
    hi[valid < 2] = np.nan
    return lo, hi
