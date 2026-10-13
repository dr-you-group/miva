"""Shared array handling for the metric modules.

Every metric in this package is a pure function of ``(y_true, y_prob)``. That
is the whole reason evaluation can be "one function plus a groupby" (spec
§4.5): if a metric knew about a site, a subgroup or a model, adding an axis
would mean editing metric code. Validation therefore lives here, once, and no
module in ``mival.metrics`` ever sees an axis name.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# Probabilities of exactly 0 or 1 have infinite logits, which no recalibration
# model can fit. Clipping is preferred over dropping: a record the model was
# certain about is a record the calibration of which we most want to measure.
LOGIT_EPS = 1e-8


def as_arrays(y_true: Sequence, y_prob: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and coerce a label/probability pair.

    Raises rather than silently coercing: a NaN probability or a label of 2
    reaching a metric produces a number that looks like a result.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"y_true and y_prob have different lengths: {y.shape[0]} vs {p.shape[0]}")
    if y.size == 0:
        raise ValueError("y_true is empty; a metric over zero records has no value")
    if not np.all(np.isfinite(y)) or not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y_true must contain only 0 and 1")
    if not np.all(np.isfinite(p)):
        raise ValueError("y_prob contains NaN or infinity")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("y_prob must lie in [0, 1]")
    return y, p


def safe_divide(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or NaN when the denominator is zero.

    An undefined metric (PPV with no predicted positives, say) must be NaN and
    not 0.0: 0.0 would be averaged into a bootstrap distribution as if it were
    a measurement.
    """
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(clipped / (1.0 - clipped))


def expit(x: np.ndarray) -> np.ndarray:
    # Clipping the linear predictor keeps exp() from overflowing on separable
    # data; at |x| = 40 the probability is already within 1e-17 of the bound.
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -40.0, 40.0)))
