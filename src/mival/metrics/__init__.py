"""Metric definitions for spec §4.5 — four categories, no axes.

``compute_metrics`` is the "one function that computes the four categories"
the spec calls for. Everything else in the evaluation stage is a groupby over
run_key and report axes, so no function in this package accepts, returns or
mentions a site, model, subgroup or perturbation. If an axis name ever appears
below, adding an axis stops being free.

Category 4 (interpretation) has its schema slot reserved here and no
implementation: attribution maps and lead/time-region agreement are a
qualitative reading against clinical criteria and are out of scope for this
plan. ``INTERPRETATION_METRICS`` is deliberately empty rather than absent, so
the category constant and the report schema already exist when it lands.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from . import calibration as _calibration
from . import discrimination as _discrimination
from . import regression as _regression
from . import survival as _survival
from . import utility as _utility
from .calibration import (
    DEFAULT_KNOTS,
    DEFAULT_MURPHY_BINS,
    brier,
    brier_scaled,
    calibration_in_the_large,
    calibration_metrics,
    eci,
    flexible_calibration_curve,
    logistic_recalibration,
    murphy_decomposition,
)
from .discrimination import (
    auprc,
    auroc,
    confusion_counts,
    discrimination_metrics,
    discrimination_slope,
    pr_points,
    roc_points,
)
from .regression import regression_metrics
from .utility import (
    DEFAULT_DECISION_THRESHOLDS,
    decision_curve,
    net_benefit,
    net_benefit_treat_all,
    utility_metrics,
)

DISCRIMINATION = "discrimination"
CALIBRATION = "calibration"
CLINICAL_UTILITY = "clinical_utility"
INTERPRETATION = "interpretation"
REGRESSION = "regression"
SURVIVAL = "survival"

#: Category 4's reserved, empty slot. See the module docstring.
INTERPRETATION_METRICS: Tuple[str, ...] = ()

#: Metric name (or, for parameterized metrics, name stem) -> category.
METRIC_CATEGORY: Dict[str, str] = {}
METRIC_CATEGORY.update({name: DISCRIMINATION for name in _discrimination.METRICS})
METRIC_CATEGORY.update({name: CALIBRATION for name in _calibration.METRICS})
METRIC_CATEGORY.update({name: CLINICAL_UTILITY for name in _utility.METRIC_STEMS})
METRIC_CATEGORY.update({name: INTERPRETATION for name in INTERPRETATION_METRICS})
METRIC_CATEGORY.update({name: REGRESSION for name in _regression.METRICS + _regression.METRIC_STEMS})
METRIC_CATEGORY.update({name: SURVIVAL for name in _survival.METRICS + _survival.METRIC_STEMS})

#: Separator between a metric name and its parameter, e.g. ``net_benefit@0.05``.
PARAMETER_SEP = "@"


def category_of(metric: str) -> str:
    """Which of the four categories a metric name belongs to.

    Parameterized names are looked up by their stem, so a new decision
    threshold never needs registering.
    """
    stem = metric.split(PARAMETER_SEP, 1)[0]
    try:
        return METRIC_CATEGORY[stem]
    except KeyError as exc:
        raise KeyError(
            f"unknown metric {metric!r}; known metrics are {sorted(METRIC_CATEGORY)}"
        ) from exc


def compute_metrics(
    y_true: Sequence,
    y_prob: Sequence,
    threshold: float,
    decision_thresholds: Sequence[float] = DEFAULT_DECISION_THRESHOLDS,
    murphy_bins: Optional[int] = DEFAULT_MURPHY_BINS,
    n_knots: int = DEFAULT_KNOTS,
) -> Dict[str, float]:
    """Every implemented metric of every category, for one slice of records.

    ``threshold`` is the model's operating point and is chosen upstream, never
    on the data being evaluated (spec §4.4). ``decision_thresholds`` are the
    clinical exchange rates of category 3 and are a different thing entirely.
    """
    values: Dict[str, float] = {}
    values.update(discrimination_metrics(y_true, y_prob, threshold))
    values.update(calibration_metrics(y_true, y_prob, murphy_bins=murphy_bins, n_knots=n_knots))
    values.update(utility_metrics(y_true, y_prob, decision_thresholds))
    return values


__all__ = [
    "CALIBRATION",
    "CLINICAL_UTILITY",
    "DEFAULT_DECISION_THRESHOLDS",
    "DEFAULT_KNOTS",
    "DEFAULT_MURPHY_BINS",
    "DISCRIMINATION",
    "INTERPRETATION",
    "INTERPRETATION_METRICS",
    "METRIC_CATEGORY",
    "PARAMETER_SEP",
    "REGRESSION",
    "auprc",
    "auroc",
    "brier",
    "brier_scaled",
    "calibration_in_the_large",
    "calibration_metrics",
    "category_of",
    "compute_metrics",
    "confusion_counts",
    "decision_curve",
    "discrimination_metrics",
    "discrimination_slope",
    "eci",
    "flexible_calibration_curve",
    "logistic_recalibration",
    "murphy_decomposition",
    "net_benefit",
    "net_benefit_treat_all",
    "pr_points",
    "regression_metrics",
    "roc_points",
    "utility_metrics",
]
