"""Category 3 — clinical utility (spec §4.5; Vickers 2006, Steyerberg 2010).

Net benefit is a category of its own and not a corner of discrimination or
calibration, because it answers a different question. It is not rank-based, so
it is not discrimination; it is not an agreement between predicted and
observed risk, so it is not calibration. It is decision-theoretic: at this
threshold, is using the model better than not using it?

    NB(p_t) = TP/N - (FP/N) * p_t/(1 - p_t)

The weight ``p_t/(1-p_t)`` is the exchange rate the threshold implies: a
clinician willing to act at a 5% risk is stating that one missed STEMI is
worth 19 unnecessary work-ups. The two reference strategies are treat-all
(intervene on everyone) and treat-none (net benefit identically zero).

For STEMI the clinically plausible band is low — roughly 1-10% (spec §4.5) —
because the cost of a miss is large. A model can have an excellent AUROC and
still sit below treat-all across that whole band, which is the case this
category exists to detect.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ._common import as_arrays

#: Spec §4.5: the clinically plausible band for STEMI is the low end.
DEFAULT_DECISION_THRESHOLDS: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)

#: Metric name stems; the reported names carry the threshold as a suffix,
#: e.g. ``net_benefit@0.05``. The threshold is a parameter of the metric, not
#: an axis of the study — it does not multiply the run_key space.
METRIC_STEMS: Tuple[str, ...] = (
    "net_benefit",
    "net_benefit_treat_all",
    "delta_net_benefit_vs_treat_all",
)


def _odds(threshold: float) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"decision threshold must lie strictly in (0, 1), got {threshold}")
    return threshold / (1.0 - threshold)


def net_benefit(y_true: Sequence, y_prob: Sequence, threshold: float) -> float:
    """Net benefit of treating everyone with ``prob >= threshold``."""
    y, p = as_arrays(y_true, y_prob)
    n = float(y.size)
    predicted = p >= threshold
    tp = float(np.sum(predicted & (y == 1.0)))
    fp = float(np.sum(predicted & (y == 0.0)))
    return float(tp / n - (fp / n) * _odds(threshold))


def net_benefit_treat_all(y_true: Sequence, y_prob: Sequence, threshold: float) -> float:
    """Net benefit of the treat-all strategy: prevalence minus weighted misfires."""
    y, _ = as_arrays(y_true, y_prob)
    prevalence = float(np.mean(y))
    return float(prevalence - (1.0 - prevalence) * _odds(threshold))


def net_benefit_treat_none(y_true: Sequence, y_prob: Sequence, threshold: float) -> float:
    """Zero by construction — kept as a function so figures read symmetrically."""
    as_arrays(y_true, y_prob)
    _odds(threshold)
    return 0.0


def decision_curve(
    y_true: Sequence, y_prob: Sequence, thresholds: Sequence[float]
) -> List[Dict[str, float]]:
    """One row per threshold: the model and both reference strategies."""
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        rows.append(
            {
                "threshold": float(threshold),
                "net_benefit": net_benefit(y_true, y_prob, threshold),
                "treat_all": net_benefit_treat_all(y_true, y_prob, threshold),
                "treat_none": 0.0,
            }
        )
    return rows


def utility_metrics(
    y_true: Sequence,
    y_prob: Sequence,
    thresholds: Sequence[float] = DEFAULT_DECISION_THRESHOLDS,
) -> Dict[str, float]:
    """All of category 3 for one slice, keyed ``<stem>@<threshold>``."""
    values: Dict[str, float] = {}
    for row in decision_curve(y_true, y_prob, thresholds):
        suffix = f"@{row['threshold']:g}"
        values["net_benefit" + suffix] = row["net_benefit"]
        values["net_benefit_treat_all" + suffix] = row["treat_all"]
        values["delta_net_benefit_vs_treat_all" + suffix] = row["net_benefit"] - row["treat_all"]
    return values
