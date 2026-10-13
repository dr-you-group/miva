"""Survival discrimination: Harrell's C and AUROC at a fixed horizon.

A higher score means higher risk. Inputs are follow-up time, a 0/1 event flag
and the score, one row per person.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

METRICS = ("c_index",)
METRIC_STEMS = ("auroc_horizon",)


def harrell_c(time: Sequence, event: Sequence, risk: Sequence) -> float:
    """Concordance over pairs where the earlier time is an event; risk ties count 1/2.

    An event and a censoring at the same time are comparable (the censored
    record outlived the event); two events at the same time are not. Exact, and O(levels x distinct risks)
    so it stays fast on a large cohort with day-resolution times.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64) > 0
    risk = np.asarray(risk, dtype=np.float64)
    if time.size == 0 or not event.any():
        return float("nan")
    values, rank = np.unique(risk, return_inverse=True)
    later = np.zeros(values.size, dtype=np.float64)  # risk-rank histogram of records with a later time
    n_later = 0
    concordant = tied = comparable = 0.0
    order = np.argsort(-time, kind="stable")
    levels, starts = np.unique(-time[order], return_index=True)
    bounds = list(starts) + [order.size]
    for k in range(len(levels)):
        group = order[bounds[k]:bounds[k + 1]]
        events = group[event[group]]
        censored = group[~event[group]]
        # A record censored at an event's own time outlived it (Harrell;
        # lifelines, survival::concordance); two events at one time are not
        # comparable. So this level's censored records join ``later`` before
        # its events are scored.
        np.add.at(later, rank[censored], 1.0)
        n_later += censored.size
        if events.size and n_later:
            below = np.cumsum(later) - later
            ranks = rank[events]
            concordant += below[ranks].sum()
            tied += later[ranks].sum()
            comparable += n_later * events.size
        np.add.at(later, rank[events], 1.0)
        n_later += events.size
    return float((concordant + 0.5 * tied) / comparable) if comparable else float("nan")


def rank_auroc(labels: Sequence, score: Sequence) -> float:
    """Mann-Whitney AUROC with average ranks for ties; any real-valued score.

    Same number as the probability-based AUROC, without the per-call overhead
    that dominates a 2,000-replicate bootstrap over a whole cohort.
    """
    from scipy.stats import rankdata

    labels = np.asarray(labels) > 0
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(np.asarray(score, dtype=np.float64))
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_at_horizon(time: Sequence, event: Sequence, risk: Sequence, horizon: float) -> float:
    """AUROC of the score for an event by ``horizon``; records censored earlier are left out."""
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64) > 0
    risk = np.asarray(risk, dtype=np.float64)
    case = event & (time <= horizon)
    control = ~case & (time >= horizon)
    keep = case | control
    if case[keep].all() or not case[keep].any():
        return float("nan")
    return rank_auroc(case[keep].astype(int), risk[keep])


def survival_metrics(time: Sequence, event: Sequence, risk: Sequence, horizons: Sequence[float]) -> Dict[str, float]:
    out = {"c_index": harrell_c(time, event, risk)}
    for horizon in horizons:
        out[f"auroc_horizon@{int(horizon) if float(horizon).is_integer() else horizon}"] = auroc_at_horizon(
            time, event, risk, horizon)
    return out
