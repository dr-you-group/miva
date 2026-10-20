"""Survival discrimination and, for an absolute risk, calibration at a horizon.

A higher score means higher risk. Inputs are follow-up time, a 0/1 event flag
and the score, one row per person.

Two families, by what the score is:

- Any score: Harrell's C over the whole follow-up (``c_index``), the
  complete-case AUROC at a horizon (``auroc_horizon_cc``: records censored
  before it are dropped), and the two censoring-weighted measures the core
  report leads with, Uno's C truncated at the horizon (``uno_c``) and the
  cumulative/dynamic AUROC (``auroc_ipcw``, Uno 2007).
- An absolute risk by the horizon only: the IPCW Brier score (Graf 1999), the
  index of prediction accuracy against the Kaplan-Meier marginal risk
  (``ipa``), and observed over expected at the horizon (``oe_ratio_horizon``).
  A risk score has no baseline hazard, so these are undefined for it.

Inverse probability of censoring weights use the Kaplan-Meier estimate of the
censoring distribution G (:func:`censoring_survival`). The data are one
cohort: G is estimated from the records being scored. Every weighted
function takes an optional fitted ``g``; :func:`survival_metrics` fits it once
per call (once per bootstrap replicate) and hands it to each, which gives the
same numbers as letting each refit it.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

METRICS = ("c_index",)
#: Every horizon metric; ``ABSOLUTE_RISK_STEMS`` need a risk by the horizon.
ABSOLUTE_RISK_STEMS = ("brier_ipcw", "ipa", "oe_ratio_horizon")
METRIC_STEMS = ("auroc_horizon_cc", "uno_c", "auroc_ipcw") + ABSOLUTE_RISK_STEMS


def horizon_metric(stem: str, horizon: float) -> str:
    """``stem@h`` with a whole-day horizon written without a decimal point."""
    value = float(horizon)
    return f"{stem}@{int(value) if value.is_integer() else value}"


def _arrays(time, event, risk=None):
    time = np.asarray(time, dtype=np.float64).ravel()
    event = np.asarray(event, dtype=np.float64).ravel() > 0
    if risk is None:
        return time, event
    return time, event, np.asarray(risk, dtype=np.float64).ravel()


def _km_steps(time: np.ndarray, event: np.ndarray, reverse: bool = False):
    """Distinct times, the estimate just after each, and the Greenwood sum.

    ``reverse`` estimates the censoring distribution: a censoring is the
    "event", and an event at a censoring's own time leaves the risk set first
    (the subject failed, so it could not be censored then).
    """
    times, inverse = np.unique(time, return_inverse=True)
    n_events = np.bincount(inverse, weights=event.astype(float), minlength=times.size)
    n_total = np.bincount(inverse, minlength=times.size).astype(float)
    at_risk = time.size - np.concatenate([[0.0], np.cumsum(n_total)[:-1]])
    if reverse:
        at_risk = at_risk - n_events
        n_events = n_total - n_events
    hazard = np.divide(n_events, at_risk, out=np.zeros(times.size), where=at_risk > 0)
    surv = np.cumprod(1.0 - hazard)
    denominator = at_risk * (at_risk - n_events)
    term = np.divide(
        n_events, denominator, out=np.full(times.size, np.inf), where=denominator > 0
    )
    term[n_events == 0] = 0.0
    return times, surv, np.cumsum(term)


def _step(times: np.ndarray, values: np.ndarray, at, left: bool, before: float) -> np.ndarray:
    """A right-continuous step function at ``at`` (``left``: its left limit)."""
    at = np.asarray(at, dtype=np.float64)
    index = np.searchsorted(times, at, side="left" if left else "right") - 1
    out = np.where(index >= 0, values[np.clip(index, 0, None)], before)
    return out.astype(np.float64)


def kaplan_meier(time: Sequence, event: Sequence, at) -> Tuple[np.ndarray, np.ndarray]:
    """``(S, se)`` at each ``at``: Kaplan-Meier survival, right-continuous, Greenwood SE.

    ``S(t)`` counts the events at ``t`` itself. Beyond the last time the last
    value holds. The SE is NaN where the estimate has reached zero.
    """
    time, event = _arrays(time, event)
    at = np.atleast_1d(np.asarray(at, dtype=np.float64))
    if time.size == 0:
        nan = np.full(at.shape, np.nan)
        return nan, nan.copy()
    times, surv, greenwood = _km_steps(time, event)
    s = _step(times, surv, at, left=False, before=1.0)
    g = _step(times, greenwood, at, left=False, before=0.0)
    with np.errstate(invalid="ignore"):
        se = s * np.sqrt(g)
    se[~np.isfinite(g) | (s <= 0.0)] = np.nan
    return s, se


def km_log_log_ci(s: np.ndarray, se: np.ndarray, z: float) -> Tuple[np.ndarray, np.ndarray]:
    """Pointwise log(-log S) interval from :func:`kaplan_meier`'s ``(S, se)``.

    Exactly 1 before the first event (no uncertainty yet), NaN where S is 0
    or its SE is undefined. The transform keeps the interval inside (0, 1).
    """
    s = np.asarray(s, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    lo = np.full(s.shape, np.nan)
    hi = np.full(s.shape, np.nan)
    one = s >= 1.0
    lo[one] = hi[one] = 1.0
    inner = (s > 0.0) & ~one & np.isfinite(se)
    log_s = np.log(s[inner])
    width = z * (se[inner] / s[inner]) / np.abs(log_s)
    lo[inner] = s[inner] ** np.exp(width)
    hi[inner] = s[inner] ** np.exp(-width)
    return lo, hi


def censoring_survival(time: Sequence, event: Sequence) -> Callable[..., np.ndarray]:
    """``G``: Kaplan-Meier estimate of staying uncensored, as ``G(t, left=False)``.

    ``left=True`` gives ``G(t-)``, the probability of being uncensored just
    before ``t`` — the weight an event at ``t`` needs, since a censoring at
    the event's own time cannot have hidden it.
    """
    time, event = _arrays(time, event)
    if time.size == 0:
        return lambda t, left=False: np.full(np.shape(np.atleast_1d(t)), np.nan)
    times, surv, _ = _km_steps(time, event, reverse=True)

    def g(t, left: bool = False) -> np.ndarray:
        return _step(times, surv, np.atleast_1d(t), left=left, before=1.0)

    return g


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


def uno_c(
    time: Sequence, event: Sequence, risk: Sequence, tau: float, g: Optional[Callable] = None
) -> float:
    """Uno's C truncated at ``tau``: Harrell's pairs, each weighted ``G(T_i-)^-2``.

    A pair counts when ``i`` is an event before ``tau`` and ``j`` outlived it
    (a later time, or censored at ``T_i``); risk ties count 1/2. The walk over
    time levels is :func:`harrell_c`'s, with the earlier event's weight
    carried into each sum. Without censoring every weight is 1 and this is
    Harrell's C over the pairs that start before ``tau``. ``g`` is
    :func:`censoring_survival` of these records, fitted here when not given.
    """
    time, event, risk = _arrays(time, event, risk)
    if time.size == 0 or not event.any():
        return float("nan")
    weight = np.zeros(time.size)
    scored = event & (time < tau)
    g = g or censoring_survival(time, event)
    weight[scored] = g(time[scored], left=True) ** -2.0
    values, rank = np.unique(risk, return_inverse=True)
    later = np.zeros(values.size, dtype=np.float64)
    n_later = 0
    concordant = tied = comparable = 0.0
    order = np.argsort(-time, kind="stable")
    levels, starts = np.unique(-time[order], return_index=True)
    bounds = list(starts) + [order.size]
    for k in range(len(levels)):
        group = order[bounds[k]:bounds[k + 1]]
        events = group[event[group]]
        censored = group[~event[group]]
        np.add.at(later, rank[censored], 1.0)
        n_later += censored.size
        scoring = events[scored[events]]
        if scoring.size and n_later:
            below = np.cumsum(later) - later
            ranks = rank[scoring]
            w = weight[scoring]
            concordant += float(np.dot(w, below[ranks]))
            tied += float(np.dot(w, later[ranks]))
            comparable += n_later * float(w.sum())
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


def horizon_weights(
    time: Sequence, event: Sequence, horizon: float, g: Optional[Callable] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """``(case, weight)`` for a binary event-by-``horizon`` question.

    A case (event at ``T <= h``) weighs ``1/G(T-)``; a record still event-free
    at ``h`` weighs ``1/G(h)``; one censored before ``h`` weighs 0, since its
    status at ``h`` is unknown and the others stand in for it. Every weight is
    1 without censoring.
    """
    time, event = _arrays(time, event)
    case = event & (time <= horizon)
    control = time > horizon
    weight = np.zeros(time.size)
    if time.size == 0:
        return case, weight
    g = g or censoring_survival(time, event)
    if case.any():
        weight[case] = 1.0 / g(time[case], left=True)
    g_h = float(g(horizon)[0])
    if control.any() and g_h > 0.0:
        weight[control] = 1.0 / g_h
    return case, weight


def auroc_ipcw(
    time: Sequence, event: Sequence, risk: Sequence, h: float, g: Optional[Callable] = None
) -> float:
    """Cumulative/dynamic AUROC at ``h`` (Uno 2007): IPCW cases against survivors.

    Cases are events by ``h`` weighted ``1/G(T-)``; controls are everyone
    still event-free after ``h``, all with the same weight ``1/G(h)``, which
    therefore cancels. A weighted Mann-Whitney over one sort of the controls;
    risk ties count 1/2.
    """
    time, event, risk = _arrays(time, event, risk)
    case, weight = horizon_weights(time, event, h, g)
    control = time > h
    if not case.any() or not control.any():
        return float("nan")
    controls = np.sort(risk[control])
    below = np.searchsorted(controls, risk[case], side="left")
    ties = np.searchsorted(controls, risk[case], side="right") - below
    w = weight[case]
    return float(np.dot(w, below + 0.5 * ties) / (w.sum() * controls.size))


def brier_ipcw(
    time: Sequence, event: Sequence, risk_h: Sequence, h: float, g: Optional[Callable] = None
) -> float:
    """IPCW Brier score of a risk by ``h`` (Graf 1999), averaged over every record.

    ``(1 - r)^2`` for a case, ``r^2`` for a survivor past ``h``, each with its
    :func:`horizon_weights` weight; a record censored before ``h`` adds 0 but
    stays in the denominator.
    """
    time, event, risk_h = _arrays(time, event, risk_h)
    if time.size == 0:
        return float("nan")
    case, weight = horizon_weights(time, event, h, g)
    loss = np.where(case, (1.0 - risk_h) ** 2, risk_h ** 2)
    return float(np.mean(weight * loss))


def km_risk(time: Sequence, event: Sequence, h: float) -> float:
    """``1 - S_KM(h)``: the cohort's observed risk by ``h``."""
    s, _ = kaplan_meier(time, event, [h])
    return float(1.0 - s[0])


def ipa(
    time: Sequence, event: Sequence, risk_h: Sequence, h: float, g: Optional[Callable] = None
) -> float:
    """Index of prediction accuracy: ``1 - Brier / Brier(KM marginal risk for everyone)``."""
    time, event, risk_h = _arrays(time, event, risk_h)
    if time.size == 0:
        return float("nan")
    g = g or censoring_survival(time, event)
    null = brier_ipcw(time, event, np.full(time.size, km_risk(time, event, h)), h, g)
    if not null > 0.0:
        return float("nan")
    return float(1.0 - brier_ipcw(time, event, risk_h, h, g) / null)


def oe_ratio_horizon(time: Sequence, event: Sequence, risk_h: Sequence, h: float) -> float:
    """Observed over expected by ``h``: ``(1 - S_KM(h)) / mean(risk_h)``."""
    time, event, risk_h = _arrays(time, event, risk_h)
    expected = float(np.mean(risk_h)) if risk_h.size else float("nan")
    if not expected > 0.0:
        return float("nan")
    return km_risk(time, event, h) / expected


def survival_metrics(
    time: Sequence,
    event: Sequence,
    risk: Sequence,
    horizons: Sequence[float],
    absolute: bool,
    risk_horizon: Optional[float] = None,
) -> Dict[str, float]:
    """Every survival metric of one slice.

    ``absolute`` says ``risk`` is a probability of the event by a horizon;
    only then are the Brier, IPA and O:E entries computed. ``risk_horizon``
    names that horizon: given, the entries are computed at it alone, since a
    risk by one horizon is not a risk by another. The censoring distribution
    is fitted once and shared by every weighted measure.
    """
    time, event, risk = _arrays(time, event, risk)
    g = censoring_survival(time, event)
    out = {"c_index": harrell_c(time, event, risk)}
    for horizon in horizons:
        out[horizon_metric("auroc_horizon_cc", horizon)] = auroc_at_horizon(time, event, risk, horizon)
        out[horizon_metric("uno_c", horizon)] = uno_c(time, event, risk, horizon, g)
        out[horizon_metric("auroc_ipcw", horizon)] = auroc_ipcw(time, event, risk, horizon, g)
        if absolute and (risk_horizon is None or float(horizon) == float(risk_horizon)):
            out[horizon_metric("brier_ipcw", horizon)] = brier_ipcw(time, event, risk, horizon, g)
            out[horizon_metric("ipa", horizon)] = ipa(time, event, risk, horizon, g)
            out[horizon_metric("oe_ratio_horizon", horizon)] = oe_ratio_horizon(
                time, event, risk, horizon)
    return out
