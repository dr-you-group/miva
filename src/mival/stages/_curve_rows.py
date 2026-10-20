"""``curves_long`` rows for one evaluated slice (site results export).

The evaluate stage decides *which* slices get curves (held-out split, enough
events and non-events); this module turns one slice's records into rows. The
arithmetic is ``mival.metrics.curves``; what is added here is the run_key and
report-axis labelling, the same split of labour as ``metrics_long``.

One row is one ``(curve, x, metric)`` point. Like ``metrics_long`` the table
is long, so a new curve or metric is new values in existing columns.

Bootstrap bands: every threshold-grid rate gets one in every slice. The
calibration curve gets one only where the caller asks (``calibration_band``):
the evaluate stage asks for subgroup ``all`` only, because refitting the
spline in each of 2,000 replicates dominates the stage's runtime and a
subgroup's calibration band is rarely read. Other slices carry the
calibration point curve with ``ci_lo``/``ci_hi`` NULL. Counts and the risk
distribution are point values everywhere. One resampling loop serves both
bands of a slice (:func:`mival.metrics.curves.curve_bootstraps`).

A survival slice follows the same rule: its censoring-weighted threshold grid
is bootstrapped wherever it is drawn, and its calibration-type curves
(Kaplan-Meier by risk group, calibration at the horizon) carry their
Greenwood intervals for the whole slice only.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mival.pipeline.runkey import AXES, REPORT_AXES, RunKey

#: Output artifact, relative to ``ctx.layout.artifacts_dir``.
CURVES_LONG = "curves_long.parquet"

#: run_key axes, the report-only axes, then one curve point.
CURVE_COLUMNS = AXES + (
    "subgroup",
    "outcome",
    "update",
    "curve",
    "x",
    "metric",
    "value",
    "ci_lo",
    "ci_hi",
    "n",
    "n_events",
)

#: ``curve`` values. ``threshold_grid``: x is the decision threshold (counts
#: and rates at ``p >= x``). ``calibration``: x is predicted risk, metric
#: ``observed``. ``risk_distribution``: x is a histogram bin's lower edge.
THRESHOLD_GRID = "threshold_grid"
CALIBRATION = "calibration"
RISK_DISTRIBUTION = "risk_distribution"
#: A regression arm's calibration: x is the mean predicted value of one of
#: ``REGRESSION_CALIBRATION_BINS`` equal-count bins of the prediction, metric
#: ``observed`` (mean truth, normal-approximation band) and ``n`` (the bin's
#: record count). A bin below ``min_events`` records is left out.
REGRESSION_CALIBRATION = "regression_calibration"
REGRESSION_CALIBRATION_BINS = 20
#: Survival arms, all at the study's first horizon ``h``.
#: ``survival_threshold_grid``: x is the decision threshold on the risk by
#: ``h``; censoring-weighted net benefit, treat-all, their difference, and the
#: unweighted ``n_predicted_positive``. Absolute-risk arms only.
#: ``km_by_risk_group``: x is days since the ECG (0, ``KM_STEP_DAYS``, ...,
#: and ``h``); metric ``survival@q<k>`` and ``at_risk@q<k>`` for the k-th
#: quarter of the slice's risk (q4 the highest). Every survival arm.
#: ``calibration_horizon``: x is the mean predicted risk of one of
#: ``SURVIVAL_CALIBRATION_BINS`` equal-count bins, metric ``observed`` (the
#: bin's Kaplan-Meier risk by ``h``) and ``n``. Absolute-risk arms only.
SURVIVAL_THRESHOLD_GRID = "survival_threshold_grid"
KM_BY_RISK_GROUP = "km_by_risk_group"
CALIBRATION_HORIZON = "calibration_horizon"
KM_STEP_DAYS = 15
KM_RISK_GROUPS = 4
SURVIVAL_CALIBRATION_BINS = 10


def curve_rows(
    key: RunKey,
    subgroup: str,
    outcome: str,
    update: str,
    curve: str,
    x: Sequence[float],
    values: Mapping[str, Sequence[float]],
    bands: Mapping[str, Tuple[Sequence[float], Sequence[float]]],
    n: int,
    n_events: Optional[int],
) -> List[Dict[str, Any]]:
    """One row per ``(metric, x)`` — the only ``curves_long`` row builder.

    ``bands`` maps a metric to its pointwise ``(lo, hi)``; a metric without
    one (a count) has no interval. A point whose value is undefined has no
    interval either, so a band never shows where the curve does not.
    ``n_events`` is None for a regression slice, which has no event.
    """
    base = key.to_dict()
    base.update(dict(zip(REPORT_AXES, (subgroup, outcome))))
    base.update(
        {
            "update": update,
            "curve": curve,
            "n": int(n),
            "n_events": None if n_events is None else int(n_events),
        }
    )
    rows: List[Dict[str, Any]] = []
    for metric, series in values.items():
        band = bands.get(metric)
        for k, at in enumerate(x):
            value = _finite(series[k])
            row = dict(base)
            row.update(
                {
                    "x": float(at),
                    "metric": metric,
                    "value": value,
                    "ci_lo": None if band is None or value is None else _finite(band[0][k]),
                    "ci_hi": None if band is None or value is None else _finite(band[1][k]),
                }
            )
            rows.append(row)
    return rows


def classification_curve_rows(
    key: RunKey,
    subgroup: str,
    outcome: str,
    update: str,
    y,
    p,
    groups: Optional[Sequence],
    grid,
    n_knots: int,
    n_replicates: int,
    seed: int,
    alpha: float,
    calibration_band: bool = True,
) -> List[Dict[str, Any]]:
    """Threshold grid, calibration and risk distribution of one probability slice.

    Counts (``tp`` ... ``fn``) and the risk histogram are point values; every
    rate and net benefit, and the calibration curve when ``calibration_band``,
    get a person-level bootstrap band from ``n_replicates`` replicates (none
    when it is 0).
    """
    import numpy as np

    from mival.metrics.curves import (
        CALIBRATION_GRID,
        RISK_EDGES,
        calibration_grid_curve,
        curve_bootstraps,
        grid_counts,
        grid_metrics,
        risk_histogram,
    )

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = int(y.size)
    n_events = int(np.sum(y == 1.0))

    def threshold_curves(y_values, p_values):
        return grid_metrics(grid_counts(y_values, p_values, grid), y_values.size, grid)

    def calibration_curve(y_values, p_values):
        return {"observed": calibration_grid_curve(y_values, p_values, CALIBRATION_GRID, n_knots)}

    counts = grid_counts(y, p, grid)
    threshold_values: Dict[str, Any] = dict(counts)
    threshold_values.update(threshold_curves(y, p))

    statistics = {THRESHOLD_GRID: threshold_curves}
    if calibration_band:
        statistics[CALIBRATION] = calibration_curve
    bands = curve_bootstraps(statistics, y, p, groups, n_replicates, seed, alpha)

    rows = curve_rows(
        key, subgroup, outcome, update, THRESHOLD_GRID, grid, threshold_values,
        bands.get(THRESHOLD_GRID, {}), n, n_events,
    )
    rows.extend(
        curve_rows(
            key, subgroup, outcome, update, CALIBRATION, CALIBRATION_GRID,
            calibration_curve(y, p), bands.get(CALIBRATION, {}), n, n_events,
        )
    )
    rows.extend(
        curve_rows(
            key, subgroup, outcome, update, RISK_DISTRIBUTION, RISK_EDGES[:-1],
            risk_histogram(y, p, RISK_EDGES), {}, n, n_events,
        )
    )
    return rows


def regression_curve_rows(
    key: RunKey,
    subgroup: str,
    outcome: str,
    update: str,
    y,
    p,
    min_n: int,
    alpha: float,
) -> List[Dict[str, Any]]:
    """The binned calibration curve of one regression slice.

    The records are sorted by prediction (stably, so ties keep their order)
    and cut into ``REGRESSION_CALIBRATION_BINS`` bins of equal count; equal
    predictions may therefore fall in neighbouring bins, which keeps every bin
    the same size where quantile edges would merge tied bins. The band is
    ``mean +/- z * sd / sqrt(n)`` at the study's ``alpha``.
    """
    from statistics import NormalDist

    import numpy as np

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    x: List[float] = []
    observed: List[float] = []
    counts: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    order = np.argsort(p, kind="mergesort")
    for members in np.array_split(order, REGRESSION_CALIBRATION_BINS):
        size = int(members.size)
        if size == 0 or size < min_n:
            continue
        mean = float(np.mean(y[members]))
        half = z * float(np.std(y[members], ddof=1)) / math.sqrt(size) if size > 1 else math.nan
        x.append(float(np.mean(p[members])))
        observed.append(mean)
        counts.append(float(size))
        lo.append(mean - half)
        hi.append(mean + half)
    if not x:
        return []
    return curve_rows(
        key, subgroup, outcome, update, REGRESSION_CALIBRATION, x,
        {"observed": observed, "n": counts}, {"observed": (lo, hi)}, int(y.size), None,
    )


def survival_curve_rows(
    key: RunKey,
    subgroup: str,
    outcome: str,
    update: str,
    time,
    event,
    risk,
    groups: Optional[Sequence],
    horizon: float,
    grid,
    absolute: bool,
    n_replicates: int,
    seed: int,
    alpha: float,
    km_band: bool = True,
) -> List[Dict[str, Any]]:
    """The survival curves of one slice at ``horizon`` (see the curve constants).

    ``absolute`` says ``risk`` is a probability of the event by ``horizon``;
    without it only the Kaplan-Meier curves by risk group are drawn, since a
    ranking score has no threshold or calibration on the risk scale.

    The threshold grid's band is a person-level bootstrap stratified on the
    event flag, refitting the censoring weights in every replicate, in every
    slice it is drawn for. The Kaplan-Meier and calibration intervals are the
    log(-log) Greenwood intervals of each estimate, given only where
    ``km_band`` (the whole slice; see the module docstring).
    """
    from statistics import NormalDist

    import numpy as np

    from mival.metrics.curves import curve_bootstraps, grid_counts, grid_metrics
    from mival.metrics.survival import horizon_weights, kaplan_meier, km_log_log_ci

    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    risk = np.asarray(risk, dtype=float)
    n = int(time.size)
    n_events = int(np.sum(event == 1.0))
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    grid = np.asarray(grid, dtype=float)

    def threshold_curves(t_values, e_values, r_values):
        case, weight = horizon_weights(t_values, e_values, horizon)
        rates = grid_metrics(grid_counts(case, r_values, grid, weights=weight), t_values.size, grid)
        return {name: rates[name] for name in ("net_benefit", "treat_all", "delta_vs_treat_all")}

    rows: List[Dict[str, Any]] = []
    if absolute:
        values: Dict[str, Any] = threshold_curves(time, event, risk)
        ordered = np.sort(risk)
        values["n_predicted_positive"] = (n - np.searchsorted(ordered, grid, side="left")).astype(float)

        def by_index(_event, index):
            # The resampler draws rows of ``event`` (its strata); the drawn
            # positions carry time and risk along with it.
            index = index.astype(int)
            return threshold_curves(time[index], event[index], risk[index])

        bands = curve_bootstraps(
            {SURVIVAL_THRESHOLD_GRID: by_index}, event, np.arange(n, dtype=float), groups,
            n_replicates, seed, alpha,
        ).get(SURVIVAL_THRESHOLD_GRID, {})
        rows.extend(
            curve_rows(key, subgroup, outcome, update, SURVIVAL_THRESHOLD_GRID, grid, values,
                       bands, n, n_events)
        )

    days = np.arange(0.0, float(horizon), float(KM_STEP_DAYS))
    days = np.append(days, float(horizon))
    km_values: Dict[str, Any] = {}
    km_bands: Dict[str, Tuple[Any, Any]] = {}
    order = np.argsort(risk, kind="mergesort")
    for k, members in enumerate(np.array_split(order, KM_RISK_GROUPS), start=1):
        if members.size == 0:
            continue
        s, se = kaplan_meier(time[members], event[members], days)
        km_values[f"survival@q{k}"] = s
        km_values[f"at_risk@q{k}"] = np.array(
            [float(np.sum(time[members] >= day)) for day in days]
        )
        if km_band:
            km_bands[f"survival@q{k}"] = km_log_log_ci(s, se, z)
    rows.extend(
        curve_rows(key, subgroup, outcome, update, KM_BY_RISK_GROUP, days, km_values,
                   km_bands, n, n_events)
    )

    if absolute:
        x: List[float] = []
        observed: List[float] = []
        counts: List[float] = []
        lo: List[float] = []
        hi: List[float] = []
        for members in np.array_split(order, SURVIVAL_CALIBRATION_BINS):
            if members.size == 0:
                continue
            s, se = kaplan_meier(time[members], event[members], [horizon])
            s_lo, s_hi = km_log_log_ci(s, se, z)
            x.append(float(np.mean(risk[members])))
            observed.append(float(1.0 - s[0]))
            counts.append(float(members.size))
            lo.append(float(1.0 - s_hi[0]))
            hi.append(float(1.0 - s_lo[0]))
        if x:
            rows.extend(
                curve_rows(
                    key, subgroup, outcome, update, CALIBRATION_HORIZON, x,
                    {"observed": observed, "n": counts},
                    {"observed": (lo, hi)} if km_band else {}, n, n_events,
                )
            )
    return rows


def _finite(value) -> Optional[float]:
    """NaN becomes NULL in the table: an undefined point is not a number."""
    number = float(value)
    return None if math.isnan(number) or math.isinf(number) else number
