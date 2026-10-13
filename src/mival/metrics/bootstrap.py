"""Uncertainty and model comparison by resampling (spec §4.5).

Two decisions here are load-bearing.

**The resampling unit is the person, not the ECG.** A patient may contribute
several recordings, and recordings from one patient are correlated. Resampling
records treats those correlated observations as independent information, which
makes the resampled distribution too tight and the confidence interval too
narrow. ``tests/test_bootstrap.py`` fixes this property: on clustered data the
person-level interval must be wider than the record-level one.

**Model comparison is a paired bootstrap of the difference, not DeLong.**
DeLong's variance estimator assumes independent observations, which clustered
ECGs are not. The paired bootstrap makes no such assumption, and pairing —
resampling persons once and evaluating both models on the same resample —
removes the between-person variance that both models share, which is exactly
the variance that swamps a difference between two good models.

The generator is always an explicit ``numpy.random.default_rng(seed)``. The
global RNG is never touched: a result that depends on how many random numbers
some other part of the process drew is not reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

#: Percentile interval coverage. Spec §4.5 reports percentile CIs.
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class BootstrapCI:
    """A point estimate with its resampled interval."""

    value: float
    ci_lo: float
    ci_hi: float
    n_valid: int
    p_value: Optional[float] = None


def event_strata(y_true: Sequence, groups: Optional[Sequence] = None) -> np.ndarray:
    """Per-row stratum label for an outcome-stratified bootstrap.

    A unit's stratum is whether it contributes any event. Stratifying keeps
    the number of event-carrying persons constant across replicates, which
    stops a rare-outcome resample from containing no events at all and
    returning NaN for every rank-based metric.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    if groups is None:
        return (y == 1.0).astype(int)
    codes = _codes(groups)
    has_event = np.zeros(int(codes.max()) + 1 if codes.size else 0, dtype=bool)
    np.logical_or.at(has_event, codes, y == 1.0)
    return has_event[codes].astype(int)


def _codes(values: Sequence) -> np.ndarray:
    array = np.asarray(values)
    _, codes = np.unique(array, return_inverse=True)
    return codes.astype(np.int64).ravel()


class Resampler:
    """Draws bootstrap row indices, resampling whole clusters.

    Built once and reused across replicates: the grouping and stratification
    of a slice do not change between replicates, and rebuilding them 2,000
    times is the difference between a fast evaluation and a slow one.
    """

    def __init__(
        self,
        n_rows: int,
        groups: Optional[Sequence] = None,
        strata: Optional[Sequence] = None,
    ) -> None:
        if n_rows <= 0:
            raise ValueError("cannot bootstrap zero rows")
        self.n_rows = int(n_rows)
        unit_codes = np.arange(self.n_rows) if groups is None else _codes(groups)
        if unit_codes.size != self.n_rows:
            raise ValueError("groups must have one entry per row")
        order = np.argsort(unit_codes, kind="stable")
        counts = np.bincount(unit_codes)
        self._rows_by_unit: List[np.ndarray] = np.split(order, np.cumsum(counts)[:-1])
        self.n_units = len(self._rows_by_unit)

        if strata is None:
            self._unit_groups = [np.arange(self.n_units)]
        else:
            stratum_per_row = np.asarray(strata).ravel()
            if stratum_per_row.size != self.n_rows:
                raise ValueError("strata must have one entry per row")
            # A cluster must sit in exactly one stratum; taking the first row's
            # label and checking the rest catches a stratum defined at record
            # level by mistake, which would otherwise silently split a person.
            labels = []
            for rows in self._rows_by_unit:
                unit_labels = stratum_per_row[rows]
                if np.any(unit_labels != unit_labels[0]):
                    raise ValueError(
                        "a resampling unit spans more than one stratum; strata must be "
                        "constant within a unit"
                    )
                labels.append(unit_labels[0])
            label_codes = _codes(np.asarray(labels))
            self._unit_groups = [
                np.flatnonzero(label_codes == code) for code in np.unique(label_codes)
            ]

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """Row indices of one replicate."""
        chosen: List[np.ndarray] = []
        for units in self._unit_groups:
            picked = rng.choice(units, size=units.size, replace=True)
            chosen.extend(self._rows_by_unit[unit] for unit in picked)
        if not chosen:  # pragma: no cover - guarded by the n_rows check
            return np.array([], dtype=np.int64)
        return np.concatenate(chosen)


def percentile_ci(samples: Sequence[float], alpha: float = DEFAULT_ALPHA):
    """Percentile interval over the finite replicates.

    Non-finite replicates are dropped rather than propagated: a resample of a
    rare-event subgroup can contain one class only, for which AUROC is
    undefined, and one such replicate would otherwise erase the interval.
    """
    values = np.asarray(samples, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan"), float("nan"), int(finite.size)
    lo = float(np.percentile(finite, 100.0 * alpha / 2.0))
    hi = float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi, int(finite.size)


def bootstrap_distribution(
    statistic: Callable[[np.ndarray], Mapping[str, float]],
    n_rows: int,
    n_replicates: int,
    seed: int,
    groups: Optional[Sequence] = None,
    strata: Optional[Sequence] = None,
) -> Dict[str, np.ndarray]:
    """Resample and evaluate ``statistic`` on each replicate.

    ``statistic`` returns a mapping so that every metric of a slice is
    computed from the *same* replicate. Bootstrapping each metric separately
    would cost as many passes as there are metrics and would make the
    intervals mutually inconsistent.
    """
    resampler = Resampler(n_rows, groups=groups, strata=strata)
    rng = np.random.default_rng(seed)
    collected: Dict[str, List[float]] = {}
    for _ in range(int(n_replicates)):
        indices = resampler.draw(rng)
        values = statistic(indices)
        for name, value in values.items():
            collected.setdefault(name, []).append(float(value))
    return {name: np.asarray(values, dtype=float) for name, values in collected.items()}


def bootstrap_ci(
    statistic: Callable[[np.ndarray], Mapping[str, float]],
    n_rows: int,
    n_replicates: int,
    seed: int,
    groups: Optional[Sequence] = None,
    strata: Optional[Sequence] = None,
    alpha: float = DEFAULT_ALPHA,
    point: Optional[Mapping[str, float]] = None,
) -> Dict[str, BootstrapCI]:
    """Point estimate plus percentile CI for every metric ``statistic`` returns.

    The point estimate is the metric on the full data, not the mean of the
    replicates: the replicate mean carries the bootstrap's own bias.
    """
    full = dict(point) if point is not None else dict(statistic(np.arange(n_rows)))
    samples = bootstrap_distribution(
        statistic,
        n_rows,
        n_replicates=n_replicates,
        seed=seed,
        groups=groups,
        strata=strata,
    )
    results: Dict[str, BootstrapCI] = {}
    for name, value in full.items():
        lo, hi, n_valid = percentile_ci(samples.get(name, np.array([])), alpha=alpha)
        results[name] = BootstrapCI(value=float(value), ci_lo=lo, ci_hi=hi, n_valid=n_valid)
    return results


def paired_bootstrap_difference(
    statistic: Callable[[np.ndarray, np.ndarray], Mapping[str, float]],
    y_true: Sequence,
    prob_a: Sequence,
    prob_b: Sequence,
    n_replicates: int,
    seed: int,
    groups: Optional[Sequence] = None,
    strata: Optional[Sequence] = None,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, BootstrapCI]:
    """Difference ``a - b`` per metric, with a paired percentile CI.

    Both arms must be aligned on the same rows — the caller joins them on
    ``image_occurrence_id`` — because pairing is what removes the shared
    between-person variance. This replaces DeLong's test, whose independence
    assumption clustered ECGs violate (spec §4.5).

    The reported ``p_value`` is the two-sided bootstrap percentile p-value,
    ``2 * min(P(d* <= 0), P(d* >= 0))``. It is secondary to the interval: the
    spec pre-specifies two confirmatory comparisons and asks for intervals
    rather than a wall of corrected p-values.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    a = np.asarray(prob_a, dtype=float).ravel()
    b = np.asarray(prob_b, dtype=float).ravel()
    if not (y.shape == a.shape == b.shape):
        raise ValueError("paired comparison requires y_true, prob_a and prob_b of equal length")

    def difference(indices: np.ndarray) -> Dict[str, float]:
        left = statistic(y[indices], a[indices])
        right = statistic(y[indices], b[indices])
        names = set(left) | set(right)
        return {
            name: float(left.get(name, float("nan")) - right.get(name, float("nan")))
            for name in names
        }

    full = difference(np.arange(y.size))
    samples = bootstrap_distribution(
        difference,
        y.size,
        n_replicates=n_replicates,
        seed=seed,
        groups=groups,
        strata=strata,
    )
    results: Dict[str, BootstrapCI] = {}
    for name, value in full.items():
        draws = samples.get(name, np.array([]))
        lo, hi, n_valid = percentile_ci(draws, alpha=alpha)
        finite = np.asarray(draws, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size < 2:
            p_value: Optional[float] = None
        else:
            below = float(np.mean(finite <= 0.0))
            above = float(np.mean(finite >= 0.0))
            p_value = float(min(1.0, 2.0 * min(below, above)))
        results[name] = BootstrapCI(
            value=float(value), ci_lo=lo, ci_hi=hi, n_valid=n_valid, p_value=p_value
        )
    return results
