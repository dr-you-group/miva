"""Clinical and statistical criteria, read from study.yaml.

Every threshold the site-results export uses (age bands, regression cuts and
their direction, the net-benefit grid, the small-cell floor) comes through
this module. The DEFAULT_* values are only what a study.yaml without the key
gets; the engine never types a clinical number anywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import numpy as np

DEFAULT_AGE_BANDS: Tuple[float, ...] = (50.0, 65.0, 80.0)
DEFAULT_MIN_CELL = 10
DEFAULT_NET_BENEFIT_GRID = {"start": 0.01, "stop": 0.99, "step": 0.01}

_OPS = ("<", "<=", ">", ">=")
_OP_NAMES = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge"}


def age_bands(spec: Mapping) -> Tuple[float, ...]:
    """Band lower edges from ``spec["age_bands"]``; strictly increasing."""
    if "age_bands" not in spec:
        return DEFAULT_AGE_BANDS
    bounds = tuple(float(b) for b in (spec["age_bands"] or ()))
    if not bounds:
        raise ValueError("age_bands must not be empty")
    if any(b2 <= b1 for b1, b2 in zip(bounds, bounds[1:])):
        raise ValueError("age_bands must be strictly increasing: %r" % (bounds,))
    return bounds


def age_band_labels(bounds) -> Tuple[str, ...]:
    """(50, 65, 80) -> ('<50', '50-64', '65-79', '>=80')."""
    b = [float(x) for x in bounds]
    labels = ["<%g" % b[0]]
    for lo, hi in zip(b, b[1:]):
        labels.append("%g-%g" % (lo, hi - 1))
    labels.append(">=%g" % b[-1])
    return tuple(labels)


def assign_age_band(ages: np.ndarray, bounds):
    """Categorical of band labels (every level kept); NaN age stays NaN."""
    import pandas as pd

    labels = list(age_band_labels(bounds))
    edges = [-np.inf] + [float(x) for x in bounds] + [np.inf]
    return pd.cut(np.asarray(ages, dtype=float), edges, right=False, labels=labels)


@dataclass(frozen=True)
class Cut:
    """A regression cut: an event is ``y <op> value``.

    ``legacy`` marks a cut written as a bare number: it keeps the ``<=`` it
    always meant and the metric name results were published under.
    """

    value: float
    op: str
    legacy: bool = False

    def events(self, y) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.op == "<":
            return y < self.value
        if self.op == "<=":
            return y <= self.value
        if self.op == ">":
            return y > self.value
        return y >= self.value

    @property
    def name(self) -> str:
        return "%s%g" % (_OP_NAMES[self.op], self.value)

    @property
    def higher_is_event(self) -> bool:
        return self.op in (">", ">=")


def parse_cuts(value: Any, where: str) -> Tuple[Cut, ...]:
    """Numbers become legacy ``<=`` cuts; ``{value, op}`` items carry their own ``op``.

    ``value`` is one list of cuts (or one item). A per-label_def mapping such
    as ``{pr: [200], qrs: [120]}`` is the caller's to split: this function
    cannot tell its keys from an item's.
    """
    if isinstance(value, Mapping):
        if not _is_cut_item(value):
            raise ValueError(
                "%s: %r is a per-label_def mapping; split it and pass each label's list"
                % (where, value)
            )
        value = [value]
    elif isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        value = [value]
    cuts = []
    for item in value:
        if isinstance(item, Mapping):
            op = item.get("op", "<=")
            if op not in _OPS or "value" not in item:
                raise ValueError("%s: bad cut %r (need value and op in %s)" % (where, item, _OPS))
            cuts.append(Cut(_number(item["value"], where), op))
        else:
            cuts.append(Cut(_number(item, where), "<=", legacy=True))
    if not cuts:
        raise ValueError("%s: cuts must not be empty" % where)
    return tuple(cuts)


def _is_cut_item(mapping: Mapping) -> bool:
    """``{value: <number>, op: ...}``, as opposed to ``{label_def: [cuts]}``."""
    if not set(mapping) <= {"value", "op"} or "value" not in mapping:
        return False
    item = mapping["value"]
    return not isinstance(item, (str, bytes, Mapping)) and not hasattr(item, "__iter__")


def _number(item: Any, where: str) -> float:
    try:
        return float(item)
    except (TypeError, ValueError):
        raise ValueError("%s: cut %r is not a number" % (where, item)) from None


def net_benefit_grid(spec: Mapping) -> np.ndarray:
    """Threshold-probability grid for decision curves; all values in (0, 1)."""
    g = dict(DEFAULT_NET_BENEFIT_GRID)
    g.update(spec.get("net_benefit_grid") or {})
    start, stop, step = float(g["start"]), float(g["stop"]), float(g["step"])
    if step <= 0:
        raise ValueError("net_benefit_grid.step must be positive")
    grid = np.round(np.arange(start, stop + step / 2, step), 10)
    if grid.size == 0 or grid.min() <= 0 or grid.max() >= 1:
        raise ValueError("net_benefit_grid values must lie inside (0, 1)")
    return grid


def min_cell(spec: Mapping) -> int:
    n = int(spec.get("min_cell", DEFAULT_MIN_CELL))
    if n < 1:
        raise ValueError("min_cell must be >= 1")
    return n
