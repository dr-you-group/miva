"""Per-sample delineation mask -> PR, QRS and QT durations in milliseconds.

A beat is one QRS run. Its P wave is the last P run that starts after the
previous QRS ended and before this QRS starts; its T wave is the first T run
that starts after this QRS and before the next one. Each interval is the
median over the beats that have it. Beats touching either edge of the window
are cut and are skipped. Mirrors the per-beat median of SemiSegECG's
``compute_numerics``.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

INTERVAL_ORDER: Tuple[str, ...] = ("pr", "qrs", "qt")
#: Class indices in the mask. Cards whose classes differ map them first.
NONE, P, QRS, T = 0, 1, 2, 3
DEFAULT_MIN_RUN_MS = 10.0


def _runs(row: np.ndarray, value: int, min_len: int) -> List[Tuple[int, int]]:
    """``(start, end)`` half-open runs of ``value`` at least ``min_len`` long."""
    hit = np.concatenate(([False], row == value, [False]))
    edges = np.flatnonzero(np.diff(hit.astype(np.int8)))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_len]


def _one(row: np.ndarray, fs: float, min_len: int) -> np.ndarray:
    length = row.size
    p_runs = _runs(row, P, min_len)
    t_runs = _runs(row, T, min_len)
    qrs_runs = _runs(row, QRS, min_len)
    pr: List[float] = []
    qrs: List[float] = []
    qt: List[float] = []
    for i, (on, off) in enumerate(qrs_runs):
        if on == 0 or off == length:
            continue
        previous_end = qrs_runs[i - 1][1] if i > 0 else 0
        next_start = qrs_runs[i + 1][0] if i + 1 < len(qrs_runs) else length
        qrs.append(off - on)
        p_candidates = [s for s, _ in p_runs if previous_end <= s < on and s > 0]
        if p_candidates:
            pr.append(on - p_candidates[-1])
        t_candidates = [e for s, e in t_runs if off <= s < next_start and e < length]
        if t_candidates:
            qt.append(t_candidates[0] - on)
    scale = 1000.0 / fs
    return np.array([np.median(v) * scale if v else np.nan for v in (pr, qrs, qt)], dtype=np.float64)


def intervals_from_mask(mask: np.ndarray, fs: float, min_run_ms: float = DEFAULT_MIN_RUN_MS) -> np.ndarray:
    """``(N, T)`` class indices -> ``(N, 3)`` PR, QRS, QT in ms (NaN where absent)."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be (records, samples), got shape {mask.shape}")
    min_len = max(1, int(round(min_run_ms * fs / 1000.0)))
    return np.stack([_one(row, fs, min_len) for row in mask]) if len(mask) else np.empty((0, 3))


def remap_classes(mask: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Map a card's class order (names like ``none``, ``P``, ``QRS``, ``T``) onto this module's."""
    lookup = {"none": NONE, "other": NONE, "p": P, "qrs": QRS, "t": T}
    table = np.array([lookup[str(name).lower()] for name in classes], dtype=np.int64)
    return table[mask]
