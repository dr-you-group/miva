"""Every output that is not a class probability, decoded in one place.

Adapters hand a numpy array of the model's raw output (after head selection)
and the card's ``output`` block; this returns one value per record (or one
vector per record for a segmentation mask). Shared by the torch and keras
adapters so a decoding rule exists once.

Types:
- ``regression``: ``value_index`` column.
- ``risk_score``: one higher-is-riskier column (``value_index``, default 0).
- ``survival_curve``: conditional probabilities of surviving each interval
  (ml4h), first ``intervals`` units; risk at ``horizon_days`` = 1 - S.
- ``mtlr``: multi-task logistic regression logits (pycox); risk at the horizon
  interpolated over equidistant cuts on [0, ``max_duration_days``].
- ``segmentation_mask``: per-sample class scores -> PR, QRS, QT in ms.
``transform`` (``mival.decode.transforms``) applies to the scalar types.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mival.decode.transforms import apply_transform

DECODED_TYPES = ("regression", "risk_score", "survival_curve", "mtlr", "segmentation_mask")


def risk_from_conditional_survival(p: np.ndarray, bin_days: float, horizon_days: float) -> np.ndarray:
    """1 - S(horizon) from per-interval conditional survival; a partial interval is geometric."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    whole = int(np.floor(horizon_days / bin_days))
    frac = horizon_days / bin_days - whole
    log_s = np.log(p[:, :whole]).sum(axis=1)
    if frac > 0 and whole < p.shape[1]:
        log_s = log_s + frac * np.log(p[:, whole])
    return 1.0 - np.exp(log_s)


def mtlr_survival(phi: np.ndarray) -> np.ndarray:
    """Survival at each cut, as pycox ``MTLR.predict_surv``: cumsum-reverse, pad, softmax."""
    phi = np.asarray(phi, dtype=np.float64)
    reverse = np.flip(np.cumsum(np.flip(phi, axis=1), axis=1), axis=1)
    padded = np.concatenate([reverse, np.zeros((phi.shape[0], 1))], axis=1)
    padded = padded - padded.max(axis=1, keepdims=True)
    pmf = np.exp(padded)
    pmf = pmf / pmf.sum(axis=1, keepdims=True)
    return 1.0 - np.cumsum(pmf, axis=1)[:, :-1]


def decode_outputs(output: Mapping[str, Any], raw: np.ndarray, fs: float) -> np.ndarray:
    kind = output.get("type")
    raw = np.asarray(raw, dtype=np.float64)
    if kind == "segmentation_mask":
        from mival.decode.intervals import intervals_from_mask, remap_classes

        mask = raw.argmax(axis=int(output.get("class_axis", 1)))
        if output.get("classes"):
            mask = remap_classes(mask, output["classes"])
        return intervals_from_mask(mask, float(output.get("mask_rate_hz", fs)))
    flat = raw.reshape(raw.shape[0], -1)
    if kind in ("regression", "risk_score"):
        values = flat[:, int(output.get("value_index", 0))]
    elif kind == "survival_curve":
        units = flat[:, : int(output["intervals"])]
        values = risk_from_conditional_survival(units, float(output["bin_days"]), float(output["horizon_days"]))
    elif kind == "mtlr":
        n_bins = int(output["n_bins"])
        surv = mtlr_survival(flat[:, :n_bins])
        cuts = np.linspace(0.0, float(output["max_duration_days"]), n_bins)
        values = 1.0 - np.array([np.interp(float(output["horizon_days"]), cuts, row) for row in surv])
    else:
        raise ValueError(f"unknown output.type {kind!r} for decoding; known: {list(DECODED_TYPES)}")
    return apply_transform(values, output.get("transform"))
