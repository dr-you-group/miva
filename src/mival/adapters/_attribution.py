"""Attribution — the backend-free half (spec §4.4 ``attribute``, §4.6 overlay).

**Method.** Integrated Gradients (Sundararajan, Taly, Yan, ICML 2017). It is
chosen over a plain gradient or gradient×input because it is the only common
method with a checkable defining property: *completeness*, the requirement that
the attributions sum to ``F(x) − F(baseline)``. That property is what
:func:`completeness_gap` tests, and it is the reason an attribution map here can
be wrong in a way the test suite notices rather than merely looking plausible.

**The baseline is required and has no default.** This is the whole argument in
the ECG saliency literature and it is not a detail. Integrated Gradients
attributes relative to a baseline, so every number it produces is a statement
about a comparison the caller chose. For images the convention is a black
image, which is at least a plausible "absence of signal". The corresponding
choice for an ECG — all leads flat at 0 mV — is not the absence of a heartbeat.
It is asystole: a maximally abnormal recording that any STEMI model has strong
opinions about. Attributing "relative to asystole" answers a question nobody
asked, and it does so while producing a picture that looks exactly like a
correct saliency map.

So a study must say what its baseline means. ``"zeros"`` remains available for
the caller who has read the above and wants it anyway; anything else arrives as
data — typically the per-lead mean or median of the development split, which
makes the attribution read as "relative to a typical recording in this cohort".

**What this does not claim.** Attribution maps are not evidence that a model
uses the region they highlight. Adebayo et al. (NeurIPS 2018) showed that
several widely used saliency methods survive randomising the model's weights,
so a map that looks anatomically sensible is not by itself a finding. Spec §4.6
therefore places these overlays in the exploratory audit, beside the waveform,
as an aid to a human reviewer — never as a reported metric.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np

#: Enough for the midpoint rule to converge on the smooth score functions here;
#: raise it if :func:`completeness_gap` reports a large residual.
DEFAULT_STEPS = 64

#: The one named baseline. Everything else is supplied as an array, because a
#: name would hide which comparison the study actually made.
ZERO_BASELINE = "zeros"


class AttributionError(ValueError):
    """Raised when an attribution cannot be computed as asked."""


def resolve_baseline(baseline: Union[None, str, Any], batch: np.ndarray) -> np.ndarray:
    """The baseline as an array shaped like ``batch``.

    Accepts the whole batch, one record broadcast over the batch, or one value
    per lead — the last being the common case, since a per-lead mean of the
    development split is a defensible "typical recording".
    """
    if baseline is None:
        raise AttributionError(
            "attribute() needs an explicit baseline. Integrated Gradients attributes "
            "relative to one, so the map depends entirely on which comparison you chose, "
            "and for an ECG the conventional all-zero baseline is asystole rather than "
            "'no signal'. Pass a per-lead reference array (the development split's mean "
            f"is the usual choice), or {ZERO_BASELINE!r} if that is genuinely what you want."
        )
    array = np.asarray(batch, dtype=np.float64)
    if isinstance(baseline, str):
        if baseline == ZERO_BASELINE:
            return np.zeros_like(array)
        raise AttributionError(
            f"unknown baseline {baseline!r}; the only named baseline is {ZERO_BASELINE!r} "
            "and every other baseline is supplied as an array"
        )
    reference = np.asarray(baseline, dtype=np.float64)
    if reference.shape == array.shape:
        return reference
    if reference.shape == array.shape[1:]:
        return np.broadcast_to(reference, array.shape).copy()
    if reference.shape == (array.shape[1],):
        return np.broadcast_to(reference[:, None], array.shape).copy()
    raise AttributionError(
        f"baseline of shape {reference.shape} does not fit a batch of shape {array.shape}; "
        f"give one value per lead {(array.shape[1],)}, one record {array.shape[1:]}, or the "
        "whole batch"
    )


def integration_points(steps: int = DEFAULT_STEPS) -> np.ndarray:
    """Midpoints of ``steps`` equal intervals of ``[0, 1]``.

    The midpoint rule rather than the endpoints: it never evaluates the model at
    the baseline itself, where a saturated score has zero gradient and
    contributes nothing but numerical noise, and it converges faster than the
    left or right rule for the same number of forward passes.
    """
    if steps < 1:
        raise AttributionError(f"steps must be at least 1, got {steps}")
    return (np.arange(steps, dtype=np.float64) + 0.5) / steps


def completeness_gap(
    attribution: np.ndarray, score_input: np.ndarray, score_baseline: np.ndarray
) -> np.ndarray:
    """``sum(attribution) − (F(x) − F(baseline))``, per record.

    Integrated Gradients satisfies this exactly; a numerical implementation
    satisfies it up to the integration error. A gap that is not small means the
    step count is too low or the gradient path is broken — which is a defect the
    picture itself would never reveal.
    """
    totals = np.asarray(attribution, dtype=np.float64).reshape(len(attribution), -1).sum(axis=1)
    return totals - (
        np.asarray(score_input, dtype=np.float64) - np.asarray(score_baseline, dtype=np.float64)
    )
