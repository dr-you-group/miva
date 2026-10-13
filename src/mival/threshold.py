"""Operating-threshold policies (spec §4.4).

Three policies are reported for every run — ``legacy``, ``refit_youden``,
``refit_sens95`` — and the primary one is ``refit_sens95``. STEMI is an
asymmetric-cost target: a missed occlusion is not exchangeable with a false
alarm, so Youden's J, which weighs sensitivity and specificity equally, is
clinically inappropriate as the headline operating point. It is still reported
because it is what most of the literature reports.

Two invariants are structural here rather than left to the caller:

* **Thresholds are never selected on test.** Every refit takes a
  :class:`SplitScores`, which carries the split its probabilities came from,
  and refuses anything but ``dev``. Passing the split as data rather than as a
  literal at the call site is what makes the check real: a stage that
  accidentally fed test probabilities in would trip it.
* **No threshold value is written in this file.** ``legacy`` reads
  ``x-mival.threshold.value`` off the ModelCard. A number typed into pipeline
  code is a number that belongs to one model, and that would falsify the claim
  that registering a model means adding one card.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Spec §2.4/§4.4: dev is the only split a threshold may be estimated on.
DEV_SPLIT = "dev"

#: Spec §4.4 ``refit_sens95``. The policy name encodes the floor; it is a
#: study-design constant, not a model property, so it lives here and not on a
#: ModelCard.
SENS95_TARGET_SENSITIVITY = 0.95

LEGACY = "legacy"
REFIT_YOUDEN = "refit_youden"
REFIT_SENS95 = "refit_sens95"


class ThresholdPolicyError(ValueError):
    """A threshold could not be produced, or was asked for on the wrong split."""


def policy_names() -> Tuple[str, ...]:
    """The policies this module implements.

    ``mival.stages.models.THRESHOLD_POLICIES`` is the pinned contract; this is
    the implementation side of it, and a test asserts the two agree. Importing
    the constant from the stage would make ``threshold`` depend on a stage,
    which is exactly the coupling spec §3.1 forbids.
    """
    return (LEGACY, REFIT_YOUDEN, REFIT_SENS95)


# --------------------------------------------------------------------------
# score containers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitScores:
    """Probabilities and binary labels, tagged with the split they came from.

    The tag is data, not documentation: :func:`fit_thresholds` reads it to
    decide whether a refit is permitted at all.
    """

    split: str
    probs: np.ndarray
    labels: np.ndarray

    @classmethod
    def of(cls, split: str, probs: Sequence[float], labels: Sequence[Any]) -> "SplitScores":
        p, y = _as_scores(probs, labels)
        return cls(split=str(split), probs=p, labels=y)

    @property
    def n(self) -> int:
        return int(self.probs.size)

    @property
    def n_events(self) -> int:
        return int(self.labels.sum())


@dataclass(frozen=True)
class RocCurve:
    """One point per distinct observed probability, ordered by descending threshold.

    A record is called positive when ``prob >= threshold``. Candidate
    thresholds are the observed probabilities themselves, so every consecutive
    pair of points differs in at least one true or false positive — which is
    what makes the maximisers below unique.
    """

    thresholds: np.ndarray
    sensitivity: np.ndarray
    specificity: np.ndarray

    @property
    def youden_j(self) -> np.ndarray:
        return self.sensitivity + self.specificity - 1.0


@dataclass(frozen=True)
class ThresholdFit:
    """One policy's threshold plus the dev operating point it lands on."""

    policy: str
    value: float
    fitted_on: str
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    youden_j: Optional[float] = None
    n: Optional[int] = None
    n_events: Optional[int] = None
    provenance: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy,
            "value": self.value,
            "fitted_on": self.fitted_on,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "youden_j": self.youden_j,
            "n": self.n,
            "n_events": self.n_events,
            "provenance": self.provenance,
        }


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def _as_scores(probs: Sequence[float], labels: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if p.size != y.size:
        raise ThresholdPolicyError(
            f"probabilities and labels differ in length: {p.size} vs {y.size}"
        )
    if p.size == 0:
        raise ThresholdPolicyError("no scores were supplied")
    if not np.all(np.isfinite(p)):
        raise ThresholdPolicyError("probabilities contain NaN or infinity")
    if p.min() < 0.0 or p.max() > 1.0:
        raise ThresholdPolicyError(
            f"probabilities must lie in [0, 1]; got [{p.min()}, {p.max()}]"
        )
    if not np.all(np.isfinite(y)) or not np.all((y == 0.0) | (y == 1.0)):
        raise ThresholdPolicyError("labels must be binary 0/1 with no missing values")
    return p, y.astype(np.int64)


def roc_curve(probs: Sequence[float], labels: Sequence[Any]) -> RocCurve:
    p, y = _as_scores(probs, labels)
    n_pos = int(y.sum())
    n_neg = int(y.size) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ThresholdPolicyError(
            f"a threshold cannot be estimated from {n_pos} positive and {n_neg} negative "
            "records; both classes must be present"
        )
    # Stable sort so that ties keep input order and the curve is reproducible
    # bit-for-bit across runs and platforms.
    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]
    # Index of the last element of each run of equal probabilities: taking a
    # threshold *inside* a tie group would split records with identical scores.
    last = np.concatenate([np.nonzero(np.diff(p_sorted))[0], [p_sorted.size - 1]])
    tp = np.cumsum(y_sorted)[last]
    fp = (last + 1) - tp
    return RocCurve(
        thresholds=p_sorted[last],
        sensitivity=tp / float(n_pos),
        specificity=1.0 - fp / float(n_neg),
    )


def operating_point(
    probs: Sequence[float], labels: Sequence[Any], value: float
) -> Tuple[Optional[float], Optional[float]]:
    """Sensitivity and specificity of a fixed threshold, ``None`` when undefined."""
    p, y = _as_scores(probs, labels)
    called = p >= float(value)
    n_pos = int(y.sum())
    n_neg = int(y.size) - n_pos
    sensitivity = float(np.sum(called & (y == 1)) / n_pos) if n_pos else None
    specificity = float(np.sum(~called & (y == 0)) / n_neg) if n_neg else None
    return sensitivity, specificity


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.concatenate([[0], np.nonzero(np.diff(ordered))[0] + 1])
    group = np.zeros(ordered.size, dtype=np.int64)
    group[starts[1:]] = 1
    np.cumsum(group, out=group)
    ends = np.concatenate([starts[1:], [ordered.size]])
    mean_rank = (starts + ends - 1) / 2.0 + 1.0
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = mean_rank[group]
    return ranks


def auroc(probs: Sequence[float], labels: Sequence[Any]) -> float:
    """Rank AUROC with tie correction.

    This exists so that hyperparameters can be selected inside dev; it is not
    the reporting path. Stage 5 owns reported metrics and their confidence
    intervals (spec §4.5), and stages do not import each other (spec §3.1).
    """
    p, y = _as_scores(probs, labels)
    n_pos = int(y.sum())
    n_neg = int(y.size) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ThresholdPolicyError(
            f"AUROC is undefined with {n_pos} positive and {n_neg} negative records"
        )
    ranks = _average_ranks(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------


def _require_dev(scores: SplitScores, policy: str) -> None:
    if scores.split != DEV_SPLIT:
        raise ThresholdPolicyError(
            f"policy {policy!r} refits the threshold and may only see the {DEV_SPLIT!r} "
            f"split, but it was given scores from {scores.split!r} (spec §4.4: thresholds "
            "are never selected on the test set)"
        )


def legacy_threshold(
    card_threshold: Mapping[str, Any],
    model_id: str,
    scores: Optional[SplitScores] = None,
) -> ThresholdFit:
    """The published operating point, read off the ModelCard.

    ``scores`` is optional and is used only to report where the published
    threshold lands on dev. It is never used to choose the value.
    """
    if "value" not in card_threshold or card_threshold.get("value") is None:
        raise ThresholdPolicyError(
            f"model {model_id!r} declares no x-mival.threshold.value, so the {LEGACY!r} "
            "policy has nothing to report; remove the policy or complete the ModelCard"
        )
    value = float(card_threshold["value"])
    sensitivity = specificity = youden = None
    n = n_events = None
    if scores is not None:
        sensitivity, specificity = operating_point(scores.probs, scores.labels, value)
        if sensitivity is not None and specificity is not None:
            youden = sensitivity + specificity - 1.0
        n, n_events = scores.n, scores.n_events
    return ThresholdFit(
        policy=LEGACY,
        value=value,
        fitted_on="model_card",
        sensitivity=sensitivity,
        specificity=specificity,
        youden_j=youden,
        n=n,
        n_events=n_events,
        provenance=card_threshold.get("provenance"),
    )


def refit_youden(scores: SplitScores) -> ThresholdFit:
    """Maximise sensitivity + specificity - 1 on dev.

    Ties in J are broken toward the *smallest* threshold, i.e. toward higher
    sensitivity. Some tie-break has to be fixed for the result to be
    reproducible, and the one that matches this study's cost asymmetry is the
    one that misses fewer STEMIs.
    """
    _require_dev(scores, REFIT_YOUDEN)
    curve = roc_curve(scores.probs, scores.labels)
    j = curve.youden_j
    winners = np.nonzero(j == j.max())[0]
    index = int(winners[-1])  # thresholds descend, so the last winner is the smallest
    return ThresholdFit(
        policy=REFIT_YOUDEN,
        value=float(curve.thresholds[index]),
        fitted_on=scores.split,
        sensitivity=float(curve.sensitivity[index]),
        specificity=float(curve.specificity[index]),
        youden_j=float(j[index]),
        n=scores.n,
        n_events=scores.n_events,
    )


def refit_sens95(
    scores: SplitScores, target_sensitivity: float = SENS95_TARGET_SENSITIVITY
) -> ThresholdFit:
    """Highest-specificity threshold on dev that still reaches the sensitivity floor.

    The sensitivity-95% "point" need not be unique, so the rule is stated as a
    constrained optimisation rather than as a search for an exact value:
    among the thresholds attaining ``sensitivity >= target``, take the one with
    the greatest specificity. Sensitivity is non-decreasing as the threshold
    falls, so the feasible thresholds form a suffix of the curve and the first
    of them is the largest — hence the most specific. Because every curve point
    is a distinct observed probability, consecutive points differ in at least
    one true or false positive, so that maximiser is unique and no arbitrary
    tie-break is left over. The floor is always attainable: the smallest
    observed probability calls every record positive, giving sensitivity 1.
    """
    _require_dev(scores, REFIT_SENS95)
    curve = roc_curve(scores.probs, scores.labels)
    feasible = np.nonzero(curve.sensitivity >= target_sensitivity)[0]
    if feasible.size == 0:  # pragma: no cover - unreachable, see docstring
        raise ThresholdPolicyError(
            f"no threshold reaches sensitivity {target_sensitivity}"
        )
    index = int(feasible[0])
    return ThresholdFit(
        policy=REFIT_SENS95,
        value=float(curve.thresholds[index]),
        fitted_on=scores.split,
        sensitivity=float(curve.sensitivity[index]),
        specificity=float(curve.specificity[index]),
        youden_j=float(curve.youden_j[index]),
        n=scores.n,
        n_events=scores.n_events,
    )


def fit_thresholds(
    policies: Iterable[str],
    scores: SplitScores,
    card_threshold: Mapping[str, Any],
    model_id: str,
) -> Dict[str, ThresholdFit]:
    """Every requested policy, keyed by name.

    A policy that cannot be produced for this model — most often ``legacy`` for
    a model whose card carries no published threshold — is omitted rather than
    faked, and the caller reports which ones came back.
    """
    fits: Dict[str, ThresholdFit] = {}
    for policy in policies:
        if policy == LEGACY:
            try:
                fits[policy] = legacy_threshold(card_threshold, model_id, scores)
            except ThresholdPolicyError:
                continue
        elif policy == REFIT_YOUDEN:
            fits[policy] = refit_youden(scores)
        elif policy == REFIT_SENS95:
            fits[policy] = refit_sens95(scores)
        else:
            raise ThresholdPolicyError(
                f"unknown threshold policy {policy!r}; implemented policies are "
                f"{list(policy_names())}"
            )
    return fits
