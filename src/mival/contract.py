"""Model input contracts and the failure vocabulary of the contract gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Tuple

from mival.ops import SUPPORTED_FILTER_KINDS, SUPPORTED_SCALINGS, SUPPORTED_UNITS

REASON_CODES: FrozenSet[str] = frozenset(
    {
        "lead_unavailable",
        "upsample_required",
        "duration_short",
        "unit_missing",
        "rate_unsupported",
        # The record's file could not be read at all (spec §10): a loader
        # failure is one record's exclusion, never the whole run's abort.
        "read_failed",
    }
)


class CompileError(Exception):
    """Raised when a recipe cannot be compiled for a record.

    Every instance carries a reason_code from the fixed vocabulary so the
    exclusion ledger stays queryable.
    """

    def __init__(self, reason_code: str, detail: str) -> None:
        if reason_code not in REASON_CODES:
            raise ValueError(f"unknown reason_code: {reason_code}")
        super().__init__(f"[{reason_code}] {detail}")
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class InputContract:
    leads: Tuple[str, ...]
    sampling_rate_hz: float
    duration_s: float
    unit: str
    scaling: str
    layout: str
    dtype: str
    filters: Tuple[Dict[str, Any], ...] = field(default=())
    gain: float = 1.0
    #: Where a longer record is cut and a shorter one is padded. ``start`` is
    #: the behaviour every card had before the keys existed.
    crop_anchor: str = "start"
    pad_anchor: str = "start"

    @property
    def n_samples(self) -> int:
        return int(round(self.duration_s * self.sampling_rate_hz))

    @classmethod
    def from_dict(cls, body: Dict[str, Any]) -> "InputContract":
        unit = str(body["unit"])
        if unit not in SUPPORTED_UNITS:
            raise ValueError(
                f"unsupported unit {unit!r}: must be one of {sorted(SUPPORTED_UNITS)}"
            )
        scaling = str(body["scaling"])
        if scaling not in SUPPORTED_SCALINGS:
            raise ValueError(
                f"unsupported scaling {scaling!r}: "
                f"must be one of {sorted(SUPPORTED_SCALINGS)}"
            )
        filters = tuple(body.get("filters", ()))
        for spec in filters:
            kind = spec.get("kind")
            if kind not in SUPPORTED_FILTER_KINDS:
                raise ValueError(
                    f"unsupported filter kind {kind!r}: "
                    f"must be one of {sorted(SUPPORTED_FILTER_KINDS)}"
                )
            # The order is required for the same reason `unit` is: it changes
            # the samples the model is given, and nothing downstream can detect
            # that the wrong one was used. A default here would be this file
            # guessing at the model's own preprocessing.
            if spec.get("order") is None:
                raise ValueError(
                    f"filter {kind!r} declares no 'order'. The filter order changes the "
                    "waveform the model sees and cannot be recovered downstream, so it is "
                    "read from the ModelCard rather than defaulted."
                )
            try:
                order = int(spec["order"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"filter {kind!r} has a non-integer order {spec['order']!r}"
                ) from exc
            if order < 1:
                raise ValueError(f"filter {kind!r} has order {order}; it must be at least 1")
        return cls(
            leads=tuple(body["leads"]),
            sampling_rate_hz=float(body["sampling_rate_hz"]),
            duration_s=float(body["duration_s"]),
            unit=unit,
            scaling=scaling,
            layout=str(body["layout"]),
            dtype=str(body["dtype"]),
            filters=filters,
            gain=float(body.get("gain", 1.0)),
            crop_anchor=_anchor(body, "crop_anchor"),
            pad_anchor=_anchor(body, "pad_anchor"),
        )


ANCHORS = ("start", "center")


def _anchor(body: Dict[str, Any], key: str) -> str:
    value = str(body.get(key, "start"))
    if value not in ANCHORS:
        raise ValueError(f"unsupported {key} {value!r}: must be one of {list(ANCHORS)}")
    return value
