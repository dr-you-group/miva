"""run_key — the coordinate of a point in the result space (spec §3.4).

A run_key names one inference run. Stages 5 and 6 do nothing but ``groupby``
over these axes, so adding an axis must not require touching evaluation or
audit code. That property only holds if the axis list lives in exactly one
place, which is ``AXES`` below.

The canonical string form is used as a filename (``predictions/<run_key>.parquet``),
so it must round-trip losslessly and stay filesystem-safe on every platform.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Spec §3.4. Order is significant: it fixes the canonical string form.
AXES: Tuple[str, ...] = (
    "site",
    "model_id",
    "training_mode",
    "recipe_id",
    "perturbation_id",
    "label_def",
    "split",
    "fold",
)

# Axes that report-only slicing uses (spec §3.4). These deliberately do NOT
# appear in AXES: they re-cut existing predictions instead of forcing another
# inference pass, which is why their cost does not multiply.
REPORT_AXES: Tuple[str, ...] = ("subgroup", "outcome")

# Axis values are identifiers, not free text. Restricting them keeps the
# canonical form parseable and safe as a path component on case-insensitive
# and POSIX filesystems alike.
_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Neither separator appears in the value charset above, so a value can never
# be mistaken for a field boundary. Choosing separators outside the charset
# makes that structural rather than something a later validator must remember
# to check: an earlier '__' separator round-tripped wrongly on a model_id like
# "ecg__founder", which the charset permits.
_FIELD_SEP = "~"
_KV_SEP = "="

# `fold` is absent for runs that are not cross-validated. It is written as a
# sentinel rather than omitted so that every canonical string has the same
# field count and can be parsed without knowing which axes are optional.
_FOLD_NONE = "na"


@dataclass(frozen=True)
class RunKey:
    """One coordinate in the run_key space.

    All axes are required. ``fold`` is the only nullable one — it is ``None``
    for runs outside an internal CV loop.
    """

    site: str
    model_id: str
    training_mode: str
    recipe_id: str
    perturbation_id: str
    label_def: str
    split: str
    fold: Optional[int] = None

    def __post_init__(self) -> None:
        for axis in AXES:
            if axis == "fold":
                continue
            value = getattr(self, axis)
            if not isinstance(value, str):
                raise TypeError(f"run_key axis {axis!r} must be str, got {type(value).__name__}")
            if not _VALUE_RE.match(value):
                raise ValueError(
                    f"run_key axis {axis}={value!r} is not a valid identifier; "
                    "allowed characters are letters, digits, '.', '_' and '-', "
                    "and the value may not start with '.', '_' or '-'"
                )
        if self.fold is not None:
            if isinstance(self.fold, bool) or not isinstance(self.fold, int):
                raise TypeError(f"run_key axis fold must be int or None, got {type(self.fold).__name__}")
            if self.fold < 0:
                raise ValueError(f"run_key axis fold must be non-negative, got {self.fold}")

    def to_dict(self) -> Dict[str, object]:
        return {axis: getattr(self, axis) for axis in AXES}

    def to_string(self) -> str:
        """Canonical, lossless, filesystem-safe string form."""
        parts = []
        for axis in AXES:
            value = getattr(self, axis)
            if axis == "fold":
                value = _FOLD_NONE if value is None else str(value)
            parts.append(f"{axis}{_KV_SEP}{value}")
        return _FIELD_SEP.join(parts)

    @classmethod
    def from_string(cls, text: str) -> "RunKey":
        parts = text.split(_FIELD_SEP)
        if len(parts) != len(AXES):
            raise ValueError(
                f"run_key string has {len(parts)} fields, expected {len(AXES)}: {text!r}"
            )
        values: Dict[str, object] = {}
        for axis, part in zip(AXES, parts):
            name, sep, value = part.partition(_KV_SEP)
            if not sep:
                raise ValueError(f"run_key field {part!r} is missing '{_KV_SEP}': {text!r}")
            if name != axis:
                raise ValueError(
                    f"run_key field {len(values)} is {name!r}, expected {axis!r}: {text!r}"
                )
            values[axis] = value
        fold_text = values["fold"]
        if fold_text == _FOLD_NONE:
            values["fold"] = None
        else:
            try:
                values["fold"] = int(fold_text)  # type: ignore[arg-type]
            except ValueError as exc:
                raise ValueError(f"run_key fold {fold_text!r} is not an integer: {text!r}") from exc
        return cls(**values)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "RunKey":
        missing = [axis for axis in AXES if axis not in values]
        if missing:
            raise ValueError(f"run_key is missing axes: {', '.join(missing)}")
        extra = [key for key in values if key not in AXES]
        if extra:
            raise ValueError(f"run_key has unknown axes: {', '.join(sorted(extra))}")
        return cls(**values)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return self.to_string()
