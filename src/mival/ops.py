"""Preprocessing primitives.

Ops are deterministic and side-effect free. The canonical application order is
fixed by the compiler, not by callers: scale_unit -> filters -> resample ->
lead selection -> crop/pad -> normalize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sps

from mival.signal import DERIVABLE_LEADS, Signal


# resample_poly builds a filter proportional to max(up, down); an exact but
# enormous ratio is a data-quality signal, not something to silently compute.
_MAX_POLYPHASE_FACTOR = 10000


def _exact_rate(hz: float) -> Optional[Fraction]:
    """Return hz as an exact Fraction, or None if it cannot be represented."""
    frac = Fraction(hz).limit_denominator(10 ** 6)
    return frac if float(frac) == float(hz) else None


def resample_factors(source_hz: float, target_hz: float) -> Tuple[int, int]:
    """Exact polyphase (up, down) factors for source_hz -> target_hz."""
    target = _exact_rate(target_hz)
    source = _exact_rate(source_hz)
    if target is None or source is None:
        raise ValueError(
            f"cannot resample {source_hz} Hz to {target_hz} Hz: "
            "one of the rates has no exact rational representation"
        )
    ratio = target / source
    up, down = ratio.numerator, ratio.denominator
    if max(up, down) > _MAX_POLYPHASE_FACTOR:
        raise ValueError(
            f"cannot resample {source_hz} Hz to {target_hz} Hz: "
            f"the exact ratio {up}/{down} exceeds the polyphase factor limit "
            f"{_MAX_POLYPHASE_FACTOR}"
        )
    return up, down


def resampled_length(n_samples: int, source_hz: float, target_hz: float) -> int:
    """Sample count resample_poly produces, which is ceil(n * up / down)."""
    if source_hz == target_hz:
        return n_samples
    up, down = resample_factors(source_hz, target_hz)
    return -(-n_samples * up // down)


class Op:
    """Base class. Subclasses set `name` and implement `apply`."""

    name: str = "op"

    @property
    def params(self) -> Dict[str, Any]:
        raise NotImplementedError

    def apply(self, sig: Signal) -> Signal:
        raise NotImplementedError


@dataclass(frozen=True)
class Resample(Op):
    target_hz: float
    name: str = field(default="resample", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"target_hz": self.target_hz}

    def apply(self, sig: Signal) -> Signal:
        if sig.sampling_rate_hz == self.target_hz:
            return sig
        up, down = resample_factors(sig.sampling_rate_hz, self.target_hz)
        resampled = sps.resample_poly(sig.data, up, down, axis=1)
        return Signal(
            data=np.ascontiguousarray(resampled, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=self.target_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class Crop(Op):
    n_samples: int
    anchor: str = "start"
    name: str = field(default="crop", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"n_samples": self.n_samples, "anchor": self.anchor}

    def apply(self, sig: Signal) -> Signal:
        if sig.n_samples < self.n_samples:
            raise ValueError(
                f"cannot crop to {self.n_samples}: source is shorter "
                f"({sig.n_samples} samples)"
            )
        if self.anchor == "center":
            start = (sig.n_samples - self.n_samples) // 2
        elif self.anchor == "start":
            start = 0
        else:
            raise ValueError(f"unknown crop anchor: {self.anchor}")
        return Signal(
            data=np.ascontiguousarray(
                sig.data[:, start : start + self.n_samples], dtype=np.float32
            ),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


SUPPORTED_PAD_MODES = frozenset({"zero"})


@dataclass(frozen=True)
class Pad(Op):
    n_samples: int
    mode: str = "zero"
    anchor: str = "start"
    name: str = field(default="pad", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"n_samples": self.n_samples, "mode": self.mode, "anchor": self.anchor}

    def apply(self, sig: Signal) -> Signal:
        if sig.n_samples > self.n_samples:
            raise ValueError(
                f"cannot pad to {self.n_samples}: source is longer "
                f"({sig.n_samples} samples)"
            )
        if self.mode not in SUPPORTED_PAD_MODES:
            raise ValueError(f"unknown pad mode: {self.mode}")
        deficit = self.n_samples - sig.n_samples
        if self.anchor == "start":
            before, after = 0, deficit
        elif self.anchor == "center":
            before, after = deficit // 2, deficit - deficit // 2
        else:
            before, after = deficit, 0
        padded = np.pad(sig.data, ((0, 0), (before, after)), mode="constant")
        return Signal(
            data=np.ascontiguousarray(padded, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


_UNIT_TO_MV = {"mV": 1.0, "uV": 1e-3, "µV": 1e-3}
SUPPORTED_UNITS = frozenset(_UNIT_TO_MV)


@dataclass(frozen=True)
class ScaleUnit(Op):
    source_unit: str
    target_unit: str
    name: str = field(default="scale_unit", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"source_unit": self.source_unit, "target_unit": self.target_unit}

    def apply(self, sig: Signal) -> Signal:
        for unit in (self.source_unit, self.target_unit):
            if unit not in _UNIT_TO_MV:
                raise ValueError(f"unsupported unit: {unit}")
        factor = _UNIT_TO_MV[self.source_unit] / _UNIT_TO_MV[self.target_unit]
        data = sig.data if factor == 1.0 else sig.data * np.float32(factor)
        return Signal(
            data=np.ascontiguousarray(data, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=self.target_unit,
        )


@dataclass(frozen=True)
class SelectLeads(Op):
    order: Tuple[str, ...]
    name: str = field(default="select_leads", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"order": list(self.order)}

    def apply(self, sig: Signal) -> Signal:
        index = {lead: i for i, lead in enumerate(sig.leads)}
        missing = [lead for lead in self.order if lead not in index]
        if missing:
            raise KeyError(f"leads not present in source: {', '.join(missing)}")
        rows = [sig.data[index[lead]] for lead in self.order]
        return Signal(
            data=np.ascontiguousarray(np.stack(rows), dtype=np.float32),
            leads=self.order,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class ReconstructLeads(Op):
    order: Tuple[str, ...]
    name: str = field(default="reconstruct_leads", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"order": list(self.order)}

    def apply(self, sig: Signal) -> Signal:
        index = {lead: i for i, lead in enumerate(sig.leads)}
        rows = []
        for lead in self.order:
            if lead in index:
                rows.append(sig.data[index[lead]])
                continue
            recipe = DERIVABLE_LEADS.get(lead)
            if recipe is None or any(src not in index for src in recipe):
                raise KeyError(f"lead cannot be derived from source: {lead}")
            acc = np.zeros(sig.n_samples, dtype=np.float32)
            for src, coeff in recipe.items():
                acc = acc + np.float32(coeff) * sig.data[index[src]]
            rows.append(acc)
        return Signal(
            data=np.ascontiguousarray(np.stack(rows), dtype=np.float32),
            leads=self.order,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class Gain(Op):
    """Multiply by a constant the model expects on top of its unit (a card-declared quirk)."""

    factor: float
    name: str = field(default="gain", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"factor": self.factor}

    def apply(self, sig: Signal) -> Signal:
        return Signal(data=np.ascontiguousarray(sig.data * np.float32(self.factor), dtype=np.float32),
                      leads=sig.leads, sampling_rate_hz=sig.sampling_rate_hz, unit=sig.unit)


SUPPORTED_SCALINGS = frozenset({"none", "global_zscore", "per_lead_zscore"})


@dataclass(frozen=True)
class Normalize(Op):
    method: str
    name: str = field(default="normalize", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"method": self.method}

    def apply(self, sig: Signal) -> Signal:
        if self.method not in SUPPORTED_SCALINGS:
            raise ValueError(f"unknown normalization method: {self.method}")
        if self.method == "none":
            return sig
        if self.method == "global_zscore":
            mean = sig.data.mean()
            std = sig.data.std()
            std = std if std > 0 else 1.0
            data = (sig.data - mean) / std
        else:
            mean = sig.data.mean(axis=1, keepdims=True)
            std = sig.data.std(axis=1, keepdims=True)
            std = np.where(std > 0, std, 1.0)
            data = (sig.data - mean) / std
        return Signal(
            data=np.ascontiguousarray(data, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


SUPPORTED_FILTER_KINDS = frozenset({"highpass", "lowpass", "notch"})


@dataclass(frozen=True)
class BandFilter(Op):
    kind: str
    cutoff_hz: Any
    order: int = 4
    name: str = field(default="filter", init=False)

    @property
    def params(self) -> Dict[str, Any]:
        return {"kind": self.kind, "cutoff_hz": self.cutoff_hz, "order": self.order}

    def apply(self, sig: Signal) -> Signal:
        if self.kind not in SUPPORTED_FILTER_KINDS:
            raise ValueError(f"unknown filter kind: {self.kind}")
        nyquist = sig.sampling_rate_hz / 2.0
        if self.kind == "notch":
            quality = 30.0
            b, a = sps.iirnotch(self.cutoff_hz / nyquist, quality)
            filtered = sps.filtfilt(b, a, sig.data, axis=1)
        else:
            sos = sps.butter(
                self.order, self.cutoff_hz / nyquist, btype=self.kind, output="sos"
            )
            filtered = sps.sosfiltfilt(sos, sig.data, axis=1)
        return Signal(
            data=np.ascontiguousarray(filtered, dtype=np.float32),
            leads=sig.leads,
            sampling_rate_hz=sig.sampling_rate_hz,
            unit=sig.unit,
        )


@dataclass(frozen=True)
class OpChain:
    ops: Tuple[Op, ...]

    def apply(self, sig: Signal) -> Signal:
        for op in self.ops:
            sig = op.apply(sig)
        return sig

    def describe(self) -> List[Dict[str, Any]]:
        return [{"name": op.name, "params": op.params} for op in self.ops]
