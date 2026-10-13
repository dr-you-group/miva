"""Perturbation grid and on-the-fly application (spec §4.3).

This is a pure function library, not a stage. Perturbed tensors are never
written to disk: spec §4.3 puts the perturbation in the dataloader and pins
reproducibility to ``seed + perturbation_id`` instead. Stage 3 therefore only
publishes the grid; stage 4 calls :func:`apply` per batch.

**The rule that gives this module its purpose.** Every perturbation restores
the signal to input-contract form. 500 Hz is degraded to 125 Hz and then
resampled *back* to 500 Hz; a 2.5 s window is zero-padded back to 10 s; a
dropped lead is zeroed rather than removed. Without that restoration the
measured quantity would be a mixture of information loss and shape mismatch,
and only the first of those is the thing under study.

Restoration includes the contract's *normalization*, which is why
:func:`apply` takes ``scaling``. Applying a 2x gain error to an already
z-scored tensor and stopping there would report a robustness failure that
cannot happen in deployment: a real mis-calibrated ECG passes through the
model's own normalization, which removes the gain. Re-applying the contract
scaling makes ``amplitude_scale`` correctly a no-op for a ``global_zscore``
contract and correctly a real gain error for a ``scaling: none`` contract.

**Constants not fixed by the spec.** Spec §4.3 names the noise *types* but no
amplitudes and no filter bands. Those live in :class:`NoiseConfig`, are stated
as configuration rather than fact, and are documented at their definition.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal as sps

from mival.ops import Crop, Normalize, Pad, Resample, SUPPORTED_SCALINGS, resample_factors
from mival.signal import LEADS_12, Signal

# --------------------------------------------------------------------------
# The grid (spec §4.3, verbatim)
# --------------------------------------------------------------------------

#: Axis order. Fixed here so that a composite perturbation_id is canonical.
AXIS_ORDER: Tuple[str, ...] = (
    "resample",
    "lead_dropout",
    "duration",
    "amplitude_scale",
    "noise",
)

#: Spec §4.3, level for level. The **first** level of every axis is that
#: axis's baseline — the spec lists them baseline-first (500 Hz, no dropout,
#: full 10 s, unity gain, no noise), which is what makes "OFAT" definable
#: without a separate baseline declaration.
DEFAULT_AXES: Mapping[str, Tuple[Any, ...]] = MappingProxyType(
    {
        "resample": (500, 250, 125, 100),
        "lead_dropout": ("none", "drop_V3V4", "precordial_only", "limb_only"),
        "duration": (10, 5, 2.5),
        "amplitude_scale": (1.0, 0.5, 2.0),
        # No powerline level by default. Spec §4.3 writes `powerline_50hz`, but
        # the mains frequency is a fact about the site's building, not about
        # this study: MIMIC-IV comes from Boston and Korean hospitals run on
        # 60 Hz, so 50 Hz interference is an artifact neither site's equipment
        # can produce, and that cell would measure robustness to something that
        # cannot happen. A wrong default is worse than an absent level because
        # it still fills a cell and still gets reported, so the level is left
        # for the study to declare:
        #     perturbation: {axes: {noise: [none, baseline_wander,
        #                                   powerline_60hz, emg]}}
        "noise": ("none", "baseline_wander", "emg"),
    }
)

#: Human-readable prefix per axis, used to build perturbation_id.
AXIS_PREFIX: Mapping[str, str] = MappingProxyType(
    {
        "resample": "resample",
        "lead_dropout": "leaddrop",
        "duration": "duration",
        "amplitude_scale": "amplitude",
        "noise": "noise",
    }
)

#: Spec §4.3: OFAT is the default policy, full cartesian is opt-in.
MODES: Tuple[str, ...] = ("ofat", "cartesian")
DEFAULT_MODE = "ofat"

BASELINE_ID = "baseline"

# A composite id joins axis tokens. '__' is outside every level token below
# (levels use at most single underscores) and is checked for at grid build
# time, so a token can never be mistaken for a field boundary.
_ID_SEP = "__"
_AXIS_KV_SEP = "-"

# The run_key / path-component charset (mival.pipeline.runkey, mival.pipeline.layout).
# A perturbation_id is a run_key axis value and a path component, so it must
# satisfy this or stage 4 cannot name its prediction files.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Lead groups, derived from the standard order rather than re-listed.
LIMB_LEADS: Tuple[str, ...] = LEADS_12[:6]
PRECORDIAL_LEADS: Tuple[str, ...] = LEADS_12[6:]

#: Which leads each ``lead_dropout`` level zeroes. "precordial_only" keeps the
#: precordial leads, so it drops the limb leads, and vice versa.
LEAD_DROPOUT_SETS: Mapping[str, frozenset] = MappingProxyType(
    {
        "none": frozenset(),
        "drop_V3V4": frozenset({"V3", "V4"}),
        "precordial_only": frozenset(LIMB_LEADS),
        "limb_only": frozenset(PRECORDIAL_LEADS),
    }
)

# The mains frequency is read out of the level name rather than configured, so
# the id and the waveform can never disagree. Adding a 60 Hz level is then a
# spec edit ("powerline_60hz") with no code change.
_POWERLINE_RE = re.compile(r"^powerline_(\d+(?:\.\d+)?)hz$")


@dataclass(frozen=True)
class NoiseConfig:
    """Noise magnitudes and bands. **Not specified by spec §4.3.**

    Spec §4.3 fixes the noise *types* and nothing else, so every number here is
    a declared default rather than a value taken from the study design. Each is
    exposed so that a reviewer can change it without touching code.

    ``snr_db`` — noise is scaled to a signal-to-noise ratio against the
    signal's own RMS rather than to an absolute millivolt amplitude. A relative
    definition is the only one that stays meaningful across contracts, because
    a contract declaring ``global_zscore`` hands this module a unit-variance
    tensor while one declaring ``none`` hands it millivolts. 10 dB is a
    declared choice, not a measured threshold.

    The band edges are the one part with outside support: the AHA/ACCF/HRS 2007
    recommendations for the standardization and interpretation of the ECG
    (Part I) put the adult diagnostic bandwidth at 0.05–150 Hz. Baseline wander
    is therefore modelled at and just above the 0.05 Hz low-frequency corner,
    and muscle artifact inside the upper part of that band (EMG energy is
    broadband above roughly 20 Hz).
    """

    snr_db: float = 10.0
    baseline_wander_band_hz: Tuple[float, float] = (0.05, 0.5)
    baseline_wander_components: int = 3
    emg_band_hz: Tuple[float, float] = (20.0, 150.0)
    emg_order: int = 4


DEFAULT_NOISE = NoiseConfig()


@dataclass(frozen=True)
class Perturbation:
    """One point of the grid. Field names are the spec's axis names."""

    id: str
    resample: float
    lead_dropout: str
    duration: float
    amplitude_scale: float
    noise: str

    def levels(self) -> Dict[str, Any]:
        return {axis: getattr(self, axis) for axis in AXIS_ORDER}

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"perturbation_id": self.id}
        payload.update(self.levels())
        return payload


# --------------------------------------------------------------------------
# Grid construction
# --------------------------------------------------------------------------


def _level_token(value: Any) -> str:
    """Render a level so that the id stays readable and round-trips."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise TypeError(f"perturbation levels may not be booleans, got {value!r}")
    number = float(value)
    return str(int(number)) if number.is_integer() else repr(number)


def resolve_axes(axes: Optional[Mapping[str, Sequence[Any]]] = None) -> Dict[str, Tuple[Any, ...]]:
    """Validate a (possibly overridden) axis table and return it normalized.

    Levels may be overridden from ``study.yaml``; axis *names* may not. A new
    axis is a change to the result space of spec §3.4, which is a spec
    decision rather than a configuration one.
    """
    if axes is None:
        return {axis: tuple(levels) for axis, levels in DEFAULT_AXES.items()}
    unknown = sorted(set(axes) - set(AXIS_ORDER))
    if unknown:
        raise ValueError(
            f"unknown perturbation axes: {', '.join(unknown)}; "
            f"the axes of spec §4.3 are {', '.join(AXIS_ORDER)}"
        )
    resolved: Dict[str, Tuple[Any, ...]] = {}
    for axis in AXIS_ORDER:
        levels = tuple(axes.get(axis, DEFAULT_AXES[axis]))
        if not levels:
            raise ValueError(f"perturbation axis {axis!r} has no levels")
        for level in levels:
            token = _level_token(level)
            if _ID_SEP in token:
                raise ValueError(
                    f"perturbation level {level!r} on axis {axis!r} contains {_ID_SEP!r}, "
                    "which separates axes inside a composite perturbation_id"
                )
            if axis == "lead_dropout" and level not in LEAD_DROPOUT_SETS:
                raise ValueError(
                    f"unknown lead_dropout level {level!r}; known levels are "
                    f"{', '.join(sorted(LEAD_DROPOUT_SETS))}"
                )
            if axis == "noise":
                _noise_kind(level)  # raises on an unknown noise level
        resolved[axis] = levels
    return resolved


def _make(axes: Mapping[str, Tuple[Any, ...]], levels: Mapping[str, Any]) -> Perturbation:
    """Build a Perturbation, deriving its id from the off-baseline axes only.

    Naming relative to baseline is what makes ``resample-125`` mean the same
    thing in OFAT and in cartesian mode, and keeps composite ids short.
    """
    tokens = []
    for axis in AXIS_ORDER:
        level = levels[axis]
        if level == axes[axis][0]:
            continue
        tokens.append(f"{AXIS_PREFIX[axis]}{_AXIS_KV_SEP}{_level_token(level)}")
    identifier = _ID_SEP.join(tokens) if tokens else BASELINE_ID
    if not _ID_RE.match(identifier):
        raise ValueError(
            f"perturbation_id {identifier!r} is not a valid run_key value; allowed "
            "characters are letters, digits, '.', '_' and '-'"
        )
    return Perturbation(id=identifier, **{axis: levels[axis] for axis in AXIS_ORDER})


def baseline(axes: Optional[Mapping[str, Sequence[Any]]] = None) -> Perturbation:
    resolved = resolve_axes(axes)
    return _make(resolved, {axis: resolved[axis][0] for axis in AXIS_ORDER})


class GridContractError(ValueError):
    """The declared grid does not fit the input contract it will be applied to."""


#: Axis → (contract attribute, human name). Both of these axes *degrade*: a
#: level at or above the contract's own value removes nothing, which
#: ``_apply_resample`` and ``_apply_duration`` implement as identity.
_DEGRADING_AXES = (
    ("resample", "sampling_rate_hz", "sampling rate"),
    ("duration", "duration_s", "duration"),
)


def check_grid_against_contract(
    axes: Optional[Mapping[str, Sequence[Any]]],
    contract: Any,
    model_id: str = "",
) -> List[str]:
    """Check a grid against the contract it will degrade, returning warnings.

    Perturbation is applied to a tensor that has already been compiled to the
    input contract, so ``resample`` and ``duration`` only mean something below
    the contract's own rate and length. Nothing else in the pipeline connects
    the two, and the failures are both silent:

    * A **baseline level below** the contract contaminates every other cell.
      :func:`apply` short-circuits the baseline cell itself to the stored
      tensor, so that one is safe — but every *other* cell carries the baseline
      levels of the axes it is not varying. With a 10 s baseline level under a
      20 s contract, ``resample-125`` would be measured on a signal that was
      also cropped to half its length, and the degradation would be attributed
      to the resample axis. Refused.
    * A **later level at or above** the contract removes nothing, so it emits a
      cell bit-identical to the baseline under a different ``perturbation_id``.
      The results table would read the two as independent measurements.
      Refused.
    * A baseline level *above* the contract is harmless — it is the identity,
      which is what a baseline should be — but its name claims a value the
      model never receives. Warned.

    A study whose cards have different contracts cannot satisfy this with one
    grid, because ``perturbation.axes`` is study-wide. That is a real limit of
    spec §4.3's absolute levels and the error says so rather than pretending.
    """
    resolved = resolve_axes(axes)
    where = f"model {model_id!r}: " if model_id else ""
    warnings: List[str] = []
    for axis, attribute, label in _DEGRADING_AXES:
        limit = float(getattr(contract, attribute))
        levels = [float(level) for level in resolved[axis]]
        if levels[0] < limit:
            raise GridContractError(
                f"{where}perturbation axis {axis!r} has baseline level {resolved[axis][0]!r}, "
                f"below the input contract's {label} ({limit}). Every cell that varies a "
                f"different axis still carries this level, so all of them would be "
                f"silently {'cropped' if axis == 'duration' else 'downsampled'} and the loss "
                f"attributed to the axis they do vary. Declare perturbation.axes.{axis} with "
                f"{limit} first. (One grid is shared by every card, so cards whose contracts "
                "differ in rate or duration cannot both be served by it.)"
            )
        degenerate = [level for level in levels[1:] if level >= limit]
        if degenerate:
            raise GridContractError(
                f"{where}perturbation axis {axis!r} has level(s) {degenerate} at or above "
                f"the input contract's {label} ({limit}). Such a level removes nothing, so "
                "it would produce a cell bit-identical to the baseline under a different "
                "perturbation_id, which the results table would read as an independent "
                "measurement."
            )
        if levels[0] > limit:
            warnings.append(
                f"{where}perturbation axis {axis!r} has baseline level {resolved[axis][0]!r} "
                f"above the input contract's {label} ({limit}). The cell is the identity, "
                "which is correct, but its name claims a value this model never receives."
            )
    return warnings


def build_grid(
    mode: str = DEFAULT_MODE,
    axes: Optional[Mapping[str, Sequence[Any]]] = None,
) -> Tuple[Perturbation, ...]:
    """The perturbation conditions to run.

    ``ofat`` moves one axis off baseline at a time (spec §4.3's default:
    "baseline 포함 약 13~16조건"). ``cartesian`` is the opt-in full product.
    """
    if mode not in MODES:
        raise ValueError(f"unknown perturbation mode {mode!r}; modes are {', '.join(MODES)}")
    resolved = resolve_axes(axes)
    base = {axis: resolved[axis][0] for axis in AXIS_ORDER}

    if mode == "cartesian":
        grid = []
        for combination in itertools.product(*(resolved[axis] for axis in AXIS_ORDER)):
            grid.append(_make(resolved, dict(zip(AXIS_ORDER, combination))))
        return tuple(grid)

    grid = [_make(resolved, base)]
    for axis in AXIS_ORDER:
        for level in resolved[axis][1:]:
            levels = dict(base)
            levels[axis] = level
            grid.append(_make(resolved, levels))
    return tuple(grid)


def from_id(
    perturbation_id: str,
    axes: Optional[Mapping[str, Sequence[Any]]] = None,
) -> Perturbation:
    """Parse a perturbation_id back into its levels."""
    resolved = resolve_axes(axes)
    levels = {axis: resolved[axis][0] for axis in AXIS_ORDER}
    if perturbation_id == BASELINE_ID:
        return _make(resolved, levels)

    by_prefix = {AXIS_PREFIX[axis]: axis for axis in AXIS_ORDER}
    seen = set()
    for token in perturbation_id.split(_ID_SEP):
        prefix, sep, value = token.partition(_AXIS_KV_SEP)
        if not sep or prefix not in by_prefix:
            raise ValueError(f"unknown perturbation token {token!r} in {perturbation_id!r}")
        axis = by_prefix[prefix]
        if axis in seen:
            raise ValueError(f"perturbation_id {perturbation_id!r} sets axis {axis!r} twice")
        seen.add(axis)
        matches = [level for level in resolved[axis] if _level_token(level) == value]
        if not matches:
            raise ValueError(
                f"level {value!r} is not on axis {axis!r}; levels are "
                f"{', '.join(_level_token(level) for level in resolved[axis])}"
            )
        levels[axis] = matches[0]
    return _make(resolved, levels)


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------


def _noise_kind(level: str) -> Tuple[str, Optional[float]]:
    """Classify a noise level, returning (kind, parameter)."""
    if level == "none":
        return ("none", None)
    if level == "baseline_wander":
        return ("baseline_wander", None)
    if level == "emg":
        return ("emg", None)
    match = _POWERLINE_RE.match(str(level))
    if match:
        return ("powerline", float(match.group(1)))
    raise ValueError(
        f"unknown noise level {level!r}; known levels are none, baseline_wander, "
        "emg and powerline_<n>hz"
    )


def record_key_of(sig: Signal) -> str:
    """Default per-record RNG material: a digest of the signal itself.

    Spec §4.3 pins reproducibility to ``seed + perturbation_id``, but a noise
    realization that is identical across every record in the cohort is a
    confound rather than noise. Deriving the third component from the tensor
    keeps :func:`apply` a pure function of its arguments while still varying
    per record, and keeps ``(seed, perturbation_id, record)`` reproducible.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(sig.data, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _rng(seed: int, perturbation_id: str, record_key: str) -> np.random.Generator:
    material = f"{int(seed)}|{perturbation_id}|{record_key}".encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def _fit_length(sig: Signal, n_samples: int) -> Signal:
    if sig.n_samples == n_samples:
        return sig
    if sig.n_samples > n_samples:
        return Crop(n_samples).apply(sig)
    return Pad(n_samples, mode="zero").apply(sig)


def _apply_lead_dropout(sig: Signal, level: str) -> Signal:
    """Zero the dropped leads. The lead count is never reduced (spec §4.3)."""
    dropped = LEAD_DROPOUT_SETS[level]
    if not dropped:
        return sig
    data = sig.data.copy()
    for position, lead in enumerate(sig.leads):
        if lead in dropped:
            data[position] = 0.0
    return Signal(
        data=np.ascontiguousarray(data, dtype=np.float32),
        leads=sig.leads,
        sampling_rate_hz=sig.sampling_rate_hz,
        unit=sig.unit,
    )


def _apply_duration(sig: Signal, seconds: float) -> Signal:
    """Keep the leading ``seconds``, then zero-pad back to contract length."""
    kept = int(round(float(seconds) * sig.sampling_rate_hz))
    if kept >= sig.n_samples:
        # A window at least as long as the record removes nothing. Saying so
        # is better than pretending a longer window adds information.
        return sig
    if kept <= 0:
        raise ValueError(f"duration level {seconds!r} keeps no samples")
    return _fit_length(Crop(kept).apply(sig), sig.n_samples)


#: Edge guard for the resample round-trip. Not a spec value: it is long enough
#: to contain the filter transient at the rates of the ``resample`` axis and
#: short enough to reflect from a 10 s record.
_GUARD_SECONDS = 0.5


def _guard_samples(sig: Signal, up: int, down: int) -> int:
    """How much edge guard the resample round-trip needs, in source samples.

    ``resample_poly`` zero-pads beyond the signal, so a record that does not
    begin and end at zero picks up a step transient at both edges. Measured on
    a 500 Hz record that transient reaches a large fraction of the signal
    amplitude in the final sample — an artifact of the round-trip, not of the
    sampling rate the axis is supposed to be measuring. Reflecting the signal
    outward before resampling and cropping the reflection away afterwards
    removes it.

    The guard is two full filter supports (``resample_poly`` builds a FIR of
    half-length ``10 * max(up, down)``), and never less than
    ``_GUARD_SECONDS`` of signal.
    """
    seconds = int(np.ceil(_GUARD_SECONDS * sig.sampling_rate_hz))
    return max(seconds, 20 * max(up, down))


def _apply_resample(sig: Signal, target_hz: float) -> Signal:
    """Degrade to ``target_hz`` and restore to the contract rate (spec §4.3)."""
    target = float(target_hz)
    if target >= sig.sampling_rate_hz:
        # Round-tripping through a higher rate adds no information; treating
        # it as identity avoids attributing interpolation ringing to an axis
        # level that is supposed to mean "no loss".
        return sig
    original_rate = sig.sampling_rate_hz
    original_n = sig.n_samples
    up, down = resample_factors(original_rate, target)

    # The guard must be a whole number of `down` source samples, so that it
    # occupies a whole number of samples at the degraded rate and crops back
    # to exactly itself. Odd reflection continues both value and slope, so the
    # filter sees no artificial edge.
    guard = int(np.ceil(_guard_samples(sig, up, down) / down) * down)
    guard = min(guard, ((original_n - 1) // down) * down)

    work = sig
    if guard > 0:
        work = Signal(
            data=np.ascontiguousarray(
                np.pad(
                    sig.data, ((0, 0), (guard, guard)), mode="reflect", reflect_type="odd"
                ),
                dtype=np.float32,
            ),
            leads=sig.leads,
            sampling_rate_hz=original_rate,
            unit=sig.unit,
        )

    # resample_poly carries its own anti-aliasing FIR, so the low-pass that
    # makes decimation honest is not a constant this module has to invent.
    degraded = Resample(target).apply(work)
    restored = Resample(original_rate).apply(degraded)
    if guard > 0:
        restored = Signal(
            data=np.ascontiguousarray(
                restored.data[:, guard : guard + original_n], dtype=np.float32
            ),
            leads=restored.leads,
            sampling_rate_hz=restored.sampling_rate_hz,
            unit=restored.unit,
        )
    return _fit_length(restored, original_n)


def _apply_amplitude(sig: Signal, factor: float) -> Signal:
    if float(factor) == 1.0:
        return sig
    return Signal(
        data=np.ascontiguousarray(sig.data * np.float32(factor), dtype=np.float32),
        leads=sig.leads,
        sampling_rate_hz=sig.sampling_rate_hz,
        unit=sig.unit,
    )


def _noise_waveform(
    kind: str,
    parameter: Optional[float],
    sig: Signal,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> np.ndarray:
    n_leads, n_samples = sig.n_leads, sig.n_samples
    times = np.arange(n_samples, dtype=np.float64) / sig.sampling_rate_hz

    if kind == "baseline_wander":
        # Electrode motion and respiration are lead-local, so each lead gets an
        # independent realization.
        low, high = config.baseline_wander_band_hz
        out = np.zeros((n_leads, n_samples), dtype=np.float64)
        for lead in range(n_leads):
            for _ in range(config.baseline_wander_components):
                frequency = rng.uniform(low, high)
                phase = rng.uniform(0.0, 2.0 * np.pi)
                weight = rng.uniform(0.5, 1.5)
                out[lead] += weight * np.sin(2.0 * np.pi * frequency * times + phase)
        return out

    if kind == "powerline":
        # Mains interference is largely common-mode: the same waveform appears
        # on every lead, which is why a notch at one frequency removes it.
        phase = rng.uniform(0.0, 2.0 * np.pi)
        tone = np.sin(2.0 * np.pi * float(parameter) * times + phase)
        return np.repeat(tone[None, :], n_leads, axis=0)

    if kind == "emg":
        white = rng.standard_normal((n_leads, n_samples))
        nyquist = sig.sampling_rate_hz / 2.0
        low, high = config.emg_band_hz
        high = min(high, 0.99 * nyquist)
        if low >= high:
            # The contract's rate leaves no room for the EMG band; broadband
            # noise is the honest fallback rather than a silently empty band.
            return white
        sos = sps.butter(
            config.emg_order, [low / nyquist, high / nyquist], btype="bandpass", output="sos"
        )
        return sps.sosfiltfilt(sos, white, axis=1)

    raise ValueError(f"unhandled noise kind: {kind}")


def _apply_noise(
    sig: Signal,
    level: str,
    rng: np.random.Generator,
    config: NoiseConfig,
) -> Signal:
    kind, parameter = _noise_kind(level)
    if kind == "none":
        return sig
    signal_rms = float(np.sqrt(np.mean(np.square(sig.data, dtype=np.float64))))
    if signal_rms == 0.0:
        # An SNR against a silent signal is undefined; adding nothing is the
        # consistent limit. Reachable only in cartesian mode (dropout + noise).
        return sig
    noise = _noise_waveform(kind, parameter, sig, rng, config)
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    if noise_rms == 0.0:  # pragma: no cover - defensive
        return sig
    target_rms = signal_rms / (10.0 ** (config.snr_db / 20.0))
    noise = noise * (target_rms / noise_rms)
    return Signal(
        data=np.ascontiguousarray(sig.data + noise, dtype=np.float32),
        leads=sig.leads,
        sampling_rate_hz=sig.sampling_rate_hz,
        unit=sig.unit,
    )


def apply(
    sig: Signal,
    perturbation: Union[str, Perturbation],
    seed: int = 0,
    scaling: str = "none",
    record_key: Optional[str] = None,
    noise: NoiseConfig = DEFAULT_NOISE,
    axes: Optional[Mapping[str, Sequence[Any]]] = None,
) -> Signal:
    """Apply one perturbation to a contract-shaped signal, restoring the shape.

    ``sig`` is a tensor already compiled to a model's input contract. The
    result has the same lead names, lead count, sample count, rate and unit —
    that invariant is the point of the axis (spec §4.3) and is asserted below.

    ``scaling`` is the contract's normalization method; it is re-applied after
    the perturbation so that the restored signal is in contract form all the
    way through, not merely the right shape. See the module docstring.

    Application order is fixed: lead dropout, duration, resample, gain, noise.
    Noise is added last because it is an amplifier-side artifact — adding it
    before the resample round-trip would let the anti-aliasing filter partly
    remove the very artifact being injected. Under the default OFAT policy at
    most one axis is off baseline, so the order only bites in cartesian mode.
    """
    if scaling not in SUPPORTED_SCALINGS:
        raise ValueError(
            f"unknown scaling {scaling!r}; must be one of {sorted(SUPPORTED_SCALINGS)}"
        )
    if isinstance(perturbation, str):
        perturbation = from_id(perturbation, axes)

    base = baseline(axes)
    if perturbation.levels() == base.levels():
        # Baseline must be exactly the stored tensor, not a re-normalized
        # near-copy of it, or every axis would be measured against a slightly
        # different reference.
        return sig

    original = (sig.leads, sig.n_samples, sig.sampling_rate_hz, sig.unit)
    key = record_key if record_key is not None else record_key_of(sig)
    rng = _rng(seed, perturbation.id, key)

    out = _apply_lead_dropout(sig, perturbation.lead_dropout)
    out = _apply_duration(out, perturbation.duration)
    out = _apply_resample(out, perturbation.resample)
    out = _apply_amplitude(out, perturbation.amplitude_scale)
    out = _apply_noise(out, perturbation.noise, rng, noise)
    out = _fit_length(out, original[1])
    out = Normalize(scaling).apply(out)

    if (out.leads, out.n_samples, out.sampling_rate_hz, out.unit) != original:
        raise AssertionError(
            f"perturbation {perturbation.id!r} did not restore the input contract form: "
            f"{original} -> {(out.leads, out.n_samples, out.sampling_rate_hz, out.unit)}"
        )
    return out
