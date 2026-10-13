"""Compile a preprocessing recipe from a model's declared input contract.

Nobody hand-writes a per-model recipe. Adding a model means adding a card.
A compile failure IS the input contract gate: the CompileError's reason_code
is what the exclusion ledger records.
"""

from __future__ import annotations

from typing import List

from mival.contract import CompileError, InputContract
from mival.ops import (
    BandFilter,
    Crop,
    Gain,
    Normalize,
    Op,
    OpChain,
    Pad,
    ReconstructLeads,
    Resample,
    ScaleUnit,
    SelectLeads,
    SUPPORTED_PAD_MODES,
    SUPPORTED_UNITS,
    resampled_length,
)
from mival.signal import DERIVABLE_LEADS, SourceMetadata


def compile_recipe(
    contract: InputContract,
    source: SourceMetadata,
    allow_upsample: bool = False,
    pad_policy: str = "reject",
) -> OpChain:
    valid_pad_policies = {"reject"} | SUPPORTED_PAD_MODES
    if pad_policy not in valid_pad_policies:
        raise ValueError(
            f"unsupported pad_policy {pad_policy!r}: "
            f"must be one of {sorted(valid_pad_policies)}"
        )

    ops: List[Op] = []

    # 1. unit. An unknown unit is never guessed: a silently wrong amplitude
    #    cannot be detected downstream.
    if source.unit is None:
        raise CompileError(
            "unit_missing", "source declares no amplitude unit or sensitivity"
        )
    if source.unit not in SUPPORTED_UNITS:
        raise CompileError(
            "unit_missing",
            "source unit {0!r} is not one of {1}".format(
                source.unit, sorted(SUPPORTED_UNITS)
            ),
        )
    ops.append(ScaleUnit(source.unit, contract.unit))

    # 2. model-declared filters
    for spec in contract.filters:
        ops.append(
            BandFilter(
                kind=spec["kind"],
                cutoff_hz=spec["cutoff_hz"],
                # Required by the input contract; never defaulted here.
                order=int(spec["order"]),
            )
        )

    # 3. sampling rate
    if source.sampling_rate_hz != contract.sampling_rate_hz:
        if source.sampling_rate_hz < contract.sampling_rate_hz and not allow_upsample:
            raise CompileError(
                "upsample_required",
                f"source is {source.sampling_rate_hz} Hz but the contract needs "
                f"{contract.sampling_rate_hz} Hz",
            )
        ops.append(Resample(contract.sampling_rate_hz))
    try:
        n_after_resample = resampled_length(
            source.n_samples, source.sampling_rate_hz, contract.sampling_rate_hz
        )
    except ValueError as exc:
        raise CompileError("rate_unsupported", str(exc)) from exc

    # 4. leads
    available = set(source.leads)
    missing = [lead for lead in contract.leads if lead not in available]
    underivable = [
        lead
        for lead in missing
        if lead not in DERIVABLE_LEADS
        or any(src not in available for src in DERIVABLE_LEADS[lead])
    ]
    if underivable:
        raise CompileError(
            "lead_unavailable",
            f"source cannot supply or derive: {', '.join(sorted(underivable))}",
        )
    ops.append(
        ReconstructLeads(contract.leads) if missing else SelectLeads(contract.leads)
    )

    # 5. length
    if n_after_resample > contract.n_samples:
        ops.append(Crop(contract.n_samples, anchor=contract.crop_anchor))
    elif n_after_resample < contract.n_samples:
        if pad_policy == "reject":
            raise CompileError(
                "duration_short",
                f"source yields {n_after_resample} samples but the contract needs "
                f"{contract.n_samples}",
            )
        ops.append(Pad(contract.n_samples, mode=pad_policy, anchor=contract.pad_anchor))

    # 6. gain: a card-declared quirk on top of the unit, applied before
    #    normalization so it changes the mean/std that step measures.
    if contract.gain != 1.0:
        ops.append(Gain(contract.gain))

    # 7. normalization
    ops.append(Normalize(contract.scaling))

    return OpChain(tuple(ops))
