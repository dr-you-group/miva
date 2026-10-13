"""Preprocess stage (spec §4.3). Implemented by Plan 3.

This file currently pins the contract only. The surface below — class name,
stage name, reason_code vocabulary, required inputs, output artifact names —
is fixed by Plan 2 because ``models`` and ``evaluate`` are written against it
in parallel. Changing any of it changes another stage's inputs.

What the stage does, in one sentence: for every record of ``cohort_index`` and
every ModelCard in the registry, compile the card's input contract against the
record's source metadata, write the resulting tensor once per distinct recipe,
and record every compile failure in the exclusion ledger.

**No model appears by name anywhere below.** Registering a model is adding one
JSON card to ``registry/models/`` (claim C1); a branch on ``model_id`` here
would falsify it. Everything model-specific is read off the card.

**recipe_id, and why it is not model_id.** ``recipe_id`` is a digest of the
compiled op chain, so two cards whose input contracts agree get the same id and
share one tensor on disk. It is also a *function of the source metadata*, so a
heterogeneous cohort yields more than one recipe_id per model — which is the
truth about that cohort, and is exactly what spec §3.4 wants the axis to carry.

**Where perturbations are.** Nowhere on disk. Spec §4.3 applies them in the
dataloader; this stage publishes the grid as ``perturbation_grid.json`` and
every row of ``preprocess_index`` carries ``perturbation_id = "baseline"``.
Stage 4 walks the grid and calls :func:`mival.perturbation.apply`.

**Loader selection.** The stage reads each record's ``local_path`` through a
loader chosen by name (spec key ``loader``, default ``"npz"``; see
:data:`LOADERS`), or an injected one when a caller passes ``loader=`` to
:class:`PreprocessStage` directly (tests do this). :func:`load_npz_record`
reads the ``.npz`` bundle used before the real cohort arrived;
:func:`load_dicom_record` reads the 12-lead ECG Waveform Storage files of the
real cohort (spec §4.2).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mival.compiler import compile_recipe
from mival.contract import CompileError, REASON_CODES
from mival.modelcard import ModelCard, load_registry
from mival.ops import OpChain
from mival.perturbation import (
    DEFAULT_MODE,
    build_grid,
    check_grid_against_contract,
    resolve_axes,
)
from mival.pipeline.hashing import canonical_json
from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import read_table, write_table
from mival.signal import Signal, SourceMetadata

#: Output artifact names, relative to ``ctx.layout.artifacts_dir``.
PREPROCESS_INDEX = "preprocess_index.parquet"
TENSOR_DIR = "tensors"

#: Sidecars. ``recipes.json`` is the "전처리 설정과 적용 순서" of
#: docs/pipeline/03-preprocess.md, and it is also how stage 4 recovers the
#: contract ``scaling`` that :func:`mival.perturbation.apply` needs without
#: re-reading the registry. ``perturbation_grid.json`` is the sweep stage 4
#: must run.
RECIPES = "recipes.json"
PERTURBATION_GRID = "perturbation_grid.json"

#: Columns of ``preprocess_index.parquet``. Stage 4 joins on these.
PREPROCESS_INDEX_COLUMNS = (
    "image_occurrence_id",
    "person_id",
    "model_id",
    "recipe_id",
    "perturbation_id",
    "tensor_path",
    "n_leads",
    "n_samples",
    "sampling_rate_hz",
)

#: Columns this stage needs from ``cohort_index`` (spec §4.1). The rest of that
#: table — labels, split keys — is carried forward by stages 4 and 5, not here.
COHORT_INDEX_COLUMNS = ("image_occurrence_id", "person_id", "local_path")

#: Every stored tensor is unperturbed; the sweep happens in stage 4's dataloader.
BASELINE_PERTURBATION_ID = "baseline"

#: Registry location when ``study.yaml`` does not name one (spec §3.2).
DEFAULT_REGISTRY = Path("registry/models")

# 12 hex characters is 48 bits. Distinct recipes number in the tens — one per
# (input contract, source metadata) pair — so collision probability is
# negligible, and a shorter id keeps ``tensors/<recipe_id>/`` readable.
RECIPE_ID_LENGTH = 12

# image_occurrence_id becomes a filename and a join key. Same charset as
# mival.pipeline.layout and mival.pipeline.runkey use for path components.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: A loader maps a record's ``local_path`` to its samples and what the Profile
#: stage observed about them.
Loader = Callable[[Path], Tuple[np.ndarray, SourceMetadata]]


def load_npz_record(path: Path) -> Tuple[np.ndarray, SourceMetadata]:
    """Read one record from an ``.npz`` bundle.

    Keys: ``data`` (n_leads, n_samples), ``leads``, ``sampling_rate_hz``, and
    an optional ``unit``. An **absent or empty** ``unit`` is passed through as
    ``None`` rather than defaulted — spec §4.3 excludes such records
    (``unit_missing``) because a guessed amplitude is silently wrong and
    nothing downstream can detect it.

    Used ahead of the real cohort's DICOM files; see :func:`load_dicom_record`.
    """
    with np.load(str(path), allow_pickle=False) as bundle:
        data = np.ascontiguousarray(bundle["data"], dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"{path}: 'data' must be 2-D (n_leads, n_samples), got {data.ndim}-D")
        leads = tuple(str(name) for name in bundle["leads"])
        sampling_rate_hz = float(bundle["sampling_rate_hz"])
        unit: Optional[str] = None
        if "unit" in bundle:
            text = str(bundle["unit"]).strip()
            unit = text or None
    source = SourceMetadata(
        leads=leads,
        sampling_rate_hz=sampling_rate_hz,
        n_samples=int(data.shape[1]),
        unit=unit,
    )
    return data, source


#: MDC lead codes (ChannelSourceSequence CodeValue) to the names this project uses.
MDC_LEADS = {"2:1": "I", "2:2": "II", "2:61": "III", "2:62": "aVR", "2:63": "aVL", "2:64": "aVF",
             "2:3": "V1", "2:4": "V2", "2:5": "V3", "2:6": "V4", "2:7": "V5", "2:8": "V6"}


def load_dicom_record(path: Path) -> Tuple[np.ndarray, SourceMetadata]:
    """Read a 12-lead ECG Waveform Storage file; pydicom applies sensitivity, so values are in the channel unit."""
    import pydicom

    dataset = pydicom.dcmread(str(path))
    item = dataset.WaveformSequence[0]
    channels = item.ChannelDefinitionSequence
    leads = tuple(_lead_name(channel) for channel in channels)
    units = {channel.ChannelSensitivityUnitsSequence[0].CodeValue
             for channel in channels if "ChannelSensitivityUnitsSequence" in channel}
    data = np.ascontiguousarray(dataset.waveform_array(0).T, dtype=np.float32)
    source = SourceMetadata(leads=leads, sampling_rate_hz=float(item.SamplingFrequency),
                            n_samples=int(data.shape[1]), unit=units.pop() if len(units) == 1 else None)
    return data, source


def _lead_name(channel) -> str:
    # Position is not trustworthy (aVF precedes aVL in these files), so the
    # MDC code is the only source of truth for which row is which lead.
    code = channel.ChannelSourceSequence[0]
    value = str(code.CodeValue)
    if value not in MDC_LEADS:
        raise ValueError(f"unrecognised MDC lead code {value!r} ({code.CodeMeaning})")
    return MDC_LEADS[value]


LOADERS: Dict[str, Loader] = {"npz": load_npz_record, "dicom": load_dicom_record}


def recipe_id(chain: OpChain) -> str:
    """A deterministic id for a compiled op chain.

    Derived from the chain's own description, so the same chain always gets
    the same id and two cards with the same input contract share a tensor.
    """
    return hashlib.sha256(
        canonical_json(chain.describe()).encode("utf-8")
    ).hexdigest()[:RECIPE_ID_LENGTH]


def _safe_id(kind: str, value: str) -> str:
    if not _SAFE_ID_RE.match(value):
        raise ValueError(
            f"{kind} {value!r} cannot be used as a path component; allowed characters "
            "are letters, digits, '.', '_' and '-', and it may not start with '.', "
            "'_' or '-'"
        )
    return value


def _optional_text(value: Any) -> Optional[str]:
    # Avoids importing pandas here — mival.pipeline.tables keeps it lazy on
    # purpose, and NaN is the only missing-value form parquet hands back.
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value)
    return text or None


def _contract_summary(card: ModelCard) -> Dict[str, Any]:
    contract = card.input_contract
    return {
        "leads": list(contract.leads),
        "sampling_rate_hz": contract.sampling_rate_hz,
        "duration_s": contract.duration_s,
        "n_samples": contract.n_samples,
        "unit": contract.unit,
        "scaling": contract.scaling,
        "layout": contract.layout,
        "dtype": contract.dtype,
    }


class PreprocessStage(Stage):
    name = "preprocess"
    # The compile failures of spec §4.3 are exactly this stage's exclusions.
    reason_codes = REASON_CODES

    def __init__(self, loader: Optional[Loader] = None) -> None:
        # Injected rather than selected by a flag: a test's injection wins
        # over the spec, so existing tests keep working unchanged.
        self._loader = loader

    def resolve_loader(self, spec: Optional[Dict[str, Any]]) -> Loader:
        """The record loader for this run: an injected one, else the spec's ``loader`` name."""
        if self._loader is not None:
            return self._loader
        name = str((spec or {}).get("loader", "npz"))
        if name not in LOADERS:
            raise ValueError(f"preprocess.loader {name!r} is not one of {sorted(LOADERS)}")
        return LOADERS[name]

    def required_inputs(self) -> tuple:
        return ("cohort_index",)

    def config_inputs(self, spec: Dict[str, Any]) -> Dict[str, Path]:
        """Every ModelCard, so that editing one invalidates this run.

        The registry arrives as a *directory*; without this the compiled
        recipes could change while config_hash did not, and the stale tensors
        would be skipped rather than rebuilt.
        """
        registry_dir = Path((spec or {}).get("registry", DEFAULT_REGISTRY))
        if not registry_dir.is_dir():
            return {}
        return {f"modelcard:{path.stem}": path for path in sorted(registry_dir.glob("*.json"))}

    def run(self, ctx: StageContext) -> StageResult:
        spec = ctx.spec or {}
        registry_dir = Path(spec.get("registry", DEFAULT_REGISTRY))
        allow_upsample = bool(spec.get("allow_upsample", False))
        pad_policy = str(spec.get("pad_policy", "reject"))

        perturbation_spec = dict(spec.get("perturbation", {}) or {})
        mode = str(perturbation_spec.get("mode", DEFAULT_MODE))
        grid = build_grid(mode=mode, axes=perturbation_spec.get("axes"))

        loader = self.resolve_loader(spec)
        cards = self._load_registry(registry_dir)
        # The grid degrades a tensor that has already been compiled to a card's
        # input contract, so its levels only mean something relative to that
        # contract. Checked once per card, before any record is read.
        grid_warnings: List[str] = []
        for card in cards.values():
            grid_warnings.extend(
                check_grid_against_contract(
                    perturbation_spec.get("axes"), card.input_contract, card.model_id
                )
            )
        cohort = read_table(ctx.input_path("cohort_index"))
        missing = [name for name in COHORT_INDEX_COLUMNS if name not in cohort.columns]
        if missing:
            raise ValueError(
                f"cohort_index is missing required columns {', '.join(missing)}; "
                f"spec §4.1 defines {', '.join(COHORT_INDEX_COLUMNS)}"
            )

        rows: List[Dict[str, Any]] = []
        recipes: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = list(grid_warnings)
        written: set = set()
        seen: set = set()
        n_in = 0
        n_out = 0
        n_partial = 0

        for record in cohort[list(COHORT_INDEX_COLUMNS)].itertuples(index=False):
            n_in += 1
            image_occurrence_id = _safe_id("image_occurrence_id", str(record.image_occurrence_id))
            person_id = _optional_text(record.person_id)
            if image_occurrence_id in seen:
                # A repeated id would put duplicate (record, model) rows in the
                # index and silently multiply stage 4's join.
                raise ValueError(
                    f"cohort_index contains image_occurrence_id {image_occurrence_id!r} "
                    "more than once; it is the join key of preprocess_index"
                )
            seen.add(image_occurrence_id)

            try:
                data, source = loader(Path(str(record.local_path)))
            except Exception as exc:  # noqa: BLE001
                # A file this stage cannot read is one record's exclusion, not
                # the run's abort (spec §10): a single corrupt or unexpected
                # file would otherwise lose every record after it.
                ctx.ledger.record(
                    image_occurrence_id,
                    person_id,
                    "read_failed",
                    f"{type(exc).__name__}: {exc}",
                )
                continue
            signal = Signal(
                data=data,
                leads=source.leads,
                sampling_rate_hz=source.sampling_rate_hz,
                unit=source.unit,
            )

            # One ledger row per (record, reason_code), not per (record, model).
            # The STARD flow of spec §3.6 counts records; the models sharing a
            # failure go into `detail` so nothing is lost.
            failures: Dict[str, List[str]] = {}
            details: Dict[str, str] = {}
            compiled = 0

            for model_id, card in sorted(cards.items()):
                try:
                    chain = compile_recipe(
                        card.input_contract,
                        source,
                        allow_upsample=allow_upsample,
                        pad_policy=pad_policy,
                    )
                except CompileError as exc:
                    failures.setdefault(exc.reason_code, []).append(model_id)
                    details.setdefault(exc.reason_code, exc.detail)
                    continue

                rid = recipe_id(chain)
                tensor = ctx.layout.artifact(TENSOR_DIR, rid, f"{image_occurrence_id}.npy")
                if tensor not in written:
                    output = chain.apply(signal)
                    self._check_contract(output, card, image_occurrence_id)
                    tensor.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(tensor), output.data)
                    written.add(tensor)

                entry = recipes.setdefault(
                    rid,
                    {
                        "recipe_id": rid,
                        "ops": chain.describe(),
                        "contract": _contract_summary(card),
                        "model_ids": [],
                    },
                )
                if model_id not in entry["model_ids"]:
                    entry["model_ids"].append(model_id)

                contract = card.input_contract
                rows.append(
                    {
                        "image_occurrence_id": image_occurrence_id,
                        "person_id": person_id,
                        "model_id": model_id,
                        "recipe_id": rid,
                        "perturbation_id": BASELINE_PERTURBATION_ID,
                        "tensor_path": str(tensor.relative_to(ctx.layout.run_dir)),
                        "n_leads": len(contract.leads),
                        "n_samples": contract.n_samples,
                        "sampling_rate_hz": contract.sampling_rate_hz,
                    }
                )
                compiled += 1

            for reason_code, model_ids in sorted(failures.items()):
                ctx.ledger.record(
                    image_occurrence_id,
                    person_id,
                    reason_code,
                    f"{details[reason_code]} [models: {', '.join(model_ids)}]",
                )
            if compiled:
                n_out += 1
                if failures:
                    n_partial += 1

        if n_partial:
            # in != out + excluded whenever this is non-zero, because the gate
            # is per (record, model). Saying so in the manifest beats leaving a
            # reader to discover the arithmetic does not close.
            warnings.append(
                f"{n_partial} record(s) compiled for some models and were excluded for "
                "others; the exclusion ledger counts records, not record-model pairs"
            )

        index_path = write_table(
            rows, ctx.layout.artifact(PREPROCESS_INDEX), PREPROCESS_INDEX_COLUMNS
        )
        recipes_path = self._write_json(
            ctx.layout.artifact(RECIPES),
            {"recipes": [recipes[rid] for rid in sorted(recipes)]},
        )
        grid_path = self._write_json(
            ctx.layout.artifact(PERTURBATION_GRID),
            {
                "mode": mode,
                "axes": {
                    axis: list(levels)
                    for axis, levels in resolve_axes(perturbation_spec.get("axes")).items()
                },
                "seed": ctx.seed,
                "perturbations": [item.to_dict() for item in grid],
            },
        )

        # Tensors are addressed through preprocess_index and deliberately not
        # listed one by one: checksumming N records x R recipes would dominate
        # this stage's wall time, and the manifest would grow without bound.
        return StageResult(
            outputs=[index_path, recipes_path, grid_path],
            counts={
                "in": n_in,
                "out": n_out,
                "index_rows": len(rows),
                "tensors": len(written),
                "recipes": len(recipes),
                "models": len(cards),
                "perturbations": len(grid),
            },
            warnings=warnings,
        )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _load_registry(directory: Path) -> Dict[str, ModelCard]:
        if not directory.is_dir():
            raise ValueError(
                f"model registry {directory} is not a directory; set 'registry' in the "
                "preprocess spec (spec §3.2)"
            )
        cards = load_registry(directory)
        if not cards:
            raise ValueError(f"model registry {directory} contains no ModelCard JSON files")
        return cards

    @staticmethod
    def _check_contract(output: Signal, card: ModelCard, image_occurrence_id: str) -> None:
        """The input contract gate of spec §3.5, on the stage-3 side.

        Compile already rejected every record it could not satisfy, so a
        mismatch here is a compiler bug rather than a data problem and must
        fail the run instead of quietly excluding a record.
        """
        contract = card.input_contract
        actual = (output.n_leads, output.n_samples, output.sampling_rate_hz, output.leads)
        expected = (
            len(contract.leads),
            contract.n_samples,
            contract.sampling_rate_hz,
            contract.leads,
        )
        if actual != expected:
            raise ValueError(
                f"compiled tensor for {image_occurrence_id} / {card.model_id} does not match "
                f"its input contract: expected {expected}, got {actual}"
            )

    @staticmethod
    def _write_json(path: Path, payload: Dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        return path
