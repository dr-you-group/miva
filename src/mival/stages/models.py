"""Models stage (spec §4.4). Implemented by Plan 4.

Contract pinned by Plan 2. ``evaluate`` reads ``PREDICTION_COLUMNS`` and is
being written in parallel against it.

This is the one stage that imports a backend, and it must do so lazily —
through ``mival.adapters.get_adapter`` — because the environment holding torch
does not hold TensorFlow.

Nothing in this file names a model. Every model-specific fact — which adapter
to use, which weights, which layer is the feature layer, which training modes
are supported, what the published threshold is, what the model was pretrained
on — is read from the ModelCard. That is claim C1: registering a model is one
JSON file and no source change. A single ``if model_id == …`` here would
falsify it, so there is none.

Three properties are enforced structurally rather than by convention:

* **The test split is touched exactly once.** Test records are held behind
  :class:`_TestSet`, which hands them out once and raises on a second request.
  Spec §4.4 requires that a second contact path not exist, not merely that it
  not be taken.
* **Thresholds are refit on dev only.** Probabilities are wrapped in a
  :class:`~mival.threshold.SplitScores` carrying the split they came from, and
  the refit policies reject anything but ``dev``.
* **Hyperparameters are chosen inside dev.** Selection runs a k-fold loop over
  the folds frozen in ``cohort_split.parquet``; the dev probabilities that feed
  the threshold refit are out-of-fold, so no threshold is chosen on data the
  model was fitted on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mival.gates import ContaminationReport, check_contamination, check_leakage
from mival.modelcard import ModelCard, load_card
from mival.pipeline.hashing import sha256_file
from mival.pipeline.runkey import RunKey
from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import read_table, write_table
from mival.stages.preprocess import PREPROCESS_INDEX_COLUMNS
from mival.threshold import SplitScores, auroc, fit_thresholds

#: Output artifacts, relative to ``ctx.layout.artifacts_dir``.
PREDICTIONS_DIR = "predictions"
TRAIN_LOG = "train_log.jsonl"

#: Spec §4.4. One parquet per run_key at ``predictions/<run_key>.parquet``.
PREDICTION_COLUMNS = (
    "image_occurrence_id",
    "person_id",
    "split",
    "fold",
    "label_primary",
    "label_sens1",
    "label_sens2",
    "label_sens3",
    "label_value",
    "label_event",
    "label_time_days",
    "label_pr",
    "label_qrs",
    "label_qt",
    "prob",
    "logit",
    "pred_value",
    "model_id",
    "training_mode",
    "recipe_id",
    "perturbation_id",
    "seed",
    "site",
)

#: The label_def of a regression arm (spec addendum: no new run_key axis, the
#: label names the task). A regression arm reads ``label_value`` and writes
#: ``pred_value``; ``prob``/``logit`` are null and no threshold is fit.
REGRESSION_LABEL_DEF = "value"
#: Every label_def scored as a continuous value against ``label_<def>``. The
#: interval ones are read off a segmentation card's decoded output.
REGRESSION_LABEL_DEFS = (REGRESSION_LABEL_DEF, "pr", "qrs", "qt")
#: A survival arm reads ``label_event`` and ``label_time_days`` and writes a
#: risk score to ``pred_value``; it has no threshold.
SURVIVAL_LABEL_DEF = "survival"
SURVIVAL_LABEL_COLUMNS = ("label_event", "label_time_days")
#: Continuous label columns (read as float); every other one is a 0/1 outcome.
#: cohort_index columns carrying a model's non-waveform inputs: ``cov_<name>``.
COVARIATE_PREFIX = "cov_"
CONTINUOUS_LABEL_COLUMNS = ("label_value", "label_time_days", "label_pr", "label_qrs", "label_qt")
#: Which card output.type an inference_only arm of each kind can score through.
_INFERENCE_OUTPUT_TYPES = {
    "binary": ("logits", "softmax", "sigmoid"),
    "value": ("regression",),
    "interval": ("regression", "segmentation_mask"),
    "survival": ("risk_score", "survival_curve", "mtlr"),
}

#: Hyperparameters for a regression arm are selected by negative dev MAE:
#: AUROC has no meaning once the target is continuous, and smaller error is
#: what a regression arm is trying to minimise.
SELECTION_METRIC_REGRESSION = "neg_mae"

#: Spec §4.4. Primary is `refit_sens95`: STEMI miss cost is asymmetric, so
#: Youden — which weighs sensitivity and specificity equally — is clinically
#: inappropriate. Thresholds are never selected on the test set.
THRESHOLD_POLICIES = ("legacy", "refit_youden", "refit_sens95")
PRIMARY_THRESHOLD_POLICY = "refit_sens95"

#: Spec §4.4 training modes.
TRAINING_MODES = ("inference_only", "linear_probe", "partial_unfreeze", "full_finetune")

#: This stage's closed exclusion vocabulary (spec §3.6). Distinct from the
#: preprocess codes: a record can reach stage 4 and still be droppable.
REASON_CODES = frozenset({"tensor_missing", "label_missing", "split_unassigned"})

# ---------------------------------------------------------------------------
# derived contract
# ---------------------------------------------------------------------------

#: Spec §2.4. ``cohort_split.parquet`` carries exactly these two split values;
#: val is the held-out fold of the dev k-fold loop, not a third file.
DEV_SPLIT = "dev"
TEST_SPLIT = "test"

#: Derived from the pinned schema so the two can never drift apart.
LABEL_COLUMNS = tuple(name for name in PREDICTION_COLUMNS if name.startswith("label_"))

#: The 0/1 label columns. Every cohort carries these; ``label_value`` is the
#: regression target and a classification-only cohort has no reason to hold it,
#: so it is read as NaN when absent rather than demanded of every cohort_index.
BINARY_LABEL_COLUMNS = ("label_primary", "label_sens1", "label_sens2", "label_sens3")

#: ``label_def`` is a run_key axis (spec §3.4) whose values name label columns.
DEFAULT_LABEL_DEF = "primary"

#: Inference batch size. A compute knob with no effect on results; overridable
#: per study with ``models.batch_size``.
DEFAULT_BATCH_SIZE = 64

#: The training modes that fit something and therefore need dev folds.
_TRAINED_MODES = tuple(mode for mode in TRAINING_MODES if mode != "inference_only")

#: Columns ``cohort_split.parquet`` must provide (spec §2.4).
COHORT_SPLIT_COLUMNS = ("person_id", "split", "fold")

#: Hyperparameters are selected by dev AUROC because AUROC is the primary
#: endpoint (spec §2.5); selecting on a different quantity than the one
#: reported would optimise for something the study does not claim.
SELECTION_METRIC = "auroc"


class TestSetContactError(RuntimeError):
    """Raised when the held-out test split is requested more than once."""


class AdapterCapabilityError(NotImplementedError):
    """Raised when a ModelCard asks for something its adapter does not implement."""


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def _expand_grid(grid: Mapping[str, Any], where: str) -> Tuple[Dict[str, Any], ...]:
    """Cartesian product of a hyperparameter grid, in a deterministic order.

    Keys are sorted so the candidate order — and therefore which candidate wins
    a tie in the selection metric — does not depend on how the YAML parser
    happened to order the mapping.
    """
    if not grid:
        return ({},)
    keys = sorted(grid)
    axes: List[List[Any]] = []
    for key in keys:
        values = grid[key]
        if not isinstance(values, (list, tuple)):
            raise ValueError(
                f"{where}: hparam_grid[{key!r}] must be a list of candidate values, got "
                f"{type(values).__name__}; use 'hparams' for a fixed setting"
            )
        if not values:
            raise ValueError(f"{where}: hparam_grid[{key!r}] is empty")
        axes.append(list(values))
    return tuple(dict(zip(keys, combination)) for combination in product(*axes))


@dataclass(frozen=True)
class Arm:
    """One (model, training mode) cell of the comparison (spec §2.5)."""

    model_id: str
    training_mode: str
    label_def: str
    hparam_grid: Tuple[Mapping[str, Any], ...]
    unfreeze_groups: Tuple[str, ...]
    early_stopping_fold: Optional[int]
    early_stopping_auto: bool

    @property
    def label_column(self) -> str:
        return "label_event" if self.is_survival else "label_" + self.label_def

    @property
    def required_label_columns(self) -> Tuple[str, ...]:
        return SURVIVAL_LABEL_COLUMNS if self.is_survival else (self.label_column,)

    @property
    def is_regression(self) -> bool:
        return self.label_def in REGRESSION_LABEL_DEFS

    @property
    def is_survival(self) -> bool:
        return self.label_def == SURVIVAL_LABEL_DEF

    @property
    def kind(self) -> str:
        if self.is_survival:
            return "survival"
        if self.label_def == REGRESSION_LABEL_DEF:
            return "value"
        return "interval" if self.is_regression else "binary"

    @property
    def objective(self) -> str:
        return "mse" if self.is_regression else "bce"

    def describe(self) -> str:
        return f"arm {self.model_id}/{self.training_mode}"

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any], index: int, default_label_def: str) -> "Arm":
        where = f"models.arms[{index}]"
        if not isinstance(body, Mapping):
            raise ValueError(f"{where} must be a mapping, got {type(body).__name__}")
        for key in ("model_id", "training_mode"):
            if not body.get(key):
                raise ValueError(f"{where}: {key} is required")
        training_mode = str(body["training_mode"])
        if training_mode not in TRAINING_MODES:
            raise ValueError(
                f"{where}: unknown training_mode {training_mode!r}; spec §4.4 defines "
                f"{list(TRAINING_MODES)}"
            )
        label_def = str(body.get("label_def", default_label_def))
        if label_def == SURVIVAL_LABEL_DEF:
            if training_mode != "inference_only":
                raise ValueError(
                    f"{where}: a survival arm scores the card's published risk and supports "
                    f"inference_only only, not {training_mode!r}"
                )
        elif "label_" + label_def not in LABEL_COLUMNS:
            raise ValueError(
                f"{where}: label_def {label_def!r} names no label column; available "
                f"columns are {list(LABEL_COLUMNS)}"
            )
        if "hparam_grid" in body and "hparams" in body:
            raise ValueError(f"{where}: give either hparams or hparam_grid, not both")
        if "hparam_grid" in body:
            grid = _expand_grid(body["hparam_grid"] or {}, where)
        else:
            fixed = body.get("hparams") or {}
            if not isinstance(fixed, Mapping):
                raise ValueError(f"{where}: hparams must be a mapping")
            grid = (dict(fixed),)
        unfreeze = tuple(str(name) for name in body.get("unfreeze_groups", ()) or ())
        if training_mode == "partial_unfreeze" and not unfreeze:
            raise ValueError(
                f"{where}: training_mode 'partial_unfreeze' requires unfreeze_groups; "
                "spec §4.4 makes the unfrozen group list a mandatory record"
            )
        declared = "early_stopping_fold" in body
        fold = body.get("early_stopping_fold")
        return cls(
            model_id=str(body["model_id"]),
            training_mode=training_mode,
            label_def=label_def,
            hparam_grid=grid,
            unfreeze_groups=unfreeze,
            early_stopping_fold=None if fold is None else int(fold),
            early_stopping_auto=not declared,
        )


@dataclass(frozen=True)
class ModelsSpec:
    """The ``models`` block of ``study.yaml``."""

    registry: Path
    arms: Tuple[Arm, ...]
    cohort_sources: Tuple[str, ...]
    batch_size: int

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "ModelsSpec":
        registry = body.get("registry")
        if not registry:
            raise ValueError(
                "models.registry is required: the path to the ModelCard directory "
                "(registry/models). Models are data, so the stage has to be told where "
                "the cards live."
            )
        arms_body = body.get("arms")
        if not arms_body:
            raise ValueError("models.arms is required and must list at least one arm")
        default_label_def = str(body.get("label_def", DEFAULT_LABEL_DEF))
        arms = tuple(
            Arm.from_mapping(arm, index, default_label_def)
            for index, arm in enumerate(arms_body)
        )
        seen = set()
        for arm in arms:
            key = (arm.model_id, arm.training_mode, arm.label_def)
            if key in seen:
                raise ValueError(
                    f"models.arms repeats {key}; two arms with the same model, training "
                    "mode and label_def would write the same run_key twice"
                )
            seen.add(key)
        batch_size = int(body.get("batch_size", DEFAULT_BATCH_SIZE))
        if batch_size < 1:
            raise ValueError(f"models.batch_size must be >= 1, got {batch_size}")
        return cls(
            registry=Path(str(registry)),
            arms=arms,
            cohort_sources=tuple(str(name) for name in body.get("cohort_sources", ()) or ()),
            batch_size=batch_size,
        )


# ---------------------------------------------------------------------------
# records and tensors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """One preprocessed tensor with its split, fold and labels attached."""

    image_occurrence_id: str
    person_id: str
    model_id: str
    recipe_id: str
    perturbation_id: str
    tensor_path: str
    split: str
    fold: Optional[int]
    labels: Mapping[str, Optional[float]]


def load_tensors(paths: Sequence[str]) -> np.ndarray:
    """Stack preprocessed tensors into one ``(B, n_leads, n_samples)`` batch.

    Preprocess writes one array per record; the canonical in-memory layout is
    lead-major and ``mival.adapters.to_model_layout`` converts to whatever the
    card declares. Pickle loading stays off: an artifact directory is not a
    trust boundary this stage should widen.
    """
    arrays = [_load_one(path) for path in paths]
    shapes = {array.shape for array in arrays}
    if len(shapes) > 1:
        raise ValueError(
            f"preprocessed tensors in one batch have differing shapes {sorted(shapes)}; "
            "a recipe must produce one fixed shape (spec §4.3)"
        )
    return np.stack(arrays, axis=0)


def _load_one(path: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            names = list(loaded.files)
            if "tensor" in names:
                return np.asarray(loaded["tensor"])
            if len(names) != 1:
                raise ValueError(
                    f"{path}: npz holds {names}; expected a single array or one named "
                    "'tensor'"
                )
            return np.asarray(loaded[names[0]])
    return np.asarray(loaded)


class _TestSet:
    """The held-out split, handed out exactly once (spec §4.4).

    Spec §4.4 requires that no code path contacts test twice. A counter checked
    after the fact would only detect the violation; holding the records behind
    a one-shot accessor means the second contact cannot happen at all. Person
    identifiers are exposed separately because the leakage gate reads the split
    membership, not the signals, and that is not a contact with the data.
    """

    def __init__(self, records: Sequence[Record], context: str) -> None:
        self._records = tuple(records)
        self._context = context
        self.touches = 0

    @property
    def person_ids(self) -> Tuple[str, ...]:
        return tuple(record.person_id for record in self._records)

    @property
    def size(self) -> int:
        return len(self._records)

    def take(self) -> Tuple[Record, ...]:
        if self.touches:
            raise TestSetContactError(
                f"the test split for {self._context} has already been used once; spec §4.4 "
                "allows a single contact, recorded in the manifest"
            )
        self.touches += 1
        return self._records


# ---------------------------------------------------------------------------
# adapter plumbing
# ---------------------------------------------------------------------------


def _adapter_method(adapter: Any, name: str, card: ModelCard) -> Callable[..., Any]:
    method = getattr(adapter, name, None)
    if not callable(method):
        raise AdapterCapabilityError(
            f"adapter {card.adapter!r} does not implement {name}(), which the spec §4.4 "
            f"adapter interface requires for model {card.model_id!r}"
        )
    return method


def _predict(
    adapter: Any,
    handle: Any,
    records: Sequence[Record],
    loader: Callable[[Sequence[str]], np.ndarray],
    batch_size: int,
    card: ModelCard,
    column: Optional[int] = None,
) -> np.ndarray:
    """The adapter's score per record, in order: a probability, or a regression value.

    ``column`` picks one value out of a per-record vector (a segmentation
    card's decoded PR, QRS and QT).
    """
    forward = _adapter_method(adapter, "forward", card)
    chunks: List[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        window = records[start : start + batch_size]
        batch = loader([record.tensor_path for record in window])
        names = _covariates(card)
        if names:
            covariates = np.array(
                [[record.labels[COVARIATE_PREFIX + name] for name in names] for record in window],
                dtype=np.float32,
            )
            probs = np.asarray(forward(handle, batch, covariates=covariates), dtype=np.float64)
        else:
            probs = np.asarray(forward(handle, batch), dtype=np.float64)
        if column is not None:
            probs = probs[:, column]
        probs = probs.ravel()
        if probs.size != len(window):
            raise ValueError(
                f"adapter {card.adapter!r} returned {probs.size} probabilities for a batch "
                f"of {len(window)} records"
            )
        chunks.append(probs)
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks)


def _fit_payload(
    train: Sequence[Record],
    validation: Sequence[Record],
    arm: Arm,
    loader: Callable[[Sequence[str]], np.ndarray],
    batch_size: int,
) -> Dict[str, Any]:
    """The ``data`` argument of ``adapter.fit(handle, data, mode, hparams)``.

    Signals are passed as paths plus a loader rather than as one materialised
    array: a dev split of tens of thousands of 12x5000 float32 ECGs does not
    fit in memory, and forcing it to would make the interface unusable at
    cohort scale. ``validation`` is the fold reserved for early stopping and is
    always dev — never test.
    """
    dtype = np.float64 if arm.is_regression else np.int64
    return {
        "tensor_paths": [record.tensor_path for record in train],
        "y": np.array([record.labels[arm.label_column] for record in train], dtype=dtype),
        "person_id": [record.person_id for record in train],
        "fold": [record.fold for record in train],
        "label_column": arm.label_column,
        "load_batch": loader,
        "batch_size": batch_size,
        "objective": arm.objective,
        "validation": (
            {
                "tensor_paths": [record.tensor_path for record in validation],
                "y": np.array(
                    [record.labels[arm.label_column] for record in validation], dtype=dtype
                ),
                "person_id": [record.person_id for record in validation],
            }
            if validation
            else None
        ),
    }


def _call_fit(
    adapter: Any,
    handle: Any,
    arm: Arm,
    hparams: Mapping[str, Any],
    data: Mapping[str, Any],
    card: ModelCard,
) -> Any:
    fit = _adapter_method(adapter, "fit", card)
    fitted = fit(handle, data, arm.training_mode, dict(hparams))
    if fitted is None:
        raise ValueError(
            f"adapter {card.adapter!r} fit() returned None; it must return the handle to "
            "use for inference (spec §4.4: fit(handle, data, mode, hparams) -> handle)"
        )
    return fitted


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------


class ModelsStage(Stage):
    name = "models"
    reason_codes = REASON_CODES

    def __init__(
        self,
        adapter_factory: Optional[Callable[[str], Any]] = None,
        tensor_loader: Optional[Callable[[Sequence[str]], np.ndarray]] = None,
    ) -> None:
        # Both hooks default to None rather than to the real implementation so
        # that constructing the stage imports no backend: get_adapter is looked
        # up only when an adapter is actually needed.
        self._adapter_factory = adapter_factory
        self._tensor_loader = tensor_loader or load_tensors

    def required_inputs(self) -> tuple:
        return ("preprocess_index", "cohort_split")

    def config_inputs(self, spec: Mapping[str, Any]) -> Dict[str, Path]:
        """Every ModelCard, so that editing one invalidates this run.

        The registry arrives as a *directory*, so without this a card's
        threshold, weights or pretraining_corpora could change while
        config_hash did not — and the stale predictions would be skipped
        instead of recomputed. ``train_log.jsonl`` records the same checksums,
        but that is an audit trail after the fact; this is the prevention.
        """
        return _registry_files(Path(str((spec or {}).get("registry", ""))))

    # -- adapters ----------------------------------------------------------

    def _adapter(self, name: str) -> Any:
        factory = self._adapter_factory
        if factory is None:
            from mival.adapters import get_adapter  # lazy: this is the backend import

            factory = get_adapter
        return factory(name)

    # -- run ---------------------------------------------------------------

    def run(self, ctx: StageContext) -> StageResult:
        spec = ModelsSpec.from_mapping(ctx.spec)
        cards = _load_cards(spec.registry)
        warnings: List[str] = []
        if not spec.cohort_sources:
            warnings.append(
                "models.cohort_sources is empty, so the contamination gate (spec §3.5) "
                "cannot fire; contamination=false here means 'not checked'"
            )

        wanted = {arm.model_id for arm in spec.arms}
        for model_id in sorted(wanted):
            if model_id not in cards:
                raise ValueError(
                    f"no ModelCard for {model_id!r} in {spec.registry}; cards found: "
                    f"{sorted(cards)}"
                )
        for arm in spec.arms:
            card = cards[arm.model_id][0]
            if arm.training_mode not in card.training_modes_supported:
                raise ValueError(
                    f"{arm.describe()}: the ModelCard declares training_modes_supported="
                    f"{list(card.training_modes_supported)}, which does not include "
                    f"{arm.training_mode!r}"
                )

        covariate_columns = sorted({
            COVARIATE_PREFIX + name for arm in spec.arms for name in _covariates(cards[arm.model_id][0])
        })
        records, n_in = self._collect(ctx, wanted, spec, covariate_columns)

        outputs: List[Path] = []
        run_keys: List[RunKey] = []
        log_entries: List[Dict[str, Any]] = []
        reports: List[ContaminationReport] = []
        counts = {"in": n_in, "out": 0, "test_contacts": 0}

        for arm in spec.arms:
            card, card_path = cards[arm.model_id]
            report = check_contamination(card.pretraining_corpora, spec.cohort_sources)
            reports.append(report)
            if report.flag:
                warnings.append(
                    f"{arm.describe()}: contaminated — pretraining corpora "
                    f"{list(report.overlapping_corpora)} overlap the evaluation cohort "
                    "(spec §3.5 gate 4 marks, it does not block)"
                )

            arm_records = [record for record in records if record.model_id == arm.model_id]
            if not arm_records:
                raise ValueError(
                    f"{arm.describe()}: preprocess_index holds no rows for model "
                    f"{arm.model_id!r}; the preprocess run supplied was compiled for "
                    f"{sorted({record.model_id for record in records})}"
                )
            adapter = self._adapter(card.adapter)
            for group_key, group in _group_by_recipe(arm_records):
                entry, group_outputs, group_keys, written, contacts = self._run_group(
                    ctx=ctx,
                    spec=spec,
                    arm=arm,
                    card=card,
                    card_path=card_path,
                    adapter=adapter,
                    records=group,
                    recipe_id=group_key[0],
                    perturbation_id=group_key[1],
                    contamination=report,
                )
                outputs.extend(group_outputs)
                run_keys.extend(group_keys)
                log_entries.append(entry)
                counts["out"] += written
                counts["test_contacts"] += contacts

        outputs.append(_write_train_log(ctx.layout.artifact(TRAIN_LOG), log_entries))
        merged = ContaminationReport.merge(reports)
        return StageResult(
            outputs=outputs,
            counts=counts,
            contamination=merged.to_dict(),
            warnings=warnings,
            run_keys=run_keys,
        )

    # -- inputs ------------------------------------------------------------

    def _collect(
        self, ctx: StageContext, wanted: Iterable[str], spec: ModelsSpec,
        covariate_columns: Sequence[str] = (),
    ) -> Tuple[List[Record], int]:
        """Join preprocess_index with the split and the labels, dropping the rest.

        A record is dropped for exactly one reason even when several apply, so
        that the ledger sums to the STARD flow (spec §3.6) instead of double
        counting one record under two codes.
        """
        index_path = ctx.input_path("preprocess_index")
        index = read_table(index_path)
        _require_columns(index, PREPROCESS_INDEX_COLUMNS, "preprocess_index")
        # Preprocess writes tensor_path relative to its own run directory
        # (``artifacts/tensors/...``), and its index sits in that run's
        # ``artifacts/``, so a relative path is anchored there — never on the
        # working directory the stage happens to be launched from.
        tensor_root = index_path.parent.parent
        splits = read_table(ctx.input_path("cohort_split"))
        _require_columns(splits, COHORT_SPLIT_COLUMNS, "cohort_split")

        split_rows = splits.to_dict("records")
        # Gate 3 (spec §3.5) against the split table itself, before the join.
        # Reading the raw rows rather than a person -> split map is what makes
        # the check able to fail: a map silently resolves a person assigned to
        # both splits by keeping whichever row came last, which is precisely
        # the defect the gate exists to catch.
        check_leakage(
            [row["person_id"] for row in split_rows if str(row["split"]) == DEV_SPLIT],
            [row["person_id"] for row in split_rows if str(row["split"]) == TEST_SPLIT],
            "cohort_split",
        )
        split_by_person: Dict[str, Tuple[str, Optional[int]]] = {}
        for row in split_rows:
            split_by_person[str(row["person_id"])] = (
                str(row["split"]),
                _as_int(row.get("fold")),
            )

        labels = self._label_source(ctx, index)
        # Every arm's label column must be present for a record to be kept, so
        # that all arms are compared on one cohort rather than on per-arm
        # cohorts whose denominators silently differ.
        needed = sorted({column for arm in spec.arms for column in arm.required_label_columns}
                        | set(covariate_columns))
        # label_value is optional on a cohort, so a regression arm run against a
        # cohort that has none would otherwise exclude every record one by one
        # and surface as "no dev records", which names the wrong problem.
        # The same holds for every continuous label a regression or survival
        # arm reads (label_qrs, label_time_days, ...).
        for column in sorted({
            column for arm in spec.arms if arm.is_regression or arm.is_survival
            for column in arm.required_label_columns
        }):
            if all(row.get(column) is None for row in labels.values()):
                raise ValueError(
                    f"a regression arm needs {column!r}, but no record in "
                    "the label source carries one: the column is absent or every value is "
                    "null. Supply a cohort_index whose "
                    f"{column!r} column is populated (spec §4.1)."
                )

        wanted = set(wanted)
        rows = [row for row in index.to_dict("records") if str(row["model_id"]) in wanted]
        kept: List[Record] = []
        for row in rows:
            image_id = str(row["image_occurrence_id"])
            person_id = str(row["person_id"])
            assignment = split_by_person.get(person_id)
            if assignment is None or assignment[0] not in (DEV_SPLIT, TEST_SPLIT):
                ctx.ledger.record(
                    image_id,
                    person_id,
                    "split_unassigned",
                    "person_id is absent from cohort_split"
                    if assignment is None
                    else f"split is {assignment[0]!r}, expected {DEV_SPLIT!r} or {TEST_SPLIT!r}",
                )
                continue
            record_labels = labels.get(image_id, {})
            missing = [column for column in needed if record_labels.get(column) is None]
            if missing:
                ctx.ledger.record(image_id, person_id, "label_missing", ",".join(missing))
                continue
            tensor_path = str(_absolute(row["tensor_path"], tensor_root))
            if not Path(tensor_path).exists():
                ctx.ledger.record(image_id, person_id, "tensor_missing", tensor_path)
                continue
            split, fold = assignment
            kept.append(
                Record(
                    image_occurrence_id=image_id,
                    person_id=person_id,
                    model_id=str(row["model_id"]),
                    recipe_id=str(row["recipe_id"]),
                    perturbation_id=str(row["perturbation_id"]),
                    tensor_path=tensor_path,
                    split=split,
                    fold=fold,
                    labels={column: record_labels.get(column)
                            for column in tuple(LABEL_COLUMNS) + tuple(covariate_columns)},
                )
            )
        return kept, len(rows)

    def _label_source(self, ctx: StageContext, index: Any) -> Dict[str, Dict[str, Optional[float]]]:
        """Labels per ``image_occurrence_id``.

        Preprocess is free to carry the labels forward on its index; when it
        does not, they come from the retrieve stage's ``cohort_index``
        (spec §4.1), passed as an extra ``--input``. It is not in
        ``required_inputs`` because that tuple is a pinned contract, but the
        stage cannot emit ``PREDICTION_COLUMNS`` without labels, so an absent
        source is a configuration error and says so.
        """
        if all(column in index.columns for column in LABEL_COLUMNS):
            return _labels_from(index)
        if "cohort_index" in ctx.inputs:
            frame = read_table(ctx.input_path("cohort_index"))
            _require_columns(
                frame, ("image_occurrence_id",) + BINARY_LABEL_COLUMNS, "cohort_index"
            )
            return _labels_from(frame)
        raise KeyError(
            "labels are unavailable: preprocess_index carries no label columns and no "
            "'cohort_index' input was supplied. Pass --input cohort_index=<path to the "
            "retrieve stage's cohort_index.parquet> (spec §4.1)."
        )

    # -- one (arm, recipe, perturbation) -----------------------------------

    def _run_group(
        self,
        ctx: StageContext,
        spec: ModelsSpec,
        arm: Arm,
        card: ModelCard,
        card_path: Path,
        adapter: Any,
        records: Sequence[Record],
        recipe_id: str,
        perturbation_id: str,
        contamination: ContaminationReport,
    ) -> Tuple[Dict[str, Any], List[Path], List[RunKey], int, int]:
        context = f"{arm.describe()} recipe={recipe_id} perturbation={perturbation_id}"
        dev = [record for record in records if record.split == DEV_SPLIT]
        test_set = _TestSet(
            [record for record in records if record.split == TEST_SPLIT], context
        )
        if not dev:
            raise ValueError(
                f"{context}: no dev records. Thresholds are refit on dev and may never be "
                "chosen on test (spec §4.4), so a run without dev has no operating point."
            )

        # Gate 3 (spec §3.5), before anything is fitted or inferred. Everyone
        # who influences the model or its operating point counts as fitting
        # data — for inference_only that is the threshold refit set.
        check_leakage([record.person_id for record in dev], test_set.person_ids, context)

        loader = self._tensor_loader
        training: Dict[str, Any] = {"mode": arm.training_mode}
        if arm.training_mode == "inference_only":
            # inference_only scores straight through the published card, so
            # the card's declared output has to match what the arm expects;
            # a trained arm's head is fitted to the arm's objective instead
            # and needs no such check.
            kind = card.output.get("type", "logits")
            allowed = _INFERENCE_OUTPUT_TYPES[arm.kind]
            if kind not in allowed:
                raise ValueError(
                    f"{context}: label_def {arm.label_def!r} needs output.type "
                    f"{'/'.join(allowed)}, but the card declares {kind!r}"
                )
            handle = _adapter_method(adapter, "load", card)(card)
            dev_probs = _predict(adapter, handle, dev, loader, spec.batch_size, card, _column(arm, card))
            final_handle = handle
            training["weights"] = [dict(weight) for weight in card.weights]
        else:
            final_handle, dev_probs, detail = self._fit_with_internal_cv(
                ctx=ctx,
                spec=spec,
                arm=arm,
                card=card,
                adapter=adapter,
                dev=dev,
                context=context,
            )
            training.update(detail)

        if arm.is_regression or arm.is_survival:
            # A regression or survival arm has no operating point to refit:
            # thresholds are a classification concept.
            thresholds = {}
        else:
            scores = _scores_of(dev, dev_probs, arm.label_column)
            thresholds = fit_thresholds(THRESHOLD_POLICIES, scores, card.threshold, card.model_id)

        test_records = test_set.take()
        test_probs = (
            _predict(adapter, final_handle, test_records, loader, spec.batch_size, card,
                     _column(arm, card) if arm.training_mode == "inference_only" else None)
            if test_records
            else np.empty(0, dtype=np.float64)
        )

        outputs: List[Path] = []
        run_keys: List[RunKey] = []
        written = 0
        for fold, subset, probs in _partition_by_fold(dev, dev_probs) + [
            (None, list(test_records), test_probs)
        ]:
            if not subset:
                continue
            split = subset[0].split
            key = RunKey(
                site=ctx.site,
                model_id=arm.model_id,
                training_mode=arm.training_mode,
                recipe_id=recipe_id,
                perturbation_id=perturbation_id,
                label_def=arm.label_def,
                split=split,
                fold=fold,
            )
            path = ctx.layout.artifact(PREDICTIONS_DIR, key.to_string() + ".parquet")
            write_table(_rows(subset, probs, arm, ctx), path, PREDICTION_COLUMNS)
            outputs.append(path)
            run_keys.append(key)
            written += len(subset)

        head_path = _persist_heads(ctx, final_handle, arm, recipe_id, perturbation_id)
        if head_path is not None:
            outputs.append(head_path)

        entry = {
            "model_id": arm.model_id,
            "training_mode": arm.training_mode,
            "label_def": arm.label_def,
            "fitted_head": None if head_path is None else str(
                head_path.relative_to(ctx.layout.run_dir)
            ),
            "recipe_id": recipe_id,
            "perturbation_id": perturbation_id,
            "site": ctx.site,
            "seed": ctx.seed,
            "model_card": {
                "path": str(card_path),
                "sha256": sha256_file(card_path),
                "adapter": card.adapter,
                "feature_layer": card.feature_layer,
                "ensemble": dict(card.ensemble),
                # Which interpreter and framework build the card was written
                # against; a result is only reproducible against that.
                "runtime": dict(card.runtime),
            },
            "training": training,
            "thresholds": {name: fit.to_dict() for name, fit in thresholds.items()},
            "primary_threshold_policy": None if arm.is_regression or arm.is_survival else PRIMARY_THRESHOLD_POLICY,
            "contamination": contamination.to_dict(),
            # Spec §4.4: the fact that test was contacted is itself a recorded
            # result. `counts.test_contacts` carries it into the manifest.
            "test": {"contacts": test_set.touches, "n": test_set.size},
            "counts": {"dev": len(dev), "test": test_set.size},
            "run_keys": [key.to_string() for key in run_keys],
        }
        return entry, outputs, run_keys, written, test_set.touches

    def _fit_with_internal_cv(
        self,
        ctx: StageContext,
        spec: ModelsSpec,
        arm: Arm,
        card: ModelCard,
        adapter: Any,
        dev: Sequence[Record],
        context: str,
    ) -> Tuple[Any, np.ndarray, Dict[str, Any]]:
        """Select hyperparameters inside dev, then fit the model that meets test.

        Returns the final handle, the out-of-fold dev probabilities, and the
        record spec §4.4 requires. Out-of-fold is what makes the threshold
        refit honest: every dev probability comes from a model that did not see
        that record.
        """
        loader = self._tensor_loader
        folds = sorted({record.fold for record in dev if record.fold is not None})
        if len(folds) != len({record.fold for record in dev}) or len(folds) < 2:
            raise ValueError(
                f"{context}: training_mode {arm.training_mode!r} needs the dev k-fold "
                "assignment frozen in cohort_split (spec §2.4), but the dev records carry "
                f"folds {sorted({record.fold for record in dev}, key=_fold_sort_key)}"
            )
        load = _adapter_method(adapter, "load", card)
        if arm.training_mode == "partial_unfreeze":
            groups = _adapter_method(adapter, "trainable_groups", card)(load(card))
            unknown = [name for name in arm.unfreeze_groups if name not in groups]
            if unknown:
                raise ValueError(
                    f"{context}: unfreeze_groups {unknown} are not trainable groups of "
                    f"{card.model_id!r}; the adapter reports {list(groups)}"
                )

        selection: List[Dict[str, Any]] = []
        best: Optional[Tuple[float, int, np.ndarray]] = None
        for candidate_index, candidate in enumerate(arm.hparam_grid):
            hparams = self._hparams(arm, candidate, ctx)
            oof = np.full(len(dev), np.nan, dtype=np.float64)
            for fold in folds:
                train = [record for record in dev if record.fold != fold]
                positions = [
                    position for position, record in enumerate(dev) if record.fold == fold
                ]
                held = [dev[position] for position in positions]
                check_leakage(
                    [record.person_id for record in train],
                    [record.person_id for record in held],
                    f"{context} inner fold {fold}",
                )
                # A fresh handle per fold: reusing a fitted one would carry the
                # previous fold's weights into this fold's estimate.
                fitted = _call_fit(
                    adapter,
                    load(card),
                    arm,
                    hparams,
                    _fit_payload(train, held, arm, loader, spec.batch_size),
                    card,
                )
                probs = _predict(adapter, fitted, held, loader, spec.batch_size, card)
                for slot, position in enumerate(positions):
                    oof[position] = probs[slot]
            score = _selection_score(oof, [record.labels[arm.label_column] for record in dev], arm)
            selection.append({"hparams": dict(hparams), _selection_metric(arm): score})
            # Strict improvement only, so the first candidate of the
            # deterministically ordered grid wins a tie.
            if best is None or score > best[0]:
                best = (score, candidate_index, oof)
        assert best is not None  # the grid always holds at least the empty candidate

        chosen = self._hparams(arm, arm.hparam_grid[best[1]], ctx)
        early_stopping_fold = arm.early_stopping_fold
        if arm.early_stopping_auto:
            # Early stopping has to watch data the fit is not learning from,
            # and test is out of bounds, so the last dev fold is reserved for
            # it. Set `early_stopping_fold: null` in the spec to fit on all of
            # dev instead — correct for adapters that need no stopping rule.
            early_stopping_fold = folds[-1]
        if early_stopping_fold is not None and early_stopping_fold not in folds:
            raise ValueError(
                f"{context}: early_stopping_fold {early_stopping_fold} is not one of the "
                f"dev folds {folds}"
            )
        train = [record for record in dev if record.fold != early_stopping_fold]
        validation = [record for record in dev if record.fold == early_stopping_fold]
        final_handle = _call_fit(
            adapter,
            load(card),
            arm,
            chosen,
            _fit_payload(train, validation, arm, loader, spec.batch_size),
            card,
        )
        detail = {
            "weights": [dict(weight) for weight in card.weights],
            "feature_layer": card.feature_layer,
            "unfreeze_groups": list(arm.unfreeze_groups),
            "hparams": dict(chosen),
            # What the adapter actually did — resolved hyperparameters, the
            # epoch early stopping chose, the trainable parameter count. Spec
            # §4.4 requires the full hyperparameters on record, and `chosen`
            # above holds only what the study asked for, not the defaults the
            # adapter filled in.
            "fit_record": getattr(final_handle, "fit_record", None),
            "internal_cv": {
                "folds": folds,
                "selection_metric": _selection_metric(arm),
                "selection_split": DEV_SPLIT,
                "candidates": selection,
                "selected": dict(chosen),
                "early_stopping_fold": early_stopping_fold,
                "final_fit_n": len(train),
            },
        }
        return final_handle, best[2], detail

    def _hparams(self, arm: Arm, candidate: Mapping[str, Any], ctx: StageContext) -> Dict[str, Any]:
        """Candidate hyperparameters plus the run's seed and mode configuration.

        The seed travels with the hyperparameters because the adapter owns the
        fit and therefore owns every source of randomness in it; the manifest
        records the same seed.
        """
        hparams = dict(candidate)
        hparams.setdefault("seed", ctx.seed)
        if arm.training_mode == "partial_unfreeze":
            hparams.setdefault("unfreeze_groups", list(arm.unfreeze_groups))
        return hparams


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


#: Where fitted heads are written, relative to ``ctx.layout.artifacts_dir``.
HEADS_DIR = "heads"


def _persist_heads(
    ctx: StageContext, handle: Any, arm: "Arm", recipe_id: str, perturbation_id: str
) -> Optional[Path]:
    """Write the fitted linear head(s) beside the predictions, or return None.

    A head fitted here otherwise exists only in this process's memory, and
    every downstream question that needs the model back — reproducing a
    prediction, drawing an attribution overlay for a probed arm in stage 6 —
    becomes impossible for exactly the arms the study trained. It is four small
    arrays per head, so the alternative to storing it is not a saving.

    ``inference_only`` arms have no fitted head and produce no file.
    """
    heads = getattr(handle, "heads", None)
    if not heads:
        single = getattr(handle, "head", None)
        heads = [single] if single is not None else []
    if not heads:
        return None
    payload: Dict[str, Any] = {}
    for index, head in enumerate(heads):
        for field_name in ("weights", "bias", "mean", "scale"):
            payload[f"head{index}.{field_name}"] = np.asarray(getattr(head, field_name))
    name = "~".join(
        [
            f"model_id={arm.model_id}",
            f"training_mode={arm.training_mode}",
            f"recipe_id={recipe_id}",
            f"perturbation_id={perturbation_id}",
            f"label_def={arm.label_def}",
        ]
    )
    path = ctx.layout.artifact(HEADS_DIR, name + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **payload)
    return path


def _registry_files(directory: Path) -> Dict[str, Path]:
    """Card paths keyed for the config_hash, or nothing if the path is unusable.

    A missing registry is not reported here: ``run`` raises a much better
    error, and ``prepare`` must not fail before the stage has had its say.
    """
    root = Path(directory)
    if not root.is_dir():
        return {}
    return {f"modelcard:{path.stem}": path for path in sorted(root.glob("*.json"))}


def _load_cards(directory: Path) -> Dict[str, Tuple[ModelCard, Path]]:
    """Every card in the registry, with its file path for the checksum record."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"models.registry {root} is not a directory")
    cards: Dict[str, Tuple[ModelCard, Path]] = {}
    for path in sorted(root.glob("*.json")):
        card = load_card(path)
        cards[card.model_id] = (card, path)
    if not cards:
        raise ValueError(f"models.registry {root} holds no ModelCard JSON files")
    return cards


def _column(arm: "Arm", card: ModelCard) -> Optional[int]:
    """The decoded interval an inference_only interval arm reads, else None."""
    if card.output.get("type") != "segmentation_mask":
        return None
    from mival.decode.intervals import INTERVAL_ORDER

    return INTERVAL_ORDER.index(arm.label_def)


def _require_columns(frame: Any, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns {missing}; got {list(frame.columns)}")


#: label_value is continuous; every other label column is a 0/1 outcome.
_REGRESSION_LABEL_COLUMN = "label_" + REGRESSION_LABEL_DEF


def _labels_from(frame: Any) -> Dict[str, Dict[str, Optional[float]]]:
    labels: Dict[str, Dict[str, Optional[float]]] = {}
    covariates = [column for column in frame.columns if str(column).startswith(COVARIATE_PREFIX)]
    for row in frame.to_dict("records"):
        values = {
            column: (_as_float if column in CONTINUOUS_LABEL_COLUMNS else _as_int)(row.get(column))
            for column in LABEL_COLUMNS
        }
        values.update({column: _as_float(row.get(column)) for column in covariates})
        labels[str(row["image_occurrence_id"])] = values
    return labels


def _covariates(card: ModelCard) -> Tuple[str, ...]:
    """Non-waveform inputs the card's module takes after the ECG (L2-3), in order."""
    return tuple(str(name) for name in (card.raw.get("x-mival", {}).get("covariates") or ()))


def _absolute(path: Any, root: Path) -> Path:
    """``path`` as written, or under ``root`` when it is relative."""
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else root / candidate


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return float(value)


def _fold_sort_key(fold: Optional[int]) -> Tuple[int, int]:
    return (1, 0) if fold is None else (0, fold)


def _group_by_recipe(
    records: Sequence[Record],
) -> List[Tuple[Tuple[str, str], List[Record]]]:
    """Split an arm's records into one group per (recipe_id, perturbation_id).

    Those two are run_key axes (spec §3.4): each combination is its own
    inference run and its own prediction file.
    """
    groups: Dict[Tuple[str, str], List[Record]] = {}
    for record in records:
        groups.setdefault((record.recipe_id, record.perturbation_id), []).append(record)
    return [(key, groups[key]) for key in sorted(groups)]


def _partition_by_fold(
    records: Sequence[Record], probs: np.ndarray
) -> List[Tuple[Optional[int], List[Record], np.ndarray]]:
    """One bucket per fold value, because ``fold`` is a run_key axis."""
    buckets: Dict[Optional[int], List[int]] = {}
    for position, record in enumerate(records):
        buckets.setdefault(record.fold, []).append(position)
    return [
        (fold, [records[position] for position in positions], probs[np.array(positions)])
        for fold, positions in sorted(buckets.items(), key=lambda item: _fold_sort_key(item[0]))
    ]


def _selection_metric(arm: Arm) -> str:
    return SELECTION_METRIC_REGRESSION if arm.is_regression else SELECTION_METRIC


def _selection_score(predicted: np.ndarray, labels: Sequence[Optional[float]], arm: Arm) -> float:
    """Higher is better either way: AUROC for classification, negative MAE for regression."""
    if arm.is_regression:
        return -float(np.mean(np.abs(predicted - np.asarray(labels, dtype=np.float64))))
    return auroc(predicted, labels)


def _scores_of(records: Sequence[Record], probs: np.ndarray, label_column: str) -> SplitScores:
    """Tag probabilities with the split their records actually came from.

    Derived from the records rather than passed as a literal so that the
    dev-only check in ``mival.threshold`` tests real provenance: a bug that
    routed test records here would trip it.
    """
    splits = {record.split for record in records}
    if len(splits) != 1:
        raise ValueError(f"scores mix splits {sorted(splits)}; one split per fit")
    return SplitScores.of(
        splits.pop(), probs, [record.labels[label_column] for record in records]
    )


def _rows(
    records: Sequence[Record], probs: np.ndarray, arm: Arm, ctx: StageContext
) -> List[Dict[str, Any]]:
    continuous = arm.is_regression or arm.is_survival
    logits = [None] * len(records) if continuous else _logits(probs)
    rows: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        value = float(probs[position])
        row: Dict[str, Any] = {
            "image_occurrence_id": record.image_occurrence_id,
            "person_id": record.person_id,
            "split": record.split,
            "fold": record.fold,
            "prob": None if continuous else value,
            "logit": logits[position],
            "pred_value": value if continuous else None,
            "model_id": arm.model_id,
            "training_mode": arm.training_mode,
            "recipe_id": record.recipe_id,
            "perturbation_id": record.perturbation_id,
            "seed": ctx.seed,
            "site": ctx.site,
        }
        for column in LABEL_COLUMNS:
            row[column] = record.labels.get(column)
        rows.append(row)
    return rows


def _logits(probs: np.ndarray) -> List[Optional[float]]:
    """log(p / (1-p)), null where it is not finite.

    Adapters return a probability (``Adapter.forward``), so the logit is
    derived here. A probability of exactly 0 or 1 has an infinite logit; that
    is written as null rather than as an infinity, because a null is a value
    every downstream aggregation already has to handle and an infinity is one
    that silently poisons a mean.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log(probs) - np.log1p(-probs)
    return [float(value) if np.isfinite(value) else None for value in values]


def _write_train_log(path: Path, entries: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, default=_jsonable) + "\n")
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"train_log entry holds a non-serialisable {type(value).__name__}")
