"""EXCLUDED 2026-09-22: this stage is unregistered in mival.pipeline.stage (see
_UNBUILT) and `mival run misclassify` refuses to run it. The code is kept as
written for when per-model misclassification review is settled.

Misclassification stage (spec §4.6). Implemented by Plan 7.

This is the implementation of a **medical algorithmic audit** (Liu, Glocker,
McCradden, Ghassemi, Denniston, Oakden-Rayner, *Lancet Digital Health* 2022).
It is deliberately a separate stage from evaluation, because the two have
opposite epistemic status: evaluation *estimates* performance for questions
fixed in advance (confirmatory), while this stage *discovers* failure modes
nobody pre-specified (exploratory). Mixing them invites reading the performance
table first and then reporting whichever subgroup looks interesting, which is
exactly the contamination the separation prevents.

Spec §4.6 states what this stage is: **not a module that performs a particular
comparison, but a query engine over run_key space.** Everything follows from
that. A selector names run_key *patterns*, never a model, so the same code
selects disagreement between models, between training modes, between
perturbation levels, between label definitions and between sites. If an axis
name ever appears inside a selector, this design has failed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mival.pipeline.runkey import AXES, RunKey
from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import read_table, write_table
from mival.stages._predictions import (
    ThresholdPolicy,
    arm_of,
    fit_operating_thresholds,
    load_predictions,
    optional_int,
    recorded_thresholds_and_contamination,
)
from mival.stages.models import LABEL_COLUMNS, REGRESSION_LABEL_DEF

#: Output artifacts, relative to ``ctx.layout.artifacts_dir``.
CASES = "cases.parquet"
REVIEW_TEMPLATE = "review_template.csv"
REVIEW_UNMATCHED = "review_unmatched.csv"
SELECTOR_LOG = "selectors.json"
FIGURE_DIR = "figures/cases"

#: Spec §4.6, the four selectors. Each is a query over run_key space.
SELECTOR_TYPES = ("error", "boundary", "contrast", "instability")

#: The guaranteed prefix of ``cases.parquet``. Context columns joined from the
#: study's own optional inputs are appended after these; the prefix is what
#: downstream code and the review loop may rely on.
CASE_COLUMNS: Tuple[str, ...] = (
    "case_id",
    "selector",
    "selector_type",
    "rank",
    "score",
    "score_name",
    "image_occurrence_id",
    "person_id",
) + AXES + (
    "prob",
    "logit",
    "threshold",
    "outcome_class",
) + LABEL_COLUMNS + (
    "peer_run_key",
    "peer_prob",
    "peer_outcome_class",
    "tensor_path",
    "figure_path",
    "attribution_status",
    "verdict",
    "note",
)

#: The columns a human fills in. Kept to two so that the review file stays a
#: file a clinician will actually complete.
REVIEW_COLUMNS = ("case_id", "verdict", "note")

#: Spec §4.6's pre-specified failure modes (FMEA), fixed before the data is
#: seen. These are the *default*; a study declares its own with
#: ``misclassify.verdicts``. Free text would make the review unaggregatable,
#: and a vocabulary hardcoded here would not be a pre-specification by the
#: study team — it would be a pre-specification by this file.
DEFAULT_VERDICTS: Tuple[str, ...] = (
    "label_error",
    "stemi_mimic",
    "reperfused",
    "signal_quality",
    "lead_misplacement",
    "true_model_error",
    "unclear",
)

#: Written when a case has no verdict yet. Distinguishable from a reviewer who
#: looked and could not decide, which is ``unclear``.
UNREVIEWED = "unreviewed"

#: ``contrast`` criteria. Spec §4.6 writes this as ``label_flip``, which is
#: ambiguous on the very axis the spec most wants it for: across ``label_def``,
#: "label" can mean the model's binarised output or the reference standard, and
#: the two select different cases. The criteria are therefore named for which
#: one is meant, and ``label_flip`` is accepted as the spec's word for the
#: first.
CONTRAST_CRITERIA = ("prediction_flip", "truth_flip", "outcome_flip", "delta_prob")
_CRITERION_ALIASES = {"label_flip": "prediction_flip"}

#: ``instability`` may vary over run_key axes or over ``seed``, which is a
#: prediction column rather than an axis: two runs that differ only in seed
#: write the same run_key into different run directories (spec §3.4).
_SEED = "seed"

#: The canonical string for "this run had no fold" — the same sentinel the
#: run_key uses, so a pattern is written the way a run_key filename reads.
_FOLD_NONE = "na"

#: Selector names become part of ``case_id`` and of a figure filename, so they
#: take the same charset as every other path component in this pipeline.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: ``case_id`` = ``<selector>~<image_occurrence_id>``. The separator is outside
#: the identifier charset, so the id round-trips. It is a readable string and
#: not a hash because a human reads it in ``review.csv``; it excludes ``k``,
#: ``tau`` and the config_hash so that tightening a selector does not orphan
#: the review already done on the cases that survive.
_CASE_SEP = "~"

#: Optional input names. None of them is required: a study that has not built
#: the Profile stage yet still gets cases, with the corresponding companion
#: fields empty rather than a failed run.
PREPROCESS_INDEX_INPUT = "preprocess_index"
TRAIN_LOG_INPUT = "train_log"
REVIEW_INPUT = "review"
#: Any further input is joined on ``image_occurrence_id`` or ``person_id`` and
#: its columns become companion context (spec §4.6: acquisition metadata,
#: cohort/index context). Naming them here would hardcode one study's schema.
_KNOWN_INPUTS = frozenset(
    {"predictions", PREPROCESS_INDEX_INPUT, TRAIN_LOG_INPUT, REVIEW_INPUT}
)
#: ModelCards enter the config_hash under this prefix (see ``config_inputs``).
#: They are not context tables and must not be joined onto the cases.
_CARD_PREFIX = "card:"


class SelectorError(ValueError):
    """Raised when a selector declaration cannot be honoured as written."""


# ---------------------------------------------------------------------------
# Patterns over run_key space
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pattern:
    """A partial coordinate: some axes pinned, the rest free.

    This is the whole reason the stage generalises. ``{"model_id": "a"}`` and
    ``{"label_def": "primary"}`` are the same kind of object, so the selector
    that contrasts two of them never learns which axis it is contrasting.
    """

    values: Mapping[str, str]

    @classmethod
    def of(cls, body: Any, where: str) -> "Pattern":
        if not isinstance(body, Mapping):
            raise SelectorError(
                f"{where}: a run_key pattern must be a mapping of axis to value, got "
                f"{type(body).__name__}"
            )
        unknown = sorted(set(body) - set(AXES))
        if unknown:
            raise SelectorError(
                f"{where}: {unknown} are not run_key axes; the axes are {list(AXES)}"
            )
        return cls({str(axis): _canonical(axis, value) for axis, value in body.items()})

    def select(self, frame):
        for axis, value in self.values.items():
            frame = frame[frame[axis] == value]
        return frame

    def describe(self) -> str:
        if not self.values:
            return "(every run_key)"
        return ", ".join(f"{axis}={self.values[axis]}" for axis in AXES if axis in self.values)


def _canonical(axis: str, value: Any) -> str:
    """One string form for an axis value, whatever a YAML file wrote.

    ``fold: 1`` and ``fold: "1"`` must select the same rows, and a parquet
    column that pandas widened to float must compare equal to both.
    """
    if axis == "fold":
        if value is None:
            return _FOLD_NONE
        text = str(value)
        if text in ("", "na", "nan", "None", "NaT", "<NA>"):
            return _FOLD_NONE
        return str(optional_int(text))
    return str(value)


def _normalise_axes(predictions):
    """A copy whose axis columns are canonical strings.

    Pattern matching is then plain string equality on every axis, including
    ``fold``, whose dtype depends on whether any run in the set was
    cross-validated.
    """
    frame = predictions.copy()
    for axis in AXES:
        frame[axis] = [_canonical(axis, value) for value in frame[axis]]
    return frame


def _run_key_of(row: Mapping[str, Any]) -> RunKey:
    values: Dict[str, Any] = {axis: str(row[axis]) for axis in AXES}
    values["fold"] = None if values["fold"] == _FOLD_NONE else int(values["fold"])
    return RunKey.from_dict(values)


def _require_one_row_per_record(frame, where: str, free_axes: Sequence[str] = ()):
    """A pattern must resolve to at most one prediction per record.

    Without this, a study that forgot to pin ``perturbation_id`` would report
    perturbation sensitivity as fold instability, or would rank the same record
    several times under one selector. The error names the axes that actually
    varied, because "your pattern is ambiguous" is not actionable and "you left
    perturbation_id free and it takes 4 values" is.
    """
    if len(frame) == 0:
        return frame
    counts = frame.groupby("image_occurrence_id").size()
    duplicated = counts[counts > 1]
    if duplicated.empty:
        return frame
    sample = frame[frame["image_occurrence_id"] == duplicated.index[0]]
    varying = [axis for axis in AXES if sample[axis].nunique() > 1]
    permitted = set(free_axes)
    offending = [axis for axis in varying if axis not in permitted]
    if not offending:
        return frame
    detail = ", ".join(
        f"{axis}={sorted(str(value) for value in sample[axis].unique())}" for axis in offending
    )
    allowed = f" (only {list(free_axes)} may vary)" if free_axes else ""
    raise SelectorError(
        f"{where}: the pattern matches {int(duplicated.iloc[0])} predictions for record "
        f"{duplicated.index[0]!r}{allowed}; these axes are left free and vary: {detail}. "
        "Pin them in the pattern."
    )


# ---------------------------------------------------------------------------
# Selector declarations
# ---------------------------------------------------------------------------


@dataclass
class Selector:
    """One declared query. Validated at load time, not at selection time."""

    name: str
    type: str
    pattern: Optional[Pattern] = None
    peer: Optional[Pattern] = None
    k: Optional[int] = None
    epsilon: Optional[float] = None
    tau: Optional[float] = None
    criterion: str = "prediction_flip"
    over: Tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, body: Any, index: int) -> "Selector":
        where = f"misclassify.selectors[{index}]"
        if not isinstance(body, Mapping):
            raise SelectorError(f"{where}: must be a mapping, got {type(body).__name__}")
        kind = str(body.get("type", "")).strip()
        if kind not in SELECTOR_TYPES:
            raise SelectorError(
                f"{where}: type {kind!r} is not one of {list(SELECTOR_TYPES)} (spec §4.6)"
            )
        name = str(body.get("name", f"{kind}{index}"))
        if not _NAME_RE.match(name):
            raise SelectorError(
                f"{where}: name {name!r} is not a valid identifier; it becomes part of a "
                "case_id and of a figure filename"
            )
        where = f"misclassify.selectors[{index}] ({name})"

        k = None if body.get("k") is None else int(body["k"])
        if k is not None and k <= 0:
            raise SelectorError(f"{where}: k must be positive, got {k}")

        if kind == "contrast":
            for required in ("a", "b"):
                if required not in body:
                    raise SelectorError(
                        f"{where}: a contrast selector needs patterns 'a' and 'b'"
                    )
            criterion = str(body.get("criterion", "prediction_flip"))
            criterion = _CRITERION_ALIASES.get(criterion, criterion)
            if criterion not in CONTRAST_CRITERIA:
                raise SelectorError(
                    f"{where}: criterion {body.get('criterion')!r} is not one of "
                    f"{list(CONTRAST_CRITERIA)}"
                )
            tau = None if body.get("tau") is None else float(body["tau"])
            if criterion == "delta_prob" and tau is None:
                raise SelectorError(
                    f"{where}: criterion 'delta_prob' needs 'tau' — without it the selector "
                    "would return every record that the two patterns both cover"
                )
            return cls(
                name=name,
                type=kind,
                pattern=Pattern.of(body["a"], f"{where}.a"),
                peer=Pattern.of(body["b"], f"{where}.b"),
                k=k,
                tau=tau,
                criterion=criterion,
            )

        pattern_body = body.get("pattern", body.get("reference"))
        if pattern_body is None:
            raise SelectorError(
                f"{where}: a {kind} selector needs a 'pattern' (spec §4.6 calls it the "
                "reference run_key)"
            )
        pattern = Pattern.of(pattern_body, f"{where}.pattern")

        if kind == "boundary":
            if body.get("epsilon") is None:
                raise SelectorError(
                    f"{where}: a boundary selector needs 'epsilon' — the width of the band "
                    "around the operating threshold"
                )
            epsilon = float(body["epsilon"])
            if epsilon <= 0:
                raise SelectorError(f"{where}: epsilon must be positive, got {epsilon}")
            return cls(name=name, type=kind, pattern=pattern, k=k, epsilon=epsilon)

        if kind == "instability":
            over = tuple(str(axis) for axis in body.get("over", ()) or ())
            if not over:
                raise SelectorError(
                    f"{where}: an instability selector needs 'over' — the axes the repeated "
                    f"predictions may differ in. Choose from {list(AXES)} or {_SEED!r}."
                )
            unknown = [axis for axis in over if axis not in AXES and axis != _SEED]
            if unknown:
                raise SelectorError(
                    f"{where}: 'over' names {unknown}, which are neither run_key axes nor "
                    f"{_SEED!r}"
                )
            pinned = [axis for axis in over if axis in pattern.values]
            if pinned:
                raise SelectorError(
                    f"{where}: {pinned} appear in both 'pattern' and 'over'. An axis that is "
                    "pinned cannot vary, so the selector would find no repeated predictions."
                )
            if body.get("tau") is None:
                raise SelectorError(
                    f"{where}: an instability selector needs 'tau' — the prob_std above which "
                    "a record counts as unstable"
                )
            return cls(
                name=name,
                type=kind,
                pattern=pattern,
                k=k,
                tau=float(body["tau"]),
                over=over,
            )

        return cls(name=name, type=kind, pattern=pattern, k=k)


@dataclass
class MisclassifySpec:
    selectors: Tuple[Selector, ...]
    verdicts: Tuple[str, ...] = DEFAULT_VERDICTS
    label_def: Optional[str] = None
    draw_figures: bool = True
    attribution: Optional[Mapping[str, Any]] = None
    threshold: ThresholdPolicy = field(default_factory=ThresholdPolicy)

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "MisclassifySpec":
        declared = spec.get("selectors", ()) or ()
        if not isinstance(declared, Sequence) or isinstance(declared, (str, bytes)):
            raise SelectorError("misclassify.selectors must be a list of selector mappings")
        selectors = tuple(
            Selector.from_mapping(body, index) for index, body in enumerate(declared)
        )
        names = [selector.name for selector in selectors]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise SelectorError(
                f"selector names {duplicates} are used more than once; a name is part of "
                "case_id, so two selectors sharing one would merge their review rows"
            )
        attribution = spec.get("attribution")
        if attribution is not None and not isinstance(attribution, Mapping):
            raise SelectorError(
                "misclassify.attribution must be a mapping declaring at least 'registry' "
                f"and 'baseline', got {type(attribution).__name__}"
            )
        verdicts = tuple(str(value) for value in spec.get("verdicts", DEFAULT_VERDICTS))
        if UNREVIEWED in verdicts:
            raise SelectorError(
                f"{UNREVIEWED!r} is the value written for a case nobody has looked at yet; "
                "it cannot also be a verdict a reviewer may choose"
            )
        return cls(
            selectors=selectors,
            verdicts=verdicts,
            label_def=None if spec.get("label_def") is None else str(spec["label_def"]),
            draw_figures=bool(spec.get("figures", True)),
            attribution=attribution,
            threshold=ThresholdPolicy.from_spec(spec),
        )


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


class MisclassifyStage(Stage):
    name = "misclassify"
    # The audit reads predictions and drops nothing. Records absent from a
    # prediction file were excluded upstream and are already in the ledger.
    reason_codes = frozenset()

    def __init__(self, adapter_factory: Optional[Any] = None) -> None:
        # Defaults to None rather than to the real lookup so that constructing
        # the stage imports no backend. Only an attribution overlay needs one.
        self._adapter_factory = adapter_factory

    def config_inputs(self, spec: Mapping[str, Any]) -> Dict[str, Path]:
        """Every ModelCard, when an attribution overlay is asked for.

        Same reason as the models stage: the registry arrives as a directory,
        so without this a card's weights could change while config_hash did
        not, and the stale overlays would be skipped instead of redrawn.
        """
        attribution = (spec or {}).get("attribution") or {}
        registry = attribution.get("registry")
        if not registry:
            return {}
        directory = Path(str(registry))
        return {
            f"card:{path.stem}": path for path in sorted(directory.glob("*.json"))
        }

    def required_inputs(self) -> tuple:
        """Only predictions.

        Every companion input of spec §4.6 is optional and is a join onto
        predictions that already exist: ``preprocess_index`` (the tensor behind
        the waveform plot), ``train_log`` (the operating threshold the models
        stage fitted), ``review`` (the human's verdicts from a previous pass),
        and any further table, whose columns become case context.
        """
        return ("predictions",)

    def run(self, ctx: StageContext) -> StageResult:
        spec = MisclassifySpec.from_spec(dict(ctx.spec or {}))
        warnings: List[str] = []

        predictions = load_predictions(ctx.input_path("predictions"), spec.label_def)
        regression = predictions["label_def"] == REGRESSION_LABEL_DEF
        if regression.any():
            warnings.append(
                f"{int(regression.sum())} regression prediction rows (label_def={REGRESSION_LABEL_DEF!r}) "
                "were skipped: the selectors are defined on probabilities"
            )
            predictions = predictions[~regression]
        frame = _normalise_axes(predictions)

        recorded: Dict[Tuple[str, ...], float] = {}
        if TRAIN_LOG_INPUT in ctx.inputs:
            recorded, _ = recorded_thresholds_and_contamination(
                ctx.inputs[TRAIN_LOG_INPUT], spec.threshold.policy, warnings
            )
        thresholds = fit_operating_thresholds(predictions, spec.threshold, warnings, recorded)

        rows: List[Dict[str, Any]] = []
        log: List[Dict[str, Any]] = []
        for selector in spec.selectors:
            selected, note = _apply(selector, frame, thresholds)
            log.append(
                {
                    "name": selector.name,
                    "type": selector.type,
                    "selected": len(selected),
                    **note,
                }
            )
            if not selected:
                message = (
                    f"selector {selector.name!r} ({selector.type}) selected no cases: "
                    f"{note.get('why', 'no record met the criterion')}"
                )
                warnings.append(message)
                log[-1]["warning"] = message
            rows.extend(selected)

        context_columns: List[str] = []
        if rows:
            rows, context_columns = _attach_context(rows, ctx, warnings)

        figure_paths: Dict[str, str] = {}
        if rows and spec.draw_figures:
            figure_paths = _draw_case_figures(
                rows, ctx, spec, warnings, self._adapter_factory
            )
        for row in rows:
            row["figure_path"] = figure_paths.get(row["case_id"])

        verdicts, unmatched = _merge_review(rows, ctx, spec, warnings)
        for row in rows:
            review = verdicts.get(row["case_id"], {})
            row["verdict"] = review.get("verdict", UNREVIEWED)
            row["note"] = review.get("note")

        columns = CASE_COLUMNS + tuple(context_columns)
        cases_path = write_table(rows, ctx.layout.artifact(CASES), columns)
        outputs = [cases_path]
        outputs.append(_write_review_template(rows, ctx))
        if unmatched:
            outputs.append(_write_unmatched(unmatched, ctx))
        outputs.append(
            _write_json(
                ctx.layout.artifact(SELECTOR_LOG),
                {"verdicts": list(spec.verdicts), "selectors": log},
            )
        )

        return StageResult(
            outputs=outputs,
            counts={"in": int(len(predictions)), "out": len(rows)},
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# The four selectors (spec §4.6)
# ---------------------------------------------------------------------------


def _apply(
    selector: Selector, frame, thresholds: Mapping[Tuple[str, ...], float]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if selector.type == "error":
        return _select_error(selector, frame, thresholds)
    if selector.type == "boundary":
        return _select_boundary(selector, frame, thresholds)
    if selector.type == "contrast":
        return _select_contrast(selector, frame, thresholds)
    return _select_instability(selector, frame, thresholds)


def _anchor(selector: Selector, row: Mapping[str, Any], threshold: float) -> Dict[str, Any]:
    """The part of a case row that every selector fills the same way."""
    label_column = f"label_{row['label_def']}"
    label = row.get(label_column)
    case: Dict[str, Any] = {
        "case_id": f"{selector.name}{_CASE_SEP}{row['image_occurrence_id']}",
        "selector": selector.name,
        "selector_type": selector.type,
        "image_occurrence_id": str(row["image_occurrence_id"]),
        "person_id": str(row["person_id"]),
        "prob": float(row["prob"]),
        "logit": _optional_float(row.get("logit")),
        "threshold": float(threshold),
        "outcome_class": _outcome_class(float(row["prob"]), label, threshold),
        "peer_run_key": None,
        "peer_prob": None,
        "peer_outcome_class": None,
        "tensor_path": None,
        "figure_path": None,
        "attribution_status": None,
        "verdict": UNREVIEWED,
        "note": None,
    }
    for axis in AXES:
        case[axis] = row[axis]
    for column in LABEL_COLUMNS:
        case[column] = _optional_int_value(row.get(column))
    return case


def _resolve(selector: Selector, frame, pattern: Pattern, side: str, free_axes=()):
    where = f"selector {selector.name!r} ({selector.type}) pattern {side}"
    selected = pattern.select(frame)
    if len(selected) == 0:
        return selected, f"pattern {side} [{pattern.describe()}] matched no predictions"
    _require_one_row_per_record(selected, where, free_axes)
    return selected, None


def _label_series(frame, where: str):
    """The label column named by this slice's ``label_def`` axis.

    Reading it from the axis rather than from the spec is what makes a
    ``label_def`` contrast work: each side of the contrast is scored against its
    own reference standard, which is the entire point of that comparison.
    """
    definitions = sorted({str(value) for value in frame["label_def"].unique()})
    if len(definitions) != 1:  # pragma: no cover - the uniqueness check precedes this
        raise SelectorError(f"{where}: slice spans several label_def values {definitions}")
    column = f"label_{definitions[0]}"
    if column not in frame.columns:
        raise SelectorError(
            f"{where}: label_def={definitions[0]!r} names column {column!r}, which the "
            f"predictions do not carry; available label columns are {list(LABEL_COLUMNS)}"
        )
    return frame[column]


def _threshold_of(row: Mapping[str, Any], thresholds: Mapping[Tuple[str, ...], float]) -> float:
    return float(thresholds[arm_of(_run_key_of(row))])


def _select_error(selector, frame, thresholds):
    """Spec §4.6: the reference run's top-k false positives and bottom-k false negatives.

    Both directions come from one selector because they are the same question
    asked of the two error types, and a study that wanted only one would set the
    other's ``k`` by omitting the class from its cohort, not by declaring two
    selectors.
    """
    selected, why = _resolve(selector, frame, selector.pattern, "pattern")
    if why:
        return [], {"candidates": 0, "why": why}
    labels = _label_series(selected, f"selector {selector.name!r}")
    thresholded = [_threshold_of(row, thresholds) for _, row in selected.iterrows()]
    work = selected.copy()
    work["_label"] = labels
    work["_threshold"] = thresholded
    labelled = work[work["_label"].notna()]
    if len(labelled) == 0:
        return [], {
            "candidates": 0,
            "why": "no record in the pattern carries the label its label_def names",
        }
    false_positive = labelled[
        (labelled["_label"] == 0) & (labelled["prob"] >= labelled["_threshold"])
    ].sort_values("prob", ascending=False)
    false_negative = labelled[
        (labelled["_label"] == 1) & (labelled["prob"] < labelled["_threshold"])
    ].sort_values("prob", ascending=True)

    rows: List[Dict[str, Any]] = []
    for group in (false_positive, false_negative):
        limited = group if selector.k is None else group.head(selector.k)
        for rank, (_, row) in enumerate(limited.iterrows()):
            case = _anchor(selector, row, float(row["_threshold"]))
            case["rank"] = rank
            case["score"] = float(row["prob"])
            case["score_name"] = "prob"
            rows.append(case)
    return rows, {
        "candidates": int(len(false_positive) + len(false_negative)),
        "false_positive": int(len(false_positive)),
        "false_negative": int(len(false_negative)),
    }


def _select_boundary(selector, frame, thresholds):
    """Spec §4.6: ``abs(prob − threshold) < ε``.

    The threshold is the one the evaluation stage reported at, not a fresh one:
    both stages call ``mival.stages._predictions``. A band around a different
    number would be a band around a decision rule nobody published.
    """
    selected, why = _resolve(selector, frame, selector.pattern, "pattern")
    if why:
        return [], {"candidates": 0, "why": why}
    work = selected.copy()
    work["_threshold"] = [_threshold_of(row, thresholds) for _, row in selected.iterrows()]
    work["_distance"] = (work["prob"] - work["_threshold"]).abs()
    band = work[work["_distance"] < selector.epsilon].sort_values("_distance")
    limited = band if selector.k is None else band.head(selector.k)
    rows = []
    for rank, (_, row) in enumerate(limited.iterrows()):
        case = _anchor(selector, row, float(row["_threshold"]))
        case["rank"] = rank
        case["score"] = float(row["prob"] - row["_threshold"])
        case["score_name"] = "prob_minus_threshold"
        rows.append(case)
    return rows, {
        "candidates": int(len(band)),
        "epsilon": selector.epsilon,
        "why": (
            f"no record fell within {selector.epsilon} of the operating threshold"
            if len(band) == 0
            else ""
        ),
    }


def _select_contrast(selector, frame, thresholds):
    """Spec §4.6: two run_key patterns, whatever axis separates them.

    This function never learns which axis it is contrasting. ``model_id``
    disagreement, what fine-tuning changed, a case that flips between 500 Hz and
    125 Hz, and a case whose reference standard changes with the label
    definition are all this one query.
    """
    left, why_left = _resolve(selector, frame, selector.pattern, "a")
    if why_left:
        return [], {"candidates": 0, "why": why_left}
    right, why_right = _resolve(selector, frame, selector.peer, "b")
    if why_right:
        return [], {"candidates": 0, "why": why_right}

    left = left.copy()
    right = right.copy()
    left["_label"] = _label_series(left, f"selector {selector.name!r} side a")
    right["_label"] = _label_series(right, f"selector {selector.name!r} side b")
    left["_threshold"] = [_threshold_of(row, thresholds) for _, row in left.iterrows()]
    right["_threshold"] = [_threshold_of(row, thresholds) for _, row in right.iterrows()]

    shared = sorted(
        set(left["image_occurrence_id"]).intersection(right["image_occurrence_id"])
    )
    if not shared:
        return [], {
            "candidates": 0,
            "why": "the two patterns cover no record in common",
        }
    left_by = {str(row["image_occurrence_id"]): row for _, row in left.iterrows()}
    right_by = {str(row["image_occurrence_id"]): row for _, row in right.iterrows()}

    hits = []
    for image_id in shared:
        a = left_by[str(image_id)]
        b = right_by[str(image_id)]
        delta = abs(float(a["prob"]) - float(b["prob"]))
        outcome_a = _outcome_class(float(a["prob"]), a["_label"], float(a["_threshold"]))
        outcome_b = _outcome_class(float(b["prob"]), b["_label"], float(b["_threshold"]))
        predicted_a = float(a["prob"]) >= float(a["_threshold"])
        predicted_b = float(b["prob"]) >= float(b["_threshold"])
        if selector.criterion == "prediction_flip":
            interesting = predicted_a != predicted_b
        elif selector.criterion == "truth_flip":
            interesting = _label_differs(a["_label"], b["_label"])
        elif selector.criterion == "outcome_flip":
            interesting = outcome_a != outcome_b
        else:
            interesting = delta > selector.tau
        # `tau` beside a flip criterion is a magnitude floor: it keeps the
        # selector from filling up with records that flipped because they sat a
        # thousandth of a probability from the threshold, which is what the
        # `boundary` selector is for.
        if selector.tau is not None and selector.criterion != "delta_prob":
            interesting = interesting and delta > selector.tau
        if interesting:
            hits.append((delta, a, b, outcome_b))

    hits.sort(key=lambda item: item[0], reverse=True)
    limited = hits if selector.k is None else hits[: selector.k]
    rows = []
    for rank, (delta, a, b, outcome_b) in enumerate(limited):
        case = _anchor(selector, a, float(a["_threshold"]))
        case["rank"] = rank
        case["score"] = float(delta)
        case["score_name"] = "abs_delta_prob"
        case["peer_run_key"] = _run_key_of(b).to_string()
        case["peer_prob"] = float(b["prob"])
        case["peer_outcome_class"] = outcome_b
        rows.append(case)
    return rows, {
        "candidates": len(hits),
        "criterion": selector.criterion,
        "shared_records": len(shared),
        "why": (
            f"no shared record met criterion {selector.criterion!r}" if not hits else ""
        ),
    }


def _select_instability(selector, frame, thresholds):
    """Spec §4.6: ``prob_std > τ`` across repeated predictions of one record.

    ``over`` is declared rather than inferred from whichever axes happen to be
    free. Inferring it would let a study that forgot to pin ``perturbation_id``
    publish perturbation sensitivity under the name "instability".

    Note what a single models run can supply: dev predictions are out-of-fold,
    so each dev record appears in exactly one fold and ``over: [fold]`` finds
    nothing. Seed variation needs the prediction directories of two runs. The
    stage reports the empty result as a warning rather than an empty table.
    """
    import numpy as np

    free = tuple(axis for axis in selector.over if axis in AXES)
    selected, why = _resolve(selector, frame, selector.pattern, "pattern", free_axes=free)
    if why:
        return [], {"candidates": 0, "why": why}

    varies_over_seed = _SEED in selector.over
    groups = []
    for image_id, group in selected.groupby("image_occurrence_id"):
        if len(group) < 2:
            continue
        if varies_over_seed and group["seed"].nunique() < 2 and not free:
            continue
        groups.append((str(image_id), group))
    if not groups:
        return [], {
            "candidates": 0,
            "repeated_records": 0,
            "why": (
                f"no record has two or more predictions differing only in {list(selector.over)}; "
                "with one models run, dev predictions are out-of-fold and appear once per "
                "record, and seed variation needs the predictions of a second run"
            ),
        }

    hits = []
    for image_id, group in groups:
        probs = group["prob"].to_numpy(dtype=float)
        spread = float(np.std(probs, ddof=1))
        if spread > selector.tau:
            ordered = group.sort_values("prob", ascending=False)
            hits.append((spread, ordered.iloc[0], ordered.iloc[-1], len(group)))
    hits.sort(key=lambda item: item[0], reverse=True)
    limited = hits if selector.k is None else hits[: selector.k]

    rows = []
    for rank, (spread, high, low, repeats) in enumerate(limited):
        threshold = _threshold_of(high, thresholds)
        case = _anchor(selector, high, threshold)
        case["rank"] = rank
        case["score"] = spread
        case["score_name"] = "prob_std"
        case["peer_run_key"] = _run_key_of(low).to_string()
        case["peer_prob"] = float(low["prob"])
        case["peer_outcome_class"] = _outcome_class(
            float(low["prob"]), _label_of(low), _threshold_of(low, thresholds)
        )
        rows.append(case)
    return rows, {
        "candidates": len(hits),
        "repeated_records": len(groups),
        "over": list(selector.over),
        "why": (
            f"no record had prob_std above {selector.tau}" if not hits else ""
        ),
    }


def _label_of(row: Mapping[str, Any]):
    return row.get(f"label_{row['label_def']}")


def _label_differs(left, right) -> bool:
    left_value = _optional_int_value(left)
    right_value = _optional_int_value(right)
    if left_value is None or right_value is None:
        return False
    return left_value != right_value


def _outcome_class(prob: float, label, threshold: float) -> str:
    """tp / fp / tn / fn, or ``unlabelled`` when the reference standard is absent."""
    value = _optional_int_value(label)
    if value is None:
        return "unlabelled"
    predicted = prob >= threshold
    if predicted and value == 1:
        return "tp"
    if predicted:
        return "fp"
    if value == 1:
        return "fn"
    return "tn"


def _optional_int_value(value) -> Optional[int]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return int(number)


def _optional_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


# ---------------------------------------------------------------------------
# Companion information (spec §4.6)
# ---------------------------------------------------------------------------


def _attach_context(
    rows: List[Dict[str, Any]], ctx: StageContext, warnings: List[str]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Join the optional tables onto the selected cases.

    Spec §4.6 wants acquisition metadata, preprocess warnings and cohort/index
    context beside each case. None of those columns is named here: a study
    supplies the tables as inputs and their columns travel through. The
    alternative — listing this study's Profile schema in this file — would have
    to change for every site.
    """
    columns: List[str] = []
    if PREPROCESS_INDEX_INPUT in ctx.inputs:
        index = read_table(ctx.inputs[PREPROCESS_INDEX_INPUT])
        if "tensor_path" in index.columns:
            wanted = ["image_occurrence_id", "recipe_id", "perturbation_id", "tensor_path"]
            available = [column for column in wanted if column in index.columns]
            lookup = {}
            for _, entry in index[available].iterrows():
                lookup[_tensor_key(entry)] = str(entry["tensor_path"])
            for row in rows:
                row["tensor_path"] = lookup.get(_tensor_key(row))
            missing = sum(1 for row in rows if row["tensor_path"] is None)
            if missing:
                warnings.append(
                    f"{missing} of {len(rows)} cases have no tensor in preprocess_index; "
                    "their waveform plots are skipped"
                )
        else:
            warnings.append(
                "preprocess_index carries no tensor_path column; waveform plots are skipped"
            )

    for name, path in sorted(ctx.inputs.items()):
        if name in _KNOWN_INPUTS or name.startswith(_CARD_PREFIX):
            continue
        extra = read_table(path)
        join_key = next(
            (key for key in ("image_occurrence_id", "person_id") if key in extra.columns),
            None,
        )
        if join_key is None:
            warnings.append(
                f"context input {name!r} joins on neither image_occurrence_id nor person_id "
                "and was skipped"
            )
            continue
        new = [
            column
            for column in extra.columns
            if column != join_key and column not in CASE_COLUMNS and column not in columns
        ]
        if not new:
            continue
        lookup = {
            str(entry[join_key]): {column: entry[column] for column in new}
            for _, entry in extra.iterrows()
        }
        for row in rows:
            values = lookup.get(str(row[join_key]), {})
            for column in new:
                row[column] = _plain(values.get(column))
        columns.extend(new)
    return rows, columns


def _tensor_key(entry: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (
        str(entry.get("image_occurrence_id")),
        str(entry.get("recipe_id")),
        str(entry.get("perturbation_id")),
    )


def _plain(value):
    """A parquet- and JSON-safe scalar. numpy types leak otherwise."""
    if value is None:
        return None
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    return value


# ---------------------------------------------------------------------------
# Review loop (spec §4.6) — the automatic/manual boundary is a file boundary
# ---------------------------------------------------------------------------


def _merge_review(
    rows: List[Dict[str, Any]], ctx: StageContext, spec: MisclassifySpec, warnings: List[str]
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """Read the human's ``review.csv`` and match it to this run's cases by ``case_id``.

    Rows that match nothing are not dropped. A reviewer's judgement is the most
    expensive artifact this pipeline produces, and a selector re-tuned between
    runs will orphan some of it; losing it silently would be the worst possible
    behaviour, so it is written out under its own name.
    """
    if REVIEW_INPUT not in ctx.inputs:
        return {}, []
    entries = _read_review(Path(ctx.inputs[REVIEW_INPUT]))
    known = {row["case_id"] for row in rows}
    verdicts: Dict[str, Dict[str, Any]] = {}
    unmatched: List[Dict[str, Any]] = []
    unknown_verdicts = set()
    for entry in entries:
        case_id = str(entry.get("case_id", "")).strip()
        if not case_id:
            continue
        verdict = str(entry.get("verdict", "") or "").strip()
        if verdict and verdict not in spec.verdicts and verdict != UNREVIEWED:
            unknown_verdicts.add(verdict)
        if case_id in known:
            verdicts[case_id] = {
                "verdict": verdict or UNREVIEWED,
                "note": entry.get("note") or None,
            }
        else:
            unmatched.append(
                {
                    "case_id": case_id,
                    "verdict": verdict,
                    "note": entry.get("note") or "",
                }
            )
    if unknown_verdicts:
        warnings.append(
            f"review.csv uses verdicts {sorted(unknown_verdicts)} that the study did not "
            f"declare; declared verdicts are {list(spec.verdicts)}. They are kept as written "
            "but will not aggregate."
        )
    if unmatched:
        warnings.append(
            f"{len(unmatched)} reviewed cases are not in this run's case set; they are "
            f"preserved in {REVIEW_UNMATCHED} rather than discarded"
        )
    return verdicts, unmatched


def _read_review(path: Path) -> List[Dict[str, Any]]:
    import csv

    if path.suffix.lower() == ".parquet":
        frame = read_table(path)
        return [dict(entry) for _, entry in frame.iterrows()]
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_review_template(rows: Sequence[Mapping[str, Any]], ctx: StageContext) -> Path:
    """The file a clinician edits, pre-filled with whatever review already exists."""
    import csv

    path = ctx.layout.artifact(REVIEW_TEMPLATE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "verdict": "" if row["verdict"] == UNREVIEWED else row["verdict"],
                    "note": row.get("note") or "",
                }
            )
    return path


def _write_unmatched(unmatched: Sequence[Mapping[str, Any]], ctx: StageContext) -> Path:
    import csv

    path = ctx.layout.artifact(REVIEW_UNMATCHED)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REVIEW_COLUMNS))
        writer.writeheader()
        for entry in unmatched:
            writer.writerow({column: entry.get(column, "") for column in REVIEW_COLUMNS})
    return path


def _write_json(path: Path, body: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Case figures (spec §4.6)
# ---------------------------------------------------------------------------


def _draw_case_figures(
    rows: List[Dict[str, Any]],
    ctx: StageContext,
    spec: MisclassifySpec,
    warnings: List[str],
    adapter_factory: Optional[Any] = None,
) -> Dict[str, str]:
    """One figure per case: the waveform the model saw, with its context.

    The tensor is plotted rather than the source signal because the tensor is
    what the model was given. A reviewer looking at a resampled, re-scaled,
    lead-subsetted input can see artefacts that the original recording does not
    show and that the model none the less scored.
    """
    from mival import figures

    leads_by_recipe = _lead_names(ctx, warnings)
    attributor = _Attributor.of(spec, warnings, adapter_factory)
    paths: Dict[str, str] = {}
    failures = 0
    for row in rows:
        if not row.get("tensor_path"):
            continue
        try:
            samples = _load_tensor(Path(row["tensor_path"]))
        except Exception as exc:  # a missing or corrupt tensor must not kill the audit
            failures += 1
            if failures == 1:
                warnings.append(
                    f"case {row['case_id']}: tensor could not be read ({exc}); its figure is "
                    "skipped"
                )
            continue
        attribution, status = attributor.of_case(row, samples)
        row["attribution_status"] = status
        target = ctx.layout.artifact(FIGURE_DIR, f"{row['case_id']}.png")
        figures.case_waveform(
            samples,
            path=target,
            leads=leads_by_recipe.get(str(row["recipe_id"])),
            title=f"{row['case_id']}  [{row['outcome_class']}]",
            annotations=_annotations(row),
            attribution=attribution,
        )
        paths[row["case_id"]] = str(target)
    if failures > 1:
        warnings.append(f"{failures} case tensors could not be read; their figures are skipped")
    attributor.report(warnings)
    return paths


class _Attributor:
    """Attribution overlays for the case figures (spec §4.6), when asked for.

    Off unless the study declares ``misclassify.attribution``, because it is
    the only part of this stage that loads model weights, and loading a backend
    inside the audit is a cost a study should opt into rather than inherit.

    It can only serve arms whose ModelCard publishes a head — in practice
    ``inference_only``. A ``linear_probe`` or fine-tuned arm scored through a
    head that the models stage fitted in memory and never wrote to disk, so its
    handle cannot be rebuilt from artifacts and its cases get a status saying
    exactly that rather than an overlay computed from the wrong model.
    """

    def __init__(self, settings, warnings, adapter_factory):
        self.baseline = settings["baseline"]
        self.steps = settings["steps"]
        self.registry = settings["registry"]
        self._factory = adapter_factory
        self._handles: Dict[str, Any] = {}
        self._cards: Optional[Mapping[str, Any]] = None
        self._counts: Dict[str, int] = {}

    @classmethod
    def of(cls, spec: MisclassifySpec, warnings: List[str], adapter_factory):
        declared = spec.attribution
        if not declared:
            return _NoAttribution()
        if not declared.get("registry"):
            warnings.append(
                "misclassify.attribution declares no 'registry', so no ModelCard can be "
                "loaded and no overlay is drawn"
            )
            return _NoAttribution()
        baseline = declared.get("baseline")
        if baseline is None:
            warnings.append(
                "misclassify.attribution declares no 'baseline'. Integrated Gradients "
                "attributes relative to one and there is no defensible default for an ECG "
                "(all-zero is asystole, not 'no signal'), so no overlay is drawn. See "
                "mival.adapters._attribution."
            )
            return _NoAttribution()
        if isinstance(baseline, str) and baseline != "zeros":
            import numpy as np

            baseline = np.load(Path(baseline))
        return cls(
            {
                "baseline": baseline,
                "steps": int(declared.get("steps", 64)),
                "registry": Path(str(declared["registry"])),
            },
            warnings,
            adapter_factory,
        )

    def of_case(self, row: Mapping[str, Any], samples):
        if str(row["training_mode"]) != "inference_only":
            return None, "unavailable_fitted_head_not_persisted"
        try:
            adapter, handle = self._handle(str(row["model_id"]))
        except Exception as exc:
            self._count(f"load_failed: {exc}")
            return None, "load_failed"
        try:
            import numpy as np

            attribution = adapter.attribute(
                handle, np.asarray(samples)[None, ...], self.baseline, self.steps
            )
        except Exception as exc:
            self._count(f"attribute_failed: {exc}")
            return None, "attribute_failed"
        self._count("ok")
        return attribution[0], "ok"

    def _handle(self, model_id: str):
        if model_id not in self._handles:
            if self._cards is None:
                from mival.modelcard import load_registry

                self._cards = load_registry(self.registry)
            factory = self._factory
            if factory is None:
                from mival.adapters import get_adapter  # lazy: the backend import

                factory = get_adapter
            card = self._cards[model_id]
            adapter = factory(card.adapter)
            self._handles[model_id] = (adapter, adapter.load(card))
        return self._handles[model_id]

    def _count(self, reason: str) -> None:
        self._counts[reason] = self._counts.get(reason, 0) + 1

    def report(self, warnings: List[str]) -> None:
        for reason, count in sorted(self._counts.items()):
            if reason != "ok":
                warnings.append(f"attribution failed for {count} case(s): {reason}")


class _NoAttribution:
    """The default: draw the waveform, claim nothing about saliency."""

    def of_case(self, row, samples):
        return None, None

    def report(self, warnings: List[str]) -> None:
        return None


def _annotations(row: Mapping[str, Any]) -> List[str]:
    """The companion facts spec §4.6 wants beside the trace."""
    lines = [
        f"prob={row['prob']:.4f}  threshold={row['threshold']:.4f}  ({row['outcome_class']})",
        f"selector={row['selector']} ({row['selector_type']})  "
        f"{row['score_name']}={_format(row['score'])}",
        "  ".join(f"{axis}={row[axis]}" for axis in ("model_id", "training_mode", "label_def")),
        "  ".join(
            f"{column}={'-' if row.get(column) is None else row[column]}"
            for column in LABEL_COLUMNS
        ),
    ]
    if row.get("peer_run_key"):
        lines.append(
            f"peer prob={_format(row['peer_prob'])} ({row['peer_outcome_class']})  "
            f"{row['peer_run_key']}"
        )
    return lines


def _format(value) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def _lead_names(ctx: StageContext, warnings: List[str]) -> Dict[str, List[str]]:
    """Lead order per recipe, from the preprocess stage's ``recipes.json``.

    Without it the plot falls back to positional lead numbers, which is honest
    but much harder to read: "the ST elevation is in the lead at index 6" is not
    a clinical sentence.
    """
    if PREPROCESS_INDEX_INPUT not in ctx.inputs:
        return {}
    recipes = Path(ctx.inputs[PREPROCESS_INDEX_INPUT]).parent / "recipes.json"
    if not recipes.exists():
        warnings.append(
            f"{recipes.name} was not found beside preprocess_index; case figures label leads "
            "by position instead of by name"
        )
        return {}
    body = json.loads(recipes.read_text(encoding="utf-8"))
    return {
        str(entry.get("recipe_id")): [str(name) for name in entry.get("leads", ())]
        for entry in body.get("recipes", ())
        if entry.get("leads")
    }


def _load_tensor(path: Path):
    import numpy as np

    array = np.load(path)
    if hasattr(array, "files"):
        array = array[array.files[0]]
    array = np.asarray(array, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"tensor at {path} is {array.ndim}-D; (n_leads, n_samples) is required")
    return array
