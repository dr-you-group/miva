"""Evaluation stage (spec §4.5). Implemented by Plan 5.

Contract pinned by Plan 2. The long format is what keeps the schema stable as
axes are added (spec §4.5), so ``METRICS_LONG_COLUMNS`` is the run_key axes
plus a fixed tail — never one column per metric.

The stage is deliberately thin. All arithmetic lives in ``mival.metrics``,
which is axis-free; everything here is loading, grouping and labelling. That
is the concrete form of the spec's "evaluation is one function that computes
four categories, and the rest is a groupby": adding an axis changes the
grouping key and nothing else.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mival.criteria import Cut, net_benefit_grid, parse_cuts
from mival.pipeline.layout import exclusions_root
from mival.pipeline.ledger import load_runs, read_exclusions
from mival.pipeline.parallel import ordered_map, resolve_workers
from mival.pipeline.runkey import AXES, REPORT_AXES, RunKey
from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import join_key, read_table, write_table
from mival.stages._comparisons import (  # noqa: F401 - _comparison_rows, _select_arm re-exported
    _comparison_rows,
    _select_arm,
    comparison_rows,
)
from mival.stages._curve_rows import (  # noqa: F401 - CURVE_COLUMNS re-exported
    CURVE_COLUMNS,
    CURVES_LONG,
    classification_curve_rows,
    regression_curve_rows,
    survival_curve_rows,
)
from mival.stages._predictions import (
    DEV_SPLIT,
    FALLBACK_THRESHOLD,
    TEST_SPLIT,
    ThresholdPolicy,
    arm_axes,
    arm_of as _arm_of,
    fit_operating_thresholds,
    group_by_run_key as _group_by_run_key,
    is_regression,
    is_survival,
    load_predictions as _load_predictions,
    recorded_thresholds_and_contamination,
)
from mival.stages.models import PRIMARY_THRESHOLD_POLICY, THRESHOLD_POLICIES
from mival.threshold import SENS95_TARGET_SENSITIVITY

#: Output artifacts, relative to ``ctx.layout.artifacts_dir``.
METRICS_LONG = "metrics_long.parquet"
FIGURE_DIR = "figures"

#: Spec §4.5: run_key columns, then the report-only axes, then the value block.
#: ``update`` says whether the row scores the model as shipped (``"none"``) or
#: after a recalibration fitted on dev; ``tier`` is
#: :func:`mival.metrics.tiers.tier_of`; ``se`` is the bootstrap SE; ``reason``
#: says why a row's value must not be read as a finding (``REASON_*``).
METRICS_LONG_COLUMNS = AXES + (
    "subgroup",
    "outcome",
    "update",
    "category",
    "metric",
    "tier",
    "value",
    "se",
    "ci_lo",
    "ci_hi",
    "n",
    "n_events",
    "suppressed",
    "reason",
    "contaminated",
)

#: Spec §4.5 categories. "Overall performance" is deliberately absent: Brier is
#: a composite, and by the Murphy decomposition its reliability term belongs to
#: calibration and its resolution term to discrimination.
CATEGORIES = (
    "discrimination",
    "calibration",
    "clinical_utility",
    "interpretation",
    "regression",
    "survival",
)

# Category 4 (interpretation) holds a schema slot and emits no rows in this
# plan: attribution maps and lead/time-region agreement are a qualitative
# reading against clinical criteria (spec §4.5, FUTURE-AI Explainability), not
# a number this stage can compute. The category constant and
# ``mival.metrics.INTERPRETATION_METRICS`` exist so that adding it later is an
# addition, not a schema change.

#: Spec §4.5. Resampling unit is person, not ECG — a patient may contribute
#: several recordings, and resampling by recording understates the CI.
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_UNIT = "person_id"

#: Spec §4.5: computed but marked, never used in conclusions.
SUBGROUP_SUPPRESS_MIN_EVENTS = 10

#: The ``update`` value of a row that scores the model as shipped.
UPDATE_NONE = "none"

#: The recalibrations reported beside the shipped model, on the whole test
#: split of the primary outcome (``fit_recalibration`` kinds / OLS).
CLASSIFICATION_UPDATES = ("intercept", "logistic")
REGRESSION_UPDATES = ("linear",)

#: A linear probe is already fitted on dev; recalibrating it is not an update.
LINEAR_PROBE = "linear_probe"

#: The models stage's policy for the card's own published operating point.
LEGACY_POLICY = "legacy"

#: ``reason`` values. A row with no reason is a reportable estimate.
#: ``insufficient_events``: events or non-events below ``evaluate.min_events``;
#: the value is still computed and the row is ``suppressed``.
#: ``no_data``: a declared subgroup level with no records; the row exists so
#: that an absent level is visible rather than silently missing (STANDING 2.2d).
#: ``no_absolute_risk``: a survival arm's Brier, IPA or O:E at a horizon when
#: its output is not a probability by that horizon (a risk score, a risk by
#: the card's own different horizon, or a scale the train_log does not
#: record); the value is NULL.
REASON_INSUFFICIENT_EVENTS = "insufficient_events"
REASON_NO_DATA = "no_data"
REASON_NO_ABSOLUTE_RISK = "no_absolute_risk"

#: A regression arm has no default cut for its derived classification
#: metrics (``auroc_below@<cut>`` and the like): a cut is the clinical line
#: of one label, so it comes from the study's ``evaluate.regression_cuts``
#: only. Without the key a regression arm gets its regression metrics and no
#: cut metrics; the export manifest says so.

# ---------------------------------------------------------------------------
# Added by Plan 5, below the pinned contract above.
# ---------------------------------------------------------------------------

#: Paired model comparisons (spec §4.5) go in their own table. A difference
#: between two arms is not a property of one point in run_key space, so writing
#: it into ``metrics_long.parquet`` would mean either inventing a run_key that
#: no run produced, or silently attributing the difference to one of the arms.
#: ``x`` is the threshold of a net-benefit grid row (NULL otherwise); ``p_holm``
#: is the Holm-adjusted p of an automatic scalar row (see ``_comparisons``).
COMPARISONS = "comparisons.parquet"
COMPARISON_COLUMNS = (
    "comparison",
    "confirmatory",
    "arm_a",
    "arm_b",
    "subgroup",
    "outcome",
    "category",
    "metric",
    "x",
    "value",
    "ci_lo",
    "ci_hi",
    "p_value",
    "p_holm",
    "n",
    "n_events",
)

#: The slice label for "no subgroup restriction". Report axes always carry a
#: value so that a filter on them never has to special-case a missing row.
SUBGROUP_ALL = "all"
#: Default value of the ``outcome`` report axis: the label the run was scored
#: against. Other outcomes (30-day mortality, PCI within 24 h) are additional
#: values of the same axis and join onto the same predictions (spec §3.4).
#: Neutral on purpose: a study's disease name belongs in its study.yaml.
OUTCOME_DIAGNOSTIC = "primary"

# ``DEV_SPLIT``, ``TEST_SPLIT`` and ``FALLBACK_THRESHOLD`` are imported from
# ``mival.stages._predictions`` and re-exported here, because the audit stage
# must use the same values: a case picked as "near the operating threshold" is
# only meaningful against the threshold this stage reported.

#: Optional input: the models stage's ``train_log.jsonl``. When supplied it is
#: the authority on both the operating threshold and the contamination flag,
#: because that stage fitted the one and ran the gate for the other. Without
#: it the threshold is refitted here from dev predictions, using the same
#: ``mival.threshold`` policies, and contamination falls back to the study
#: spec's restatement.
TRAIN_LOG_INPUT = "train_log"

#: Optional input: ``runs.json`` of this invocation (stage -> run dirs). With it
#: the STARD flow reads only this invocation's ledger parts, so exclusions of a
#: stale earlier run cannot leak into the counts.
RUNS_INPUT = "runs"

#: STARD box for the development split, drawn beside the analysed (test) count.
STARD_DEV_BOX = "development (dev)"
#: Manifest ``counts`` keys of the STARD analysed (test) and dev ECG counts.
STARD_COUNT_ANALYSED = "analysed"
STARD_COUNT_DEV = "dev"


class EvaluateStage(Stage):
    name = "evaluate"
    # Evaluation aggregates; it does not drop records. Records absent from a
    # prediction file were already excluded upstream and are already in the
    # ledger, so re-recording them here would double-count the STARD flow.
    reason_codes = frozenset()

    def required_inputs(self) -> tuple:
        """The two inputs Plan 2 pinned.

        Three optional inputs are also read when supplied, all of them joins
        onto predictions that already exist rather than reasons to run
        inference again (spec §3.4):

        ``train_log``
            The models stage's ``train_log.jsonl``; the authority on the
            operating threshold (spec §4.4) and the contamination flag
            (spec §3.5).
        ``outcomes``
            Extra label columns — 30-day mortality, PCI within 24 h — which
            become further values of the ``outcome`` report axis.
        ``attributes``
            Extra person- or record-level columns to slice the ``subgroup``
            report axis by. A Categorical column contributes every declared
            level, observed or not.
        ``runs``
            ``runs.json`` of this invocation, which restricts the STARD flow
            to this invocation's exclusion ledger parts.
        """
        return ("predictions", "cohort_split")

    def run(self, ctx: StageContext) -> StageResult:
        from mival.metrics import DEFAULT_MURPHY_BINS

        spec = dict(ctx.spec or {})
        warnings: List[str] = []

        # No decision thresholds unless the study names them: the net-benefit
        # rows at a named threshold are supplementary, and the full curve is
        # in curves_long at every grid point.
        settings = _Settings.from_spec(spec, (), DEFAULT_MURPHY_BINS)
        if settings.seed is None:
            # The bootstrap seed is the run's seed unless the evaluation spec
            # overrides it, so the manifest's `seed` field is the one that
            # actually reproduces the intervals.
            settings.seed = int(ctx.seed) if ctx.seed is not None else 0
        predictions = _load_predictions(ctx.input_path("predictions"), settings.label_def)
        cohort = read_table(ctx.input_path("cohort_split"))
        predictions = _attach_cohort_columns(predictions, cohort)
        for name in ("outcomes", "attributes"):
            if name in ctx.inputs:
                predictions = _attach_table(predictions, read_table(ctx.inputs[name]), name, warnings)

        recorded: Dict[Tuple[str, ...], float] = {}
        if TRAIN_LOG_INPUT in ctx.inputs:
            (
                recorded,
                settings.recorded_contamination,
                settings.output_scales,
            ) = _recorded_thresholds_and_contamination(
                ctx.inputs[TRAIN_LOG_INPUT], settings, warnings
            )
        thresholds = _fit_thresholds(predictions, settings, warnings, recorded)
        legacy: Dict[Tuple[str, ...], float] = {}
        if TRAIN_LOG_INPUT in ctx.inputs:
            # The log's own legacy fit: the card's published operating point.
            # An arm without one simply has no legacy rows, so its absence is
            # not a warning.
            legacy, _, _ = recorded_thresholds_and_contamination(
                ctx.inputs[TRAIN_LOG_INPUT], LEGACY_POLICY, []
            )

        workers = resolve_workers(ctx.execution.get("workers"))
        run_keys: List[RunKey] = []
        curves: Dict[str, List[Tuple[str, Any, Any]]] = {}
        scatter: Dict[str, List[Tuple[str, Any, Any]]] = {}
        recalibrations: Dict[Tuple[Any, ...], Any] = {}

        def units():
            """``(tag, args)`` of every slice unit, in the order the rows are written.

            Everything that touches ``warnings`` or the figure series runs
            here, in this process and in loop order; a unit itself only
            computes (``_slice_unit``), so the rows do not depend on where it
            ran. ``tag`` is what the assembly below needs besides the result.
            """
            for key, group in _group_by_run_key(predictions):
                run_keys.append(key)
                threshold = thresholds[_arm_of(key)]
                contaminated = settings.is_contaminated(key)
                if is_survival(key) and _arm_of(key) not in settings.output_scales:
                    message = (
                        f"no output_scale recorded for survival arm {_arm_of(key)} (train_log "
                        "absent or older); its absolute-risk metrics are reported as "
                        f"{REASON_NO_ABSOLUTE_RISK}"
                    )
                    if message not in warnings:
                        warnings.append(message)
                for outcome, label_column in settings.outcome_columns(key, group, warnings):
                    if outcome == settings.primary_outcome:
                        for update, updated, updated_threshold in _update_units(
                            key, group, predictions, label_column, threshold,
                            settings, recalibrations, warnings,
                        ):
                            yield None, (
                                key, _unit_frame(updated, label_column), label_column, outcome,
                                SUBGROUP_ALL, updated_threshold, settings, contaminated, None,
                                update,
                            )
                    for subgroup, slice_frame in _subgroup_slices(group, settings.subgroups):
                        yield (key, subgroup, outcome, contaminated), (
                            key, _unit_frame(slice_frame, label_column), label_column, outcome,
                            subgroup, threshold, settings, contaminated,
                            legacy.get(_arm_of(key)), UPDATE_NONE,
                        )
                        if (
                            subgroup == SUBGROUP_ALL
                            and outcome == settings.primary_outcome
                            and key.split == TEST_SPLIT
                        ):
                            _collect_figure_series(
                                key, slice_frame, label_column, outcome, curves, scatter
                            )

        rows: List[Dict[str, Any]] = []
        curve_rows: List[Dict[str, Any]] = []
        # The whole slice comes first; its metric names are what an empty
        # subgroup level reports as no_data.
        whole_metrics: List[str] = []
        for tag, (slice_rows, slice_curves) in ordered_map(
            _slice_unit, units(), workers, describe=_describe_slice_unit
        ):
            if tag is not None:
                key, subgroup, outcome, contaminated = tag
                if subgroup == SUBGROUP_ALL:
                    whole_metrics = [row["metric"] for row in slice_rows]
                elif not slice_rows:
                    slice_rows = _metric_rows(
                        key, subgroup, outcome, UPDATE_NONE,
                        point={metric: None for metric in whole_metrics},
                        intervals={}, n=0, n_events=0, n_nonevents=0,
                        settings=settings, contaminated=contaminated,
                    )
            rows.extend(slice_rows)
            curve_rows.extend(slice_curves)

        metrics_path = write_table(rows, ctx.layout.artifact(METRICS_LONG), METRICS_LONG_COLUMNS)
        outputs = [metrics_path]
        outputs.append(write_table(curve_rows, ctx.layout.artifact(CURVES_LONG), CURVE_COLUMNS))

        outputs.append(
            write_table(
                comparison_rows(predictions, settings, thresholds, warnings, workers=workers),
                ctx.layout.artifact(COMPARISONS),
                COMPARISON_COLUMNS,
            )
        )

        # STARD "analysed" is the held-out split; dev is its own box. Both go
        # in the manifest too, so the export can redraw the flow from it.
        split = predictions["split"].astype(str)
        images = predictions["image_occurrence_id"]
        n_analysed = int(images[split == TEST_SPLIT].nunique())
        n_dev = int(images[split == settings.dev_split].nunique())
        if settings.draw_figures:
            outputs.extend(
                _draw_figures(ctx, rows, curves, scatter, settings, n_analysed, n_dev, warnings)
            )

        return StageResult(
            outputs=outputs,
            counts={
                "in": int(len(predictions)),
                "out": len(rows),
                STARD_COUNT_ANALYSED: n_analysed,
                STARD_COUNT_DEV: n_dev,
            },
            warnings=warnings,
            run_keys=run_keys,
            compute={"workers": workers},
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _attach_cohort_columns(predictions, cohort):
    """Join the frozen split table (spec §2.4) by person.

    ``cohort_split.parquet`` is the authority on split and fold, and is also
    where person-level attributes such as sex or age band travel. Columns
    already present in the predictions win, so a join can add report axes but
    can never quietly rewrite what the model actually scored.
    """
    if "person_id" not in cohort.columns:
        raise ValueError("cohort_split must have a person_id column (spec §2.4)")
    new_columns = [
        column
        for column in cohort.columns
        if column == "person_id" or column not in predictions.columns
    ]
    if len(new_columns) == 1:
        return predictions
    right = cohort[new_columns].drop_duplicates(subset=["person_id"])
    merged = predictions.merge(right, on="person_id", how="left")
    return merged


def _attach_table(predictions, extra, name="input", warnings=None):
    """Join an optional report-axis table on whichever key it carries.

    Report axes are cheap precisely because they need no new inference (spec
    §3.4): a downstream outcome or a hidden stratum is a join onto predictions
    that already exist. Keys are matched through :func:`join_key`, since
    prediction files carry ids as strings and CDM-built tables as integers.
    A table that matches no prediction row is an error (every report axis
    would be empty); one that misses some rows is warned about.
    """
    for key in ("image_occurrence_id", "person_id"):
        if key in extra.columns and key in predictions.columns:
            new_columns = [
                column
                for column in extra.columns
                if column == key or column not in predictions.columns
            ]
            if len(new_columns) == 1:
                return predictions
            right = extra[new_columns].rename(columns={key: "_join_key"})
            right["_join_key"] = join_key(right["_join_key"])
            right = right[right["_join_key"].notna()].drop_duplicates(subset=["_join_key"])
            left_key = join_key(predictions[key])
            matched = left_key.isin(set(right["_join_key"]))
            if len(predictions) and not matched.any():
                raise ValueError(
                    f"evaluate input {name!r} matches no prediction row on {key}; "
                    "its ids are not the ones the predictions carry"
                )
            if not matched.all() and warnings is not None:
                warnings.append(
                    f"evaluate input {name!r}: {1 - matched.mean():.1%} of prediction rows "
                    f"have no row on {key}; their columns from it are missing"
                )
            merged = predictions.assign(_join_key=left_key).merge(right, on="_join_key", how="left")
            return merged.drop(columns="_join_key")
    raise ValueError(
        "an optional evaluation input must join on image_occurrence_id or person_id"
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class _Settings:
    """The evaluation spec, with every default stated once."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        default_decision_thresholds: Sequence[float],
        default_murphy_bins: Optional[int],
    ):
        self.subgroups: Tuple[str, ...] = tuple(spec.get("subgroups", ()) or ())
        self.outcomes: Mapping[str, Any] = dict(
            spec.get("outcomes", {OUTCOME_DIAGNOSTIC: None}) or {OUTCOME_DIAGNOSTIC: None}
        )
        self.primary_outcome = next(iter(self.outcomes), OUTCOME_DIAGNOSTIC)
        self.label_columns: Mapping[str, str] = dict(spec.get("label_columns", {}) or {})
        #: Only used when predictions arrive as one file without the axis.
        self.label_def: Optional[str] = (
            None if spec.get("label_def") is None else str(spec["label_def"])
        )
        self.decision_thresholds: Tuple[float, ...] = tuple(
            spec.get("decision_thresholds", default_decision_thresholds)
        )
        # Shaded on the decision-curve figure only when the study states one;
        # a plausible threshold range is a clinical judgement per outcome.
        self.decision_band: Optional[Tuple[float, float]] = (
            None
            if spec.get("decision_band") is None
            else tuple(float(value) for value in spec["decision_band"])
        )
        self.net_benefit_grid = net_benefit_grid(spec)
        # An explicit ``murphy_bins: null`` asks for the exact decomposition on
        # unique forecast values; omitting the key gets the binned one, which
        # is the only meaningful decomposition for a continuous score.
        self.murphy_bins = spec["murphy_bins"] if "murphy_bins" in spec else default_murphy_bins
        self.calibration_knots = int(spec.get("calibration_knots", 4))
        self.dev_split = str(spec.get("dev_split", DEV_SPLIT))
        self.threshold: Optional[float] = (
            None if spec.get("threshold") is None else float(spec["threshold"])
        )
        self.threshold_policy = str(spec.get("threshold_policy", PRIMARY_THRESHOLD_POLICY))
        if self.threshold_policy not in THRESHOLD_POLICIES:
            raise ValueError(
                f"threshold_policy {self.threshold_policy!r} is not one of {list(THRESHOLD_POLICIES)}"
            )
        self.sensitivity_target = float(
            spec.get("sensitivity_target", SENS95_TARGET_SENSITIVITY)
        )
        self.legacy_threshold = float(spec.get("legacy_threshold", FALLBACK_THRESHOLD))
        self.bootstrap = bool(spec.get("bootstrap", True))
        self.bootstrap_replicates = int(spec.get("bootstrap_replicates", BOOTSTRAP_REPLICATES))
        self.bootstrap_alpha = float(spec.get("bootstrap_alpha", 0.05))
        self.bootstrap_metrics: Optional[Tuple[str, ...]] = (
            tuple(spec["bootstrap_metrics"]) if spec.get("bootstrap_metrics") else None
        )
        # None means "take the study seed", which is what the manifest records.
        self.seed: Optional[int] = None if spec.get("seed") is None else int(spec["seed"])
        self.contamination: Mapping[str, bool] = dict(spec.get("contamination", {}) or {})
        #: Filled from the models stage's train_log when that input is supplied.
        self.recorded_contamination: Dict[Tuple[str, ...], bool] = {}
        #: Each arm's ``{"scale", "horizon_days"}``, from the same log.
        self.output_scales: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        self.comparisons: Sequence[Mapping[str, Any]] = tuple(spec.get("comparisons", ()) or ())
        self.comparison_metrics: Tuple[str, ...] = tuple(
            spec.get("comparison_metrics", ("auroc", "auprc"))
        )
        #: Every arm pair within a label_def, before the explicit comparisons.
        self.auto_comparisons = bool(spec.get("auto_comparisons", True))
        self.draw_figures = bool(spec.get("figures", True))
        self.min_events = int(spec.get("min_events", SUBGROUP_SUPPRESS_MIN_EVENTS))
        cuts = spec.get("regression_cuts")
        # Not set: no cut metrics, said in the export manifest. An empty, null
        # or scalar setting is refused instead: written that way it looks like
        # a cut was meant, and a regression arm would quietly lose it.
        # A mapping gives each label_def its own cuts (age, PR, QRS, QT differ).
        # Each list holds bare numbers (legacy ``<=``) and ``{value, op}`` items.
        def _cuts(value, where):
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError(
                    f"evaluate.regression_cuts{where} must be a non-empty list of cuts "
                    f"(numbers or {{value, op}} items), got {value!r}"
                )
            return parse_cuts(value, f"evaluate.regression_cuts{where}")

        self.regression_cuts_by_label: Optional[Dict[str, Tuple[Cut, ...]]] = None
        self.regression_cuts: Tuple[Cut, ...] = ()
        if "regression_cuts" not in spec:
            pass
        elif isinstance(cuts, Mapping):
            self.regression_cuts_by_label = {
                str(label): _cuts(value, f"[{label!r}]") for label, value in cuts.items()
            }
        else:
            self.regression_cuts_by_label = None
            self.regression_cuts = _cuts(cuts, "")
        self.horizons: Tuple[float, ...] = tuple(
            float(day) for day in (spec.get("horizons_days") or [spec.get("horizon_days", 365)])
        )

    def cuts_for(self, label_def: str) -> Tuple[Cut, ...]:
        if self.regression_cuts_by_label is None:
            return self.regression_cuts
        if label_def not in self.regression_cuts_by_label:
            raise ValueError(
                f"evaluate.regression_cuts names no cuts for label_def {label_def!r}; "
                f"it has {sorted(self.regression_cuts_by_label)}"
            )
        return self.regression_cuts_by_label[label_def]

    @classmethod
    def from_spec(cls, spec, default_decision_thresholds, default_murphy_bins) -> "_Settings":
        return cls(spec, default_decision_thresholds, default_murphy_bins)

    def is_contaminated(self, key: RunKey) -> bool:
        """Spec §3.5: contamination is flagged, not blocked.

        The models stage is the authority — it ran the gate against the
        ModelCard's ``pretraining_corpora`` — so its train_log wins when it is
        available. The study spec's per-model restatement is the fallback for
        evaluating a prediction table on its own.
        """
        arm = _arm_of(key)
        if arm in self.recorded_contamination:
            return bool(self.recorded_contamination[arm])
        return bool(self.contamination.get(key.model_id, False))

    def risk_horizon(self, key: RunKey) -> Optional[float]:
        """The horizon a survival arm's prediction is an absolute risk by, or None.

        None unless the train_log records the arm's output as a probability
        with the card's ``output.horizon_days``: only the log knows, and a
        probability without a stated horizon is a risk by no known day.
        """
        recorded = self.output_scales.get(_arm_of(key)) or {}
        if recorded.get("scale") != "probability" or recorded.get("horizon_days") is None:
            return None
        return float(recorded["horizon_days"])

    def threshold_policy_settings(self) -> ThresholdPolicy:
        """The subset the audit stage must share (``mival.stages._predictions``)."""
        return ThresholdPolicy(
            threshold=self.threshold,
            policy=self.threshold_policy,
            legacy_threshold=self.legacy_threshold,
            dev_split=self.dev_split,
            sensitivity_target=self.sensitivity_target,
        )

    def label_column_for(self, label_def: str) -> str:
        default = "label_event" if label_def == "survival" else f"label_{label_def}"
        return self.label_columns.get(label_def, default)

    def outcome_columns(self, key: RunKey, frame, warnings: List[str]):
        """``(outcome, column)`` pairs for one run_key group.

        ``None`` as a configured column means "the label this run was scored
        against", which is the ``label_def`` axis of its own run_key.
        """
        pairs = []
        for outcome, column in self.outcomes.items():
            resolved = self.label_column_for(key.label_def) if column is None else str(column)
            if resolved not in frame.columns:
                message = (
                    f"outcome {outcome!r} needs column {resolved!r}, which is absent for "
                    f"run_key {key.to_string()}; the outcome is skipped"
                )
                if message not in warnings:
                    warnings.append(message)
                continue
            pairs.append((outcome, resolved))
        return pairs


# ---------------------------------------------------------------------------
# Grouping — the only place that knows the axes exist
# ---------------------------------------------------------------------------


def _subgroup_slices(frame, subgroups: Sequence[str]):
    """``(name, frame)``: the whole slice, then one slice per level of each subgroup column.

    A Categorical column's levels are its declared categories, so a level with
    no records still yields (an empty frame) and is reported as ``no_data``.
    Any other column's levels are its observed values.
    """
    import pandas as pd

    yield SUBGROUP_ALL, frame
    for column in subgroups:
        if column not in frame.columns:
            continue
        values = frame[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            levels = list(values.cat.categories)
        else:
            levels = sorted(values.dropna().unique(), key=str)
        for value in levels:
            yield f"{column}={value}", frame[values == value]


def _curve_label(key: RunKey) -> str:
    return f"{key.model_id}/{key.training_mode}/{key.split}"


# ---------------------------------------------------------------------------
# The one function that computes the four categories, per slice
# ---------------------------------------------------------------------------


def _rows_for_slice(
    key: RunKey,
    frame,
    label_column: str,
    outcome: str,
    subgroup: str,
    threshold: float,
    settings: _Settings,
    contaminated: bool,
    legacy_threshold: Optional[float] = None,
    update: str = UPDATE_NONE,
) -> List[Dict[str, Any]]:
    if is_regression(key):
        return _regression_rows(
            key, frame, label_column, outcome, subgroup, settings, contaminated, update
        )
    if is_survival(key):
        return _survival_rows(key, frame, label_column, outcome, subgroup, settings, contaminated)

    import numpy as np

    from mival.metrics import compute_metrics
    from mival.metrics.bootstrap import bootstrap_ci, event_strata

    usable = frame[frame[label_column].notna()]
    y = usable[label_column].to_numpy(dtype=float)
    p = usable["prob"].to_numpy(dtype=float)
    n = int(y.size)
    n_events = int(np.sum(y == 1.0))
    if n == 0:
        return []

    # The card's published operating point is reported on held-out data only,
    # as a second 2x2 table beside the refitted one.
    with_legacy = legacy_threshold is not None and key.split == TEST_SPLIT

    def statistic(indices) -> Dict[str, float]:
        values = compute_metrics(
            y[indices],
            p[indices],
            threshold=threshold,
            decision_thresholds=settings.decision_thresholds,
            murphy_bins=settings.murphy_bins,
            n_knots=settings.calibration_knots,
        )
        if with_legacy:
            values.update(_legacy_metrics(y[indices], p[indices], legacy_threshold))
        return values

    point = statistic(np.arange(n))
    intervals: Dict[str, Any] = {}
    if settings.bootstrap and settings.bootstrap_replicates > 0 and n_events > 0:
        wanted = settings.bootstrap_metrics
        selected = point if wanted is None else {k: v for k, v in point.items() if k in wanted}

        def restricted(indices) -> Dict[str, float]:
            values = statistic(indices)
            return values if wanted is None else {k: values[k] for k in selected}

        groups = (
            usable[BOOTSTRAP_UNIT].to_numpy() if BOOTSTRAP_UNIT in usable.columns else None
        )
        intervals = bootstrap_ci(
            restricted,
            n_rows=n,
            n_replicates=settings.bootstrap_replicates,
            seed=settings.seed,
            groups=groups,
            strata=event_strata(y, groups),
            alpha=settings.bootstrap_alpha,
            point=selected,
        )

    return _metric_rows(
        key, subgroup, outcome, update, point, intervals,
        n=n, n_events=n_events, n_nonevents=n - n_events,
        settings=settings, contaminated=contaminated,
    )


def _curve_rows_for_slice(
    key: RunKey,
    frame,
    label_column: str,
    outcome: str,
    subgroup: str,
    settings: _Settings,
    update: str = UPDATE_NONE,
) -> List[Dict[str, Any]]:
    """Fixed-grid curves of one slice, or none.

    Curves are a held-out result: only the test split gets them. A slice
    whose events or non-events fall below ``min_events`` gets none at all —
    a curve through a handful of records is noise drawn as a line, and unlike
    a marked metrics row it has nowhere to carry the mark. A survival arm
    gets its curves at the first horizon (the threshold grid and calibration
    only when its output is an absolute risk by that horizon); a regression arm gets its
    binned calibration curve instead, each bin held to ``min_events``
    records. The calibration-type bands (calibration, Kaplan-Meier) are given
    for the whole slice only (see ``_curve_rows``).
    """
    if key.split != TEST_SPLIT:
        return []
    if is_survival(key):
        usable = frame[
            frame[label_column].notna()
            & frame["label_time_days"].notna()
            & frame["pred_value"].notna()
        ]
        event = usable[label_column].to_numpy(dtype=float)
        n_events = int((event == 1.0).sum())
        if event.size == 0 or min(n_events, int(event.size) - n_events) < settings.min_events:
            return []
        groups = usable[BOOTSTRAP_UNIT].to_numpy() if BOOTSTRAP_UNIT in usable.columns else None
        return survival_curve_rows(
            key, subgroup, outcome, UPDATE_NONE,
            usable["label_time_days"].to_numpy(dtype=float),
            event,
            usable["pred_value"].to_numpy(dtype=float),
            groups,
            horizon=settings.horizons[0],
            grid=settings.net_benefit_grid,
            absolute=settings.risk_horizon(key) == settings.horizons[0],
            n_replicates=settings.bootstrap_replicates if settings.bootstrap else 0,
            seed=settings.seed,
            alpha=settings.bootstrap_alpha,
            km_band=subgroup == SUBGROUP_ALL,
        )
    if is_regression(key):
        # The same records _regression_rows scores.
        usable = frame[frame[label_column].notna() & frame["pred_value"].notna()]
        if not len(usable):
            return []
        return regression_curve_rows(
            key, subgroup, outcome, UPDATE_NONE,
            usable[label_column].to_numpy(dtype=float),
            usable["pred_value"].to_numpy(dtype=float),
            min_n=settings.min_events,
            alpha=settings.bootstrap_alpha,
        )
    usable = frame[frame[label_column].notna()]
    y = usable[label_column].to_numpy(dtype=float)
    n_events = int((y == 1.0).sum())
    # The size check comes first so that ``min_events: 0`` cannot let an
    # empty level through to a curve of zero records.
    if y.size == 0 or min(n_events, int(y.size) - n_events) < settings.min_events:
        return []
    groups = usable[BOOTSTRAP_UNIT].to_numpy() if BOOTSTRAP_UNIT in usable.columns else None
    return classification_curve_rows(
        key, subgroup, outcome, update,
        y, usable["prob"].to_numpy(dtype=float), groups,
        grid=settings.net_benefit_grid,
        n_knots=settings.calibration_knots,
        n_replicates=settings.bootstrap_replicates if settings.bootstrap else 0,
        seed=settings.seed,
        alpha=settings.bootstrap_alpha,
        calibration_band=subgroup == SUBGROUP_ALL,
    )


def _update_units(
    key: RunKey,
    frame,
    predictions,
    label_column: str,
    threshold: float,
    settings: _Settings,
    fitted: Dict[Tuple[Any, ...], Any],
    warnings: List[str],
):
    """``(update, updated_frame, updated_threshold)`` of each dev-fitted update of this slice.

    None unless this is the whole test slice of a classification or
    regression arm that is not a linear probe. Each update is fitted once per
    arm, on every record of that arm's development split with a label (folds
    ignored) — the arm includes its perturbation, so a perturbed arm is
    updated by what its own development split says, and has no update rows
    when that split is absent. An update that cannot be fitted gives nothing
    and one warning. The operating threshold goes through the same map, which
    for an increasing update leaves the 2x2 table as it was. The fits and the
    warnings happen here, in the calling process; scoring the updated frame
    is a ``_slice_unit``.
    """
    if key.split != TEST_SPLIT or key.training_mode == LINEAR_PROBE or is_survival(key):
        return
    regression = is_regression(key)
    value_column = "pred_value" if regression else "prob"
    arm = _arm_of(key)
    for update in REGRESSION_UPDATES if regression else CLASSIFICATION_UPDATES:
        if (arm, update) not in fitted:
            development = predictions
            for axis, value in zip(arm_axes(), arm):
                development = development[development[axis] == value]
            development = development[development["split"] == settings.dev_split]
            development = development[
                development[label_column].notna() & development[value_column].notna()
            ]
            y_dev = development[label_column].to_numpy(dtype=float)
            p_dev = development[value_column].to_numpy(dtype=float)
            if regression:
                from mival.metrics.regression import fit_linear_recalibration

                fitted[(arm, update)] = fit_linear_recalibration(y_dev, p_dev)
            else:
                from mival.metrics.calibration import fit_recalibration

                fitted[(arm, update)] = fit_recalibration(y_dev, p_dev, update)
            if fitted[(arm, update)] is None:
                warnings.append(
                    f"the {update!r} update could not be fitted on the development split of "
                    f"arm {arm} (no records, one class, a constant score or no convergence); "
                    "it has no update rows"
                )
        mapping = fitted[(arm, update)]
        if mapping is None:
            continue
        updated = frame.copy()
        updated[value_column] = mapping(frame[value_column].to_numpy(dtype=float))
        yield update, updated, _updated_threshold(mapping, threshold)


#: The prediction columns a slice unit reads besides its label column.
_UNIT_COLUMNS = ("prob", "pred_value", "label_time_days", BOOTSTRAP_UNIT)


def _unit_frame(frame, label_column: str):
    """The columns of ``frame`` a ``_slice_unit`` reads, so a worker is sent only those."""
    return frame[[label_column] + [
        column for column in _UNIT_COLUMNS if column != label_column and column in frame.columns
    ]]


def _slice_unit(
    key: RunKey,
    frame,
    label_column: str,
    outcome: str,
    subgroup: str,
    threshold: float,
    settings: _Settings,
    contaminated: bool,
    legacy_threshold: Optional[float],
    update: str,
):
    """``(metrics rows, curve rows)`` of one slice — the unit a worker process runs.

    Module level and argument-only so that it pickles under ``spawn``; it
    reads nothing but its arguments and resamples from its own
    ``default_rng(settings.seed)``, so its rows are the same in any process
    and in any order. An update's curves are drawn for a probability arm only.
    """
    rows = _rows_for_slice(
        key=key, frame=frame, label_column=label_column, outcome=outcome,
        subgroup=subgroup, threshold=threshold, settings=settings,
        contaminated=contaminated, legacy_threshold=legacy_threshold, update=update,
    )
    if update != UPDATE_NONE and is_regression(key):
        return rows, []
    return rows, _curve_rows_for_slice(
        key, frame, label_column, outcome, subgroup, settings, update
    )


def _describe_slice_unit(args) -> str:
    """Which slice a failed ``_slice_unit`` was scoring (its error message prefix)."""
    key, outcome, subgroup, update = args[0], args[3], args[4], args[9]
    return f"evaluate {key.to_string()} outcome={outcome} subgroup={subgroup} update={update}"


def _collect_figure_series(key: RunKey, slice_frame, label_column: str, outcome: str, curves, scatter):
    """Add one whole test slice of the primary outcome to the figure series.

    The figures compare arms on held-out data: one series per arm. Dev folds
    are internal-CV predictions and stay in metrics_long only. Same
    restriction the metrics use: a record without a label for this outcome is
    not part of this curve.
    """
    labelled = slice_frame[slice_frame[label_column].notna()]
    if not len(labelled) or is_survival(key):
        # No survival figure yet (spec: stage 2).
        return
    if is_regression(key):
        # The same rows _regression_rows scores: a null pred_value would plot
        # as a gap the metrics never saw.
        labelled = labelled[labelled["pred_value"].notna()]
        if not len(labelled):
            return
        scatter.setdefault(outcome, []).append(
            (
                _curve_label(key),
                labelled[label_column].to_numpy(dtype=float),
                labelled["pred_value"].to_numpy(dtype=float),
            )
        )
    else:
        curves.setdefault(outcome, []).append(
            (
                _curve_label(key),
                labelled[label_column].to_numpy(dtype=float),
                labelled["prob"].to_numpy(dtype=float),
            )
        )


def _updated_threshold(mapping, threshold: float) -> float:
    import numpy as np

    if threshold != threshold:
        return threshold
    return float(mapping(np.array([threshold]))[0])


def _legacy_metrics(y, p, threshold: float) -> Dict[str, float]:
    """The threshold-dependent measures at the legacy operating point."""
    from mival.metrics import PARAMETER_SEP, confusion_counts
    from mival.metrics.discrimination import contingency_metrics

    counts = confusion_counts(y, p, threshold)
    derived = contingency_metrics(*counts)
    values = {name: derived[name] for name in ("sensitivity", "specificity", "ppv", "npv")}
    values.update(zip(("tp", "fp", "tn", "fn"), (float(count) for count in counts)))
    return {f"{name}{PARAMETER_SEP}{LEGACY_POLICY}": value for name, value in values.items()}


def _regression_rows(
    key: RunKey,
    frame,
    label_column: str,
    outcome: str,
    subgroup: str,
    settings: _Settings,
    contaminated: bool,
    update: str = UPDATE_NONE,
) -> List[Dict[str, Any]]:
    """Category 'regression': error, fit, calibration and one AUROC per cut.

    A regression arm has no operating threshold (spec addendum), so this is a
    parallel path to :func:`_rows_for_slice` rather than a branch inside it:
    ``compute_metrics`` needs a probability and a threshold that this arm does
    not have.

    The core metrics are not about an event: one unstratified person-level
    bootstrap, ``n_events`` NULL, and the suppression floor is on ``n``. Each
    cut is its own binary question, so each gets its own bootstrap stratified
    on that cut's events, and its row carries that cut's event count and mark.
    """
    import numpy as np

    from mival.metrics.bootstrap import bootstrap_ci, event_strata
    from mival.metrics.regression import auroc_cut, core_regression_metrics, cut_metric_name

    usable = frame[frame[label_column].notna() & frame["pred_value"].notna()]
    y = usable[label_column].to_numpy(dtype=float)
    p = usable["pred_value"].to_numpy(dtype=float)
    n = int(y.size)
    if n == 0:
        return []
    groups = usable[BOOTSTRAP_UNIT].to_numpy() if BOOTSTRAP_UNIT in usable.columns else None
    resample = settings.bootstrap and settings.bootstrap_replicates > 0

    def intervals_of(statistic, point, strata):
        if not resample:
            return {}
        return bootstrap_ci(
            statistic,
            n_rows=n,
            n_replicates=settings.bootstrap_replicates,
            seed=settings.seed,
            groups=groups,
            strata=strata,
            alpha=settings.bootstrap_alpha,
            point=point,
        )

    def core(indices) -> Dict[str, float]:
        return core_regression_metrics(y[indices], p[indices])

    point = core(np.arange(n))
    rows = _metric_rows(
        key, subgroup, outcome, update, point, intervals_of(core, point, None),
        n=n, n_events=None, n_nonevents=None,
        settings=settings, contaminated=contaminated,
    )
    for cut in settings.cuts_for(key.label_def):
        name = cut_metric_name(cut)
        events = cut.events(y).astype(float)
        n_events = int(events.sum())

        def statistic(indices, cut=cut, name=name) -> Dict[str, float]:
            return {name: auroc_cut(y[indices], p[indices], cut)}

        cut_point = statistic(np.arange(n))
        rows.extend(
            _metric_rows(
                key, subgroup, outcome, update, cut_point,
                intervals_of(statistic, cut_point, event_strata(events, groups)),
                n=n, n_events=n_events, n_nonevents=n - n_events,
                settings=settings, contaminated=contaminated,
            )
        )
    return rows


def _survival_rows(
    key: RunKey,
    frame,
    label_column: str,
    outcome: str,
    subgroup: str,
    settings: _Settings,
    contaminated: bool,
) -> List[Dict[str, Any]]:
    """Category 'survival': discrimination at each horizon, and calibration for an absolute risk.

    ``label_column`` is the event flag; the follow-up time is ``label_time_days``.
    One person-level bootstrap covers every metric. A risk score carries no
    baseline hazard, so an arm whose output is not a probability by the
    horizon (``_Settings.risk_horizon``) gets its Brier, IPA and O:E rows
    with no value and reason ``no_absolute_risk`` — present, so the absence
    is visible, but never a number.
    """
    import numpy as np

    from mival.metrics.bootstrap import bootstrap_ci, event_strata
    from mival.metrics.survival import ABSOLUTE_RISK_STEMS, horizon_metric, survival_metrics

    usable = frame[
        frame[label_column].notna() & frame["label_time_days"].notna() & frame["pred_value"].notna()
    ]
    event = usable[label_column].to_numpy(dtype=float)
    time = usable["label_time_days"].to_numpy(dtype=float)
    risk = usable["pred_value"].to_numpy(dtype=float)
    n = int(event.size)
    if n == 0:
        return []
    n_events = int(event.sum())
    risk_horizon = settings.risk_horizon(key)

    def statistic(indices) -> Dict[str, float]:
        return survival_metrics(
            time[indices], event[indices], risk[indices], settings.horizons,
            absolute=risk_horizon is not None, risk_horizon=risk_horizon,
        )

    point = statistic(np.arange(n))
    intervals: Dict[str, Any] = {}
    if settings.bootstrap and settings.bootstrap_replicates > 0:
        groups = usable[BOOTSTRAP_UNIT].to_numpy() if BOOTSTRAP_UNIT in usable.columns else None
        intervals = bootstrap_ci(
            statistic,
            n_rows=n,
            n_replicates=settings.bootstrap_replicates,
            seed=settings.seed,
            groups=groups,
            strata=event_strata(event, groups),
            alpha=settings.bootstrap_alpha,
            point=point,
        )
    # Every horizon the risk is not a risk by: present, NULL, and marked.
    reasons: Dict[str, str] = {}
    for horizon in settings.horizons:
        for stem in ABSOLUTE_RISK_STEMS:
            name = horizon_metric(stem, horizon)
            if name not in point:
                point[name] = None
                reasons[name] = REASON_NO_ABSOLUTE_RISK
    return _metric_rows(
        key, subgroup, outcome, UPDATE_NONE, point, intervals,
        n=n, n_events=n_events, n_nonevents=n - n_events,
        settings=settings, contaminated=contaminated, reasons=reasons,
    )


def _metric_rows(
    key: RunKey,
    subgroup: str,
    outcome: str,
    update: str,
    point: Mapping[str, Any],
    intervals: Mapping[str, Any],
    n: int,
    n_events: Optional[int],
    n_nonevents: Optional[int],
    settings: _Settings,
    contaminated: bool,
    reasons: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    """One ``metrics_long`` row per entry of ``point`` — the only row builder.

    ``intervals`` maps a metric to its ``BootstrapCI``; a metric without one
    gets no interval. ``n == 0`` marks a declared level with no records
    (``no_data``); otherwise too few events or non-events is
    ``insufficient_events`` (spec §4.5: computed, marked, and not used in
    conclusions). A metric that is not about an event (``n_events`` None, the
    regression core) is held to the same floor on ``n`` instead. ``reasons``
    gives a metric its own reason, which wins over the slice's (a metric that
    is undefined for this arm is that, whatever the event count).
    """
    from mival.metrics import category_of
    from mival.metrics.tiers import tier_of

    if n == 0:
        reason: Optional[str] = REASON_NO_DATA
    elif n_events is None:
        reason = REASON_INSUFFICIENT_EVENTS if n < settings.min_events else None
    elif min(n_events, n_nonevents) < settings.min_events:
        reason = REASON_INSUFFICIENT_EVENTS
    else:
        reason = None
    base = key.to_dict()
    rows: List[Dict[str, Any]] = []
    reasons = reasons or {}
    for metric, value in point.items():
        interval = intervals.get(metric)
        row_reason = reasons.get(metric, reason)
        row = dict(base)
        # The report-only axes (spec §3.4) are written from REPORT_AXES rather
        # than by name, so they stay in step with the run_key module.
        row.update(dict(zip(REPORT_AXES, (subgroup, outcome))))
        row.update(
            {
                "update": update,
                "category": category_of(metric),
                "metric": metric,
                "tier": tier_of(metric),
                "value": _finite(value),
                "se": _finite(interval.se) if interval is not None else None,
                "ci_lo": _finite(interval.ci_lo) if interval is not None else None,
                "ci_hi": _finite(interval.ci_hi) if interval is not None else None,
                "n": int(n),
                "n_events": None if n_events is None else int(n_events),
                "suppressed": row_reason is not None,
                "reason": row_reason,
                "contaminated": bool(contaminated),
            }
        )
        rows.append(row)
    return rows


def _finite(value) -> Optional[float]:
    """NaN becomes NULL in the table: an undefined metric is not a number."""
    if value is None:
        return None
    number = float(value)
    return None if math.isnan(number) or math.isinf(number) else number


# ---------------------------------------------------------------------------
# Operating thresholds (spec §4.4) — fitted off the test split, never on it
# ---------------------------------------------------------------------------


def _fit_thresholds(
    predictions,
    settings: _Settings,
    warnings: List[str],
    recorded: Optional[Mapping[Tuple[str, ...], float]] = None,
) -> Dict[Tuple[str, ...], float]:
    """One operating threshold per arm — see ``mival.stages._predictions``."""
    return fit_operating_thresholds(
        predictions,
        settings.threshold_policy_settings(),
        warnings,
        recorded,
        settings.label_column_for,
    )


def _recorded_thresholds_and_contamination(
    path: Path, settings: _Settings, warnings: List[str]
):
    """Read the models stage's ``train_log.jsonl`` (spec §4.4, §3.5)."""
    return recorded_thresholds_and_contamination(
        path, settings.threshold_policy, warnings
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _draw_figures(
    ctx: StageContext,
    rows,
    curves,
    scatter,
    settings: _Settings,
    n_analysed: int,
    n_dev: int,
    warnings: List[str],
):
    if importlib.util.find_spec("matplotlib") is None:
        warnings.append("matplotlib is not installed; figures were skipped")
        return []

    from mival import figures as figure_module
    from mival.metrics import decision_curve

    directory = ctx.layout.artifact(FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    stages, exclusions, side = _stard_counts(ctx, n_analysed, n_dev)
    if stages:
        written.append(
            figure_module.stard_flow(stages, exclusions, directory / "stard_flow.png", side=side)
        )

    primary = curves.get(settings.primary_outcome, [])
    if primary:
        written.append(figure_module.roc_pr_curves(primary, directory / "roc_pr.png"))
        written.append(
            figure_module.calibration_curves(
                primary, directory / "calibration.png", n_knots=settings.calibration_knots
            )
        )
        grid = _decision_grid(settings)
        written.append(
            figure_module.decision_curves(
                [(label, decision_curve(y, p, grid)) for label, y, p in primary],
                directory / "decision_curve.png",
                band=settings.decision_band,
            )
        )

    points = scatter.get(settings.primary_outcome, [])
    if points:
        written.append(figure_module.regression_scatter(points, directory / "regression_scatter.png"))

    written.extend(_slice_figures(rows, directory, figure_module))
    return written


def _decision_grid(settings: _Settings):
    """Thresholds of the decision-curve figure.

    With a stated band the figure zooms around it, as it always has; without
    one it spans the study's net-benefit grid.
    """
    import numpy as np

    if settings.decision_band is None:
        return [float(value) for value in settings.net_benefit_grid]
    lo, hi = float(settings.decision_band[0]), float(settings.decision_band[1])
    return [float(value) for value in np.linspace(max(lo / 2.0, 0.001), max(hi * 3.0, 0.3), 60)]


def _slice_figures(rows, directory: Path, figure_module):
    """Degradation, forest and rank-reversal figures, all read off the long table.

    They are built from ``metrics_long`` rather than from the raw predictions
    so that what a figure shows is exactly what the table reports.
    """
    written = []
    # Recalibrated rows are a separate question; the figures show the model as shipped.
    shipped = [row for row in rows if row["update"] == UPDATE_NONE]
    auroc_rows = [row for row in shipped if row["metric"] == "auroc"]
    f1_rows = [row for row in shipped if row["metric"] == "f1"]

    degradation: Dict[str, List[Tuple[str, float, float, float]]] = {}
    for row in auroc_rows:
        if row["subgroup"] != SUBGROUP_ALL or row["value"] is None:
            continue
        degradation.setdefault(f"{row['model_id']}/{row['training_mode']}", []).append(
            (
                str(row["perturbation_id"]),
                row["value"],
                _or_nan(row["ci_lo"]),
                _or_nan(row["ci_hi"]),
            )
        )
    degradation = {name: sorted(points, key=lambda item: item[0]) for name, points in degradation.items()}
    if any(len(points) > 1 for points in degradation.values()):
        written.append(
            figure_module.degradation_curve(
                degradation, directory / "perturbation_degradation.png", metric_label="AUROC"
            )
        )

    forest = [
        (
            f"{row['model_id']}/{row['subgroup']}",
            row["value"],
            _or_nan(row["ci_lo"]),
            _or_nan(row["ci_hi"]),
        )
        for row in auroc_rows
        if row["subgroup"] != SUBGROUP_ALL and row["value"] is not None
    ]
    if forest:
        flags = [
            row["suppressed"]
            for row in auroc_rows
            if row["subgroup"] != SUBGROUP_ALL and row["value"] is not None
        ]
        written.append(
            figure_module.forest_plot(
                forest,
                directory / "subgroup_forest.png",
                reference=0.5,
                metric_label="AUROC",
                suppressed=flags,
            )
        )

    conditions: Dict[str, List[Tuple[str, float]]] = {}
    for row in f1_rows:
        if row["value"] is None:
            continue
        conditions.setdefault(f"{row['perturbation_id']}|{row['subgroup']}", []).append(
            (f"{row['model_id']}/{row['training_mode']}", row["value"])
        )
    conditions = {name: values for name, values in sorted(conditions.items()) if len(values) > 1}
    if len(conditions) > 1:
        written.append(
            figure_module.rank_reversal(
                conditions, directory / "rank_reversal_f1.png", metric_label="F1"
            )
        )
    return written


def _or_nan(value) -> float:
    return float("nan") if value is None else float(value)


def _stard_counts(
    ctx: StageContext, n_analysed: int, n_dev: int
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, str, int]], List[Tuple[str, int]]]:
    """Build the STARD flow from the exclusion ledger (spec §3.6).

    The diagram is a query, never a hand-kept count. The ledger records what
    was dropped and this stage knows what survived, so the top box — the
    number assessed for eligibility — is their sum and cannot drift from the
    per-stage boxes below it.

    ``n_analysed`` counts the test split's ECGs only: STARD's "analysed" is
    the set the reported accuracy is estimated on. The development split's
    ECGs are a side box (third return value) branching off the last chain box,
    not a step of the chain: dev was set aside, not filtered into test. The
    two add up to what survived the exclusions.
    """
    from mival.pipeline.stage import STAGE_ORDER

    root = exclusions_root(ctx.layout.runs_root, ctx.study_id)
    runs = load_runs(ctx.inputs[RUNS_INPUT]) if RUNS_INPUT in ctx.inputs else None
    ledger = read_exclusions(root, runs=runs)
    tallies = ledger.groupby(["stage", "reason_code"]).size().reset_index(name="n")
    exclusions = [
        (str(row.stage), str(row.reason_code), int(row.n)) for row in tallies.itertuples()
    ]
    per_stage = {str(stage): int(count) for stage, count in ledger.groupby("stage").size().items()}
    ordered = [stage for stage in STAGE_ORDER if stage in per_stage]
    ordered.extend(sorted(stage for stage in per_stage if stage not in STAGE_ORDER))
    entered = int(n_analysed) + int(n_dev) + int(sum(per_stage.values()))
    stages: List[Tuple[str, int]] = [("assessed for eligibility", entered)]
    remaining = entered
    for stage in ordered:
        remaining -= per_stage[stage]
        stages.append((stage, remaining))
    stages.append(("analysed", int(n_analysed)))
    return stages, exclusions, [(STARD_DEV_BOX, int(n_dev))]
