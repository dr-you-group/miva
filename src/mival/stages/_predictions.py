"""Reading the models stage output — shared by evaluation and the audit.

Stages 5 and 6 both consume ``predictions/<run_key>.parquet`` and both need the
same three things from it: the axes recovered from the file name, a grouping by
run_key, and the operating threshold of an arm.

They must agree exactly. A case selected as "just below the operating
threshold" is meaningless if the audit's threshold is not the number evaluation
reported its sensitivity at, and two copies of a threshold policy drift. So the
code lives here once and both stages call it, rather than being duplicated with
a test asserting the two copies still match.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from mival.pipeline.runkey import AXES, RunKey
from mival.pipeline.tables import read_table
from mival.stages.models import (
    PREDICTION_COLUMNS,
    PRIMARY_THRESHOLD_POLICY,
    REGRESSION_LABEL_DEF,
    REGRESSION_LABEL_DEFS,
    SURVIVAL_LABEL_DEF,
    THRESHOLD_POLICIES,
)
from mival.threshold import SENS95_TARGET_SENSITIVITY

#: Spec §2.4: the two splits of the frozen cohort table. Only dev may carry a
#: threshold fit (spec §4.4).
DEV_SPLIT = "dev"
TEST_SPLIT = "test"

#: Fallback when no threshold can be fitted. 0.5 is not a clinical choice; it is
#: the neutral value used only when the run carries no development split, and it
#: is always accompanied by a warning in the manifest.
FALLBACK_THRESHOLD = 0.5

#: The axes a threshold is *not* a property of. Refitting one per split would
#: mean the test estimate used an operating point chosen on the test set.
_NON_ARM_AXES = ("split", "fold")

#: Numeric prediction columns that are entirely null on one arm's kind (a
#: regression arm's ``prob``/``logit``, a classification arm's ``pred_value``).
#: Concatenating an all-null (object-dtype) column against a typed one of the
#: same name is a pandas ``FutureWarning``; casting every frame the same way
#: first keeps the dtypes identical so the concat has nothing to warn about.
_NUMERIC_PREDICTION_COLUMNS = tuple(
    name for name in PREDICTION_COLUMNS if name.startswith("label_")
) + ("prob", "logit", "pred_value")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_predictions(path: Path, label_def: Optional[str] = None):
    """Read the models stage output, whether a directory or a single file.

    Prediction files are named by their run_key (spec §3.4), which is the only
    place ``label_def`` is recorded — it is not a column of
    ``PREDICTION_COLUMNS``. Parsing the name back into a ``RunKey`` recovers
    every axis, so the axes never have to be re-derived from the contents.

    A single file is also accepted, because the CLI addresses inputs as files
    (an input is hashed into the config_hash, and a directory has no hash).
    Such a file must carry the axes as columns; ``label_def`` may instead come
    from the stage spec, and is never guessed.
    """
    import pandas

    target = Path(path)
    if target.is_dir():
        frames = []
        for part in sorted(target.glob("*.parquet")):
            frame = read_table(part)
            try:
                key = RunKey.from_string(part.stem)
            except ValueError as exc:
                raise ValueError(
                    f"prediction file {part.name} is not named by a run_key: {exc}"
                ) from exc
            check_axes_agree(frame, key, part.name)
            for axis, value in key.to_dict().items():
                frame[axis] = value
            # See _NUMERIC_PREDICTION_COLUMNS: keep dtypes identical across arms.
            for column in _NUMERIC_PREDICTION_COLUMNS:
                if column in frame.columns:
                    frame[column] = frame[column].astype("float64")
            frames.append(frame)
        if not frames:
            raise ValueError(f"no prediction parquet files under {target}")
        predictions = pandas.concat(frames, ignore_index=True)
    else:
        predictions = read_table(target)
        if "label_def" not in predictions.columns and label_def is not None:
            predictions["label_def"] = label_def
        # See _NUMERIC_PREDICTION_COLUMNS: the same dtypes a directory gets, so
        # one file and a directory of one file do not read differently.
        for column in _NUMERIC_PREDICTION_COLUMNS:
            if column in predictions.columns:
                predictions[column] = predictions[column].astype("float64")

    missing = [
        column
        for column in ("image_occurrence_id", "person_id", "prob")
        if column not in predictions.columns
    ]
    if missing:
        raise ValueError(
            f"predictions are missing required columns {missing}; expected the schema "
            f"{list(PREDICTION_COLUMNS)}"
        )
    absent_axes = [axis for axis in AXES if axis not in predictions.columns]
    if absent_axes:
        raise ValueError(
            f"predictions are missing run_key axes {absent_axes}. A single prediction file "
            "must carry every axis as a column (set 'label_def' in the stage spec if it "
            "is not one of them); a directory of files takes the axes from the run_key in "
            "each file name instead."
        )
    return predictions


def check_axes_agree(frame, key: RunKey, filename: str) -> None:
    """A file's contents must match the run_key in its name.

    The name is what the axes are taken from, so a mismatch would silently
    attribute one run's predictions to another coordinate. Rather than pick a
    winner, refuse.
    """
    import pandas

    for axis, expected in key.to_dict().items():
        if axis not in frame.columns or expected is None:
            continue
        present = frame[axis].dropna()
        if present.empty:
            continue
        found = {str(value) for value in pandas.unique(present)}
        if found != {str(expected)}:
            raise ValueError(
                f"prediction file {filename} is named {axis}={expected!r} but its rows carry "
                f"{sorted(found)}"
            )


# ---------------------------------------------------------------------------
# Grouping — the only place that knows the axes exist
# ---------------------------------------------------------------------------


def group_by_run_key(predictions):
    """Yield ``(RunKey, frame)`` for every distinct coordinate present."""
    grouped = predictions.groupby(list(AXES), dropna=False, sort=True)
    for values, frame in grouped:
        if not isinstance(values, tuple):
            values = (values,)
        fields = dict(zip(AXES, values))
        fields["fold"] = optional_int(fields.get("fold"))
        yield RunKey.from_dict({axis: fields[axis] for axis in AXES}), frame


def optional_int(value) -> Optional[int]:
    if value is None:
        return None
    text = str(value)
    if text in ("", "nan", "None", "NaT", "na", "<NA>"):
        return None
    return int(float(text))


def arm_of(key: RunKey) -> Tuple[str, ...]:
    """The identity of a model configuration, ignoring split and fold.

    Thresholds are a property of the arm, not of the split it is measured on:
    refitting one per split would mean the test estimate used an operating point
    chosen on the test set (spec §4.4).
    """
    return tuple(getattr(key, axis) for axis in AXES if axis not in _NON_ARM_AXES)


def arm_axes() -> Tuple[str, ...]:
    """The axis names that :func:`arm_of` returns values for, in order."""
    return tuple(axis for axis in AXES if axis not in _NON_ARM_AXES)


def is_regression(key: RunKey) -> bool:
    """A regression arm reads ``label_<def>``/``pred_value`` and has no threshold."""
    return key.label_def in REGRESSION_LABEL_DEFS


def is_survival(key: RunKey) -> bool:
    """A survival arm reads ``label_event``/``label_time_days``/``pred_value``."""
    return key.label_def == SURVIVAL_LABEL_DEF


# ---------------------------------------------------------------------------
# Operating thresholds (spec §4.4) — fitted off the test split, never on it
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThresholdPolicy:
    """Everything the operating point depends on, read from a stage spec.

    Both stages parse it the same way, so an audit configured with a different
    ``threshold_policy`` than the evaluation it accompanies is a visible
    difference in two spec files rather than a hidden difference in two code
    paths.
    """

    threshold: Optional[float] = None
    policy: str = PRIMARY_THRESHOLD_POLICY
    legacy_threshold: float = FALLBACK_THRESHOLD
    dev_split: str = DEV_SPLIT
    sensitivity_target: float = SENS95_TARGET_SENSITIVITY

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "ThresholdPolicy":
        policy = str(spec.get("threshold_policy", PRIMARY_THRESHOLD_POLICY))
        if policy not in THRESHOLD_POLICIES:
            raise ValueError(
                f"threshold_policy {policy!r} is not one of {list(THRESHOLD_POLICIES)}"
            )
        return cls(
            threshold=None if spec.get("threshold") is None else float(spec["threshold"]),
            policy=policy,
            legacy_threshold=float(spec.get("legacy_threshold", FALLBACK_THRESHOLD)),
            dev_split=str(spec.get("dev_split", DEV_SPLIT)),
            sensitivity_target=float(
                spec.get("sensitivity_target", SENS95_TARGET_SENSITIVITY)
            ),
        )


def default_label_column(label_def: str) -> str:
    return f"label_{label_def}"


def fit_operating_thresholds(
    predictions,
    policy: ThresholdPolicy,
    warnings: List[str],
    recorded: Optional[Mapping[Tuple[str, ...], float]] = None,
    label_column_for: Callable[[str], str] = default_label_column,
) -> Dict[Tuple[str, ...], float]:
    """One operating threshold per arm.

    Preference order: an explicit value in the spec, then the value the models
    stage recorded when it fitted the policy, then a refit here on dev rows.
    The models stage is preferred because it is the stage that held the
    ModelCard and the out-of-fold dev probabilities; refitting is the fallback
    for the case where only a prediction table survived.
    """
    import numpy as np

    thresholds: Dict[Tuple[str, ...], float] = {}
    for key, _ in group_by_run_key(predictions):
        arm = arm_of(key)
        if is_regression(key) or is_survival(key):
            thresholds[arm] = float("nan")
            continue
        if arm in thresholds:
            continue
        if policy.threshold is not None:
            thresholds[arm] = policy.threshold
            continue
        if recorded and arm in recorded:
            thresholds[arm] = float(recorded[arm])
            continue
        development = predictions
        for axis, value in zip(arm_axes(), arm):
            development = development[development[axis] == value]
        development = development[development["split"] == policy.dev_split]
        label_column = label_column_for(key.label_def)
        if label_column in development.columns:
            development = development[development[label_column].notna()]
        else:
            development = development.iloc[0:0]
        y = development[label_column].to_numpy(dtype=float) if len(development) else np.array([])
        if y.size == 0 or len(np.unique(y)) < 2:
            thresholds[arm] = policy.legacy_threshold
            warnings.append(
                f"no usable development split for arm {arm}; falling back to the neutral "
                f"threshold {policy.legacy_threshold} — threshold-dependent metrics for "
                "this arm are not a refitted operating point"
            )
            continue
        thresholds[arm] = refit_threshold(
            development["prob"].to_numpy(dtype=float), y, policy
        )
    return thresholds


def refit_threshold(p, y, policy: ThresholdPolicy) -> float:
    """Delegate to the policies of ``mival.threshold`` (spec §4.4).

    The policy implementation is not repeated here. Beyond avoiding a second
    copy, ``SplitScores`` carries the split its probabilities came from and the
    refit functions refuse anything but dev, so the "no threshold is ever chosen
    on test" rule is enforced by the type rather than by this call site
    remembering to filter.
    """
    from mival.threshold import (
        REFIT_SENS95,
        REFIT_YOUDEN,
        SplitScores,
        refit_sens95,
        refit_youden,
    )

    scores = SplitScores.of(policy.dev_split, p, y)
    if policy.policy == REFIT_YOUDEN:
        return float(refit_youden(scores).value)
    if policy.policy == REFIT_SENS95:
        return float(refit_sens95(scores, policy.sensitivity_target).value)
    return policy.legacy_threshold


def recorded_thresholds_and_contamination(
    path: Path, policy_name: str, warnings: List[str]
):
    """Read the models stage's ``train_log.jsonl`` (spec §4.4, §3.5).

    That stage fitted every threshold policy on out-of-fold dev probabilities
    and ran the contamination gate against each ModelCard. Both belong to the
    run that produced the predictions, so when the log is available it is read
    rather than reconstructed.
    """
    import json

    thresholds: Dict[Tuple[str, ...], float] = {}
    contaminated: Dict[Tuple[str, ...], bool] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        arms = set()
        for text in entry.get("run_keys", ()):
            try:
                arms.add(arm_of(RunKey.from_string(text)))
            except ValueError:  # pragma: no cover - a malformed log line
                warnings.append(f"train_log holds an unparseable run_key: {text!r}")
        policy = policy_name or entry.get(
            "primary_threshold_policy", PRIMARY_THRESHOLD_POLICY
        )
        fit = (entry.get("thresholds") or {}).get(policy)
        flag = bool((entry.get("contamination") or {}).get("flag", False))
        for arm in arms:
            contaminated[arm] = flag
            if fit is not None and fit.get("value") is not None:
                thresholds[arm] = float(fit["value"])
            elif entry.get("primary_threshold_policy") is None:
                # A regression arm never fits a threshold; a missing fit here
                # is the expected shape, not a gap worth a warning.
                continue
            else:
                warnings.append(
                    f"train_log has no {policy!r} threshold for arm {arm}; it will be refitted "
                    "from the dev predictions"
                )
    return thresholds, contaminated
