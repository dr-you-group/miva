"""Describe stage: per-ECG attributes, Table 1 and the exclusion flow by group.

Runs after preprocess and drops nothing. It joins what earlier stages already
wrote — retrieve's ``ecg_attributes`` (age, sex, year for every queried ECG)
and preprocess's ``source_index`` (device and recording size) — onto the
cohort, and writes three tables:

- ``attributes.parquet``: one row per cohort ECG. ``sex`` and ``age_band`` are
  stored as Categoricals with every level, so evaluate can tell an empty
  subgroup from one that does not exist.
- ``table1_long.csv``: Table 1 in long form, by overall/dev/test and, when
  ``describe.outcome`` is set, by outcome within test.
- ``exclusions_long.csv``: exclusion counts per stage and reason, overall and
  by sex and age band.

Nothing is masked here. Small cells are hidden at export, which knows the
site's ``min_cell``; masking twice would hide the numbers the site itself
needs to check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from mival import criteria
from mival.pipeline.layout import exclusions_root
from mival.pipeline.ledger import load_runs, read_exclusions
from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import join_key, read_table

#: Output artifacts, relative to ``ctx.layout.artifacts_dir``.
ATTRIBUTES = "attributes.parquet"
TABLE1_LONG = "table1_long.csv"
EXCLUSIONS_LONG = "exclusions_long.csv"

ATTRIBUTE_COLUMNS = (
    "image_occurrence_id", "person_id", "age_years", "age_band", "sex", "ecg_year",
    "device_manufacturer", "device_model", "sampling_rate_hz", "duration_s", "n_leads",
)
TABLE1_COLUMNS = ("variable", "level", "column", "stat", "value", "n")
EXCLUSIONS_LONG_COLUMNS = ("stage", "reason_code", "group", "level", "n")

#: Optional input: ``runs.json`` of this invocation. Without it every part file
#: of the ledger is read.
RUNS_INPUT = "runs"

SEX_LEVELS = ("M", "F", "unknown")
#: The level for a missing age band or sex, and for a missing device tag.
UNKNOWN = "unknown"
COLUMN_OVERALL = "overall"
SPLITS = ("dev", "test")
EXCLUSION_GROUPS = ("all", "sex", "age_band")

#: Continuous summaries. min and max are left out on purpose: each is one
#: person's value, and a site may not export a single person's value.
CONTINUOUS_STATS = ("mean", "sd", "median", "q1", "q3", "p5", "p95")

#: Variables summarised as continuous when the cohort carries them with values.
_CONTINUOUS_OPTIONAL = ("label_value", "label_delta_days", "label_time_days")


@dataclass(frozen=True)
class DescribeSpec:
    age_bands: Tuple[float, ...]
    outcome: Optional[str]

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "DescribeSpec":
        if body.get("conditions"):
            raise ValueError("describe.conditions is reserved; v1 reads no condition_occurrence")
        outcome = body.get("outcome")
        return cls(age_bands=criteria.age_bands(body), outcome=None if outcome is None else str(outcome))


class DescribeStage(Stage):
    name = "describe"
    reason_codes = frozenset()  # describe reports; it never drops a record

    def required_inputs(self) -> tuple:
        return ("cohort_index", "cohort_split", "ecg_attributes", "source_index")

    def run(self, ctx: StageContext) -> StageResult:
        spec = DescribeSpec.from_mapping(ctx.spec or {})
        cohort = read_table(ctx.input_path("cohort_index"))
        if spec.outcome is not None and spec.outcome not in cohort.columns:
            raise ValueError(
                f"describe.outcome {spec.outcome!r} is not a cohort_index column; "
                f"columns are {sorted(cohort.columns)}"
            )
        ecg_attributes = read_table(ctx.input_path("ecg_attributes"))
        attributes = build_attributes(
            cohort, ecg_attributes, read_table(ctx.input_path("source_index")), spec.age_bands
        )
        split = read_table(ctx.input_path("cohort_split"))

        attributes_path = ctx.layout.artifact(ATTRIBUTES)
        attributes_path.parent.mkdir(parents=True, exist_ok=True)
        attributes[list(ATTRIBUTE_COLUMNS)].to_parquet(attributes_path, index=False)

        table = table1(cohort, attributes, split, spec.outcome)
        table1_path = ctx.layout.artifact(TABLE1_LONG)
        table.to_csv(table1_path, index=False, lineterminator="\n")

        runs = load_runs(ctx.inputs[RUNS_INPUT]) if RUNS_INPUT in ctx.inputs else None
        ledger = read_exclusions(exclusions_root(ctx.layout.runs_root, ctx.study_id), runs)
        flow = exclusions_by_group(ledger, ecg_attributes, spec.age_bands)
        exclusions_path = ctx.layout.artifact(EXCLUSIONS_LONG)
        flow.to_csv(exclusions_path, index=False, lineterminator="\n")

        return StageResult(
            outputs=[attributes_path, table1_path, exclusions_path],
            counts={"in": len(cohort), "out": len(attributes)},
        )


def _key(series):
    """Join key: ledger rows carry ids as strings, artifacts may carry integers."""
    return join_key(series)


def _with_age_and_sex(frame, bounds):
    """Add ``age_band`` and a Categorical ``sex`` with every level."""
    import pandas as pd

    frame = frame.copy()
    frame["age_band"] = criteria.assign_age_band(frame["age_years"].to_numpy(dtype=float), bounds)
    sex = frame["sex"].where(frame["sex"].isin(SEX_LEVELS), UNKNOWN)
    frame["sex"] = pd.Categorical(sex, categories=list(SEX_LEVELS))
    return frame


def build_attributes(cohort, ecg_attributes, source_index, bounds):
    """One row per cohort ECG, in cohort order.

    A cohort ECG absent from ``ecg_attributes`` gets sex ``unknown`` and a
    missing age; one absent from ``source_index`` (unreadable, so excluded by
    preprocess) gets device ``unknown`` and missing recording size.
    """
    left = cohort[["image_occurrence_id", "person_id"]].copy()
    left["_key"] = _key(left["image_occurrence_id"])
    ecg = ecg_attributes[["image_occurrence_id", "ecg_year", "age_years", "sex"]].copy()
    ecg["_key"] = _key(ecg.pop("image_occurrence_id"))
    source = source_index[[
        "image_occurrence_id", "device_manufacturer", "device_model",
        "sampling_rate_hz", "duration_s", "n_leads",
    ]].copy()
    source["_key"] = _key(source.pop("image_occurrence_id"))
    frame = (
        left.merge(ecg.drop_duplicates("_key"), on="_key", how="left")
        .merge(source.drop_duplicates("_key"), on="_key", how="left")
        .drop(columns="_key")
    )
    frame["sex"] = frame["sex"].fillna(UNKNOWN)
    for column in ("device_manufacturer", "device_model"):
        frame[column] = frame[column].fillna(UNKNOWN)
    return _with_age_and_sex(frame, bounds)


def _continuous(values) -> List[Tuple[str, float]]:
    """The fixed continuous summaries of the non-missing values."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return []
    q = np.quantile(x, [0.5, 0.25, 0.75, 0.05, 0.95])
    sd = float(np.std(x, ddof=1)) if x.size > 1 else float("nan")
    return list(zip(CONTINUOUS_STATS, (float(x.mean()), sd, *[float(v) for v in q])))


def _level(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return UNKNOWN
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "%g" % float(value)
    return str(value)


def _levels(series, categories=None) -> List[Tuple[str, int]]:
    """Counts per level; declared categories are listed even when empty."""
    labels = [_level(v) for v in series.tolist()]
    tally: Dict[str, int] = {}
    for label in labels:
        tally[label] = tally.get(label, 0) + 1
    order = list(categories or [])
    order += sorted(label for label in tally if label not in order)
    return [(label, tally.get(label, 0)) for label in order]


def _column_rows(frame, column: str, outcome: Optional[str] = None) -> List[Dict[str, Any]]:
    """Table 1 rows for one column (overall, a split, or an outcome group)."""
    rows: List[Dict[str, Any]] = []
    n_ecgs = len(frame)

    def add(variable, level, stat, value, n):
        rows.append({"variable": variable, "level": level, "column": column,
                     "stat": stat, "value": value, "n": int(n)})

    def continuous(variable, values):
        n = int(np.sum(~np.isnan(np.asarray(values, dtype=float))))
        for stat, value in _continuous(values):
            add(variable, "", stat, value, n)

    def categorical(variable, series, categories=None, percent=True):
        for level, count in _levels(series, categories):
            add(variable, level, "count", count, n_ecgs)
            if percent:
                add(variable, level, "percent", 100.0 * count / n_ecgs if n_ecgs else float("nan"), n_ecgs)

    def binary(variable, series):
        # A label's distribution is over the ECGs that have the label, so the
        # denominator (and ``n``) is the non-missing count and 1 + 0 = 100%.
        import pandas as pd

        values = pd.to_numeric(series, errors="coerce").dropna()
        n_known = len(values)
        for level, value in (("1", 1), ("0", 0)):
            count = int((values == value).sum())
            add(variable, level, "count", count, n_known)
            add(variable, level, "percent", 100.0 * count / n_known if n_known else float("nan"), n_known)

    per_person = frame.groupby("person_id").size()
    add("persons", "", "n_persons", int(len(per_person)), len(per_person))
    add("ecgs", "", "n_ecgs", n_ecgs, n_ecgs)
    continuous("ecgs_per_person", per_person.to_numpy())
    continuous("age_years", frame["age_years"].to_numpy())
    bands = list(frame["age_band"].cat.categories)
    categorical("age_band", frame["age_band"].astype(object), bands)
    categorical("sex", frame["sex"].astype(object), list(SEX_LEVELS))
    for variable in _CONTINUOUS_OPTIONAL:
        if variable in frame.columns and frame[variable].notna().any():
            continuous(variable, frame[variable].to_numpy())
    if "label_event" in frame.columns and frame["label_event"].notna().any():
        binary("label_event", frame["label_event"])
    if outcome is not None and outcome != "label_event":
        binary(outcome, frame[outcome])
    categorical("ecg_year", frame["ecg_year"], percent=False)
    categorical("device_manufacturer", frame["device_manufacturer"])
    categorical("sampling_rate_hz", frame["sampling_rate_hz"], percent=False)
    continuous("duration_s", frame["duration_s"].to_numpy())
    return rows


def table1(cohort, attributes, split, outcome: Optional[str]):
    """Table 1 in long form. ``age_years`` and the device rows are per ECG."""
    import pandas as pd

    labels = [c for c in ("label_value", "label_delta_days", "label_time_days", "label_event")
              if c in cohort.columns]
    extra = [outcome] if outcome is not None and outcome not in labels else []
    frame = attributes.copy()
    for column in labels + extra:
        frame[column] = cohort[column].to_numpy()
    by_person = split.drop_duplicates("person_id").set_index("person_id")["split"]
    frame["_split"] = frame["person_id"].map(by_person)

    rows = _column_rows(frame, COLUMN_OVERALL, outcome)
    for name in SPLITS:
        rows += _column_rows(frame[frame["_split"] == name], name, outcome)
    if outcome is not None:
        test = frame[frame["_split"] == "test"]
        values = pd.to_numeric(test[outcome], errors="coerce")
        for value in (1, 0):
            rows += _column_rows(test[values == value], f"test:{outcome}={value}", outcome)
    return pd.DataFrame(rows, columns=list(TABLE1_COLUMNS))


def exclusions_by_group(ledger, ecg_attributes, bounds):
    """Exclusions per (stage, reason_code), overall and by sex and age band.

    Attributes come from ``ecg_attributes``, which holds every queried ECG.
    An excluded ECG not found there counts as sex ``unknown`` and age band
    ``unknown``. Sex and age-band levels are listed even when zero; age band
    ``unknown`` only when some excluded ECG has no age.
    """
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    ecg = ecg_attributes[["image_occurrence_id", "age_years", "sex"]].copy()
    ecg["_key"] = _key(ecg.pop("image_occurrence_id"))
    frame = ledger[["image_occurrence_id", "stage", "reason_code"]].copy()
    frame["_key"] = _key(frame["image_occurrence_id"])
    frame = frame.merge(ecg.drop_duplicates("_key"), on="_key", how="left")
    frame["sex"] = frame["sex"].fillna(UNKNOWN)
    frame = _with_age_and_sex(frame, bounds)
    band_levels = list(frame["age_band"].cat.categories)
    if frame["age_band"].isna().any():
        band_levels.append(UNKNOWN)
    for (stage, reason), group in frame.groupby(["stage", "reason_code"], sort=True):
        rows.append({"stage": stage, "reason_code": reason, "group": "all", "level": "all", "n": len(group)})
        for level, count in _levels(group["sex"].astype(object), list(SEX_LEVELS)):
            rows.append({"stage": stage, "reason_code": reason, "group": "sex", "level": level, "n": count})
        for level, count in _levels(group["age_band"].astype(object), band_levels):
            rows.append({"stage": stage, "reason_code": reason, "group": "age_band", "level": level, "n": count})
    return pd.DataFrame(rows, columns=list(EXCLUSIONS_LONG_COLUMNS))
