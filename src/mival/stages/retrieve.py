"""Retrieve stage (spec §4.1): every ECG in the CDM with its nearest label."""

from __future__ import annotations

import decimal
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.study import REMOVED_KEYS
from mival.pipeline.tables import write_table

COHORT_INDEX = "cohort_index.parquet"
SUMMARY = "retrieve_summary.json"
SQL_DIR = "sql"
SQL_FILE = Path(__file__).with_name("sql") / "retrieve_ecg_label.sql"
#: retrieve.dbms. The label query uses a lateral join and date arithmetic, so it
#: has one file per DBMS; the person and snapshot queries are plain SQL shared
#: by both.
DBMS = ("postgresql", "sqlserver")
LABEL_SQL_FILES = {
    "postgresql": SQL_FILE,
    "sqlserver": Path(__file__).with_name("sql") / "retrieve_ecg_label.sqlserver.sql",
}
SNAPSHOT_SQL_FILE = Path(__file__).with_name("sql") / "label_snapshot.sql"
PERSON_SQL_FILE = Path(__file__).with_name("sql") / "retrieve_ecg_person.sql"

#: ``label_source.kind``. ``measurement`` is the original nearest-measurement
#: label; the others are derived from the person, death and visit tables or
#: from a per-ECG file, and leave the measurement cutoff columns empty.
LABEL_KINDS = ("measurement", "age_at_ecg", "death_within", "machine_measurement", "measurement_file")
#: Extra cohort_index columns written by each derived kind.
DERIVED_COLUMNS = {
    "age_at_ecg": (),
    "death_within": ("label_event", "label_time_days"),
    "machine_measurement": ("label_pr", "label_qrs", "label_qt"),
    "measurement_file": (),
}
#: Kinds whose label comes from a file named by ``label_source.path``.
FILE_KINDS = ("machine_measurement", "measurement_file")
#: Person covariates every derived kind writes, for cards that take them (L2-3).
COVARIATE_COLUMNS = ("cov_age_years", "cov_sex_male")
MALE_CONCEPT_ID = 8507
FEMALE_CONCEPT_ID = 8532
#: Written once per run for every ECG the query returned, before any exclusion,
#: so a later stage can describe the people the cohort left out. It stays at
#: the site: it is the only artifact that holds age and sex of excluded ECGs.
ECG_ATTRIBUTES = "ecg_attributes.parquet"
ECG_ATTRIBUTE_COLUMNS = ("image_occurrence_id", "person_id", "ecg_year", "age_years", "sex")
#: Plausible machine-measured intervals in ms; outside is a device artefact or
#: a missing-value code (29999, 65535, ...).
INTERVAL_BOUNDS = {"label_pr": (60.0, 400.0), "label_qrs": (40.0, 250.0), "label_qt": (200.0, 700.0)}
#: label_source.follow_up_after_last_visit_days default: how long after the last
#: hospital contact a death is still taken as recorded (death_within).
DEFAULT_FOLLOW_UP_DAYS = 365

COHORT_INDEX_COLUMNS = (
    "image_occurrence_id", "person_id", "local_path", "index_datetime",
    "label_datetime", "label_delta_days", "label_value",
    "label_primary", "label_sens1", "label_sens2", "label_sens3",
)
#: retrieve.ecg_selection. ``per_person: all`` keeps every eligible ECG;
#: ``one`` keeps one per person, chosen by ``rule``. CDM image_occurrence
#: carries a date and no time, so ECGs on the same day are ordered by
#: image_occurrence_id.
PER_PERSON = ("all", "one")
SELECTION_RULES = ("first", "last", "nearest_label", "random")
#: Kinds whose label is a measurement near the ECG, so label_delta_days exists.
NEAREST_LABEL_KINDS = ("measurement", "measurement_file")

#: The ledger detail of ``path_unresolvable``.
UNRESOLVABLE_DETAIL = "empty image_occurrence.local_path"

REASON_CODES = frozenset({"label_missing", "label_implausible", "path_unresolvable", "file_missing", "not_selected"})

Query = Callable[[str, Mapping[str, Any]], Any]
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
#: SQL Server names a schema in another database as ``database.schema`` and is
#: case-insensitive, so one dot and upper case are allowed there.
_SQLSERVER_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


@dataclass(frozen=True)
class RetrieveSpec:
    dsn_env: str
    schema: str
    label_concept_id: Optional[int]  # required for label_kind measurement, unused otherwise
    modality_concept_id: int
    window_days: int
    window_days_sens2: int
    implausible_below: Optional[float]  # None: no lower bound, nothing is excluded as implausible
    require_local_file: bool
    primary_cutoff: Optional[float] = None  # label_primary = value <= primary_cutoff; None: no class labels
    sens1_cutoff: Optional[float] = None    # label_sens1 = value < sens1_cutoff
    follow_up_after_last_visit_days: int = DEFAULT_FOLLOW_UP_DAYS
    label_kind: str = "measurement"
    horizon_days: int = 365
    measurements_path: Optional[str] = None
    per_person: str = "all"
    selection_rule: Optional[str] = None
    dbms: str = "postgresql"

    @property
    def one_per_person(self) -> bool:
        return self.per_person == "one"

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "RetrieveSpec":
        if not body.get("dsn_env"):
            raise ValueError("retrieve.dsn_env is required")
        if body.get("modality_concept_id") is None:
            raise ValueError("retrieve.modality_concept_id is required")
        dbms = str(body.get("dbms", "postgresql"))
        if dbms not in DBMS:
            raise ValueError(f"retrieve.dbms {dbms!r} is not one of {list(DBMS)}")
        schema = str(body.get("schema", "cdm"))
        pattern = _SQLSERVER_IDENTIFIER if dbms == "sqlserver" else _IDENTIFIER
        if not pattern.match(schema):
            raise ValueError(f"retrieve.schema {schema!r} is not a plain identifier")
        if body.get("local_path_root") is not None:
            # load_study refuses it first; this covers a spec given as a file path.
            raise ValueError(f"retrieve.local_path_root: {REMOVED_KEYS[('retrieve', 'local_path_root')]}")
        source = body.get("label_source") or {}
        kind = str(source.get("kind", "measurement"))
        if kind not in LABEL_KINDS:
            raise ValueError(f"retrieve.label_source.kind {kind!r} is not one of {list(LABEL_KINDS)}")
        if kind in FILE_KINDS and not source.get("path"):
            raise ValueError(f"retrieve.label_source.path is required for kind {kind!r}")
        if kind == "measurement" and body.get("label_concept_id") is None:
            raise ValueError("retrieve.label_concept_id is required")
        per_person, rule = _ecg_selection(body, kind)
        return cls(
            label_kind=kind,
            horizon_days=int(source.get("horizon_days", 365)),
            follow_up_after_last_visit_days=int(
                source.get("follow_up_after_last_visit_days", DEFAULT_FOLLOW_UP_DAYS)),
            measurements_path=None if source.get("path") is None else str(source["path"]),
            per_person=per_person,
            selection_rule=rule,
            dsn_env=str(body["dsn_env"]),
            dbms=dbms,
            schema=schema,
            label_concept_id=None if body.get("label_concept_id") is None else int(body["label_concept_id"]),
            modality_concept_id=int(body["modality_concept_id"]),
            window_days=int(body.get("window_days", 7)),
            window_days_sens2=int(body.get("window_days_sens2", 30)),
            implausible_below=None if body.get("implausible_below", 5) is None else float(body.get("implausible_below", 5)),
            require_local_file=bool(body.get("require_local_file", True)),
            primary_cutoff=None if body.get("primary_cutoff") is None else float(body["primary_cutoff"]),
            sens1_cutoff=None if body.get("sens1_cutoff") is None else float(body["sens1_cutoff"]),
        )

    def params(self, window_days: int) -> dict:
        return {
            "label_concept_id": self.label_concept_id,
            "window_days": window_days,
            "modality_concept_id": self.modality_concept_id,
        }


def _ecg_selection(body: Mapping[str, Any], kind: str):
    """``(per_person, rule)`` from ``ecg_selection``, or from the older ``one_per_person`` flag."""
    selection = body.get("ecg_selection")
    if selection is None:
        # one_per_person: true was the only way to ask for one ECG, and it meant random.
        return ("one", "random") if body.get("one_per_person") else ("all", None)
    if "one_per_person" in body:
        raise ValueError("retrieve: give ecg_selection or one_per_person, not both")
    per_person = str(selection.get("per_person", ""))
    if per_person not in PER_PERSON:
        raise ValueError(f"retrieve.ecg_selection.per_person {per_person!r} is not one of {list(PER_PERSON)}")
    rule = selection.get("rule")
    if per_person == "all":
        if rule is not None:
            raise ValueError("retrieve.ecg_selection.rule only applies to per_person: one")
        return "all", None
    if rule not in SELECTION_RULES:
        raise ValueError(f"retrieve.ecg_selection.rule {rule!r} is not one of {list(SELECTION_RULES)}")
    if rule == "nearest_label" and kind not in NEAREST_LABEL_KINDS:
        raise ValueError(f"retrieve.ecg_selection.rule nearest_label needs a label kind in {list(NEAREST_LABEL_KINDS)}, not {kind!r}")
    return per_person, rule


class RetrieveStage(Stage):
    name = "retrieve"
    reason_codes = REASON_CODES

    def __init__(self, query: Optional[Query] = None) -> None:
        self._query = query  # injected in tests; None means the spec's DBMS via dsn_env

    def config_inputs(self, spec: Mapping[str, Any]) -> dict:
        """The per-ECG label file of ``machine_measurement``, so its content is hashed."""
        source = (spec or {}).get("label_source") or {}
        if source.get("kind") in FILE_KINDS and source.get("path"):
            # The machine-measurement key predates the file kind; kept so
            # existing delineation runs keep their config_hash.
            name = "machine_measurements" if source["kind"] == "machine_measurement" else "label_file"
            return {name: Path(str(source["path"]))}
        return {}

    def run(self, ctx: StageContext) -> StageResult:
        spec = RetrieveSpec.from_mapping(ctx.spec or {})
        query = self._query or (lambda sql, params: QUERY_FUNCTIONS[spec.dbms](sql, params, spec.dsn_env))
        if spec.label_kind != "measurement":
            return self._run_derived(ctx, spec, query)
        sql = LABEL_SQL_FILES[spec.dbms].read_text(encoding="utf-8").format(schema=spec.schema)

        snapshot_sql = SNAPSHOT_SQL_FILE.read_text(encoding="utf-8").format(schema=spec.schema)
        snapshot_params = {"label_concept_id": spec.label_concept_id}

        narrow = query(sql, spec.params(spec.window_days))
        wide = query(sql, spec.params(spec.window_days_sens2))
        snapshot = read_snapshot(query(snapshot_sql, snapshot_params))
        if narrow["label_value"].notna().sum() == 0:
            raise ValueError(
                f"no label within {spec.window_days} days for any ECG; check retrieve.label_concept_id"
            )
        frame = narrow.merge(
            wide[["image_occurrence_id", "label_value"]].rename(columns={"label_value": "label_value_sens2"}),
            on="image_occurrence_id", how="left",
        )
        frame["local_path"] = frame["local_path"].map(resolve_local_path)

        attributes_path = write_ecg_attributes(ctx, frame)
        kept = exclude(frame, spec, ctx.ledger)
        if spec.one_per_person:
            kept = one_per_person(kept, ctx.seed, ctx.ledger, spec.selection_rule)
        kept = attach_labels(kept, spec)
        index_path = write_table(kept[list(COHORT_INDEX_COLUMNS)].to_dict("records"),
                                 ctx.layout.artifact(COHORT_INDEX), COHORT_INDEX_COLUMNS)
        sql_paths = [write_sql(ctx, sql, spec.params(days), f"window_{days}")
                     for days in (spec.window_days, spec.window_days_sens2)]
        sql_paths.append(write_sql(ctx, snapshot_sql, snapshot_params, "label_snapshot"))
        summary_path = write_summary(ctx, spec, frame, kept, snapshot)
        return StageResult(outputs=[index_path, summary_path, attributes_path, *sql_paths],
                           counts={"in": len(frame), "out": len(kept)})


    def _run_derived(self, ctx: StageContext, spec: RetrieveSpec, query: Query) -> StageResult:
        """Labels computed per ECG rather than read from a nearby measurement."""
        import numpy as np

        sql = PERSON_SQL_FILE.read_text(encoding="utf-8").format(schema=spec.schema)
        params = {"modality_concept_id": spec.modality_concept_id}
        frame = query(sql, params)
        frame["local_path"] = frame["local_path"].map(resolve_local_path)
        frame = frame.assign(label_datetime=None, label_delta_days=np.nan, label_primary=np.nan,
                             label_sens1=np.nan, label_sens2=np.nan, label_sens3=np.nan)
        frame = derive_labels(frame, spec)
        attributes_path = write_ecg_attributes(ctx, frame)
        kept = exclude_derived(frame, spec, ctx.ledger)
        if spec.one_per_person:
            kept = one_per_person(kept, ctx.seed, ctx.ledger, spec.selection_rule)
        columns = COHORT_INDEX_COLUMNS + DERIVED_COLUMNS[spec.label_kind] + COVARIATE_COLUMNS
        index_path = write_table(kept[list(columns)].to_dict("records"),
                                 ctx.layout.artifact(COHORT_INDEX), columns)
        sql_path = write_sql(ctx, sql, params, "ecg_person")
        payload = {
            "label_kind": spec.label_kind,
            "n_ecg": int(len(frame)),
            "n_out": int(len(kept)),
            "n_persons_out": int(kept["person_id"].nunique()),
            "excluded": ctx.ledger.counts(),
        }
        for column in ("label_value",) + DERIVED_COLUMNS[spec.label_kind]:
            values = kept[column].dropna()
            if len(values):
                payload[f"{column}_quantiles"] = {
                    f"p{int(q * 100)}": float(v) for q, v in values.quantile([0.05, 0.5, 0.95]).items()}
        if spec.label_kind == "death_within":
            payload["events"] = int(kept["label_event"].sum())
        summary_path = ctx.layout.artifact(SUMMARY)
        summary_path.write_bytes(json.dumps(payload, indent=2).encode("utf-8"))
        return StageResult(outputs=[index_path, summary_path, attributes_path, sql_path],
                           counts={"in": len(frame), "out": len(kept)})


def derive_labels(frame, spec: RetrieveSpec):
    """Fill ``label_value`` or the kind's extra columns; NaN where none can be derived."""
    import numpy as np
    import pandas

    index = pandas.to_datetime(frame["index_datetime"])
    frame["cov_age_years"] = (index.dt.year - frame["year_of_birth"]).astype(float)
    gender = frame["gender_concept_id"] if "gender_concept_id" in frame else pandas.Series(np.nan, index=frame.index)
    frame["cov_sex_male"] = (gender == MALE_CONCEPT_ID).astype(float).where(gender.notna())
    if spec.label_kind == "age_at_ecg":
        frame["label_value"] = (index.dt.year - frame["year_of_birth"]).astype(float)
        return frame
    if spec.label_kind == "measurement_file":
        return _nearest_file_measurement(frame, index, spec)
    frame["label_value"] = np.nan
    if spec.label_kind == "death_within":
        death = pandas.to_datetime(frame["death_date"])
        last = pandas.to_datetime(frame["last_visit_end"])
        end = np.minimum(index + pandas.Timedelta(days=spec.horizon_days),
                         last + pandas.Timedelta(days=spec.follow_up_after_last_visit_days))
        event = death.notna() & (death <= end)
        stop = death.where(event, end)
        frame["label_event"] = event.astype(float).where(last.notna())
        frame["label_time_days"] = (stop - index).dt.days.astype(float).where(last.notna())
        return frame
    measurements = pandas.read_csv(spec.measurements_path,
                                   usecols=["study_id", "p_onset", "qrs_onset", "qrs_end", "t_end"])
    intervals = pandas.DataFrame({
        "study_id": measurements["study_id"].astype("int64"),
        "label_pr": measurements["qrs_onset"] - measurements["p_onset"],
        "label_qrs": measurements["qrs_end"] - measurements["qrs_onset"],
        "label_qt": measurements["t_end"] - measurements["qrs_onset"],
    }).astype({"label_pr": float, "label_qrs": float, "label_qt": float})
    for column, (low, high) in INTERVAL_BOUNDS.items():
        raw = intervals[column]
        intervals[column] = raw.where(raw.between(low, high), -1.0)  # -1: present but implausible
    frame["study_id"] = frame["local_path"].map(_study_id)
    merged = frame.merge(intervals.drop_duplicates("study_id"), on="study_id", how="left")
    return merged


def _nearest_file_measurement(frame, index, spec: RetrieveSpec):
    """Nearest value within ``window_days`` from a (person_id, measurement_datetime, value) file.

    Same rule as the measurement SQL: smallest day distance, ties to the
    earlier measurement. For labels whose numbers did not survive the CDM load
    (MIMIC NT-proBNP: value '___', valuenum present).
    """
    import pandas

    labs = pandas.read_csv(spec.measurements_path)
    labs = labs[labs["value"].notna()].copy()
    labs["measurement_datetime"] = pandas.to_datetime(labs["measurement_datetime"])
    labs["measurement_date"] = labs["measurement_datetime"].dt.normalize()
    ecg = frame[["image_occurrence_id", "person_id"]].assign(_index_date=index.dt.normalize())
    pairs = ecg.merge(labs, on="person_id", how="inner")
    pairs["_delta"] = (pairs["measurement_date"] - pairs["_index_date"]).dt.days
    pairs = pairs[pairs["_delta"].abs() <= spec.window_days]
    pairs = pairs.assign(_abs=pairs["_delta"].abs()).sort_values(["image_occurrence_id", "_abs", "measurement_datetime"])
    best = pairs.drop_duplicates("image_occurrence_id")[["image_occurrence_id", "value", "measurement_datetime", "_delta"]]
    best = best.rename(columns={"value": "label_value", "measurement_datetime": "label_datetime", "_delta": "label_delta_days"})
    out = frame.drop(columns=["label_value", "label_datetime", "label_delta_days"], errors="ignore")
    return out.merge(best, on="image_occurrence_id", how="left")


def _study_id(path: Any) -> Optional[int]:
    """The ``s<digits>`` directory of a MIMIC-IV-ECG path."""
    match = re.search(r"/s(\d+)/", str(path))
    return int(match.group(1)) if match else None


def exclude_derived(frame, spec: RetrieveSpec, ledger):
    def drop(frame, mask, code, detail):
        for row in frame[mask].itertuples(index=False):
            ledger.record(row.image_occurrence_id, row.person_id, code, detail(row))
        return frame[~mask]

    if spec.label_kind == "measurement_file":
        frame = drop(frame, frame["label_value"].isna(), "label_missing",
                     lambda r: f"no value within {spec.window_days} days in the label file")
        frame = drop(frame, _implausible(frame, spec), "label_implausible",
                     lambda r: f"label_value={r.label_value}")
    elif spec.label_kind == "age_at_ecg":
        frame = drop(frame, frame["label_value"].isna(), "label_missing", lambda r: "no year_of_birth")
        frame = drop(frame, frame["label_value"] < 18, "label_implausible", lambda r: f"age={r.label_value}")
    elif spec.label_kind == "death_within":
        frame = drop(frame, frame["label_time_days"].isna(), "label_implausible",
                     lambda r: "no visit to bound follow-up")
        # A death on the ECG's own day is an event at time 0 and stays; no
        # follow-up at all (censored at or before the ECG) is excluded.
        no_follow_up = (frame["label_time_days"] < 0) | (
            (frame["label_time_days"] == 0) & (frame["label_event"] == 0))
        frame = drop(frame, no_follow_up, "label_implausible",
                     lambda r: f"follow-up {r.label_time_days} days")
    else:
        columns = list(DERIVED_COLUMNS["machine_measurement"])
        frame = drop(frame, frame[columns].isna().any(axis=1), "label_missing",
                     lambda r: "no machine measurement for this study_id")
        frame = drop(frame, (frame[columns] < 0).any(axis=1), "label_implausible",
                     lambda r: "interval outside plausible bounds or a missing-value code")
    frame = drop(frame, frame["local_path"].isna(), "path_unresolvable",
                 lambda r: UNRESOLVABLE_DETAIL)
    if spec.require_local_file:
        frame = drop(frame, ~frame["local_path"].map(lambda p: Path(p).is_file()), "file_missing",
                     lambda r: r.local_path)
    return frame


def one_per_person(frame, seed, ledger, rule="random"):
    """One ECG per person chosen by ``rule``; the rest are ledgered as not selected.

    random: seeded; first/last: by ECG date, ties to the lower image_occurrence_id;
    nearest_label: smallest |label_delta_days|, ties to the earlier ECG.
    """
    import numpy as np
    import pandas

    ordered = frame.sort_values("image_occurrence_id").reset_index(drop=True)
    day = pandas.to_datetime(ordered["index_datetime"])
    if rule == "random":
        keys = ordered.assign(_k1=np.random.default_rng(seed).random(len(ordered)))[["_k1"]]
    elif rule == "first":
        keys = pandas.DataFrame({"_k1": day})
    elif rule == "last":
        keys = pandas.DataFrame({"_k1": -day.astype("int64")})
    elif rule == "nearest_label":
        keys = pandas.DataFrame({"_k1": ordered["label_delta_days"].abs(), "_k2": day})
    else:
        raise ValueError(f"unknown ECG selection rule {rule!r}")
    ranked = pandas.concat([ordered[["person_id", "image_occurrence_id"]], keys], axis=1)
    chosen = ranked.sort_values(["person_id", *keys.columns, "image_occurrence_id"]).groupby("person_id").head(1).index
    mask = ordered.index.isin(chosen)
    for row in ordered[~mask].itertuples(index=False):
        ledger.record(row.image_occurrence_id, row.person_id, "not_selected", f"one ECG per person ({rule})")
    return ordered[mask]


def resolve_local_path(cdm_path: Any) -> Optional[str]:
    """The DICOM file of one ECG: the CDM ``image_occurrence.local_path`` as it is stored.

    Only surrounding whitespace is stripped; an empty or missing value is
    ``None`` (``path_unresolvable``). The path is never rewritten, so the CDM
    must hold paths this server can open.
    """
    if cdm_path is None or cdm_path != cdm_path or not str(cdm_path).strip():
        return None
    return str(cdm_path).strip()


def _implausible(frame, spec: RetrieveSpec):
    """Rows whose label is at or below ``implausible_below``; none when it is null."""
    if spec.implausible_below is None:
        import pandas

        return pandas.Series(False, index=frame.index)
    return frame["label_value"] <= spec.implausible_below


def exclude(frame, spec: RetrieveSpec, ledger):
    """Drop rows in the spec's order, so each record is excluded for one reason only."""
    def drop(frame, mask, code, detail):
        for row in frame[mask].itertuples(index=False):
            ledger.record(row.image_occurrence_id, row.person_id, code, detail(row))
        return frame[~mask]

    frame = drop(frame, frame["label_value"].isna(), "label_missing",
                 lambda r: f"no label within {spec.window_days} days")
    frame = drop(frame, _implausible(frame, spec), "label_implausible",
                 lambda r: f"label_value={r.label_value}")
    frame = drop(frame, frame["local_path"].isna(), "path_unresolvable",
                 lambda r: UNRESOLVABLE_DETAIL)
    if spec.require_local_file:
        frame = drop(frame, ~frame["local_path"].map(lambda p: Path(p).is_file()), "file_missing",
                     lambda r: r.local_path)
    return frame


def attach_labels(frame, spec: RetrieveSpec):
    import numpy as np

    value, wide = frame["label_value"], frame["label_value_sens2"]
    # Each class label comes from its own cutoff; a label whose cutoff the
    # study does not give stays NaN. Nullable labels are float with NaN:
    # parquet round-trips that, and the models stage already reads NaN as
    # "no label".
    primary, sens1, sens2 = (np.nan,) * 3
    if spec.primary_cutoff is not None:
        primary = (value <= spec.primary_cutoff).astype("int64")
        sens2 = (wide <= spec.primary_cutoff).astype("float64").where(wide.notna())
    if spec.sens1_cutoff is not None:
        sens1 = (value < spec.sens1_cutoff).astype("int64")
    return frame.assign(label_primary=primary, label_sens1=sens1, label_sens2=sens2, label_sens3=np.nan)


def read_snapshot(frame) -> dict:
    """The label table's row count and highest measurement_id when this run read it."""
    if len(frame) != 1:
        raise ValueError(
            f"the label snapshot query returned {len(frame)} rows; it aggregates and "
            "must return exactly one"
        )
    row = frame.iloc[0]
    # Postgres returns count as bigint and max(measurement_id) as whatever the
    # column is, which psycopg may hand back as Decimal; JSON wants neither.
    return {
        "label_rows": _as_int(row["n_rows"]),
        "max_measurement_id": _as_int(row["max_measurement_id"]),
    }


def _as_int(value: Any) -> Optional[int]:
    return None if value is None or value != value else int(value)


def write_sql(ctx: StageContext, sql: str, params: Mapping[str, Any], name: str) -> Path:
    filled = sql
    for key, value in params.items():
        filled = filled.replace(f"%({key})s", str(value))
    path = ctx.layout.artifact(SQL_DIR, f"{name}.sql")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(filled.encode("utf-8"))
    return path


def write_summary(ctx: StageContext, spec: RetrieveSpec, frame, kept, snapshot: Mapping[str, Any]) -> Path:
    quantiles = kept["label_value"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    payload = {
        "snapshot": dict(snapshot),
        "n_ecg": int(len(frame)),
        "n_labelled": int(frame["label_value"].notna().sum()),
        "n_out": int(len(kept)),
        "n_persons_out": int(kept["person_id"].nunique()),
        "excluded": ctx.ledger.counts(),
        "label_value_quantiles": {f"p{int(q * 100)}": float(v) for q, v in quantiles.items()},
        "label_delta_days": {str(k): int(v) for k, v in kept["label_delta_days"].value_counts().sort_index().items()},
    }
    # A label is reported when its cutoff is set, even if no ECG is kept (count 0).
    positives = {name: int(kept[name].sum())
                 for name, cutoff in (("label_primary", spec.primary_cutoff), ("label_sens1", spec.sens1_cutoff),
                                      ("label_sens2", spec.primary_cutoff)) if cutoff is not None}
    if positives:
        payload["positives"] = positives
    path = ctx.layout.artifact(SUMMARY)
    path.write_bytes(json.dumps(payload, indent=2).encode("utf-8"))
    return path


def write_ecg_attributes(ctx: StageContext, frame) -> Path:
    """Age at the ECG and sex of every ECG the query returned, excluded or not."""
    import numpy as np
    import pandas

    index = pandas.to_datetime(frame["index_datetime"])
    birth = pandas.to_numeric(frame["year_of_birth"], errors="coerce") if "year_of_birth" in frame \
        else pandas.Series(np.nan, index=frame.index)
    gender = frame["gender_concept_id"] if "gender_concept_id" in frame else pandas.Series(None, index=frame.index, dtype=object)
    sex = gender.map({MALE_CONCEPT_ID: "M", FEMALE_CONCEPT_ID: "F"}).fillna("unknown")
    table = pandas.DataFrame({
        "image_occurrence_id": frame["image_occurrence_id"],
        "person_id": frame["person_id"],
        "ecg_year": index.dt.year,
        "age_years": (index.dt.year - birth).astype(float),
        "sex": sex,
    })
    return write_table(table.to_dict("records"), ctx.layout.artifact(ECG_ATTRIBUTES), ECG_ATTRIBUTE_COLUMNS)


def _decimals_to_float(frame):
    """Cast every column whose non-null values are ``decimal.Decimal`` to float64."""
    # psycopg maps Postgres numeric to decimal.Decimal, and numpy cannot mix
    # Decimal with float (quantile's linear interpolation, say), so every such
    # column is normalized once, right where a query result enters this
    # process. Nothing here is specific to any one column or label.
    for column in frame.columns:
        present = frame[column].dropna()
        if not present.empty and isinstance(present.iloc[0], decimal.Decimal):
            frame[column] = frame[column].astype(float)
    return frame


#: Connection keys an env file may use. The PG* names are libpq's own; the DB_*
#: names say the same for either DBMS. Values are host, port, database name,
#: user and password whatever the driver later calls them.
CONNECTION_KEYS = {
    "PGHOST": "host", "PGPORT": "port", "PGDATABASE": "dbname", "PGUSER": "user", "PGPASSWORD": "password",
    "DB_HOST": "host", "DB_PORT": "port", "DB_NAME": "dbname", "DB_USER": "user", "DB_PASSWORD": "password",
}


def read_connection_settings(dsn_env: str) -> dict:
    """The connection keywords held in an env file; the values are never logged."""
    # Kept out of os.environ: exporting them would leak the credentials into
    # every subprocess of this run, and into anything that dumps the
    # environment. Only the path of this file ever enters a hash.
    settings = {}
    for line in Path(dsn_env).read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if sep and key in CONNECTION_KEYS:
            settings[CONNECTION_KEYS[key]] = value.strip().strip("'\"")
    return settings


def _frame(cursor):
    import pandas

    columns = [column[0] for column in cursor.description]
    return _decimals_to_float(pandas.DataFrame(cursor.fetchall(), columns=columns))


def query_postgres(sql: str, params: Mapping[str, Any], dsn_env: str):
    """Run one query with libpq settings read from an env file; nothing is logged."""
    import psycopg

    settings = read_connection_settings(dsn_env)
    with psycopg.connect(**settings) as connection, connection.cursor() as cursor:
        cursor.execute(sql, params)
        return _frame(cursor)


def query_sqlserver(sql: str, params: Mapping[str, Any], dsn_env: str):
    """Run one query on SQL Server through pymssql (bundles FreeTDS, no ODBC driver needed)."""
    import pymssql

    settings = read_connection_settings(dsn_env)
    keywords = {"server": settings.get("host", "localhost"), "database": settings.get("dbname"),
                "user": settings.get("user"), "password": settings.get("password"),
                # 7.3 is the first TDS version that returns DATE columns as dates, not strings.
                "tds_version": "7.3"}
    if settings.get("port"):
        keywords["port"] = str(settings["port"])
    connection = pymssql.connect(**{k: v for k, v in keywords.items() if v is not None})
    try:
        cursor = connection.cursor()
        cursor.execute(sql, dict(params))
        return _frame(cursor)
    finally:
        connection.close()


QUERY_FUNCTIONS = {"postgresql": query_postgres, "sqlserver": query_sqlserver}
