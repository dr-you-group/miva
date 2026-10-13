"""Profile stage (spec §4.2): freeze the person-level split and check the event count."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from mival.gates import EventCountError
from mival.pipeline.stage import Stage, StageContext, StageResult
from mival.pipeline.tables import read_table, write_table

COHORT_SPLIT = "cohort_split.parquet"
SUMMARY = "profile_summary.json"
COHORT_SPLIT_COLUMNS = ("person_id", "split", "fold")
#: ``stratify_on: none`` puts every person in one stratum.
STRATIFY_NONE = "none"


@dataclass(frozen=True)
class ProfileSpec:
    test_fraction: float
    n_folds: int
    stratify_on: str
    min_test_positives: int

    @classmethod
    def from_mapping(cls, body: Mapping[str, Any]) -> "ProfileSpec":
        return cls(
            test_fraction=float(body.get("test_fraction", 0.2)),
            n_folds=int(body.get("n_folds", 5)),
            stratify_on=str(body.get("stratify_on", "label_primary")),
            min_test_positives=int(body.get("min_test_positives", 100)),
        )


class ProfileStage(Stage):
    name = "profile"
    reason_codes = frozenset()  # profile describes; it never drops a record

    def required_inputs(self) -> tuple:
        return ("cohort_index",)

    def run(self, ctx: StageContext) -> StageResult:
        spec = ProfileSpec.from_mapping(ctx.spec or {})
        cohort = read_table(ctx.input_path("cohort_index"))
        if spec.stratify_on == STRATIFY_NONE:
            # One stratum: a continuous or survival label has no 0/1 column to
            # balance, and the event-count gate is then a row count.
            cohort = cohort.assign(**{STRATIFY_NONE: 1})
        # A person is positive if any of their ECGs is, so the split is stratified on people.
        persons = cohort.groupby("person_id")[spec.stratify_on].max().reset_index()
        split = assign_splits(persons, spec.stratify_on, spec.test_fraction, spec.n_folds, ctx.seed)

        test_positives = int(cohort[cohort["person_id"].isin(split.loc[split["split"] == "test", "person_id"])][spec.stratify_on].sum())
        if test_positives < spec.min_test_positives:
            raise EventCountError(
                f"test split holds {test_positives} positive ECGs; profile.min_test_positives is {spec.min_test_positives}"
            )

        split_path = write_table(split.to_dict("records"), ctx.layout.artifact(COHORT_SPLIT), COHORT_SPLIT_COLUMNS)
        summary_path = write_summary(ctx, cohort, split, spec, test_positives)
        return StageResult(outputs=[split_path, summary_path], counts={"in": len(cohort), "out": len(split)})


def assign_splits(persons, stratify_on: str, test_fraction: float, n_folds: int, seed):
    """Stratified person-level test split, then stratified folds over dev. Same seed, same split."""
    import pandas

    rng = np.random.default_rng(seed)
    parts = []
    for _, stratum in persons.sort_values("person_id").groupby(stratify_on):
        order = stratum.iloc[rng.permutation(len(stratum))]
        n_test = int(round(len(order) * test_fraction))
        n_dev = len(order) - n_test
        parts.append(pandas.DataFrame({
            "person_id": order["person_id"].to_numpy(),
            "split": ["test"] * n_test + ["dev"] * n_dev,
            # NaN rather than a nullable integer: parquet round-trips it and
            # the models stage reads NaN as "not cross-validated".
            "fold": [np.nan] * n_test + list((np.arange(n_dev) % n_folds).astype(float)),
        }))
    return pandas.concat(parts, ignore_index=True)


def write_summary(ctx: StageContext, cohort, split, spec: ProfileSpec, test_positives: int) -> Path:
    joined = cohort.merge(split, on="person_id")
    by_split = {
        name: {
            "n_persons": int(group["person_id"].nunique()),
            "n_ecgs": int(len(group)),
            "n_positive_ecgs": int(group[spec.stratify_on].sum()),
        }
        for name, group in joined.groupby("split")
    }
    payload = {
        "splits": by_split,
        "gate": {"min_test_positives": spec.min_test_positives, "test_positives": test_positives, "passed": True},
        "ecgs_per_person": {str(k): int(v) for k, v in cohort.groupby("person_id").size().value_counts().sort_index().items()},
    }
    # Both describe a continuous label and the delay to it; a cohort built for a
    # binary outcome carries neither, and their absence is not a failure.
    if "label_value" in cohort.columns:
        payload["label_value_quantiles"] = {
            f"p{int(q * 100)}": float(v)
            for q, v in cohort["label_value"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).items()
        }
    if "label_delta_days" in cohort.columns:
        payload["label_delta_days"] = {
            str(k): int(v)
            for k, v in cohort["label_delta_days"].value_counts().sort_index().items()
        }
    path = ctx.layout.artifact(SUMMARY)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
