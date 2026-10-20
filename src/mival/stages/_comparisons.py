"""Paired model comparisons of the evaluation stage (spec §4.5).

Two kinds of row go to ``comparisons.parquet``, automatic ones first:

automatic
    Every pair of distinct arms that share a ``label_def``, on the whole test
    split of the primary outcome, as shipped (no update) and unperturbed. A
    classification pair gets ``delta_auroc`` and ``delta_net_benefit`` at
    every point of the study's net-benefit grid (the threshold in ``x``); a
    regression pair ``delta_mae``; a survival pair ``delta_uno_c@h`` at each
    horizon. All are exploratory (``confirmatory=False``). The family a
    p-value is Holm-adjusted in is ``(label_def, metric)``: every pair of one
    label on one scalar metric (each horizon its own metric), so adding an
    unrelated label to a study leaves another label's ``p_holm`` as it was.
    A grid point is a curve, not a test, so its ``p_holm`` is NULL.
explicit
    The study's ``evaluate.comparisons`` entries, unchanged in meaning: two
    arms named by axis filters, compared on ``evaluate.comparison_metrics``.

Both are a paired bootstrap of the difference (``a - b``), never DeLong: the
two arms are joined on the record, and each replicate resamples persons once
and scores both arms on that one resample.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from mival.perturbation import BASELINE_ID
from mival.pipeline.runkey import AXES
from mival.stages._predictions import (
    TEST_SPLIT,
    arm_of as _arm_of,
    group_by_run_key as _group_by_run_key,
    is_regression,
    is_survival,
)

#: The ``comparison`` value of every automatic row.
AUTO_COMPARISON = "auto"


#: The record key the two arms are joined on.
RECORD_KEY = "image_occurrence_id"


def comparison_rows(
    predictions, settings, thresholds, warnings: List[str], workers: int = 1
) -> List[Dict[str, Any]]:
    """Automatic rows (unless ``evaluate.auto_comparisons`` is false), then explicit ones.

    ``workers`` spreads the automatic pairs over processes
    (``mival.pipeline.parallel``); the explicit ones, a handful the study
    names, run here.
    """
    rows: List[Dict[str, Any]] = []
    if settings.auto_comparisons:
        rows.extend(auto_comparison_rows(predictions, settings, thresholds, warnings, workers))
    rows.extend(_comparison_rows(predictions, settings, thresholds, warnings))
    return rows


# ---------------------------------------------------------------------------
# Automatic: every arm pair within a label_def
# ---------------------------------------------------------------------------


def auto_comparison_rows(
    predictions, settings, thresholds, warnings: List[str], workers: int = 1
) -> List[dict]:
    """Every pair of arms within each ``label_def``, ``a < b`` in arm order.

    Each pair is one unit of ``mival.pipeline.parallel.ordered_map``: its
    outcome column is resolved here (``_pair_label``, the only step that
    reads or dedupes against ``warnings``), the bootstrap runs in
    ``_pair_unit``, and the unit's own warnings are appended in pair order.
    """
    from mival.metrics.bootstrap import holm
    from mival.pipeline.parallel import ordered_map

    by_label: Dict[str, List[Tuple[Tuple[str, ...], Any, Any]]] = {}
    candidates: Dict[Tuple[str, ...], List[Tuple[Any, Any]]] = {}
    for key, frame in _group_by_run_key(predictions):
        if key.split != TEST_SPLIT or str(key.perturbation_id) != BASELINE_ID:
            continue
        candidates.setdefault(_arm_of(key), []).append((key, frame))
    if not candidates:
        warnings.append(
            f"no test run_key has perturbation_id {BASELINE_ID!r}; there are no automatic "
            "arm-pair comparisons"
        )
        return []
    for arm, keyed in sorted(candidates.items()):
        if len(keyed) != 1:
            warnings.append(
                f"arm {arm} has {len(keyed)} test run_keys (folds); it is left out of the "
                "automatic comparisons, which need one test prediction per arm"
            )
            continue
        key, frame = keyed[0]
        by_label.setdefault(key.label_def, []).append((arm, key, frame))

    def units():
        for label_def in sorted(by_label):
            arms = by_label[label_def]
            for i in range(len(arms)):
                for j in range(i + 1, len(arms)):
                    (key_a, frame_a), (key_b, frame_b) = arms[i][1:], arms[j][1:]
                    label = _pair_label(key_a, frame_a, key_b, frame_b, settings, warnings)
                    if label is None:
                        continue
                    columns = _pair_columns(key_a, label)
                    yield label_def, (
                        (key_a, frame_a[[c for c in columns if c in frame_a.columns]]),
                        (key_b, frame_b[[c for c in columns if c in frame_b.columns]]),
                        label, settings,
                    )

    by_label_rows: Dict[str, List[Dict[str, Any]]] = {}
    for label_def, (pair_rows, pair_warnings) in ordered_map(
        _pair_unit, units(), workers, describe=_describe_pair_unit
    ):
        warnings.extend(pair_warnings)
        by_label_rows.setdefault(label_def, []).extend(pair_rows)

    rows: List[Dict[str, Any]] = []
    for label_def in sorted(by_label):
        label_rows = by_label_rows.get(label_def, [])
        # Holm per (label_def, scalar metric), across this label's pairs;
        # grid rows are not tests.
        families: Dict[str, List[Dict[str, Any]]] = {}
        for row in label_rows:
            if row["x"] is None:
                families.setdefault(row["metric"], []).append(row)
        for family in families.values():
            for row, adjusted in zip(family, holm([row["p_value"] for row in family])):
                row["p_holm"] = adjusted
        rows.extend(label_rows)
    return rows


def _pair_label(key_a, frame_a, key_b, frame_b, settings, warnings: List[str]) -> Optional[str]:
    """The primary outcome's label column of a pair, or None when either arm lacks it."""
    outcome = settings.primary_outcome
    columns = dict(settings.outcome_columns(key_a, frame_a, warnings))
    columns_b = dict(settings.outcome_columns(key_b, frame_b, warnings))
    if outcome not in columns or outcome not in columns_b:
        return None
    return columns[outcome]


def _value_and_outcome_columns(key, label: str):
    if is_survival(key):
        return "pred_value", [label, "label_time_days"]
    if is_regression(key):
        return "pred_value", [label]
    return "prob", [label]


def _pair_columns(key, label: str) -> List[str]:
    """The columns ``_pair_rows`` reads, so a worker is sent only those."""
    value_column, outcome_columns = _value_and_outcome_columns(key, label)
    return [RECORD_KEY, "person_id", value_column] + outcome_columns


def _describe_pair_unit(args) -> str:
    (key_a, _), (key_b, _) = args[0], args[1]
    return f"comparison {key_a.to_string()} vs {key_b.to_string()}"


def _pair_unit(left, right, label: str, settings):
    """``(rows, warnings)`` of one pair — the unit a worker process runs."""
    warnings: List[str] = []
    return _pair_rows(left, right, label, settings, warnings), warnings


def _pair_rows(left, right, label: str, settings, warnings: List[str]) -> List[Dict[str, Any]]:
    """The comparison rows of one pair of test run_keys with a shared ``label_def``.

    ``label`` is the primary outcome's column (``_pair_label``).
    """
    import numpy as np

    from mival.metrics.bootstrap import event_strata, paired_bootstrap
    from mival.stages.evaluate import SUBGROUP_ALL

    key_a, frame_a = left
    key_b, frame_b = right
    outcome = settings.primary_outcome
    name = f"{key_a.to_string()} vs {key_b.to_string()}"
    value_column, outcome_columns = _value_and_outcome_columns(key_a, label)
    merged = _paired_frame(
        frame_a, frame_b, value_column, outcome_columns, name, warnings
    )
    if merged is None:
        return []

    groups = merged["person_id"].to_numpy() if "person_id" in merged.columns else None
    y = merged[f"{label}_a"].to_numpy(dtype=float)
    a = merged["value_a"].to_numpy(dtype=float)
    b = merged["value_b"].to_numpy(dtype=float)
    n = int(y.size)
    replicates = settings.bootstrap_replicates if settings.bootstrap else 0
    grid: Optional[np.ndarray] = None

    if is_survival(key_a):
        from mival.metrics.survival import censoring_survival, horizon_metric, uno_c

        time = merged["label_time_days_a"].to_numpy(dtype=float)
        event = y.astype(bool)
        names = [(horizon, horizon_metric("uno_c", horizon)) for horizon in settings.horizons]

        def difference(indices) -> Dict[str, float]:
            t, e = time[indices], event[indices]
            # One censoring fit per replicate: both arms score the same records.
            g = censoring_survival(t, e)
            return {
                f"delta_{metric}": uno_c(t, e, a[indices], horizon, g)
                - uno_c(t, e, b[indices], horizon, g)
                for horizon, metric in names
            }

        strata = event_strata(y, groups)
        n_events: Optional[int] = int(event.sum())
    elif is_regression(key_a):
        from mival.metrics.regression import mae

        def difference(indices) -> Dict[str, float]:
            return {"delta_mae": mae(y[indices], a[indices]) - mae(y[indices], b[indices])}

        strata = None
        n_events = None
    else:
        from mival.metrics.curves import grid_counts, grid_metrics
        from mival.metrics.discrimination import auroc

        grid = np.asarray(settings.net_benefit_grid, dtype=float)

        def net_benefit(y_values, p_values):
            counts = grid_counts(y_values, p_values, grid)
            return grid_metrics(counts, y_values.size, grid)["net_benefit"]

        def difference(indices) -> Dict[str, float]:
            y_draw = y[indices]
            values = {"delta_auroc": auroc(y_draw, a[indices]) - auroc(y_draw, b[indices])}
            delta = net_benefit(y_draw, a[indices]) - net_benefit(y_draw, b[indices])
            values.update((f"delta_net_benefit#{k}", float(d)) for k, d in enumerate(delta))
            return values

        strata = event_strata(y, groups)
        n_events = int(np.sum(y == 1.0))

    results = paired_bootstrap(
        difference, n, n_replicates=replicates, seed=settings.seed,
        groups=groups, strata=strata, alpha=settings.bootstrap_alpha,
    )
    rows = []
    for metric in sorted(results, key=_grid_order):
        result = results[metric]
        x: Optional[float] = None
        if "#" in metric:
            metric, index = metric.split("#", 1)
            x = float(grid[int(index)])
        rows.append(
            _row(
                AUTO_COMPARISON, False, key_a, key_b, SUBGROUP_ALL, outcome, metric,
                result, n, n_events, x,
            )
        )
    return rows


def _grid_order(metric: str):
    stem, _, index = metric.partition("#")
    return (stem, int(index) if index else -1)


def _paired_frame(frame_a, frame_b, value_column, outcome_columns, name, warnings):
    """The two arms joined on the record, each keeping its value and outcome columns.

    An inner join: a record scored by one arm only cannot be paired, and the
    number of such records (and of records without an outcome or a value) is
    reported in one warning. Columns are renamed ``*_a``/``*_b`` before the
    join rather than left to merge suffixes, which would apply to one side
    only when the other lacks the column. Two arms that disagree on the
    outcome of a shared record are an error: a paired comparison needs the
    same outcome on both sides.
    """
    needed = [value_column] + list(outcome_columns)
    keep_a = [RECORD_KEY] + (["person_id"] if "person_id" in frame_a.columns else []) + needed
    left = frame_a[keep_a].rename(
        columns={value_column: "value_a", **{c: f"{c}_a" for c in outcome_columns}}
    )
    right = frame_b[[RECORD_KEY] + needed].rename(
        columns={value_column: "value_b", **{c: f"{c}_b" for c in outcome_columns}}
    )
    merged = left.merge(right, on=RECORD_KEY, how="inner")
    complete = merged["value_a"].notna() & merged["value_b"].notna()
    for column in outcome_columns:
        complete &= merged[f"{column}_a"].notna() & merged[f"{column}_b"].notna()
    merged = merged[complete]
    dropped = len(frame_a) + len(frame_b) - 2 * len(merged)
    if merged.empty:
        warnings.append(f"comparison {name!r} has no records in common; skipped")
        return None
    for column in outcome_columns:
        if not merged[f"{column}_a"].astype(float).equals(merged[f"{column}_b"].astype(float)):
            raise ValueError(
                f"comparison {name!r}: the two arms disagree on {column!r} for records they "
                "share; a paired comparison requires the same outcome on both sides"
            )
    if dropped:
        warnings.append(
            f"comparison {name!r}: {dropped} prediction rows ({len(frame_a)} in a, "
            f"{len(frame_b)} in b, {len(merged)} paired) are not in both arms with a value "
            "and an outcome; the comparison uses the paired records only"
        )
    return merged


def _row(
    name, confirmatory, key_a, key_b, subgroup, outcome, metric, result, n, n_events, x
) -> Dict[str, Any]:
    from mival.metrics import category_of
    from mival.stages.evaluate import _finite

    return {
        "comparison": name,
        "confirmatory": bool(confirmatory),
        "arm_a": key_a.to_string(),
        "arm_b": key_b.to_string(),
        "subgroup": subgroup,
        "outcome": outcome,
        "category": category_of(metric[len("delta_"):]),
        "metric": metric,
        "x": x,
        "value": _finite(result.value),
        "ci_lo": _finite(result.ci_lo),
        "ci_hi": _finite(result.ci_hi),
        "p_value": _finite(result.p_value),
        "p_holm": None,
        "n": int(n),
        "n_events": None if n_events is None else int(n_events),
    }


# ---------------------------------------------------------------------------
# Explicit: the study's evaluate.comparisons
# ---------------------------------------------------------------------------


def _comparison_rows(
    predictions, settings, thresholds, warnings: List[str]
) -> List[Dict[str, Any]]:
    import numpy as np

    from mival.metrics import compute_metrics
    from mival.metrics.bootstrap import event_strata, paired_bootstrap_difference
    from mival.stages.evaluate import SUBGROUP_ALL

    rows: List[Dict[str, Any]] = []
    for comparison in settings.comparisons:
        name = str(comparison.get("name", "comparison"))
        key_a, frame_a = _select_arm(predictions, comparison["a"], name)
        key_b, frame_b = _select_arm(predictions, comparison["b"], name)
        skipped = [key for key in (key_a, key_b) if is_regression(key) or is_survival(key)]
        if skipped:
            # Comparison metrics (auroc, auprc, ...) are defined on
            # probabilities; a regression or survival arm has none. The
            # automatic comparisons score those arms on their own metrics.
            kind = "regression" if is_regression(skipped[0]) else "survival"
            warnings.append(
                f"comparison {name!r} involves a {kind} arm (label_def="
                f"{skipped[0].label_def!r}); skipped"
            )
            continue
        label_a = settings.label_column_for(key_a.label_def)
        label_b = settings.label_column_for(key_b.label_def)
        # Renamed before the join rather than relying on merge suffixes: when
        # both arms share a label_def the two label columns have the same name
        # and pandas would suffix them too, silently changing which column the
        # outcome is read from.
        left = frame_a[[RECORD_KEY, "person_id", "prob", label_a]].rename(
            columns={"prob": "prob_a", label_a: "y_a"}
        )
        right = frame_b[[RECORD_KEY, "prob", label_b]].rename(
            columns={"prob": "prob_b", label_b: "y_b"}
        )
        merged = left.merge(right, on=RECORD_KEY, how="inner")
        merged = merged[merged["y_a"].notna() & merged["y_b"].notna()]
        if merged.empty:
            warnings.append(f"comparison {name!r} has no records in common; skipped")
            continue
        if label_a == label_b and not merged["y_a"].equals(merged["y_b"]):
            raise ValueError(
                f"comparison {name!r}: the two arms disagree on {label_a!r} for records they "
                "share; a paired comparison requires the same outcome on both sides"
            )
        y = merged["y_a"].to_numpy(dtype=float)
        prob_a = merged["prob_a"].to_numpy(dtype=float)
        prob_b = merged["prob_b"].to_numpy(dtype=float)
        groups = merged["person_id"].to_numpy()

        def statistic(y_values, p_values) -> Dict[str, float]:
            values = compute_metrics(
                y_values,
                p_values,
                threshold=thresholds[_arm_of(key_a)],
                decision_thresholds=settings.decision_thresholds,
                murphy_bins=settings.murphy_bins,
                n_knots=settings.calibration_knots,
            )
            return {metric: values[metric] for metric in settings.comparison_metrics}

        differences = paired_bootstrap_difference(
            statistic,
            y,
            prob_a,
            prob_b,
            n_replicates=settings.bootstrap_replicates if settings.bootstrap else 0,
            seed=settings.seed,
            groups=groups,
            strata=event_strata(y, groups),
            alpha=settings.bootstrap_alpha,
        )
        for metric, result in sorted(differences.items()):
            rows.append(
                _row(
                    name,
                    comparison.get("confirmatory", False),
                    key_a,
                    key_b,
                    str(comparison.get("subgroup", SUBGROUP_ALL)),
                    str(comparison.get("outcome", settings.primary_outcome)),
                    f"delta_{metric}",
                    result,
                    y.size,
                    int(np.sum(y == 1.0)),
                    None,
                )
            )
    return rows


def _select_arm(predictions, filters: Mapping[str, Any], name: str):
    """Resolve a partial axis filter to exactly one run_key.

    Ambiguity is an error rather than a silent pool: quietly concatenating two
    folds into "the model" is how a comparison ends up measuring something
    nobody specified.
    """
    frame = predictions
    for axis, value in filters.items():
        if axis not in AXES:
            raise ValueError(f"comparison {name!r} filters on {axis!r}, which is not a run_key axis")
        frame = frame[frame[axis].astype(str) == str(value)]
    matches = list(_group_by_run_key(frame)) if len(frame) else []
    if len(matches) != 1:
        raise ValueError(
            f"comparison {name!r} arm {dict(filters)} selects {len(matches)} run_keys; "
            "it must select exactly one"
        )
    return matches[0]
