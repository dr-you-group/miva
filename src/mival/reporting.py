"""Reporting drafts written into the export bundle: TRIPOD+AI checklist and Methods.

Both are drafts. The checklist says which TRIPOD+AI items the bundle already
holds an output for (``auto``) and which the authors write by hand
(``manual``); the Methods draft states the facts the engine knows — data
source, cohort rules, split, models, metrics and masking — so that nobody
retypes a window length or a grid from memory. Neither replaces reading the
results.

The checklist items and the bundle files that fill them live in
``reporting_tripod_ai.yaml`` beside this module, so a new mapping is a data
edit, not a code change.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

CHECKLIST_FILE = Path(__file__).with_name("reporting_tripod_ai.yaml")

AUTO = "auto"
MANUAL = "manual"


def load_checklist() -> List[Dict[str, Any]]:
    """The TRIPOD+AI items, in checklist order."""
    import yaml

    body = yaml.safe_load(CHECKLIST_FILE.read_text(encoding="utf-8"))
    return [dict(item, item=str(item["item"]), files=list(item.get("files") or ()))
            for item in body["items"]]


def tripod_checklist(bundle_index: Sequence[str]) -> str:
    """Markdown table: item, title, ``auto``/``manual``, the bundle files that fill it."""
    paths = sorted(str(p).replace("\\", "/") for p in bundle_index)
    lines = [
        "# TRIPOD+AI checklist (draft)",
        "",
        "`auto`: the bundle holds an output for this item (listed). `manual`: written by "
        "the authors. An `auto` item still needs its text in the manuscript.",
        "",
        "| 항목 | 제목 | 상태 | 파일 |",
        "|---|---|---|---|",
    ]
    for item in load_checklist():
        found = [p for p in paths if any(fnmatch.fnmatchcase(p, pat) for pat in item["files"])]
        status = AUTO if found else MANUAL
        lines.append(f"| {item['item']} | {item['title']} | {status} | {', '.join(found)} |")
    return "\n".join(lines) + "\n"


def cut_text(cuts: Any) -> str:
    """``[{value, op}]`` (or a per-label mapping of them) as ``> 65; <= 40``."""
    if isinstance(cuts, Mapping):
        return "; ".join(f"{label}: {cut_text(items)}" for label, items in cuts.items())
    return ", ".join(f"{item['op']} {float(item['value']):g}" for item in cuts or ())


def _grid_text(grid: Mapping[str, Any]) -> str:
    return (f"{float(grid['start']):g} to {float(grid['stop']):g} in steps of "
            f"{float(grid['step']):g} ({int(grid['n_points'])} thresholds)")


def _tier_table() -> List[str]:
    from mival.metrics.tiers import TIER_STEMS

    lines = ["| tier | metrics |", "|---|---|"]
    for tier, stems in TIER_STEMS.items():
        lines.append(f"| {tier} | {', '.join(stems)} |")
    return lines


def _subgroup_text(study: Mapping[str, Any]) -> str:
    """Which subgroups were prespecified and which exploratory, as the study states it."""
    subgroups = list(study.get("subgroups") or ())
    if not subgroups:
        return "No subgroup analyses were run."
    named = study.get("prespecified_subgroups")
    if named is None:
        return (f"Subgroups: {', '.join(subgroups)} [prespecified or exploratory: fill in; "
                "set export.prespecified_subgroups].")
    pre = [s for s in subgroups if s in named]
    rest = [s for s in subgroups if s not in named]
    parts = []
    if pre:
        parts.append(f"{', '.join(pre)} {'was' if len(pre) == 1 else 'were'} prespecified")
    if rest:
        parts.append(f"{', '.join(rest)} {'is' if len(rest) == 1 else 'are'} exploratory")
    return "Subgroups: " + "; ".join(parts) + "."


def _study_paragraphs(study: Mapping[str, Any]) -> List[str]:
    retrieve = dict(study.get("retrieve") or {})
    profile = dict(study.get("profile") or {})
    criteria = dict(study.get("criteria") or {})
    years = study.get("ecg_years")
    out = [f"## {study['study_id']}", ""]

    span = (f"ECGs were recorded from {years[0]} to {years[1]} (Table 1, ECGs by year)."
            if years else "The ECG years are in Table 1 (ECGs by year).")
    out += [
        "**Data source.** Data were read from the OMOP Common Data Model with its medical "
        f"imaging extension (MI-CDM) at site `{study['site']}`. {span}",
        "",
    ]

    rules = []
    if retrieve.get("label_concept_id") is not None:
        rules.append(f"the label is measurement concept {retrieve['label_concept_id']}")
    source = retrieve.get("label_source") if isinstance(retrieve.get("label_source"), Mapping) else {}
    if source.get("kind") is not None:
        text = f"the label is derived per ECG ({source['kind']}"
        if source.get("horizon_days") is not None:
            text += f", horizon {source['horizon_days']} days"
        follow_up = criteria.get("follow_up_after_last_visit_days")
        if follow_up is not None:
            text += (f"; a death is taken as recorded up to {follow_up} days after the last "
                     "visit, later follow-up is censored")
        rules.append(text + ")")
    if retrieve.get("window_days") is not None:
        text = f"an ECG is paired with a label within {retrieve['window_days']} days"
        if retrieve.get("window_days_sens2") is not None:
            text += f" ({retrieve['window_days_sens2']} days in a sensitivity analysis)"
        rules.append(text)
    if retrieve.get("ecg_selection") is not None:
        rules.append(f"ECG selection per person: {retrieve['ecg_selection']}")
    selection = retrieve.get("ecg_selection") if isinstance(retrieve.get("ecg_selection"), Mapping) else {}
    unit = ("Each person contributes one ECG, so Table 1 counts persons and ECGs alike."
            if selection.get("per_person") == "one" else
            "A person may contribute several ECGs: Table 1 counts and summarises ECGs (age is "
            "per ECG), its `persons` row gives the number of persons, and its `unit` column says "
            "which each row counts.")
    out += [
        "**Cohort.** " + ("; ".join(rules) + "." if rules else "[retrieve rules: fill in]")
        + " Exclusions by stage and reason are in the exclusion table and the STARD flow. " + unit,
        "",
    ]

    fraction = profile.get("test_fraction")
    split = (f"{float(fraction) * 100:g}% of persons" if fraction is not None
             else "[test fraction: fill in]")
    folds = (f", with {profile['n_folds']} cross-validation folds in development"
             if profile.get("n_folds") else "")
    out += [
        f"**Split.** {split} were held out for testing (seed {study.get('seed')}){folds}. "
        "The split is by person: all ECGs of a person fall in the same split.",
        "",
    ]

    arms = []
    for arm in study.get("arms") or ():
        corpora = ", ".join(arm.get("pretraining_corpora") or ()) or "not stated on the card"
        output = (arm.get("output") or {}).get("type", "not stated on the card")
        arms.append(f"- `{arm['model_id']}` ({arm['training_mode']}, label `{arm['label_def']}`): "
                    f"pretraining data {corpora}; output {output}.")
    out += ["**Models.**", ""] + (arms or ["- [models: fill in]"]) + [""]

    measures = [
        f"Intervals are percentile intervals from {study.get('bootstrap_replicates')} "
        "bootstrap replicates resampling persons, not ECGs.",
    ]
    if criteria.get("regression_cuts"):
        measures.append(
            f"Regression outputs are also scored as a classifier at the cuts "
            f"{cut_text(criteria['regression_cuts'])}.")
    if criteria.get("net_benefit_grid"):
        measures.append(
            "Decision curves and threshold metrics are reported on a fixed grid of threshold "
            f"probabilities from {_grid_text(criteria['net_benefit_grid'])}.")
    if criteria.get("age_bands"):
        from mival.criteria import age_band_labels

        labels = criteria.get("age_band_labels") or age_band_labels(criteria["age_bands"])
        measures.append(f"Age bands (years): {', '.join(labels)}.")
    measures += [
        "Recalibration fitted on the development split (intercept-only and logistic for "
        "classification, linear for regression) is reported beside the original test "
        "metrics; linear-probe arms are reported as fitted.",
        "All arm pairs within a label are compared by paired bootstrap; comparisons are "
        "exploratory and reported with Holm-adjusted p values.",
        _subgroup_text(study),
        f"A subgroup with fewer than {criteria.get('min_events')} events or non-events is "
        "computed but marked `insufficient_events`.",
        "In the exported tables `masked` marks a withheld count: every count from 1 to "
        f"{int(study['min_cell']) - 1}, and the cells withheld to protect them where counts add "
        "up to a total (subgroup levels, splits, outcome groups, exclusion reasons; "
        "complementary suppression). A linear-programming audit over all exported tables "
        "together (Table 1, metrics, comparisons, exclusions and the STARD flow, with each "
        "evaluated slice as a subset of the Table 1 records of its split and level) "
        f"checks that each small count's feasible range reaches {study['min_cell']} and spans "
        f"at least {int(study['min_cell']) - 1}.",
    ]
    out += ["**Metrics.** " + " ".join(measures), ""]
    return out


def methods_draft(studies: Sequence[Mapping[str, Any]]) -> str:
    """Methods paragraphs per study from the facts the export collected.

    Each study mapping holds ``study_id``, ``site``, ``seed``, ``ecg_years``
    (first and last year, or None), the ``retrieve`` and ``profile`` specs,
    ``arms`` (model_id, training_mode, label_def, pretraining_corpora,
    output), ``subgroups`` and ``prespecified_subgroups`` (None when the
    study does not say), ``criteria`` (regression_cuts, net_benefit_grid,
    age_bands, min_events), ``bootstrap_replicates`` and ``min_cell``.
    """
    lines = [
        "# Methods (draft)",
        "",
        "Generated from study.yaml, the model cards and the bundle tables. Check every "
        "sentence against the study protocol before use.",
        "",
        "Metric tiers (Van Calster/STRATOS core set; Metrics Reloaded for the rest):",
        "",
    ]
    lines += _tier_table() + [""]
    for study in studies:
        lines += _study_paragraphs(study)
    return "\n".join(lines)
