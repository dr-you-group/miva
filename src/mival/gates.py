"""Leakage and contamination gates (spec §3.5, gates 3 and 4).

The two gates sit at the same point of the pipeline and behave in opposite
ways, deliberately.

**Leakage stops the run.** A person whose ECGs were used to fit the model or to
choose its operating point, and who also appears in the test split, makes the
test estimate meaningless. There is no version of that result worth reporting,
so the run raises.

**Contamination only marks the run.** A model pretrained on the evaluation
cohort's own corpus is not disqualified: its performance is still worth
reporting, and whether it was contaminated is one of the axes along which the
results are read. MIMIC-IV-ECG appears in the pretraining of 9 of the 12
foundation models reviewed, so on a MIMIC run this gate genuinely fires; making
it blocking would leave nothing to report.

Neither gate knows anything model-specific. The pretraining corpora come from
the ModelCard and the cohort's own sources come from the study spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple


class LeakageError(RuntimeError):
    """Raised when a person used for fitting also appears in the test split."""


# How many overlapping identifiers to name in the exception. Enough to start
# debugging, few enough that the message stays readable and does not spray a
# cohort's worth of person_ids into a log.
_MAX_REPORTED = 5

_SEPARATORS = re.compile(r"[\s_.]+")


def check_leakage(
    fitting_person_ids: Iterable[Any],
    test_person_ids: Iterable[Any],
    context: str = "",
) -> None:
    """Gate 3 of spec §3.5. Raises on any overlap; returns ``None`` otherwise.

    ``fitting_person_ids`` is every person who influenced the model that will
    be applied to test — those whose records were trained on *and* those whose
    records chose the operating threshold. The spec words the gate as "person
    used in training"; a threshold refit on dev is also a decision fitted to
    those people, so an `inference_only` arm is held to the same rule. Split
    integrity is a property of ``cohort_split.parquet``, so a violation is a
    broken upstream split rather than a stray record, and there is nothing this
    stage could drop to repair it.
    """
    fitting = {str(value) for value in fitting_person_ids}
    held_out = {str(value) for value in test_person_ids}
    overlap = sorted(fitting & held_out)
    if not overlap:
        return
    shown = ", ".join(overlap[:_MAX_REPORTED])
    if len(overlap) > _MAX_REPORTED:
        shown += f", … ({len(overlap)} in total)"
    where = f" for {context}" if context else ""
    raise LeakageError(
        f"leakage gate{where}: {len(overlap)} person_id(s) are used for fitting and are "
        f"also in the test split: {shown}. Splits are frozen at the profile stage and "
        "are person-level by construction (spec §2.4), so this is an upstream split "
        "defect, not a droppable record."
    )


def _normalize(name: Any) -> str:
    """Fold a corpus name to its comparison form.

    Case, surrounding whitespace and the choice of separator are folded away,
    so ``MIMIC-IV-ECG``, ``mimic_iv_ecg`` and ``mimic iv ecg`` all match. The
    asymmetry of the two error modes justifies the leniency: a missed overlap
    silently turns a contaminated result into a clean-looking one, whereas a
    spurious match only adds a conservative annotation. What is deliberately
    *not* done is fuzzy or substring matching — corpus names are a controlled
    vocabulary, and guessing at them would make the flag unreproducible.
    """
    return _SEPARATORS.sub("-", str(name).strip().casefold())


@dataclass(frozen=True)
class ContaminationReport:
    """Gate 4 of spec §3.5. Marks the run; never stops it."""

    flag: bool
    overlapping_corpora: Tuple[str, ...] = ()
    #: False when the study declared no cohort sources, in which case a clean
    #: report means "not checked", not "not contaminated". The caller warns.
    checked: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Exactly the manifest's ``contamination`` block (spec §3.7)."""
        return {"flag": self.flag, "overlapping_corpora": list(self.overlapping_corpora)}

    @classmethod
    def merge(cls, reports: Sequence["ContaminationReport"]) -> "ContaminationReport":
        """Combine per-arm reports into the one block the manifest carries.

        A stage run that evaluates several models is contaminated if any of
        them is: the manifest describes the run, and a run containing one
        contaminated arm cannot be read as clean.
        """
        if not reports:
            return cls(flag=False, overlapping_corpora=(), checked=False)
        corpora: List[str] = []
        for report in reports:
            for name in report.overlapping_corpora:
                if name not in corpora:
                    corpora.append(name)
        return cls(
            flag=any(report.flag for report in reports),
            overlapping_corpora=tuple(sorted(corpora)),
            checked=all(report.checked for report in reports),
        )


def check_contamination(
    pretraining_corpora: Iterable[Any], cohort_sources: Iterable[Any]
) -> ContaminationReport:
    """Intersect the card's pretraining corpora with the cohort's own sources.

    Reports the card's spelling of each overlapping corpus, not the folded
    form, so the manifest stays traceable back to the ModelCard.
    """
    sources = {_normalize(name) for name in cohort_sources}
    if not sources:
        return ContaminationReport(flag=False, overlapping_corpora=(), checked=False)
    overlap = sorted(
        {str(name) for name in pretraining_corpora if _normalize(name) in sources}
    )
    return ContaminationReport(
        flag=bool(overlap), overlapping_corpora=tuple(overlap), checked=True
    )


class EventCountError(RuntimeError):
    """Spec §2.4 gate 2: too few events in the held-out split to report anything."""
