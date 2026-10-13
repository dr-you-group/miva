"""Exclusion ledger (spec §3.6).

Every stage appends a row here when it drops a record. Because it is one table
across all stages, the STARD participant flow diagram is a query over it rather
than a hand-maintained figure.

The vocabulary of ``reason_code`` is closed *per stage*. This is the same
property Plan 1 built into ``CompileError``: a free-text reason makes the
ledger unqueryable, and an excluded record that nobody can count is an
excluded record that silently biases the cohort.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, FrozenSet, List, Optional, Union

from .tables import read_dataset, write_table

COLUMNS = (
    "image_occurrence_id",
    "person_id",
    "stage",
    "reason_code",
    "detail",
    "timestamp",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ExclusionLedger:
    """Collects exclusions for one stage run, then writes them as one part.

    Rows are buffered rather than appended to disk one at a time: parquet has
    no append, and a per-row rewrite would be quadratic over a cohort.
    """

    stage: str
    allowed_codes: FrozenSet[str]
    clock: Callable[[], str] = _utc_now
    rows: List[dict] = field(default_factory=list)

    def record(
        self,
        image_occurrence_id: str,
        person_id: Optional[str],
        reason_code: str,
        detail: str = "",
    ) -> None:
        if reason_code not in self.allowed_codes:
            raise ValueError(
                f"{reason_code!r} is not a reason_code declared by stage {self.stage!r}; "
                f"declared codes are {sorted(self.allowed_codes)}"
            )
        self.rows.append(
            {
                "image_occurrence_id": str(image_occurrence_id),
                "person_id": None if person_id is None else str(person_id),
                "stage": self.stage,
                "reason_code": reason_code,
                "detail": detail,
                "timestamp": self.clock(),
            }
        )

    def __len__(self) -> int:
        return len(self.rows)

    def counts(self) -> dict:
        """Rows per reason_code, for the manifest and for fast sanity checks."""
        tally: dict = {}
        for row in self.rows:
            tally[row["reason_code"]] = tally.get(row["reason_code"], 0) + 1
        return tally

    def flush(self, path: Union[str, Path]) -> Path:
        """Write this run's part file.

        Written even when empty. An absent part file is ambiguous — it could
        mean "nothing was excluded" or "the stage crashed before writing" —
        and the STARD flow needs those distinguished.
        """
        return write_table(self.rows, path, COLUMNS)


def read_exclusions(root: Union[str, Path]):
    """Reassemble the single logical ledger table of spec §3.6."""
    return read_dataset(root, COLUMNS)
