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
import json
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Union

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


def load_runs(path: Union[str, Path]) -> Dict[str, List[str]]:
    """Read ``runs.json`` (stage -> run directories) as stage -> config hashes.

    The hash is the last part of the run directory whether it was written
    with '/' or '\\', so a runs.json written on Windows reads the same here.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {stage: [PureWindowsPath(run_dir).name for run_dir in run_dirs]
            for stage, run_dirs in raw.items()}


def read_exclusions(root: Union[str, Path], runs: Optional[Mapping[str, Sequence[str]]] = None):
    """Reassemble the single logical ledger table of spec §3.6.

    The ledger directory accumulates part files from every earlier invocation
    of the study. With ``runs`` (stage -> config hashes of this invocation) only
    those parts are read, so a stale run's exclusions cannot leak into the flow;
    a stage absent from ``runs`` is not read at all.
    """
    if runs is None:
        return read_dataset(root, COLUMNS)
    base = Path(root)
    wanted = {stage: set(hashes) for stage, hashes in runs.items()}
    for stage, hashes in sorted(wanted.items()):
        # A stage that keeps no ledger has no directory; but a listed run whose
        # part file is missing in an existing directory would understate the flow.
        if (base / stage).is_dir():
            for config_hash in sorted(hashes):
                if not (base / stage / f"{config_hash}.parquet").is_file():
                    raise ValueError(
                        f"runs lists {stage}/{config_hash} but {base / stage} has no "
                        f"{config_hash}.parquet"
                    )
    return read_dataset(
        root, COLUMNS, include=lambda part: part.stem in wanted.get(part.parent.name, ())
    )
