"""Artifact addressing (spec §3.4, §3.6).

    runs/<study_id>/<stage>/<config_hash>/{artifacts,manifest.json,logs}

Every path a stage writes is derived here. A stage that builds paths by string
concatenation is a stage that will one day disagree with the manifest reader
about where its own output lives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

# Spec §3.4: all artifacts live on the 400 GB EBS volume. /scratch (instance
# store) is a DICOM cache only and is not assumed to survive a stop/start.
DEFAULT_RUNS_ROOT = Path("/data/mi-val/runs")

MANIFEST_NAME = "manifest.json"
# A crashed run records its manifest here instead. `is_complete` deliberately
# does not look at this name, so a failure is re-run rather than skipped.
FAILED_MANIFEST_NAME = "manifest.failed.json"
ARTIFACTS_DIR = "artifacts"
LOGS_DIR = "logs"
EXCLUSIONS_DIR = "exclusions"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check(kind: str, value: str) -> str:
    if not _NAME_RE.match(value):
        raise ValueError(
            f"{kind} {value!r} is not a valid path component; allowed characters are "
            "letters, digits, '.', '_' and '-', and it may not start with '.', '_' or '-'"
        )
    return value


@dataclass(frozen=True)
class RunLayout:
    """Resolved paths for one (study, stage, config_hash) triple."""

    runs_root: Path
    study_id: str
    stage: str
    config_hash: str

    @property
    def study_dir(self) -> Path:
        return self.runs_root / self.study_id

    @property
    def stage_dir(self) -> Path:
        return self.study_dir / self.stage

    @property
    def run_dir(self) -> Path:
        return self.stage_dir / self.config_hash

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / ARTIFACTS_DIR

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / LOGS_DIR

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / MANIFEST_NAME

    @property
    def exclusions_path(self) -> Path:
        """This run's slice of the study-wide exclusion ledger.

        Spec §3.6 describes one table. Parquet files cannot be appended to, so
        the physical layout is one part file per (stage, config_hash) and
        ``read_exclusions`` reassembles the single logical table the spec
        specifies. Writing parts also means a re-run replaces its own rows
        instead of duplicating them into a shared file.
        """
        return self.study_dir / EXCLUSIONS_DIR / self.stage / f"{self.config_hash}.parquet"

    def artifact(self, *parts: str) -> Path:
        return self.artifacts_dir.joinpath(*parts)

    def is_complete(self) -> bool:
        """A run is complete when its manifest exists.

        The manifest is written last, so its presence means the artifacts it
        lists were fully written. A half-finished run leaves artifacts but no
        manifest and is correctly re-run.
        """
        return self.manifest_path.is_file()

    def create_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.exclusions_path.parent.mkdir(parents=True, exist_ok=True)


def run_layout(
    study_id: str,
    stage: str,
    config_hash: str,
    runs_root: Union[str, Path, None] = None,
) -> RunLayout:
    return RunLayout(
        runs_root=Path(runs_root) if runs_root is not None else DEFAULT_RUNS_ROOT,
        study_id=_check("study_id", study_id),
        stage=_check("stage", stage),
        config_hash=_check("config_hash", config_hash),
    )


def exclusions_root(runs_root: Union[str, Path], study_id: str) -> Path:
    return Path(runs_root) / _check("study_id", study_id) / EXCLUSIONS_DIR
