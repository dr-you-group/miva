"""Pipeline runner: run coordinates, addressing, manifests, ledger, stage CLI.

Nothing here imports a machine-learning backend, pandas, or pyarrow at module
scope. ``import mival.pipeline`` must succeed in both the Python 3.10 torch
environment and the Python 3.9 Keras 2.7 environment.
"""

from .hashing import config_hash, sha256_file
from .layout import RunLayout, run_layout
from .ledger import ExclusionLedger, read_exclusions
from .manifest import ArtifactRef, RunManifest
from .runkey import AXES, REPORT_AXES, RunKey
from .stage import Stage, StageContext, StageResult, execute, get_stage, prepare, stage_names
from .study import Study, load_study

__all__ = [
    "AXES",
    "REPORT_AXES",
    "ArtifactRef",
    "ExclusionLedger",
    "RunKey",
    "RunLayout",
    "RunManifest",
    "Stage",
    "StageContext",
    "StageResult",
    "Study",
    "config_hash",
    "execute",
    "get_stage",
    "load_study",
    "prepare",
    "read_exclusions",
    "run_layout",
    "sha256_file",
    "stage_names",
]
