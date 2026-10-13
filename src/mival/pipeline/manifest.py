"""Run manifest (spec §3.7).

The manifest is what makes a result auditable: it records what went in, what
came out, on which code and which machine. Spec §3.7 also states that the
Technical Report is a rendering of manifests plus the exclusion ledger plus the
stage 5/6 artifacts, so every field here is eventually a cell in a table
somebody reads.

Writing the manifest is also the completion marker (see ``RunLayout.is_complete``),
so it must be written last and written atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mival import __version__

from .hashing import sha256_file, sha256_path

SCHEMA_VERSION = 1

# Libraries whose version changes can change numeric results. Recorded when
# present; a stage running in the Keras environment has no torch and vice
# versa, and an absent library is recorded as absent rather than omitted.
_TRACKED_LIBS = ("numpy", "scipy", "pandas", "pyarrow", "torch", "tensorflow", "keras", "sklearn")


@dataclass
class ArtifactRef:
    path: str
    sha256: str

    @classmethod
    def of(cls, path: Union[str, Path], root: Union[str, Path, None] = None) -> "ArtifactRef":
        """Checksum a file, recording its path relative to ``root`` when given.

        Relative paths keep the manifest identical across machines that mount
        the study at different points.
        """
        resolved = Path(path)
        recorded = str(resolved.relative_to(root)) if root is not None else str(resolved)
        return cls(path=recorded, sha256=sha256_path(resolved))


def _git(args: List[str], cwd: Union[str, Path]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def capture_git(cwd: Union[str, Path, None] = None) -> Dict[str, Any]:
    """Commit and dirty flag, or nulls outside a repository.

    ``dirty`` is not merely cosmetic: a result produced from an uncommitted
    tree cannot be reproduced from the commit hash alone, and the report must
    be able to say so.
    """
    cwd = Path(cwd) if cwd is not None else Path.cwd()
    commit = _git(["rev-parse", "HEAD"], cwd)
    if commit is None:
        return {"commit": None, "dirty": None}
    status = _git(["status", "--porcelain"], cwd)
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _source_digest() -> Optional[str]:
    """A checksum over ``mival``'s own Python source.

    ``capture_git`` records the commit, but a commit is not enough to decide
    whether a finished run may be skipped: it is absent for an installed copy,
    it does not change when the tree is dirty, and the working tree is what
    actually ran. Hashing the source files is available in every deployment and
    changes exactly when the code does.
    """
    root = Path(__file__).resolve().parent.parent
    try:
        parts = sorted(root.rglob("*.py"))
    except OSError:  # pragma: no cover - unreadable install
        return None
    if not parts:  # pragma: no cover - zipped install
        return None
    digest = hashlib.sha256()
    for part in parts:
        try:
            body = part.read_bytes()
        except OSError:  # pragma: no cover - unreadable file
            return None
        digest.update(str(part.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(body).digest())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def code_identity() -> Dict[str, Any]:
    """Which code produced this run.

    Recorded in every manifest and compared before a finished run is skipped.
    Without that comparison, editing a default in this package leaves
    ``config_hash`` unchanged — it hashes the study spec and the input files,
    not the code — so the next run is skipped as complete and the old numbers
    are reported as new ones.

    Deliberately not part of ``config_hash``: the hash names the run directory,
    and folding the source into it would scatter a study's artifacts across a
    new directory for every edit, including edits to a comment. Naming and
    staleness are different questions, so they get different mechanisms.
    """
    return {
        "mival_version": __version__,
        "source_sha256": _source_digest(),
        "container_digest": os.environ.get("MIVAL_CONTAINER_DIGEST"),
    }


def capture_env() -> Dict[str, Any]:
    """Python, library versions, and accelerator identity.

    Imports nothing eagerly: probing for torch inside the Keras environment
    must not fail, and probing must not pull a 2 GB framework into a stage that
    does not use one. Only modules already imported by this process are
    reported, plus a metadata lookup for the rest.
    """
    versions: Dict[str, Optional[str]] = {}
    for name in _TRACKED_LIBS:
        module = sys.modules.get(name)
        version = getattr(module, "__version__", None) if module is not None else None
        if version is None:
            version = _installed_version(name)
        versions[name] = version

    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": versions,
        "container_digest": os.environ.get("MIVAL_CONTAINER_DIGEST"),
        "cuda": None,
        "gpu": None,
    }
    torch = sys.modules.get("torch")
    if torch is not None:  # only when the stage already loaded it
        try:
            env["cuda"] = getattr(torch.version, "cuda", None)
            if torch.cuda.is_available():
                env["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # pragma: no cover - driver-dependent
            pass
    return env


def _installed_version(name: str) -> Optional[str]:
    dist = {"sklearn": "scikit-learn"}.get(name, name)
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        return metadata.version(dist)
    except Exception:
        return None


@dataclass
class RunManifest:
    """Spec §3.7, field for field."""

    run_id: str
    stage: str
    study_id: str
    site: str
    config_hash: str
    seed: Optional[int] = None
    git: Dict[str, Any] = field(default_factory=capture_git)
    #: Which code produced this run. Compared before a finished run is skipped.
    code: Dict[str, Any] = field(default_factory=code_identity)
    env: Dict[str, Any] = field(default_factory=capture_env)
    input_artifacts: List[ArtifactRef] = field(default_factory=list)
    output_artifacts: List[ArtifactRef] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=lambda: {"in": 0, "out": 0, "excluded": 0})
    contamination: Dict[str, Any] = field(
        default_factory=lambda: {"flag": False, "overlapping_corpora": []}
    )
    compute: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write(self, path: Union[str, Path]) -> Path:
        """Write atomically.

        The manifest doubles as the run's completion marker, so a partially
        written one would make a crashed run look finished and permanently
        skippable.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + ".tmp")
        temp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, target)
        return target

    @classmethod
    def read(cls, path: Union[str, Path]) -> "RunManifest":
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        version = body.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: manifest schema_version is {version!r}, this build reads "
                f"{SCHEMA_VERSION}"
            )
        for key in ("input_artifacts", "output_artifacts"):
            body[key] = [ArtifactRef(**ref) for ref in body.get(key, [])]
        return cls(**body)
