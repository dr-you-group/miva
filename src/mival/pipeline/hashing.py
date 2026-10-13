"""Checksums and config_hash (spec §3.4).

``config_hash`` is the hash of the stage spec plus the checksums of its input
artifacts. Same inputs and same settings means the same hash, which is what
makes re-runs skippable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Union

_CHUNK = 1 << 20

# 16 hex characters is 64 bits. At 10^6 runs the birthday collision
# probability is about 2.7e-8, which is far below the rate at which any other
# part of this pipeline fails. Longer hashes only make paths harder to read.
CONFIG_HASH_LENGTH = 16


def sha256_file(path: Union[str, Path]) -> str:
    """Full SHA-256 of a file, streamed so large tensors do not load into RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Union[str, Path]) -> str:
    """A file's SHA-256, or a directory's: the hash of every file's relative path and SHA-256.

    A stage output that is a directory of files (the models stage's
    predictions) can then be passed as one input and still enter config_hash
    by content.
    """
    path = Path(path)
    if not path.is_dir():
        return sha256_file(path)
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256_file(child).encode("ascii") + b"\n")
    return digest.hexdigest()


def canonical_json(obj: object) -> str:
    """Serialize deterministically.

    ``sort_keys`` makes the hash independent of dict insertion order, which
    otherwise varies with how a YAML/JSON parser walked the file.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(spec: Mapping[str, object], inputs: Mapping[str, str]) -> str:
    """Hash a stage spec together with its input artifact checksums.

    ``inputs`` maps a *logical* input name to that artifact's SHA-256 — not a
    filesystem path. Paths differ between the laptop and DICOM-MIVA while the
    content does not, and a hash that changed with the mount point would make
    every result non-reproducible across machines.
    """
    for name, checksum in inputs.items():
        if not isinstance(checksum, str) or len(checksum) != 64:
            raise ValueError(
                f"input artifact {name!r} checksum must be a 64-character SHA-256 hex "
                f"digest, got {checksum!r}"
            )
    payload = canonical_json({"spec": spec, "inputs": dict(inputs)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:CONFIG_HASH_LENGTH]
