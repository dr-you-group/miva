"""Where a conda env keeps its interpreter, on POSIX and on Windows.

A model card names its runtime env as a directory (``${MIVAL_HOME}/envs/...``).
On Linux and macOS the interpreter is ``<env>/bin/python``; a Windows conda
env keeps ``python.exe`` at its root. The run scripts and the bundle check
all ask here, so the two layouts are told apart in one place.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def env_python(env_dir: Union[str, Path]) -> Path:
    """The interpreter of the env at ``env_dir``; ``FileNotFoundError`` when it has none."""
    root = Path(env_dir)
    posix = root / "bin" / "python"
    windows = root / "python.exe"
    candidates = (windows, posix) if os.name == "nt" else (posix, windows)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no interpreter in env {root} (looked for {posix} and {windows})")
