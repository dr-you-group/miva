"""Load a torch extension the card's code would JIT-compile from an already built copy.

A card whose module compiles a CUDA extension at load time (through
``torch.utils.cpp_extension.load``) may declare::

    "runtime": {"prebuilt_torch_extensions": true, "run_env": {"TORCH_EXTENSIONS_DIR": ...}}

While such a card's module is built (:func:`prebuilt_extensions` around
``TorchAdapter.load``) two things change in this process, and nothing else:

- ``cpp_extension.load(name, ...)`` returns the module imported from
  ``$TORCH_EXTENSIONS_DIR/<name>/<name><suffix>`` (``.so``; ``.pyd`` on
  Windows) when that file exists, the way torch imports it at the end of its
  own build, and skips the build check. torch's check re-runs ninja in every
  new process, and ninja rebuilds when anything it recorded differs: the
  absolute install path, the compiler and CUDA variables, or a toolkit or
  compiler header that is not on this machine. A bundle built once can
  therefore not be loaded through it on a machine without that toolchain.
  Without the file (or without ``TORCH_EXTENSIONS_DIR``) the call goes to
  torch's own ``load`` unchanged.
- when torch found no CUDA toolkit (``cpp_extension.CUDA_HOME`` is None),
  ``CUDA_HOME`` reads as a placeholder path, so code that only asks for the
  toolkit's include path at import time (and compiles nothing, e.g. a
  non-compiled backend) does not fail. A real build in the fallback sees
  ``CUDA_HOME`` unset again and fails with torch's own message.

Outside the context torch is as it was: ``load`` and ``CUDA_HOME`` are
restored. Code that bound ``load`` at import keeps a reference to the wrapper,
which passes every call through to torch when no context is active.

The wrapper reaches code that imports ``load`` by name only if that import
happens inside the context, i.e. when the card's module is first imported
while it is built. That is the case in a models-stage arm process (one process
per arm, the card's code imported only by ``build_module``). If the code was
imported earlier in the process, it holds torch's own ``load`` and builds with
ninja; when a prebuilt file was there but nothing was loaded from it, the
modules holding torch's ``load`` are named in a warning and in the record
(``load_bound_before``).

The extensions directory is read from the environment only
(``TORCH_EXTENSIONS_DIR``); run_models sets it from the card's ``run_env``. A
process that loads the card without that variable falls back to torch.

What was loaded is recorded per card (:func:`record`): extension name, file
and sha256, plus whether the placeholder was used. The models stage writes it
to ``train_log.jsonl`` (``model_card.prebuilt_torch_extensions``) and the
export carries the names per arm. The file is trusted to have been built from
the same sources and flags (the bundle builds it from the same env); its name
is the only thing matched, so the record keeps the sha256.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

#: The card key (``x-mival.runtime``) that opts in.
CARD_KEY = "prebuilt_torch_extensions"
#: Where the prebuilt extensions are looked up (as torch's own cache root).
EXTENSIONS_DIR_ENV = "TORCH_EXTENSIONS_DIR"
#: What ``CUDA_HOME`` reads as while a card that opted in is built without a toolkit.
CUDA_HOME_PLACEHOLDER = os.path.join(tempfile.gettempdir(), "mival-no-cuda-toolkit")

_state: Dict[str, Any] = {"active": [], "original_load": None}
_records: Dict[str, Dict[str, Any]] = {}


def enabled(card: Any) -> bool:
    return bool((getattr(card, "runtime", None) or {}).get(CARD_KEY))


def library_suffix() -> str:
    return ".pyd" if os.name == "nt" else ".so"


def prebuilt_path(name: str, root: Optional[str] = None) -> Optional[Path]:
    """``<root>/<name>/<name><suffix>`` when it is a file; root defaults to ``$TORCH_EXTENSIONS_DIR``."""
    root = root if root is not None else os.environ.get(EXTENSIONS_DIR_ENV)
    if not root:
        return None
    path = Path(root) / name / f"{name}{library_suffix()}"
    return path if path.is_file() else None


def _holding(original: Any) -> List[str]:
    """Modules outside torch whose namespace holds ``original`` (bound by ``from ... import load``)."""
    found = []
    for name, module in list(sys.modules.items()):
        if module is None or name == "torch" or name.startswith("torch."):
            continue
        try:
            values = list(vars(module).values())
        except TypeError:
            continue
        if any(value is original for value in values):
            found.append(name)
    return sorted(found)


def _any_prebuilt() -> bool:
    root = os.environ.get(EXTENSIONS_DIR_ENV)
    return bool(root) and any(Path(root).glob(f"*/*{library_suffix()}"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_library(name: str, path: Path, is_python_module: bool) -> Any:
    """Import as torch does after a build (``_import_module_from_library``)."""
    from torch.utils import cpp_extension

    importer = getattr(cpp_extension, "_import_module_from_library", None)
    if importer is not None:
        return importer(name, str(path.parent), is_python_module)
    if not is_python_module:  # pragma: no cover - torch without the helper
        import torch

        torch.ops.load_library(str(path))
        return str(path)
    spec = importlib.util.spec_from_file_location(name, str(path))  # pragma: no cover
    module = importlib.util.module_from_spec(spec)  # pragma: no cover
    spec.loader.exec_module(module)  # pragma: no cover
    return module  # pragma: no cover


def _wrapped_load(name, *args, **kwargs):
    """``cpp_extension.load``: the prebuilt file while a card context is active, else torch's own."""
    from torch.utils import cpp_extension

    original = _state["original_load"]
    active = _state["active"]
    if active:
        path = prebuilt_path(str(name))
        if path is not None:
            module = _import_library(str(name), path, bool(kwargs.get("is_python_module", True)))
            found = _records.setdefault(active[-1], {"extensions": [], "cuda_home_placeholder": False})
            if all(e["name"] != str(name) for e in found["extensions"]):
                found["extensions"].append({"name": str(name), "file": str(path), "sha256": _sha256(path)})
            return module
    if cpp_extension.CUDA_HOME == CUDA_HOME_PLACEHOLDER:
        # A real build must not run against the placeholder: torch's own error instead.
        cpp_extension.CUDA_HOME = None
        try:
            return original(name, *args, **kwargs)
        finally:
            cpp_extension.CUDA_HOME = CUDA_HOME_PLACEHOLDER
    return original(name, *args, **kwargs)


@contextmanager
def prebuilt_extensions(card: Any) -> Iterator[None]:
    """Active while the card's module is built; a no-op for a card that does not opt in."""
    if not enabled(card):
        yield
        return
    from torch.utils import cpp_extension

    key = str(getattr(card, "model_id", "?"))
    outermost = not _state["active"]
    placeholder = False
    if outermost:
        _state["original_load"] = cpp_extension.load
        cpp_extension.load = _wrapped_load
        if cpp_extension.CUDA_HOME is None:
            cpp_extension.CUDA_HOME = CUDA_HOME_PLACEHOLDER
            placeholder = True
    _state["active"].append(key)
    try:
        yield
    finally:
        _state["active"].pop()
        found = _records.setdefault(key, {"extensions": [], "cuda_home_placeholder": False})
        if outermost:
            cpp_extension.load = _state["original_load"]
            if placeholder:
                cpp_extension.CUDA_HOME = None
                found["cuda_home_placeholder"] = True
            if not found["extensions"] and _any_prebuilt():
                holders = _holding(_state["original_load"])
                if holders:
                    found["load_bound_before"] = holders
                    warnings.warn(
                        f"{key}: runtime.{CARD_KEY} is set and a prebuilt extension is in "
                        f"${EXTENSIONS_DIR_ENV}, but nothing was loaded from it; "
                        f"{', '.join(holders)} imported torch's cpp_extension.load before the card's "
                        "module was built, so torch's own build ran instead", RuntimeWarning)


def record(card: Any) -> Dict[str, Any]:
    """What this process loaded for the card: ``{"prebuilt_torch_extensions": {...}}``, else empty."""
    if not enabled(card):
        return {}
    found = _records.get(str(getattr(card, "model_id", "?")))
    if not found or not (found["extensions"] or found["cuda_home_placeholder"]
                         or found.get("load_bound_before")):
        return {}
    out = {"extensions": [dict(e) for e in found["extensions"]],
           "cuda_home_placeholder": bool(found["cuda_home_placeholder"])}
    if found.get("load_bound_before"):
        out["load_bound_before"] = list(found["load_bound_before"])
    return {CARD_KEY: out}


def loaded_names(entry: Any) -> List[str]:
    """Extension names from a recorded ``prebuilt_torch_extensions`` value (train_log)."""
    if not isinstance(entry, dict):
        return []
    return [str(e.get("name")) for e in entry.get("extensions") or [] if isinstance(e, dict)]
