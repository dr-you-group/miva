"""Platform runtime overrides a ModelCard declares for itself.

A card's ``x-mival.runtime`` may hold one block per platform it needs to run
differently on. Today there is one platform key, ``windows`` (``sys.platform
== "win32"``)::

    "runtime": {
      ...,
      "compile_env": {...},
      "windows": {
        "builder_kwargs": {"config": {"backend": "vanilla"}},
        "compile_env": null,
        "slstm_backend": "vanilla"
      }
    }

On that platform the block is applied once, when the card is loaded
(:func:`mival.modelcard.load_card`), so every consumer (the torch adapter's
``build_module``, the site scripts that read ``run_env``, the train_log
record) sees the same card:

- ``builder_kwargs`` is deep-merged into ``x-mival.builder.kwargs``: a nested
  mapping is merged key by key, any other value (``null`` included, since a
  builder argument may legitimately be ``None``) replaces the one there.
- every other key replaces that key of ``runtime``; ``null`` removes it
  (``"compile_env": null``: nothing is compiled on this platform).

On any other platform nothing is applied and the card is exactly as written.
``MIVAL_RUNTIME_PLATFORM`` (e.g. ``win32``) replaces ``sys.platform`` for this
decision, so the override can be exercised, and compared against the
unmodified card, on a machine that is not that platform.

The block itself stays in ``runtime`` after it is applied, and the card
carries what was applied as ``ModelCard.runtime_override`` (name, platform and
the list of changes), which the models stage writes to train_log and the
export carries per arm.

``runtime.state_dict_transform`` (usually set only inside a platform block)
names a function the torch adapter calls on the loaded state dict before the
strict ``load_state_dict``, for a backend whose parameter layout differs from
the checkpoint's::

    "state_dict_transform": "module:function"
    "state_dict_transform": {"function": "module:function", "code_path": "<dir>"}

The string form resolves ``module`` from the card's ``code_path``, as the
builder module is; the object form from its own ``code_path`` (e.g. model
glue that ships with the bundle rather than with the author's code). The
function takes the state dict and returns the one to load. When the override
sets it, the record also carries ``state_dict_transform`` (the function name).
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

#: Replaces ``sys.platform`` when deciding which override applies.
PLATFORM_ENV = "MIVAL_RUNTIME_PLATFORM"
#: ``sys.platform`` value -> the ``runtime`` key holding that platform's override.
PLATFORM_KEYS = {"win32": "windows"}
BUILDER_KWARGS = "builder_kwargs"
#: ``runtime`` key naming a state-dict transform (see the module docstring).
STATE_DICT_TRANSFORM = "state_dict_transform"


def runtime_platform(platform: Optional[str] = None) -> str:
    """The platform overrides are chosen for: ``platform``, else ``$MIVAL_RUNTIME_PLATFORM``, else ``sys.platform``."""
    if platform:
        return platform
    return os.environ.get(PLATFORM_ENV) or sys.platform


def override_key(platform: Optional[str] = None) -> Optional[str]:
    """The ``runtime`` key whose block applies on this platform, if any platform key does."""
    return PLATFORM_KEYS.get(runtime_platform(platform))


def deep_merge(base: Mapping[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    """``base`` with ``patch`` merged in: mappings recursively, every other value replaced."""
    merged = copy.deepcopy(dict(base))
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _leaves(patch: Mapping[str, Any], prefix: str) -> List[str]:
    out = []
    for key, value in patch.items():
        path = f"{prefix}.{key}"
        if isinstance(value, Mapping) and value:
            out.extend(_leaves(value, path))
        else:
            out.append(f"{path}={json.dumps(value, ensure_ascii=False)}")
    return out


def apply_platform_override(
    ext: Mapping[str, Any], where: Any = "card", platform: Optional[str] = None
) -> Tuple[Mapping[str, Any], Optional[Dict[str, Any]]]:
    """``(ext, record)``: the card's ``x-mival`` with this platform's override applied.

    With no override for this platform the same ``ext`` object comes back and
    ``record`` is None. Otherwise a modified deep copy comes back with
    ``{"name", "platform", "changes"}``.
    """
    key = override_key(platform)
    runtime = ext.get("runtime") or {}
    if key is None or key not in runtime:
        return ext, None
    block = runtime[key]
    if not isinstance(block, Mapping):
        raise ValueError(f"{where}: x-mival.runtime.{key} must be an object, got {type(block).__name__}")
    for name in PLATFORM_KEYS.values():
        if name in block:
            raise ValueError(f"{where}: x-mival.runtime.{key} cannot hold another platform block {name!r}")

    out = copy.deepcopy(dict(ext))
    out_runtime = dict(out.get("runtime") or {})
    changes: List[str] = []
    kwargs_patch = block.get(BUILDER_KWARGS)
    if kwargs_patch is not None:
        if not isinstance(kwargs_patch, Mapping):
            raise ValueError(f"{where}: x-mival.runtime.{key}.{BUILDER_KWARGS} must be an object")
        builder = out.get("builder")
        if not isinstance(builder, Mapping):
            raise ValueError(
                f"{where}: x-mival.runtime.{key}.{BUILDER_KWARGS} is given but the card declares "
                "no x-mival.builder to pass them to"
            )
        builder = dict(builder)
        builder["kwargs"] = deep_merge(builder.get("kwargs") or {}, kwargs_patch)
        out["builder"] = builder
        changes.extend(_leaves(kwargs_patch, "builder.kwargs"))
    for name, value in block.items():
        if name == BUILDER_KWARGS:
            continue
        if value is None:
            if name in out_runtime:
                del out_runtime[name]
                changes.append(f"runtime.{name} removed")
        elif name == STATE_DICT_TRANSFORM:
            # Only the function name is recorded: a code_path would carry a
            # site's absolute path into train_log and error messages.
            out_runtime[name] = copy.deepcopy(value)
            function = state_dict_transform_spec(value, f"{where}: x-mival.runtime.{key}")[0]
            changes.append(f"runtime.{name}={function}")
        else:
            out_runtime[name] = copy.deepcopy(value)
            changes.append(f"runtime.{name}={json.dumps(value, ensure_ascii=False)}")
    out["runtime"] = out_runtime
    record: Dict[str, Any] = {"name": key, "platform": runtime_platform(platform), "changes": changes}
    transform = block.get(STATE_DICT_TRANSFORM)
    if transform is not None:
        record[STATE_DICT_TRANSFORM] = state_dict_transform_spec(transform, f"{where}: x-mival.runtime.{key}")[0]
    return out, record


def state_dict_transform_spec(value: Any, where: Any = "card") -> Tuple[str, Optional[str]]:
    """``("module:function", code_path or None)`` from ``runtime.state_dict_transform`` as the card gives it."""
    if isinstance(value, str):
        function, code_path = value, None
    elif isinstance(value, Mapping):
        unknown = sorted(set(value) - {"function", "code_path"})
        if unknown:
            raise ValueError(f"{where}: runtime.{STATE_DICT_TRANSFORM} has unknown keys {unknown}; "
                             "it takes function and code_path")
        function, code_path = value.get("function"), value.get("code_path")
        if code_path is not None and (not isinstance(code_path, str) or not code_path):
            raise ValueError(f"{where}: runtime.{STATE_DICT_TRANSFORM}.code_path must be a path")
    else:
        raise ValueError(f"{where}: runtime.{STATE_DICT_TRANSFORM} must be \"module:function\" or "
                         f"{{\"function\": ..., \"code_path\": ...}}, got {type(value).__name__}")
    module, sep, attribute = function.partition(":") if isinstance(function, str) else ("", "", "")
    if not (module and sep and attribute):
        raise ValueError(f"{where}: runtime.{STATE_DICT_TRANSFORM} function must read "
                         f"\"module:function\", got {function!r}")
    return function, code_path


def override_digests(directory: Any, prefix: str, platform: Optional[str] = None) -> Dict[str, str]:
    """A checksum per card in ``directory`` whose platform override applies here.

    Stages add these to their config_hash so a run with an override applied
    never reuses the output of a run without it (the same card file gives the
    same file checksum either way). Empty when no override applies, so the
    hash on a platform without overrides is what it always was. The block is
    read from the card as written, without placeholder expansion.
    """
    key = override_key(platform)
    root = Path(str(directory))
    if key is None or not root.is_dir():
        return {}
    found: Dict[str, str] = {}
    for path in sorted(root.glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        runtime = ((body.get("x-mival") or {}).get("runtime") or {}) if isinstance(body, dict) else {}
        if key in runtime:
            payload = json.dumps({"override": key, "block": runtime[key]}, sort_keys=True,
                                 separators=(",", ":"), ensure_ascii=False)
            found[f"{prefix}{path.stem}"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return found
