"""ModelCard loading.

A card is ROADMAP-compatible descriptive metadata plus an `x-mival` execution
extension. Cards are data, not code: adding a model must not require a source
change anywhere in this package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from mival.contract import InputContract
from mival.placeholders import expand
from mival.runtime_platform import STATE_DICT_TRANSFORM, apply_platform_override, state_dict_transform_spec

_REQUIRED_TOP_FIELDS = (
    "model_id",
    "Name",
)

_REQUIRED_MIVAL_FIELDS = (
    "adapter",
    "weights",
    "input_contract",
    "output",
    "training_modes_supported",
    "pretraining_corpora",
)


@dataclass(frozen=True)
class ModelCard:
    model_id: str
    name: str
    adapter: str
    weights: Tuple[Dict[str, Any], ...]
    ensemble: Dict[str, Any]
    input_contract: InputContract
    output: Dict[str, Any]
    feature_layer: Optional[str]
    threshold: Dict[str, Any]
    runtime: Dict[str, Any]
    training_modes_supported: Tuple[str, ...]
    pretraining_corpora: Tuple[str, ...]
    raw: Dict[str, Any]
    #: The platform override applied at load (``mival.runtime_platform``):
    #: ``{"name", "platform", "changes"}``, or None when the card is as written.
    runtime_override: Optional[Dict[str, Any]] = None


def load_card(path: Path, platform: Optional[str] = None) -> ModelCard:
    """Read a card, with ``${NAME}`` placeholders filled and this platform's runtime override applied.

    ``platform`` replaces ``sys.platform`` / ``$MIVAL_RUNTIME_PLATFORM`` when
    choosing the override (see :mod:`mival.runtime_platform`).
    """
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    override = None
    if "x-mival" in body:
        body["x-mival"] = expand(body["x-mival"], Path(path))
        body["x-mival"], override = apply_platform_override(body["x-mival"], Path(path), platform)
    for key in _REQUIRED_TOP_FIELDS:
        if key not in body:
            raise ValueError(f"{path}: {key} is required")
    ext = body.get("x-mival")
    if ext is None:
        raise ValueError(f"{path}: missing x-mival extension block")
    for key in _REQUIRED_MIVAL_FIELDS:
        if key not in ext:
            raise ValueError(f"{path}: x-mival.{key} is required")
    transform = (ext.get("runtime") or {}).get(STATE_DICT_TRANSFORM)
    if transform is not None:
        state_dict_transform_spec(transform, path)
    return ModelCard(
        model_id=body["model_id"],
        name=body["Name"],
        adapter=ext["adapter"],
        weights=tuple(ext["weights"]),
        ensemble=ext.get("ensemble", {"method": "none", "members": 1}),
        input_contract=InputContract.from_dict(ext["input_contract"]),
        output=ext["output"],
        feature_layer=ext.get("feature_layer"),
        threshold=ext.get("threshold", {}),
        runtime=ext.get("runtime", {}),
        training_modes_supported=tuple(ext["training_modes_supported"]),
        pretraining_corpora=tuple(ext["pretraining_corpora"]),
        raw=body,
        runtime_override=override,
    )


def load_registry(directory: Path, platform: Optional[str] = None) -> Dict[str, ModelCard]:
    cards = {}
    for path in sorted(Path(directory).glob("*.json")):
        card = load_card(path, platform)
        cards[card.model_id] = card
    return cards
