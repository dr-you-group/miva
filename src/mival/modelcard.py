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


def load_card(path: Path) -> ModelCard:
    body = json.loads(Path(path).read_text())
    if "x-mival" in body:
        body["x-mival"] = expand(body["x-mival"], Path(path))
    for key in _REQUIRED_TOP_FIELDS:
        if key not in body:
            raise ValueError(f"{path}: {key} is required")
    ext = body.get("x-mival")
    if ext is None:
        raise ValueError(f"{path}: missing x-mival extension block")
    for key in _REQUIRED_MIVAL_FIELDS:
        if key not in ext:
            raise ValueError(f"{path}: x-mival.{key} is required")
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
    )


def load_registry(directory: Path) -> Dict[str, ModelCard]:
    cards = {}
    for path in sorted(Path(directory).glob("*.json")):
        card = load_card(path)
        cards[card.model_id] = card
    return cards
