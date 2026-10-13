"""Adapter lookup. Adding a backend means adding a module and one entry here."""

from __future__ import annotations

from typing import Dict, List

from mival.adapters.base import Adapter, to_model_layout

_FACTORIES = {
    "torch": "mival.adapters.torch_adapter:TorchAdapter",
    "keras": "mival.adapters.keras_adapter:KerasAdapter",
}

_CACHE: Dict[str, Adapter] = {}


def available_adapters() -> List[str]:
    return sorted(_FACTORIES)


def get_adapter(name: str) -> Adapter:
    if name not in _FACTORIES:
        raise KeyError(f"no adapter named {name!r}; have {available_adapters()}")
    if name not in _CACHE:
        import importlib

        module_path, class_name = _FACTORIES[name].split(":")
        module = importlib.import_module(module_path)
        _CACHE[name] = getattr(module, class_name)()
    return _CACHE[name]


__all__ = ["Adapter", "available_adapters", "get_adapter", "to_model_layout"]
