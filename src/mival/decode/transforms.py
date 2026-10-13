"""Map a model's output back onto its label's scale (L2-8).

Declared on the card as ``output.transform``. Applied after ensemble
averaging, so a model that predicts a z-scored or log value is averaged on the
scale it was trained on and then converted once.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

TRANSFORM_KINDS = ("affine", "exp")


def apply_transform(values: np.ndarray, transform: Optional[Mapping[str, Any]]) -> np.ndarray:
    if not transform:
        return values
    kind = transform.get("kind")
    if kind == "affine":
        return values * float(transform.get("scale", 1.0)) + float(transform.get("offset", 0.0))
    if kind == "exp":
        return np.exp(values)
    raise ValueError(f"unknown output.transform kind {kind!r}; known: {list(TRANSFORM_KINDS)}")
