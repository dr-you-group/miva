"""Frozen-backbone representations, computed once per arm instead of once per fit.

Why this exists. ``models`` selects a ``linear_probe`` arm's hyperparameters by
inner cross-validation: every candidate x inner fold loads a fresh handle,
fits a head on the train folds (``features`` over train and the early-stopping
fold) and predicts the held fold (``features`` again, then the head). The final
fit and the test prediction follow. With the backbone frozen, every one of
those passes recomputes the same representation of the same tensor, and on a
GPU-sized backbone that recomputation is nearly all of the stage's wall time.
The cache within one fit (``Adapter._fit_linear_probe``) only spans epochs.

What makes reuse sound. A representation is a function of the backbone's
weights, the layer it is read from, the feature set (ensemble member) and the
tensor. The key therefore names the backbone through the ModelCard — the only
description of a model this package has — and never through the Python
handle: ``models`` calls ``load(card)`` per fold, so every fold holds a new
handle around the same published weights. A handle whose backbone was trained
(``partial_unfreeze``, ``full_finetune``) is a different function of the same
tensor, so a handle opts in explicitly by carrying ``backbone_as_published =
True``; backends set it on ``load`` and carry it through ``with_head(s)``, and
nothing else ever sets it. Anything without the flag — a fine-tuned handle, or
a third-party adapter's handle — is computed exactly as before.

Scope and memory. A cache lives only inside ``Adapter.feature_cache_scope``,
which ``models`` opens around one (arm, recipe, perturbation) group of a
``linear_probe`` arm, so nothing outlives the group and a re-run of the stage
in the same process (tests, a notebook) never sees an earlier run's tensors.
Its size is bounded by a byte budget; once the budget is reached new rows are
computed and returned but not stored. Declining rather than evicting is
deliberate: the access pattern is repeated full scans of the same dev split,
which an LRU policy turns into zero hits as soon as the split exceeds the
budget, whereas keeping the first rows keeps their hits on every pass.

Opt-out. ``MIVAL_FEATURE_CACHE=0`` (or ``false``/``off``/``no``) turns the
cache off; ``MIVAL_FEATURE_CACHE_MAX_GB`` sets the budget. They are process
settings rather than study keys because they change how fast a result is
computed, never which result: they do not belong in a study's config hash.
Whether the cache ran, and how much it reused, is written to the arm's
``training.feature_cache`` in ``train_log.jsonl``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

ENV_SWITCH = "MIVAL_FEATURE_CACHE"
ENV_BUDGET_GB = "MIVAL_FEATURE_CACHE_MAX_GB"

#: About five feature sets of 100k ECGs x 2k float32 dimensions. Large enough
#: for one cohort-scale arm of an ordinary model, small enough to sit beside a
#: backbone on a modest site server. Raise it with ``MIVAL_FEATURE_CACHE_MAX_GB``.
DEFAULT_BUDGET_GB = 4.0

_OFF = frozenset({"0", "false", "off", "no"})


def feature_cache_setting() -> Tuple[bool, int, Optional[str]]:
    """(enabled, budget in bytes, why it is off) from the environment."""
    switch = os.environ.get(ENV_SWITCH, "").strip().lower()
    if switch in _OFF:
        return False, 0, f"{ENV_SWITCH}={os.environ[ENV_SWITCH]}"
    raw = os.environ.get(ENV_BUDGET_GB, "").strip()
    try:
        gigabytes = float(raw) if raw else DEFAULT_BUDGET_GB
    except ValueError:
        raise ValueError(f"{ENV_BUDGET_GB} must be a number of gigabytes, got {raw!r}") from None
    if not np.isfinite(gigabytes) or gigabytes <= 0:
        raise ValueError(
            f"{ENV_BUDGET_GB} must be > 0 gigabytes, got {raw!r}; set {ENV_SWITCH}=0 to "
            "turn the cache off instead"
        )
    return True, int(gigabytes * 1024 ** 3), None


def backbone_identity(card: Any) -> str:
    """A digest of everything on the card that can change a representation.

    The whole card goes in — weights (uri, sha256, member specs), feature
    layer, input contract, builder, ensemble — rather than a hand-picked
    subset, because a field left out is a field whose change would silently
    serve another backbone's features. Fields that cannot change a
    representation (threshold, provenance) only cost a cache miss, and within
    one group the card never changes anyway.
    """
    body = {
        "model_id": getattr(card, "model_id", None),
        "adapter": getattr(card, "adapter", None),
        "weights": getattr(card, "weights", None),
        "feature_layer": getattr(card, "feature_layer", None),
        "ensemble": getattr(card, "ensemble", None),
        "input_contract": _plain(getattr(card, "input_contract", None)),
        "raw": getattr(card, "raw", None),
    }
    text = json.dumps(body, sort_keys=True, default=repr)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) and not isinstance(value, type) else value


class FeatureCache:
    """Rows of ``features()`` output keyed by (backbone, feature set, tensor path)."""

    def __init__(self, budget_bytes: int) -> None:
        self.budget_bytes = int(budget_bytes)
        self._rows: Dict[Tuple[str, int, str], np.ndarray] = {}
        self.stored_bytes = 0
        self.computed = 0
        self.reused = 0
        self.declined = 0
        # id(card) -> (card, identity). The card is held so its id cannot be
        # reused by another object while the scope is open; hashing the card
        # once per scope instead of once per batch.
        self._identities: Dict[int, Tuple[Any, str]] = {}

    def identity(self, card: Any) -> str:
        known = self._identities.get(id(card))
        if known is None or known[0] is not card:
            known = (card, backbone_identity(card))
            self._identities[id(card)] = known
        return known[1]

    def get(self, key: Tuple[str, int, str]) -> Optional[np.ndarray]:
        return self._rows.get(key)

    def put(self, key: Tuple[str, int, str], row: np.ndarray) -> None:
        previous = self._rows.get(key)
        freed = 0 if previous is None else previous.nbytes
        nbytes = np.asarray(row).nbytes
        if self.stored_bytes - freed + nbytes > self.budget_bytes:
            self.declined += 1
            return
        # A copy, so that nothing holds the whole batch alive through one row
        # and no caller can mutate a stored row in place.
        self._rows[key] = np.array(row, copy=True)
        self.stored_bytes += nbytes - freed

    def report(self) -> Mapping[str, Any]:
        """What goes into ``training.feature_cache`` of the train log."""
        return {
            "enabled": True,
            "used": bool(self.computed or self.reused),
            "computed": self.computed,
            "reused": self.reused,
            "declined": self.declined,
            "stored_bytes": self.stored_bytes,
            "budget_bytes": self.budget_bytes,
        }
