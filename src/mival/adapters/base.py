"""Adapter interface (spec §4.4).

Backends are imported lazily inside adapter methods so that `mival` core
imports cleanly in both the torch and the keras27 environment.

``fit`` is concrete here rather than abstract. ``linear_probe`` trains one
logistic layer on a frozen representation, and once ``features`` has produced
that representation nothing about the fit is backend-specific — so it is
implemented once, in NumPy, and both backends inherit exactly the same
optimiser, regularisation and stopping rule. Only the modes that actually
propagate gradients into the backbone are delegated to the backend, through
``_finetune``.

Across fits, a frozen backbone's representation is reused through
``cached_features`` (see ``mival.adapters._feature_cache`` for the key, the
rule for which handles qualify, the memory bound and the opt-out). Both
backends inherit that one mechanism: ``_collect_features`` reads through it,
and ``forward_cached`` runs a fitted head over it for the prediction path.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

import numpy as np

from mival.adapters._attribution import DEFAULT_STEPS
from mival.adapters._feature_cache import FeatureCache, feature_cache_setting
from mival.adapters._training import (
    FitData,
    FitHParams,
    FitSplit,
    LinearHead,
    TrainingError,
    fit_linear_head,
    parse_fit_data,
    parse_hparams,
)
from mival.modelcard import ModelCard


def to_model_layout(batch: np.ndarray, layout: str) -> np.ndarray:
    """Convert canonical (B, n_leads, n_samples) into the model's layout."""
    if layout == "lead_time":
        return batch
    if layout == "time_lead":
        return np.ascontiguousarray(np.transpose(batch, (0, 2, 1)))
    raise ValueError(f"unknown layout: {layout}")


class Adapter:
    name: str = "adapter"

    #: The open ``feature_cache_scope``'s cache, or ``None`` outside one.
    _feature_cache: Optional[FeatureCache] = None

    def load(self, card: ModelCard) -> Any:
        raise NotImplementedError

    def forward(self, handle: Any, batch: np.ndarray) -> np.ndarray:
        """Return positive-class probability, shape (B,)."""
        raise NotImplementedError

    def features(self, handle: Any, batch: np.ndarray, index: int = 0) -> np.ndarray:
        """Return representations, shape (B, D).

        ``index`` selects among ``feature_set_names(handle)``. It is 0 for every
        model with a single representation, which is every model that is not an
        ensemble.
        """
        raise NotImplementedError

    def trainable_groups(self, handle: Any) -> List[str]:
        raise NotImplementedError

    def attribute(
        self,
        handle: Any,
        batch: np.ndarray,
        baseline: Any = None,
        steps: int = DEFAULT_STEPS,
    ) -> np.ndarray:
        """Return per-sample saliency over the input, shape of ``batch``.

        Optional in spec §4.4. The method and — more importantly — the baseline
        are chosen in ``mival.adapters._attribution``; read that module before
        calling this, because ``baseline`` has no default and the reason it has
        none is the substance of the choice.
        """
        raise NotImplementedError(
            f"adapter {self.name!r} implements no attribute(); spec §4.4 marks it optional"
        )

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        handle: Any,
        data: Mapping[str, Any],
        mode: str,
        hparams: Mapping[str, Any],
    ) -> Any:
        """Fit ``handle`` for ``mode`` and return the handle to run inference with.

        The returned handle is a *new* object, not the argument mutated in
        place: ``models`` fits one handle per inner CV fold and compares their
        held-out predictions, so a fit that leaked into its input would carry
        one fold's weights into the next fold's estimate.
        """
        payload = parse_fit_data(data)
        # pos_weight='balanced' divides by n_pos, which is meaningless once
        # labels are continuous; a regression fit never needs class counts.
        # Silently resolving it to 1.0 would let a study record a setting that
        # never took effect, so an arm that asks for one is rejected instead.
        if payload.objective != "bce" and hparams.get("pos_weight", 1.0) != 1.0:
            raise ValueError(
                f"pos_weight={hparams['pos_weight']!r} weighs the positive class of a "
                f"classification loss, but this arm fits objective {payload.objective!r} "
                "against a continuous target; drop pos_weight from its hparams"
            )
        counts = (payload.train.n_pos, payload.train.n_neg) if payload.objective == "bce" else (1, 1)
        hp = parse_hparams(hparams, mode, *counts)
        if mode == "linear_probe":
            return self._fit_linear_probe(handle, payload, hp)
        return self._finetune(handle, payload, mode, hp)

    def feature_set_names(self, handle: Any) -> List[str]:
        """The representations this handle offers, in the order ``features`` uses.

        One name for an ordinary model. An ensemble has one per member, because
        independently trained members share no coordinate system: their
        representations cannot be averaged, and probing only the first would
        make a "head-retrained" arm a single-member arm while the
        ``inference_only`` arm keeps all of them — which would break the very
        symmetry the spec §2.5 comparison depends on.

        A head is fitted per name and the resulting probabilities are pooled by
        the rule the ModelCard already declares for ``inference_only``, so both
        arms of an ensemble pool at the same place.
        """
        return [self.name]

    def _fit_linear_probe(self, handle: Any, data: FitData, hp: FitHParams) -> Any:
        """Freeze the backbone, cache its representation, fit one logistic layer.

        Caching is what makes this cheap: with the backbone frozen the
        representation does not change between epochs, so the signals are read
        from disk once instead of once per epoch. It is exactly equivalent
        because ``features`` runs the module in inference mode — no dropout, no
        batch-norm update.
        """
        names = list(self.feature_set_names(handle))
        if not names:
            raise TrainingError(f"adapter {self.name!r} offers no representation to probe")
        heads: List[LinearHead] = []
        records: List[Mapping[str, Any]] = []
        for index, name in enumerate(names):
            train_features = self._collect_features(handle, data.train, data, index)
            validation = None
            if data.validation is not None:
                validation = (
                    self._collect_features(handle, data.validation, data, index),
                    data.validation.y,
                )
            head, record = fit_linear_head(
                train_features,
                data.train.y,
                hp,
                validation,
                context=f"linear_probe[{name}]",
                objective=data.objective,
            )
            record["feature_set"] = name
            heads.append(head)
            records.append(record)

        combined: Dict[str, Any] = dict(records[0])
        combined.update(
            {
                "mode": "linear_probe",
                "hparams": hp.to_record(),
                "n_validation": 0 if data.validation is None else len(data.validation),
                "frozen": "all",
                "feature_sets": names,
                "objective": data.objective,
            }
        )
        if len(records) > 1:
            # An ensemble's per-member fits differ; keeping only the first
            # would report one member's epoch count as the arm's.
            combined["members"] = records
        return self.with_heads(handle, heads, combined)

    def with_heads(
        self, handle: Any, heads: List[LinearHead], record: Mapping[str, Any]
    ) -> Any:
        """Attach one fitted head per feature set.

        The single-head case delegates to ``with_head`` so that a backend with
        one representation implements only that.
        """
        if len(heads) == 1:
            return self.with_head(handle, heads[0], record)
        raise NotImplementedError(
            f"adapter {self.name!r} produced {len(heads)} representations but does not "
            "implement with_heads()"
        )

    def _collect_features(
        self, handle: Any, split: FitSplit, data: FitData, index: int = 0
    ) -> np.ndarray:
        """Representations for every record of ``split``, in order."""
        chunks: List[np.ndarray] = []
        for start in range(0, len(split), data.batch_size):
            paths = split.tensor_paths[start : start + data.batch_size]
            # The float64 conversion stays here, on whatever features()
            # returned, so a cached row reaches the head exactly as a freshly
            # computed one would.
            block = np.asarray(
                self.cached_features(handle, paths, data.load_batch, index), dtype=np.float64
            )
            if block.ndim != 2 or block.shape[0] != len(paths):
                raise TrainingError(
                    f"adapter {self.name!r} features() returned shape {block.shape} for a "
                    f"batch of {len(paths)} records; (B, D) is required"
                )
            chunks.append(block)
        if not chunks:
            raise TrainingError(f"adapter {self.name!r}: no records to extract features from")
        return np.concatenate(chunks, axis=0)

    # -- the cross-fit feature cache ------------------------------------------

    @contextmanager
    def feature_cache_scope(self) -> Iterator[Callable[[], Mapping[str, Any]]]:
        """Reuse frozen-backbone representations until the block exits.

        Yields a callable returning the record ``models`` writes to the train
        log. With the cache switched off (``MIVAL_FEATURE_CACHE=0``) nothing is
        installed and every call below computes exactly as it would outside a
        scope. A scope opened inside another one replaces it until it exits.
        """
        enabled, budget, reason = feature_cache_setting()
        if not enabled:
            yield lambda: {"enabled": False, "used": False, "reason": reason}
            return
        outer = self._feature_cache
        cache = FeatureCache(budget)
        self._feature_cache = cache
        try:
            yield cache.report
        finally:
            self._feature_cache = outer

    def _serves_cached_features(self, handle: Any) -> bool:
        """Whether ``handle``'s representations may come from the cache.

        Only a backbone exactly as the card publishes it: a fresh ``load``, or
        a ``with_head(s)`` copy of one, which shares the unmodified module. The
        flag is opt-in and nothing but ``load`` sets it, so a fine-tuned handle
        or an adapter that never heard of the cache computes as before.
        """
        return (
            self._feature_cache is not None
            and getattr(handle, "backbone_as_published", False) is True
            and getattr(handle, "card", None) is not None
        )

    def cached_features(
        self,
        handle: Any,
        paths: Sequence[str],
        load_batch: Callable[[Sequence[str]], np.ndarray],
        index: int = 0,
    ) -> np.ndarray:
        """``features`` of the tensors at ``paths``, each computed once per scope.

        Returned as ``features`` returned it — same dtype, no conversion — so
        callers keep their own conversion where it was. A batch with nothing
        cached is loaded and featurised as one call, exactly as without the
        cache; a batch with some rows cached loads and featurises only the
        rest. Outside a scope, or for a handle that does not qualify, this is
        ``features(handle, load_batch(paths), index)``.
        """
        paths = [str(path) for path in paths]
        cache = self._feature_cache
        if cache is None or not self._serves_cached_features(handle):
            return self.features(handle, load_batch(list(paths)), index)
        identity = cache.identity(handle.card)
        keys = [(identity, int(index), path) for path in paths]
        rows: List[Optional[np.ndarray]] = [cache.get(key) for key in keys]
        missing = [slot for slot, row in enumerate(rows) if row is None]
        cache.reused += len(paths) - len(missing)
        if not missing:
            return np.stack(rows)  # type: ignore[arg-type]
        wanted = paths if len(missing) == len(paths) else [paths[slot] for slot in missing]
        block = np.asarray(self.features(handle, load_batch(list(wanted)), index))
        if block.ndim != 2 or block.shape[0] != len(wanted):
            raise TrainingError(
                f"adapter {self.name!r} features() returned shape {block.shape} for a "
                f"batch of {len(wanted)} records; (B, D) is required"
            )
        cache.computed += len(wanted)
        for row, slot in zip(block, missing):
            cache.put(keys[slot], row)
            rows[slot] = row
        if len(missing) == len(paths):
            return block
        return np.stack(rows)  # type: ignore[arg-type]

    def forward_cached(
        self,
        handle: Any,
        paths: Sequence[str],
        load_batch: Callable[[Sequence[str]], np.ndarray],
    ) -> Optional[np.ndarray]:
        """A probe-headed handle's scores over cached representations, else ``None``.

        ``forward`` receives arrays only, so it cannot know which tensor a row
        came from; the prediction path in ``models`` offers paths here first
        and falls back to ``forward`` on ``None``. ``None`` is returned for
        every handle the cache does not serve — no open scope, a fine-tuned
        backbone, or no fitted head (``inference_only`` scores through the
        published head, not through ``features``) — so those paths are
        untouched.
        """
        if not self._serves_cached_features(handle):
            return None
        loaded: Dict[tuple, np.ndarray] = {}

        def load_once(batch_paths: Sequence[str]) -> np.ndarray:
            # An ensemble reads one batch per member; without the cache
            # forward() loads it once for all of them, and so does this.
            key = tuple(batch_paths)
            if key not in loaded:
                loaded[key] = load_batch(list(batch_paths))
            return loaded[key]

        return self._fitted_head_scores(
            handle, lambda index: self.cached_features(handle, paths, load_once, index)
        )

    def _fitted_head_scores(
        self, handle: Any, features_of: Callable[[int], np.ndarray]
    ) -> Optional[np.ndarray]:
        """What ``forward`` returns for a handle with fitted head(s), else ``None``.

        ``features_of(index)`` supplies feature set ``index``. A backend
        implements this once and calls it from ``forward`` with
        ``features(handle, batch, index)``, so the cached and the uncached
        prediction run the same head arithmetic. The default declines, which
        keeps a backend that does not implement it on its own ``forward``.
        """
        return None

    def with_head(self, handle: Any, head: LinearHead, record: Mapping[str, Any]) -> Any:
        """A copy of ``handle`` whose ``forward`` runs ``head`` over ``features``."""
        raise NotImplementedError(
            f"adapter {self.name!r} cannot attach a fitted head, so it supports no "
            "training mode beyond inference_only"
        )

    def _finetune(self, handle: Any, data: FitData, mode: str, hp: FitHParams) -> Any:
        """Fit with gradients reaching the backbone. Backend-specific."""
        raise NotImplementedError(
            f"adapter {self.name!r} does not implement training mode {mode!r}"
        )
