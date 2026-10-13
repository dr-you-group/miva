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
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np

from mival.adapters._attribution import DEFAULT_STEPS
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
            batch = data.load_batch(list(paths))
            block = np.asarray(self.features(handle, batch, index), dtype=np.float64)
            if block.ndim != 2 or block.shape[0] != len(paths):
                raise TrainingError(
                    f"adapter {self.name!r} features() returned shape {block.shape} for a "
                    f"batch of {len(paths)} records; (B, D) is required"
                )
            chunks.append(block)
        if not chunks:
            raise TrainingError(f"adapter {self.name!r}: no records to extract features from")
        return np.concatenate(chunks, axis=0)

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
