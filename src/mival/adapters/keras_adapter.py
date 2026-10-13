"""Keras 2.7 backend.

The archived PROPHECG runtime is intentionally CPU-only: it predates the CUDA
toolkit on this instance. TensorFlow is imported inside load() so that the
torch environment can import mival.adapters without TensorFlow installed.

This adapter fits a head (``linear_probe``) but does not backpropagate into the
backbone; see ``_finetune`` for why.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

import numpy as np

from mival.adapters._training import FitData, FitHParams, LinearHead, TrainingError
from mival.adapters.base import Adapter, to_model_layout
from mival.decode.outputs import DECODED_TYPES
from mival.modelcard import ModelCard


@dataclass
class KerasHandle:
    members: List[Any]
    card: ModelCard
    #: One fitted head per ensemble member, in member order. Empty until a
    #: ``linear_probe`` has run; the card's own pooling rule combines them.
    heads: List[LinearHead] = field(default_factory=list)
    #: What the fit did, copied into ``train_log.jsonl`` by the models stage.
    fit_record: Optional[Mapping[str, Any]] = None


def _as_two_class(positive: np.ndarray) -> np.ndarray:
    """``(B,)`` positive-class probability as the ``(B, 2)`` form of the card.

    The card's ``ensemble.method`` is defined over the model's own output
    columns, so pooling reads the same shape whether the probabilities came
    from the published head or from a fitted one.
    """
    column = np.asarray(positive, dtype=np.float64).reshape(-1, 1)
    return np.hstack([1.0 - column, column])


def pool_ensemble(
    outputs: List[np.ndarray], method: str, positive_index: int
) -> np.ndarray:
    """Reduce per-member class probabilities to one positive-class score.

    Members are averaged first and the positive column is taken second. The
    reverse order gives a different answer for any non-degenerate ensemble.
    """
    if method == "mean_probability":
        pooled = np.mean(np.stack(outputs, axis=0), axis=0)
    elif method == "none":
        pooled = outputs[0]
    else:
        raise ValueError(f"unsupported ensemble method: {method}")
    return pooled[:, positive_index].astype(np.float64)


def _load_member(tf: Any, card: ModelCard, spec: Mapping[str, Any]) -> Any:
    """A saved model, or (``weights_format: keras_weights``) a built one with weights loaded.

    Some checkpoints ship weights only; the card's ``builder`` names a function
    (under ``code_path``) that returns the uncompiled architecture.
    """
    ext = card.raw.get("x-mival", {})
    if ext.get("weights_format") != "keras_weights":
        return tf.keras.models.load_model(spec["uri"], compile=False)
    import importlib
    import sys

    code_path = ext.get("code_path")
    if code_path and code_path not in sys.path:
        sys.path.insert(0, code_path)
    module_name, _, attribute = ext["builder"]["module"].partition(":")
    build = getattr(importlib.import_module(module_name), attribute)
    model = build(**ext["builder"].get("kwargs", {}))
    model.load_weights(spec["uri"])
    return model


def _select_head(member: Any, raw: Any, head: Optional[str]) -> np.ndarray:
    """One output of a multi-output model, by the card's ``output.head`` name."""
    if isinstance(raw, dict):
        if head is None:
            raise ValueError(f"model returns heads {sorted(raw)}; the card must name output.head")
        return np.asarray(raw[head])
    if isinstance(raw, (list, tuple)):
        names = list(getattr(member, "output_names", []) or [])
        if head is None:
            if len(raw) != 1:
                raise ValueError(f"model returns {len(raw)} heads {names}; the card must name output.head")
            return np.asarray(raw[0])
        if head not in names:
            raise ValueError(f"output.head {head!r} is not one of the model's heads {names}")
        return np.asarray(raw[names.index(head)])
    return np.asarray(raw)


class KerasAdapter(Adapter):
    name = "keras"

    def load(self, card: ModelCard) -> KerasHandle:
        # The archived PROPHECG runtime predates this instance's CUDA toolkit,
        # so the CPU pin must win even if the surrounding process exported a
        # device. setdefault would silently leave an inherited value in place.
        if card.runtime.get("device") != "cuda":
            # A card whose runtime says cuda (the keras 3 environment) keeps the GPU.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf

        members = [_load_member(tf, card, spec) for spec in card.weights]
        return KerasHandle(members=members, card=card)

    def forward(self, handle: KerasHandle, batch: np.ndarray) -> np.ndarray:
        if handle.heads:
            # A probed ensemble pools where the unprobed one pools: at the
            # probability level, by the card's own rule. Each member's head
            # emits a positive-class probability, so the two-column form
            # `pool_ensemble` expects is rebuilt from it.
            # `_as_two_class` also fits an identity head: column 1 is then the
            # predicted value rather than a probability, and `pool_ensemble`
            # still averages it across members the same way.
            outputs = [
                _as_two_class(head.predict(self.features(handle, batch, index)))
                for index, head in enumerate(handle.heads)
            ]
            return pool_ensemble(outputs, self._pooling(handle), 1)
        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        kind = handle.card.output.get("type")
        if kind in DECODED_TYPES and not (
            kind == "regression" and self._pooling(handle) == "mean_probability"
        ):
            return self._decoded(handle, arranged)
        outputs = [
            np.asarray(member(arranged, training=False)) for member in handle.members
        ]
        if handle.card.output.get("type") == "regression":
            column = handle.card.output.get("value_index")
            if column is None:
                raise NotImplementedError(
                    f"{handle.card.model_id} declares output.type 'regression' but no "
                    "output.value_index, so its checkpoint has no column to read a value "
                    "from. Declare which output column is the value."
                )
        else:
            column = handle.card.output["positive_index"]
        return pool_ensemble(outputs, self._pooling(handle), column)

    def _decoded(self, handle: KerasHandle, arranged: np.ndarray) -> np.ndarray:
        """Head by name, optional framing, member mean, then the shared decoder."""
        from mival.decode.frames import predict_in_frames
        from mival.decode.outputs import decode_outputs

        output = handle.card.output
        frame = output.get("frame_samples")
        fs = handle.card.input_contract.sampling_rate_hz
        raws = []
        for member in handle.members:
            def predict(x, member=member):
                return _select_head(member, member(x, training=False), output.get("head"))

            raw = (predict_in_frames(predict, arranged, int(frame), output.get("frame_zscore_eps"))
                   if frame else predict(arranged))
            raws.append(np.asarray(raw, dtype=np.float64))
        if output.get("type") == "segmentation_mask":
            return decode_outputs(output, np.mean(np.stack(raws), axis=0), fs)
        return np.mean(np.stack([decode_outputs(output, raw, fs) for raw in raws]), axis=0)

    def _pooling(self, handle: KerasHandle) -> str:
        return handle.card.ensemble.get("method", "none")

    def feature_set_names(self, handle: KerasHandle) -> List[str]:
        """One representation per ensemble member.

        Independently trained members share no coordinate system, so there is
        no single representation for the ensemble — but there is one per
        member, and the card already says how member outputs combine. Probing
        every member and pooling by that rule keeps the ``linear_probe`` arm
        the same size and shape as the ``inference_only`` arm, which is what
        makes the spec §2.5 comparison a comparison.
        """
        if handle.card.feature_layer is None:
            raise NotImplementedError(
                f"{handle.card.model_id} declares no feature_layer; "
                "representation extraction is unavailable"
            )
        if len(handle.members) > 1 and self._pooling(handle) != "mean_probability":
            raise NotImplementedError(
                f"{handle.card.model_id} is a {len(handle.members)}-member ensemble whose "
                f"card declares ensemble.method={self._pooling(handle)!r}. Per-member heads "
                "can only be pooled by a rule the card states; declare "
                "'mean_probability' or reduce the card to a single member."
            )
        return [f"member{index}" for index in range(len(handle.members))]

    def features(self, handle: KerasHandle, batch: np.ndarray, index: int = 0) -> np.ndarray:
        names = self.feature_set_names(handle)
        if not 0 <= index < len(names):
            raise IndexError(
                f"{handle.card.model_id} has {len(names)} representations; feature set "
                f"{index} does not exist"
            )
        import tensorflow as tf

        member = handle.members[index]
        extractor = tf.keras.Model(
            inputs=member.input, outputs=member.get_layer(handle.card.feature_layer).output
        )
        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        return np.asarray(extractor(arranged, training=False))

    def trainable_groups(self, handle: KerasHandle) -> List[str]:
        return [layer.name for layer in handle.members[0].layers]

    # -- fitting -----------------------------------------------------------

    def with_heads(
        self, handle: KerasHandle, heads: List[LinearHead], record: Mapping[str, Any]
    ) -> KerasHandle:
        if len(heads) != len(handle.members):
            raise TrainingError(
                f"{handle.card.model_id}: {len(heads)} heads were fitted for "
                f"{len(handle.members)} ensemble members"
            )
        # Members are shared rather than copied: linear_probe never writes to
        # them, and the archived models are large.
        return KerasHandle(
            members=handle.members,
            card=handle.card,
            heads=list(heads),
            fit_record=dict(record),
        )

    def _finetune(
        self, handle: KerasHandle, data: FitData, mode: str, hp: FitHParams
    ) -> KerasHandle:
        """Not implemented, deliberately.

        No arm in spec §2.5 needs it: the study's trained arms are two linear
        probes and one torch full fine-tune. The archived runtime is also
        CPU-only by construction (the card's ``runtime.device`` is ``cpu``,
        because Keras 2.7 predates this instance's CUDA toolkit), so
        backpropagating through five ensemble members over a hospital-scale dev
        split is not something the available hardware can run in any case.

        A card that declares one of these modes is declaring something this
        adapter cannot deliver, and saying so here is better than discovering
        it after a long run.
        """
        raise TrainingError(
            f"the keras adapter does not implement training mode {mode!r} for "
            f"{handle.card.model_id!r}. Only 'inference_only' and 'linear_probe' are "
            "available: the archived Keras 2.7 runtime is CPU-only, so fine-tuning a "
            f"{len(handle.members)}-member ensemble is not runnable here. Remove "
            f"{mode!r} from the card's training_modes_supported, or move the model to a "
            "backend that can train it."
        )
