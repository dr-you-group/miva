"""PyTorch backend.

torch is imported inside methods so that importing `mival.adapters` stays
cheap and works in the keras27 environment where torch is absent.

Inference has one shape whatever produced the handle: run the module to get a
representation, then run a linear head over it. ``inference_only`` on a card
whose checkpoint already carries the right head is the case where the head
comes from the checkpoint; ``linear_probe`` and the fine-tuning modes fit one.
Keeping a single path means a fitted handle cannot accidentally be scored
through a different code path than an unfitted one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from mival.adapters._attribution import (
    DEFAULT_STEPS,
    integration_points,
    resolve_baseline,
)
from mival.adapters._training import (
    FitData,
    FitHParams,
    FitSplit,
    LinearHead,
    TrainingError,
    EarlyStopping,
    epoch_batches,
    validation_score,
)
from mival.adapters.base import Adapter, to_model_layout
from mival.modelcard import ModelCard


@dataclass
class TorchHandle:
    module: Any
    device: str
    card: ModelCard
    #: Present once the handle has been fitted. ``None`` means the module
    #: emits no positive-class probability yet.
    head: Optional[LinearHead] = None
    #: What the fit did, copied into ``train_log.jsonl`` by the models stage.
    fit_record: Optional[Mapping[str, Any]] = None


DEFAULT_RETURNS = ("logits", "features")
ALLOWED_RETURNS = frozenset({"logits", "features", "ignore"})


def _validate_returns(card: ModelCard) -> None:
    """Reject a runtime.returns name this adapter does not know."""
    # A misnamed position (a typo, say) would silently read as None in
    # _module_outputs. Checked once at load rather than on every forward call,
    # since every handle is built through load first.
    names = card.runtime.get("returns", DEFAULT_RETURNS)
    for name in names:
        if name not in ALLOWED_RETURNS:
            raise ValueError(
                f"{card.model_id!r}: runtime.returns names {name!r}, which is not one of "
                f"{sorted(ALLOWED_RETURNS)}"
            )


def _module_outputs(handle: "TorchHandle", tensor: Any, module: Any = None, extra: Any = None) -> Tuple[Optional[Any], Optional[Any]]:
    """``(logits, features)`` read off the module's return value by the card's ``runtime.returns``.

    ``extra`` is the covariate tensor of a card that declares ``covariates``.
    """
    target = module or handle.module
    output = target(tensor) if extra is None else target(tensor, extra)
    outputs = output if isinstance(output, tuple) else (output,)
    names = tuple(handle.card.runtime.get("returns", DEFAULT_RETURNS))
    if len(outputs) > len(names):
        raise TrainingError(
            f"{handle.card.model_id!r} returned {len(outputs)} tensors but runtime.returns "
            f"names {len(names)}: {list(names)}"
        )
    named = dict(zip(names, outputs))
    return named.get("logits"), named.get("features")


def _module_features(handle: "TorchHandle", tensor: Any, module: Any = None) -> Any:
    features = _module_outputs(handle, tensor, module)[1]
    if features is None:
        raise TrainingError(
            f"{handle.card.model_id!r} names no 'features' in runtime.returns; "
            "there is no representation to probe"
        )
    return features


def _module_dtype(module: Any) -> Any:
    """The dtype the module's own parameters use.

    Attribution builds its input tensors rather than receiving them, so it has
    to match the module instead of hoping the caller's array already did.
    """
    import torch

    for parameter in module.parameters():
        return parameter.dtype
    return torch.float32


def _as_tensor(array: np.ndarray, device: str, dtype: Any) -> Any:
    import torch

    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=dtype)


def build_module(card: ModelCard, weight: Optional[Mapping[str, Any]] = None) -> Any:
    """Construct the module the card declares and load its weights into it."""
    import sys

    import torch

    ext = card.raw["x-mival"]
    uri = (weight or card.weights[0])["uri"]
    weights_format = ext.get("weights_format", "checkpoint")
    code_path = ext.get("code_path")
    if code_path and code_path not in sys.path:
        sys.path.insert(0, code_path)

    if weights_format == "jit":
        return torch.jit.load(uri, map_location="cpu")

    builder = ext.get("builder")
    if not builder:
        raise ValueError(f"{card.model_id!r}: weights_format {weights_format!r} needs x-mival.builder")
    module_name, _, attribute = builder["module"].partition(":")
    constructor = getattr(_import_builder_module(module_name, code_path, card.model_id), attribute)
    module = constructor(*builder.get("args", ()), **builder.get("kwargs", {}))

    state = _read_state_dict(uri, weights_format, ext.get("state_dict_key"))
    for old, new in ext.get("rename_keys", {}).items():
        state = {key.replace(old, new, 1) if key.startswith(old) else key: value for key, value in state.items()}
    _check_n_outputs(card, state)
    module.load_state_dict(state, strict=True)
    return module


def _import_builder_module(module_name: str, code_path: Optional[str], model_id: str) -> Any:
    """Import the builder module from this card's own code_path."""
    import importlib
    import sys

    if not code_path:
        return importlib.import_module(module_name)
    root = Path(code_path).resolve()
    own = _module_file_in(root, module_name)
    if own is None:
        # The card's code_path does not provide this module, so it comes from
        # the environment and there is nothing to disambiguate.
        return importlib.import_module(module_name)

    module = importlib.import_module(module_name)
    if _module_is_under(module, root):
        return module

    # code_path entries accumulate on sys.path across the cards of one run and
    # sys.modules caches by bare name, so a second card declaring a module name
    # another card already imported is handed that card's code. Drop the cached
    # one and import this card's file, with its own root searched first.
    for name in [key for key in list(sys.modules)
                 if key == module_name or key.startswith(module_name + ".")]:
        del sys.modules[name]
    saved = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.path[:] = saved
    if not _module_is_under(module, root):
        raise ValueError(
            f"{model_id!r}: builder module {module_name!r} resolved to "
            f"{getattr(module, '__file__', None)!r}, but this card's x-mival.code_path is "
            f"{str(root)!r}, which holds {str(own)!r}. A module of that name from another "
            "card's code_path was found first on sys.path or in sys.modules; give one of "
            "the two a distinct top-level name."
        )
    return module


def _module_file_in(root: Path, module_name: str) -> Optional[Path]:
    top = module_name.partition(".")[0]
    for candidate in (root / f"{top}.py", root / top / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _module_is_under(module: Any, root: Path) -> bool:
    origin = getattr(module, "__file__", None)
    if origin is None:
        return False
    resolved = Path(origin).resolve()
    return resolved == root or root in resolved.parents


def _read_state_dict(uri: str, weights_format: str, state_dict_key: Optional[str]) -> Dict[str, Any]:
    import torch

    if weights_format == "safetensors":
        from safetensors.torch import load_file

        return load_file(uri)
    # Pickle stays off. A checkpoint's non-tensor metadata (e.g. saved metrics)
    # can carry any of NumPy's common scalar dtypes, not just float64, so the
    # allow-list covers the dtype classes rather than one instance of one; the
    # scalar reconstructor is allow-listed both under NumPy's pre-2.0 module
    # path (what older checkpoints were pickled with) and its own current
    # path (what a checkpoint written in this environment's NumPy pickles).
    safe_globals = [
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np._core.multiarray.scalar,
        (np.dtype, "numpy.dtype"),
        *[type(np.dtype(name)) for name in ("float32", "float64", "int32", "int64", "bool")],
    ]
    with torch.serialization.safe_globals(safe_globals):
        loaded = torch.load(uri, map_location="cpu", weights_only=True)
    if weights_format == "checkpoint":
        return loaded[state_dict_key or "state_dict"]
    if weights_format == "state_dict":
        return loaded
    raise ValueError(f"unknown weights_format {weights_format!r}; use state_dict, checkpoint, safetensors or jit")


def _check_n_outputs(card: ModelCard, state: Dict[str, Any]) -> None:
    """A card that disagrees with its own checkpoint would silently score the wrong head."""
    key = card.output.get("weight_key")
    declared = card.output.get("n_outputs")
    if key is None or declared is None:
        return
    actual = int(state[key].shape[0])
    if actual != int(declared):
        raise ValueError(f"card declares {declared} outputs but the checkpoint has {actual}")


#: ``ensemble.method`` values that average the members' raw outputs.
MEAN_ENSEMBLE_METHODS = ("mean", "mean_regression", "mean_probability")


def _build_with_ensemble(card: ModelCard) -> Any:
    """One module, or the mean of one module per weight file (L2-14)."""
    import torch

    method = (card.ensemble or {}).get("method", "none")
    members = int((card.ensemble or {}).get("members", 1))
    if len(card.weights) == 1 and members <= 1:
        return build_module(card)
    if method not in MEAN_ENSEMBLE_METHODS:
        raise ValueError(
            f"{card.model_id!r} lists {len(card.weights)} weights with ensemble.method "
            f"{method!r}; the torch adapter averages members for {list(MEAN_ENSEMBLE_METHODS)}"
        )
    if members != len(card.weights):
        raise ValueError(
            f"{card.model_id!r}: ensemble.members is {members} but the card lists "
            f"{len(card.weights)} weights"
        )

    class MeanEnsemble(torch.nn.Module):
        def __init__(self, modules):
            super().__init__()
            self.members = torch.nn.ModuleList(modules)

        def forward(self, *inputs):
            outputs = [member(*inputs) for member in self.members]
            if isinstance(outputs[0], tuple):
                return tuple(torch.stack(parts).mean(dim=0) for parts in zip(*outputs))
            return torch.stack(outputs).mean(dim=0)

    return MeanEnsemble([build_module(card, weight) for weight in card.weights])


#: Output types read without a positive_index or value_index check and
#: decoded by ``mival.decode.outputs`` (shared with the keras adapter).
SCORE_FREE_OUTPUT_TYPES = ("risk_score", "segmentation_mask", "survival_curve", "mtlr")


class TorchAdapter(Adapter):
    name = "torch"

    def load(self, card: ModelCard) -> TorchHandle:
        import torch

        _validate_returns(card)
        module = _build_with_ensemble(card)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return TorchHandle(module=module.to(device).eval(), device=device, card=card)

    def features(self, handle: TorchHandle, batch: np.ndarray, index: int = 0) -> np.ndarray:
        import torch

        if index != 0:
            raise IndexError(
                f"{handle.card.model_id!r} has one representation; feature set {index} "
                "does not exist"
            )

        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        tensor = torch.from_numpy(np.ascontiguousarray(arranged)).to(handle.device)
        with torch.inference_mode():
            features = _module_features(handle, tensor)
        return features.detach().cpu().numpy()

    def forward(self, handle: TorchHandle, batch: np.ndarray, covariates: Optional[np.ndarray] = None) -> np.ndarray:
        if handle.head is not None:
            return handle.head.predict(self.features(handle, batch))
        return self._published_scores(handle, batch, covariates)

    def _published_scores(
        self, handle: TorchHandle, batch: np.ndarray, covariates: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """The checkpoint's own score: a positive-class probability, or a regression value.

        Which column is the positive class (or the regression value), and
        whether the head is a softmax or per-class logits, are card facts. A
        classification checkpoint whose card names no ``positive_index``
        publishes no probability for this outcome, and ``inference_only`` is
        genuinely unavailable for it. That is a property of the model, so the
        error says which model and what to do instead. A regression card names
        no ``positive_index`` at all, so that check does not apply to it, but it
        must name a ``value_index`` instead, checked the same way.
        """
        import torch

        kind = handle.card.output.get("type", "logits")
        if kind in SCORE_FREE_OUTPUT_TYPES:
            pass
        elif kind == "regression":
            if handle.card.output.get("value_index") is None:
                raise NotImplementedError(
                    f"{handle.card.model_id!r} declares output.type 'regression' but no "
                    "output.value_index, so its checkpoint has no column to read a value "
                    "from. Declare which output column is the value."
                )
        elif handle.card.output.get("positive_index") is None:
            raise NotImplementedError(
                f"{handle.card.model_id!r} declares no output.positive_index, so its "
                "checkpoint emits no probability for this outcome. Fit a head with "
                "linear_probe, partial_unfreeze or full_finetune before calling forward."
            )
        arranged = to_model_layout(batch, handle.card.input_contract.layout)
        tensor = torch.from_numpy(np.ascontiguousarray(arranged)).to(handle.device)
        extra = None
        if covariates is not None:
            extra = torch.from_numpy(np.ascontiguousarray(covariates, dtype=np.float32)).to(handle.device)
        with torch.inference_mode():
            logits, _features = _module_outputs(handle, tensor, extra=extra)
        if logits is None:
            raise NotImplementedError(
                f"{handle.card.model_id!r} returns a representation only; there is no "
                "published head to read a score from"
            )
        if kind in SCORE_FREE_OUTPUT_TYPES or kind == "regression":
            from mival.decode.outputs import decode_outputs

            return decode_outputs(handle.card.output, logits.detach().float().cpu().numpy(),
                                  handle.card.input_contract.sampling_rate_hz)
        return self._score_from_outputs(handle, logits).detach().cpu().numpy().astype(np.float64)

    def _score_from_outputs(self, handle: TorchHandle, outputs: Any) -> Any:
        """Positive-class probability, or the value column for a regression card.

        ``forward`` detaches it; ``attribute`` differentiates through it. Both
        must read the same column of the same head, so the branch lives once.
        """
        import torch

        kind = handle.card.output.get("type", "logits")
        if kind == "regression":
            return outputs[:, int(handle.card.output["value_index"])]
        if kind == "risk_score":
            return outputs.reshape(outputs.shape[0], -1)[:, int(handle.card.output.get("value_index", 0))]
        if kind == "segmentation_mask":
            raise NotImplementedError(
                f"{handle.card.model_id!r}: a segmentation mask has no single score to attribute"
            )
        index = int(handle.card.output["positive_index"])
        if kind == "softmax":
            return torch.softmax(outputs, dim=-1)[:, index]
        if kind in ("logits", "sigmoid"):
            return torch.sigmoid(outputs[:, index])
        raise ValueError(
            f"unknown output.type {kind!r}; adapter reads 'softmax', 'logits', 'sigmoid' or "
            "'regression'"
        )

    def trainable_groups(self, handle: TorchHandle) -> List[str]:
        return [name for name, _ in handle.module.named_children()]

    # -- attribution (spec §4.4, §4.6) --------------------------------------

    def attribute(
        self,
        handle: TorchHandle,
        batch: np.ndarray,
        baseline: Any = None,
        steps: int = DEFAULT_STEPS,
    ) -> np.ndarray:
        """Integrated Gradients of the positive-class probability.

        Read ``mival.adapters._attribution`` for why ``baseline`` is required.
        The returned array is in the canonical ``(B, n_leads, n_samples)``
        layout whatever the card's layout is, so a caller plotting a lead never
        has to know how the model wanted its input transposed.
        """
        import torch

        array = np.asarray(batch, dtype=np.float64)
        reference = resolve_baseline(baseline, array)
        layout = handle.card.input_contract.layout
        module = handle.module
        dtype = _module_dtype(module)

        x = _as_tensor(to_model_layout(array, layout), handle.device, dtype)
        base = _as_tensor(to_model_layout(reference, layout), handle.device, dtype)

        was_training = module.training
        module.eval()
        try:
            gradients = torch.zeros_like(x)
            for alpha in integration_points(steps):
                point = (base + float(alpha) * (x - base)).detach().requires_grad_(True)
                probability = self._probability_tensor(handle, point)
                (gradient,) = torch.autograd.grad(probability.sum(), point)
                gradients = gradients + gradient
            attribution = (x - base) * (gradients / float(steps))
        finally:
            if was_training:
                module.train()

        result = attribution.detach().cpu().numpy().astype(np.float64)
        if layout == "time_lead":
            result = np.transpose(result, (0, 2, 1))
        return np.ascontiguousarray(result)

    def _probability_tensor(self, handle: TorchHandle, tensor: Any) -> Any:
        """The positive-class probability as a differentiable tensor.

        The fitted-head branch re-expresses the NumPy ``LinearHead`` in torch
        rather than calling it, because attribution needs the gradient to reach
        the input through the head. It is the same arithmetic — standardise,
        one linear layer, sigmoid — so a probed handle and an ``inference_only``
        handle are attributed through paths that agree with what ``forward``
        returns.
        """
        import torch

        if handle.head is None:
            if handle.card.output.get("type") == "regression":
                raise NotImplementedError(
                    "attribution is defined for probabilities; regression arms are "
                    "skipped in stage 6"
                )
            logits, _features = _module_outputs(handle, tensor)
            if logits is None:
                raise NotImplementedError(
                    f"{handle.card.model_id!r} returns a representation only; there is no "
                    "published head to attribute through"
                )
            if handle.card.output.get("positive_index") is None:
                raise NotImplementedError(
                    f"{handle.card.model_id!r} declares no output.positive_index, so there "
                    "is no probability to attribute. Fit a head first."
                )
            return self._score_from_outputs(handle, logits)

        head = handle.head
        if head.link == "identity":
            raise NotImplementedError(
                "attribution is defined for probabilities; regression arms are "
                "skipped in stage 6"
            )
        features = _module_features(handle, tensor)
        as_tensor = lambda values: torch.as_tensor(  # noqa: E731 - three identical conversions
            np.asarray(values, dtype=np.float64), dtype=features.dtype, device=features.device
        )
        standardised = (features - as_tensor(head.mean)) / as_tensor(head.scale)
        return torch.sigmoid(standardised @ as_tensor(head.weights) + float(head.bias))

    # -- fitting -----------------------------------------------------------

    def with_head(
        self, handle: TorchHandle, head: LinearHead, record: Mapping[str, Any]
    ) -> TorchHandle:
        # The module is shared rather than copied: linear_probe never wrote to
        # it, and copying a backbone per inner CV fold would cost hundreds of
        # megabytes for nothing. The fine-tuning path, which does write, copies.
        return TorchHandle(
            module=handle.module,
            device=handle.device,
            card=handle.card,
            head=head,
            fit_record=dict(record),
        )

    def _finetune(
        self, handle: TorchHandle, data: FitData, mode: str, hp: FitHParams
    ) -> TorchHandle:
        """Backpropagate into the backbone: ``partial_unfreeze`` or ``full_finetune``.

        The fitted module is a deep copy. ``models`` fits a fresh handle per
        inner CV fold and compares their held-out predictions, so mutating the
        loaded module would carry one fold's weights into the next fold's
        estimate — the exact leak the internal CV exists to avoid.
        """
        import torch

        if hp.seed is None:
            raise TrainingError(
                f"training mode {mode!r} draws randomness (parameter initialisation, "
                "batch order, and any dropout in the backbone) and is not reproducible "
                "without a seed. Run the models stage with --seed."
            )

        module = copy.deepcopy(handle.module).to(handle.device)
        frozen = self._set_trainable(module, mode, hp)

        # torch's initialisers and dropout draw from the process-global
        # generator, so seeding it is the only way to make the fit
        # reproducible. The previous state is restored afterwards: a fit that
        # left the global stream reseeded would change the behaviour of
        # everything downstream in the same process.
        rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(hp.seed)
            return self._run_finetune(handle, module, frozen, data, mode, hp)
        finally:
            torch.set_rng_state(rng_state)

    def _set_trainable(self, module: Any, mode: str, hp: FitHParams) -> Tuple[str, ...]:
        """Mark which parameters get gradients. Returns the frozen group names."""
        children = dict(module.named_children())
        if mode == "full_finetune":
            for parameter in module.parameters():
                parameter.requires_grad_(True)
            return ()
        unknown = [name for name in hp.unfreeze_groups if name not in children]
        if unknown:
            raise TrainingError(
                f"unfreeze_groups {unknown} are not trainable groups of this module; "
                f"it reports {sorted(children)}"
            )
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        for name in hp.unfreeze_groups:
            for parameter in children[name].parameters():
                parameter.requires_grad_(True)
        return tuple(name for name in children if name not in set(hp.unfreeze_groups))

    def _run_finetune(
        self,
        handle: TorchHandle,
        module: Any,
        frozen: Tuple[str, ...],
        data: FitData,
        mode: str,
        hp: FitHParams,
    ) -> TorchHandle:
        import torch

        device = handle.device
        layout = handle.card.input_contract.layout

        width = self._feature_width(handle, module, data, layout, device)
        head = torch.nn.Linear(width, 1).to(device)
        # Starting the bias at the training log-odds (or, for a regression
        # objective, the training mean) means epoch 0 already predicts the
        # base rate, so the first gradients carry signal about the features
        # rather than about the prevalence.
        with torch.no_grad():
            if data.objective == "mse":
                head.bias.fill_(float(data.train.y.mean()))
            else:
                prevalence = float(data.train.y.mean())
                head.bias.fill_(
                    float(np.log(prevalence / (1.0 - prevalence)))
                    if 0.0 < prevalence < 1.0
                    else 0.0
                )

        parameters = [p for p in module.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            list(head.parameters()) + parameters, lr=hp.lr, weight_decay=hp.weight_decay
        )
        criterion = (
            torch.nn.MSELoss()
            if data.objective == "mse"
            else torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(hp.pos_weight, dtype=torch.float32, device=device)
            )
        )

        rng = np.random.default_rng(hp.seed)
        stopper = EarlyStopping(hp.patience, greater_is_better=True)
        best: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
        trace: List[Dict[str, float]] = []
        epochs_run = 0

        for epoch in range(1, hp.epochs + 1):
            epochs_run = epoch
            module.train()
            # Frozen groups stay in eval mode so their batch-norm running
            # statistics are not updated by data they are not learning from.
            children = dict(module.named_children())
            for name in frozen:
                children[name].eval()
            head.train()

            total = 0.0
            for indices in epoch_batches(len(data.train), data.batch_size, rng):
                tensor = self._batch_tensor(data, data.train, indices, layout, device)
                targets = torch.from_numpy(
                    data.train.y[indices].astype(np.float32)
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = head(_module_features(handle, tensor, module=module)).squeeze(-1)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu()) * len(indices)
            train_loss = total / len(data.train)

            if data.validation is None:
                best = (_cpu_state(module), _cpu_state(head))
                trace.append({"epoch": epoch, "train_loss": train_loss})
                continue

            outputs = self._outputs(handle, module, head, data, data.validation, layout, device)
            score = validation_score(
                outputs, data.validation.y, hp.early_stopping_metric, f"{mode} fit", data.objective
            )
            trace.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_score": score.primary,
                    "validation_loss": -score.tiebreak,
                }
            )
            if stopper.update(epoch, score):
                best = (_cpu_state(module), _cpu_state(head))
            if stopper.should_stop:
                break

        if best is not None:
            module.load_state_dict(best[0])
            head.load_state_dict(best[1])
        module = module.to(device).eval()

        # The fitted head is linear, so it is lifted out of torch and inference
        # runs the same features-then-head path as linear_probe. Nothing about
        # the fit is lost: the backbone's learned weights live in `module`.
        with torch.no_grad():
            lifted = LinearHead.of(
                head.weight.detach().cpu().numpy(),
                float(head.bias.detach().cpu()),
                link="identity" if data.objective == "mse" else "logistic",
            )
        record = {
            "mode": mode,
            "head": "linear" if data.objective == "mse" else "logistic",
            "objective": data.objective,
            "optimizer": "adamw",
            "hparams": hp.to_record(),
            "n_train": len(data.train),
            "n_validation": 0 if data.validation is None else len(data.validation),
            "n_features": int(lifted.weights.size),
            "frozen": list(frozen),
            "trainable_parameters": int(
                sum(p.numel() for p in module.parameters() if p.requires_grad)
            ),
            "epochs_run": epochs_run,
            "best_epoch": stopper.best_epoch,
            "best_validation_score": stopper.best_primary,
            "validation_metric": (
                ("mae" if data.objective == "mse" else hp.early_stopping_metric)
                if data.validation is not None
                else None
            ),
            "trace": trace,
        }
        return TorchHandle(
            module=module,
            device=device,
            card=handle.card,
            head=lifted,
            fit_record=record,
        )

    # -- fine-tuning helpers ------------------------------------------------

    def _feature_width(
        self, handle: TorchHandle, module: Any, data: FitData, layout: str, device: str
    ) -> int:
        """Probe the representation size with one batch.

        Read from the module rather than from the card: ``feature_width`` is
        not a required ModelCard field, and a card that disagreed with its own
        checkpoint would silently build the wrong head.
        """
        import torch

        indices = np.arange(min(data.batch_size, len(data.train)))
        tensor = self._batch_tensor(data, data.train, indices, layout, device)
        was_training = module.training
        module.eval()
        with torch.no_grad():
            features = _module_features(handle, tensor, module=module)
        module.train(was_training)
        if features.dim() != 2:
            raise TrainingError(
                f"module produced features of shape {tuple(features.shape)}; a linear "
                "head needs (B, D)"
            )
        return int(features.shape[1])

    def _batch_tensor(
        self, data: FitData, split: FitSplit, indices: np.ndarray, layout: str, device: str
    ) -> Any:
        import torch

        paths = [split.tensor_paths[index] for index in indices]
        arranged = to_model_layout(data.load_batch(paths), layout)
        return torch.from_numpy(np.ascontiguousarray(arranged)).to(device)

    def _outputs(
        self,
        handle: TorchHandle,
        module: Any,
        head: Any,
        data: FitData,
        split: FitSplit,
        layout: str,
        device: str,
    ) -> np.ndarray:
        """Validation predictions: probabilities for ``bce``, values for ``mse``."""
        import torch

        module.eval()
        head.eval()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for indices in epoch_batches(len(split), data.batch_size):
                tensor = self._batch_tensor(data, split, indices, layout, device)
                logits = head(_module_features(handle, tensor, module=module)).squeeze(-1)
                values = logits if data.objective == "mse" else torch.sigmoid(logits)
                chunks.append(values.cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)


def _cpu_state(module: Any) -> Dict[str, Any]:
    """A detached CPU copy of a module's parameters, for restoring the best epoch."""
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}
