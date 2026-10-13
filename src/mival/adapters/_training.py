"""Backend-independent parts of ``Adapter.fit`` (spec §4.4).

Three things live here rather than in a backend module.

**The hyperparameter contract.** ``fit(handle, data, mode, hparams)`` receives
whatever the study's ``hparam_grid`` produced. Parsing it in one place means
torch and Keras cannot drift into understanding different keys, and an
unrecognised key is an error rather than a silent no-op — a misspelled
``learning_rate`` that is ignored turns a grid search into several identical
candidates whose AUROC differences are noise.

**The linear head.** ``linear_probe`` trains one logistic layer on top of a
frozen representation. Nothing about that is backend-specific once
``Adapter.features`` has produced the representation, so it is implemented once
in NumPy: both adapters get the same optimiser, the same regularisation and the
same early-stopping rule, and the whole thing is testable in an environment
with no backend installed at all.

**The stopping rule.** ``models`` selects hyperparameters by dev AUROC
(``SELECTION_METRIC``). Stopping on a different quantity than the one that
picks the winner would optimise for something the study does not report, so
early stopping watches AUROC too, on the validation fold the stage reserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from mival.threshold import ThresholdPolicyError, auroc


class TrainingError(RuntimeError):
    """A fit could not proceed: bad data, bad hyperparameters, or an unsupported mode."""


#: Modes that fit something. ``inference_only`` never reaches an adapter's fit.
TRAINED_MODES = ("linear_probe", "partial_unfreeze", "full_finetune")

#: Mode-dependent defaults. A frozen backbone tolerates a large step size; a
#: pretrained backbone being fine-tuned does not, because a large step size
#: destroys the pretrained representation before the head has learned anything
#: (the discriminative-fine-tuning argument in Howard & Ruder 2018). These are
#: starting points, not choices the study should rely on: put ``lr`` in the
#: arm's ``hparam_grid`` so it is selected on dev like every other
#: hyperparameter. Whatever is actually used is recorded in ``fit_record`` and
#: from there in ``train_log.jsonl``.
_MODE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "linear_probe": {"epochs": 200, "lr": 1e-2, "weight_decay": 1e-4, "patience": 20},
    "partial_unfreeze": {"epochs": 20, "lr": 1e-4, "weight_decay": 1e-4, "patience": 3},
    "full_finetune": {"epochs": 20, "lr": 1e-5, "weight_decay": 1e-4, "patience": 3},
}

#: Defaults that do not depend on the mode.
#:
#: ``pos_weight`` is 1.0 — the loss is *not* class-balanced by default. STEMI is
#: rare, so reweighting is tempting, but it inflates predicted risk and the
#: study reports calibration as its own evaluation category (spec §4.5, Murphy
#: decomposition). The operating point is what needs to account for the
#: asymmetric cost of a missed STEMI, and the threshold policy
#: ``refit_sens95`` is where that decision is made and recorded. Set
#: ``pos_weight: balanced`` to weigh positives by the negative/positive ratio.
_COMMON_DEFAULTS: Dict[str, Any] = {
    "pos_weight": 1.0,
    "early_stopping_metric": "auroc",
    "seed": None,
    "unfreeze_groups": (),
}

_ALLOWED_KEYS = frozenset(_COMMON_DEFAULTS) | {"epochs", "lr", "weight_decay", "patience"}

_METRICS = ("auroc", "loss")


@dataclass(frozen=True)
class FitHParams:
    """Resolved hyperparameters, plus the record of what they resolved to."""

    epochs: int
    lr: float
    weight_decay: float
    patience: int
    pos_weight: float
    early_stopping_metric: str
    seed: Optional[int]
    unfreeze_groups: Tuple[str, ...]
    given: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        """What ``train_log.jsonl`` should carry (spec §4.4 requires full hparams)."""
        return {
            "epochs": self.epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "pos_weight": self.pos_weight,
            "early_stopping_metric": self.early_stopping_metric,
            "seed": self.seed,
            "unfreeze_groups": list(self.unfreeze_groups),
            "given": dict(self.given),
        }


def parse_hparams(
    hparams: Mapping[str, Any], mode: str, n_pos: int, n_neg: int
) -> FitHParams:
    """Fill in the mode's defaults and reject anything unrecognised."""
    if mode not in _MODE_DEFAULTS:
        raise TrainingError(
            f"unknown training mode {mode!r}; adapters fit {list(TRAINED_MODES)} "
            "('inference_only' never calls fit)"
        )
    unknown = sorted(set(hparams) - _ALLOWED_KEYS)
    if unknown:
        raise TrainingError(
            f"unknown hyperparameter(s) {unknown} for mode {mode!r}; understood keys are "
            f"{sorted(_ALLOWED_KEYS)}. A silently ignored key would make two grid "
            "candidates identical while the study recorded them as different."
        )

    resolved = dict(_COMMON_DEFAULTS)
    resolved.update(_MODE_DEFAULTS[mode])
    resolved.update(hparams)

    epochs = _positive_int(resolved["epochs"], "epochs")
    patience = _positive_int(resolved["patience"], "patience")
    lr = _positive_float(resolved["lr"], "lr")
    weight_decay = _non_negative_float(resolved["weight_decay"], "weight_decay")

    pos_weight = resolved["pos_weight"]
    if isinstance(pos_weight, str):
        if pos_weight != "balanced":
            raise TrainingError(
                f"pos_weight must be a positive number or 'balanced', got {pos_weight!r}"
            )
        if n_pos == 0:
            raise TrainingError(
                "pos_weight='balanced' needs at least one positive training record"
            )
        pos_weight = float(n_neg) / float(n_pos)
    else:
        pos_weight = _positive_float(pos_weight, "pos_weight")

    metric = str(resolved["early_stopping_metric"])
    if metric not in _METRICS:
        raise TrainingError(
            f"early_stopping_metric must be one of {list(_METRICS)}, got {metric!r}"
        )

    seed = resolved["seed"]
    groups = tuple(str(name) for name in (resolved["unfreeze_groups"] or ()))
    if mode == "partial_unfreeze" and not groups:
        raise TrainingError(
            "mode 'partial_unfreeze' requires unfreeze_groups; spec §4.4 makes the "
            "unfrozen group list a mandatory record"
        )

    return FitHParams(
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        pos_weight=pos_weight,
        early_stopping_metric=metric,
        seed=None if seed is None else int(seed),
        unfreeze_groups=groups,
        given=dict(hparams),
    )


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise TrainingError(f"{name} must be an integer, got {value!r}") from None
    if number < 1:
        raise TrainingError(f"{name} must be >= 1, got {number}")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = _non_negative_float(value, name)
    if number <= 0:
        raise TrainingError(f"{name} must be > 0, got {number}")
    return number


def _non_negative_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TrainingError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(number) or number < 0:
        raise TrainingError(f"{name} must be a finite number >= 0, got {value!r}")
    return number


# ---------------------------------------------------------------------------
# the data argument
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitSplit:
    """One side of the fit: signals to load, and the label for each."""

    tensor_paths: Tuple[str, ...]
    y: np.ndarray
    person_id: Tuple[str, ...]

    def __len__(self) -> int:
        return len(self.tensor_paths)

    @property
    def n_pos(self) -> int:
        return int(self.y.sum())

    @property
    def n_neg(self) -> int:
        return len(self) - self.n_pos


@dataclass(frozen=True)
class FitData:
    """The parsed ``data`` argument of ``fit``."""

    train: FitSplit
    validation: Optional[FitSplit]
    load_batch: Callable[[Sequence[str]], np.ndarray]
    batch_size: int
    label_column: str
    #: ``"bce"`` (default) for a classification card, ``"mse"`` for a
    #: regression card. Threaded through the head, the loss and early stopping
    #: so a regression arm never accidentally scores as a probability.
    objective: str = "bce"


def parse_fit_data(data: Mapping[str, Any]) -> FitData:
    """Validate the payload ``models`` builds, and fail early if it is malformed.

    Signals arrive as paths plus a loader, never as one materialised array: a
    dev split of tens of thousands of 12x5000 float32 ECGs does not fit in
    memory.
    """
    if not isinstance(data, Mapping):
        raise TrainingError(f"fit data must be a mapping, got {type(data).__name__}")
    for key in ("tensor_paths", "y", "load_batch", "batch_size"):
        if key not in data:
            raise TrainingError(f"fit data is missing required key {key!r}")
    loader = data["load_batch"]
    if not callable(loader):
        raise TrainingError("fit data 'load_batch' must be callable")
    batch_size = _positive_int(data["batch_size"], "batch_size")
    objective = str(data.get("objective", "bce"))
    if objective not in ("bce", "mse"):
        raise TrainingError(f"objective must be 'bce' or 'mse', got {objective!r}")
    train = _split(data, "train", objective)
    if not len(train):
        raise TrainingError("fit was given no training records")
    if objective == "bce" and (train.n_pos == 0 or train.n_neg == 0):
        raise TrainingError(
            f"the training records carry {train.n_pos} positive and {train.n_neg} "
            "negative labels; a single-class fit has no decision to learn. Check the "
            "dev fold assignment in cohort_split (spec §2.4)."
        )
    body = data.get("validation")
    validation = None
    if body:
        validation = _split(body, "validation", objective)
        if not len(validation):
            validation = None
    return FitData(
        train=train,
        validation=validation,
        load_batch=loader,
        batch_size=batch_size,
        label_column=str(data.get("label_column", "")),
        objective=objective,
    )


def _split(body: Mapping[str, Any], where: str, objective: str = "bce") -> FitSplit:
    paths = tuple(str(path) for path in body["tensor_paths"])
    labels = np.asarray(body["y"])
    if labels.ndim != 1 or labels.size != len(paths):
        raise TrainingError(
            f"{where}: y has shape {labels.shape} but there are {len(paths)} tensor paths"
        )
    persons = tuple(str(value) for value in body.get("person_id", ()) or ())
    if persons and len(persons) != len(paths):
        raise TrainingError(
            f"{where}: {len(persons)} person_ids for {len(paths)} tensor paths"
        )
    if objective == "mse":
        return FitSplit(tensor_paths=paths, y=labels.astype(np.float64), person_id=persons)
    values = set(np.unique(labels).tolist()) if labels.size else set()
    if not values <= {0, 1}:
        raise TrainingError(
            f"{where}: labels must be 0 or 1, got {sorted(values)}; adapters fit a binary "
            "outcome (spec §4.4)"
        )
    return FitSplit(tensor_paths=paths, y=labels.astype(np.int64), person_id=persons)


def epoch_batches(n: int, batch_size: int, rng: Optional[Any] = None) -> List[np.ndarray]:
    """Index batches for one epoch, shuffled when ``rng`` is given.

    ``rng`` is a local ``numpy.random.Generator``. The global NumPy state is
    never touched: a fit that reseeded it would change the behaviour of every
    later bootstrap in the same process.
    """
    order = np.arange(n) if rng is None else rng.permutation(n)
    return [order[start : start + batch_size] for start in range(0, n, batch_size)]


# ---------------------------------------------------------------------------
# early stopping
# ---------------------------------------------------------------------------


class Score(NamedTuple):
    """An early-stopping score, oriented so that larger is always better.

    Two numbers rather than one because AUROC saturates. On a validation fold
    the model already ranks perfectly, every later epoch ties at 1.0, strict
    improvement never fires, and the fit keeps its first epoch — a model that
    ranks correctly while predicting nearly the base rate for everything. That
    is a real failure: calibration is its own reported evaluation category
    (spec §4.5).

    ``tiebreak`` is the negated cross-entropy. Log loss is a strictly proper
    scoring rule, so among epochs that rank identically it prefers the better
    calibrated one, and the epoch-order tie-break only decides between models
    that are indistinguishable on both.
    """

    primary: float
    tiebreak: float


class EarlyStopping:
    """Keep the best epoch by validation score, and stop after ``patience`` misses.

    Improvement must be strict, so the earliest epoch reaching a score wins —
    the less-trained model of two that are indistinguishable.
    """

    def __init__(self, patience: int, greater_is_better: bool = True) -> None:
        self.patience = patience
        self.greater_is_better = greater_is_better
        self.best_score: Optional[Any] = None
        self.best_epoch: Optional[int] = None
        self.since_best = 0

    def update(self, epoch: int, score: Any) -> bool:
        """Record ``score`` for ``epoch``; return True when it is a new best."""
        better = self.best_score is None or (
            score > self.best_score if self.greater_is_better else score < self.best_score
        )
        if better:
            self.best_score = score
            self.best_epoch = epoch
            self.since_best = 0
            return True
        self.since_best += 1
        return False

    @property
    def best_primary(self) -> Optional[float]:
        """The reportable part of the best score."""
        if self.best_score is None:
            return None
        if isinstance(self.best_score, Score):
            return float(self.best_score.primary)
        return float(self.best_score)

    @property
    def should_stop(self) -> bool:
        return self.since_best >= self.patience


def validation_score(
    probs: np.ndarray, labels: np.ndarray, metric: str, context: str, objective: str = "bce"
) -> Score:
    """The early-stopping score, oriented so that higher is always better."""
    if objective == "mse":
        residual = np.asarray(probs, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
        return Score(primary=-float(np.abs(residual).mean()), tiebreak=-float((residual**2).mean()))
    loss = float(binary_cross_entropy(probs, labels))
    if metric == "loss":
        return Score(primary=-loss, tiebreak=0.0)
    try:
        discrimination = auroc(probs, labels)
    except ThresholdPolicyError as exc:
        raise TrainingError(
            f"{context}: early stopping on AUROC is impossible because the validation "
            f"fold is single-class ({exc}). Either the dev fold assignment in "
            "cohort_split is not stratified, or this arm needs "
            "early_stopping_metric: loss."
        ) from exc
    return Score(primary=discrimination, tiebreak=-loss)


def binary_cross_entropy(
    probs: np.ndarray, labels: np.ndarray, pos_weight: float = 1.0
) -> float:
    """Mean weighted BCE. Probabilities are clipped so a 0 or 1 is not infinite."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    y = np.asarray(labels, dtype=np.float64)
    weights = np.where(y == 1, pos_weight, 1.0)
    losses = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float((weights * losses).sum() / weights.sum())


def mean_squared_error(pred: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error, the regression loss ``fit_record["train_loss"]`` reports."""
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
    return float(np.mean(diff * diff))


# ---------------------------------------------------------------------------
# the linear head
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearHead:
    """A logistic layer over a fixed representation, with its standardisation.

    Both ``linear_probe`` and the fine-tuning modes end here: a fine-tuned
    backbone's final layer is also linear, so its weights are converted into
    one of these and inference takes a single code path — features, then head.
    ``mean``/``scale`` are the training-set statistics; for a head lifted out of
    a backend module they are 0 and 1.
    """

    weights: np.ndarray
    bias: float
    mean: np.ndarray
    scale: np.ndarray
    #: ``"logistic"`` for a probability head, ``"identity"`` for a regression
    #: head that predicts the outcome's own value.
    link: str = "logistic"

    @classmethod
    def of(cls, weights: Any, bias: float, link: str = "logistic") -> "LinearHead":
        w = np.asarray(weights, dtype=np.float64).ravel()
        return cls(
            weights=w,
            bias=float(bias),
            mean=np.zeros_like(w),
            scale=np.ones_like(w),
            link=link,
        )

    def logits(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise TrainingError(
                f"features must be 2-D (B, D), got shape {matrix.shape}"
            )
        if matrix.shape[1] != self.weights.size:
            raise TrainingError(
                f"head expects {self.weights.size}-dimensional features, got "
                f"{matrix.shape[1]}"
            )
        return ((matrix - self.mean) / self.scale) @ self.weights + self.bias

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Values for an identity head, probabilities for a logistic one."""
        return self.logits(features) if self.link == "identity" else self.probabilities(features)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    values = np.asarray(z, dtype=np.float64)
    out = np.empty_like(values)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    out[~positive] = exponent / (1.0 + exponent)
    return out


def fit_linear_head(
    features: np.ndarray,
    labels: np.ndarray,
    hp: FitHParams,
    validation: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    context: str = "linear_probe",
    objective: str = "bce",
) -> Tuple[LinearHead, Dict[str, Any]]:
    """Logistic (``bce``) or linear (``mse``) regression on a frozen representation, by full-batch Adam."""
    # Full batch rather than mini-batch: the representation is already in
    # memory and the objective is convex, so gradient noise buys nothing. The
    # weights start at zero and the bias at the training log-odds for bce, at
    # the training mean for mse; with a convex objective and a deterministic
    # start the fit consumes no randomness at all, which is why linear_probe
    # reproduces exactly regardless of the seed.
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).ravel()
    if X.ndim != 2 or X.shape[0] != y.size:
        raise TrainingError(
            f"{context}: features shape {X.shape} does not match {y.size} labels"
        )
    if y.size == 0:
        raise TrainingError(f"{context}: no training rows")

    # Standardised on the training rows: Adam is scale-sensitive in practice
    # and these representations have wildly differing per-unit variances, so an
    # unstandardised fit converges at very different rates per dimension. The
    # statistics travel inside the returned head, so inference applies the same
    # transform.
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    # A constant feature carries no information; dividing by its zero spread
    # would produce NaNs that silently poison every later gradient.
    scale = np.where(scale > 0, scale, 1.0)
    Z = (X - mean) / scale

    weights = np.zeros(Z.shape[1], dtype=np.float64)
    if objective == "mse":
        # The bias starts at the training mean rather than at zero: epoch 0
        # already predicts the base rate, so the first gradients carry signal
        # about the features rather than about the outcome's mean.
        bias = float(y.mean())
        sample_weight = np.ones_like(y)
    else:
        n_pos = float(y.sum())
        prevalence = n_pos / y.size
        bias = float(np.log(prevalence / (1.0 - prevalence))) if 0 < prevalence < 1 else 0.0
        sample_weight = np.where(y == 1, hp.pos_weight, 1.0)
    total_weight = sample_weight.sum()

    validation_features = None
    if validation is not None:
        validation_features = (
            np.asarray(validation[0], dtype=np.float64),
            np.asarray(validation[1]).ravel(),
        )
    stopper = EarlyStopping(hp.patience, greater_is_better=True)
    best = (weights.copy(), bias)

    # Adam (Kingma & Ba 2015) with the paper's default moments.
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_w = np.zeros_like(weights)
    v_w = np.zeros_like(weights)
    m_b = v_b = 0.0
    trace: List[Dict[str, float]] = []
    epochs_run = 0

    for epoch in range(1, hp.epochs + 1):
        epochs_run = epoch
        p = (Z @ weights + bias) if objective == "mse" else sigmoid(Z @ weights + bias)
        residual = sample_weight * (p - y)
        grad_w = Z.T @ residual / total_weight
        grad_b = float(residual.sum() / total_weight)

        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * grad_w**2
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * grad_b**2
        correction1 = 1 - beta1**epoch
        correction2 = 1 - beta2**epoch
        # Decoupled weight decay (Loshchilov & Hutter 2019): with Adam, folding
        # L2 into the gradient makes the effective penalty depend on each
        # coordinate's gradient history. Decoupling it is also what torch's
        # AdamW does, so the two fitting paths regularise identically.
        weights = weights * (1.0 - hp.lr * hp.weight_decay) - hp.lr * (
            m_w / correction1
        ) / (np.sqrt(v_w / correction2) + eps)
        bias = bias - hp.lr * (m_b / correction1) / (np.sqrt(v_b / correction2) + eps)

        if validation_features is None:
            best = (weights.copy(), bias)
            continue
        link = "identity" if objective == "mse" else "logistic"
        head = LinearHead(weights=weights, bias=bias, mean=mean, scale=scale, link=link)
        score = validation_score(
            head.predict(validation_features[0]),
            validation_features[1],
            hp.early_stopping_metric,
            context,
            objective,
        )
        trace.append(
            {
                "epoch": epoch,
                "validation_score": score.primary,
                "validation_loss": -score.tiebreak,
            }
        )
        if stopper.update(epoch, score):
            best = (weights.copy(), bias)
        if stopper.should_stop:
            break

    link = "identity" if objective == "mse" else "logistic"
    head = LinearHead(weights=best[0], bias=best[1], mean=mean, scale=scale, link=link)
    train_loss = (
        mean_squared_error(head.predict(X), y)
        if objective == "mse"
        else binary_cross_entropy(head.probabilities(X), y, hp.pos_weight)
    )
    record = {
        "head": "linear" if objective == "mse" else "logistic",
        "objective": objective,
        "optimizer": "adam",
        "n_train": int(y.size),
        "n_features": int(Z.shape[1]),
        "epochs_run": epochs_run,
        "best_epoch": stopper.best_epoch,
        "best_validation_score": stopper.best_primary,
        "validation_metric": (
            ("mae" if objective == "mse" else hp.early_stopping_metric)
            if validation is not None
            else None
        ),
        "train_loss": train_loss,
        "trace": trace,
    }
    return head, record
