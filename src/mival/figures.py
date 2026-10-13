"""Figures for spec §4.5 — every one of them a code artifact.

No figure in this study is drawn by hand, including the STARD participant
flow: it is a query over the exclusion ledger (spec §3.6), so it cannot drift
away from the counts in the manifest.

matplotlib is imported lazily and pinned to the ``Agg`` backend. The instances
this runs on are headless, and importing ``pyplot`` at module scope would make
``import mival.figures`` fail there — and with it, anything that merely wanted
to read a metric.

Every function here takes plain sequences and returns a path. None of them
knows what a subgroup, a perturbation or a model is: the evaluation stage does
the axis-aware selection and hands over labelled series. That is the same
separation the metric functions keep, and for the same reason — a new axis
must not require editing plotting code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

_MISSING = (
    "drawing figures requires matplotlib. Install it with `pip install 'mival[figures]'`."
)

#: A labelled series of predictions: ``(label, y_true, y_prob)``.
Curve = Tuple[str, Sequence[float], Sequence[float]]
#: A labelled estimate with its interval: ``(label, value, ci_lo, ci_hi)``.
Estimate = Tuple[str, float, float, float]


def _pyplot():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(_MISSING) from exc
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as pyplot

    return pyplot


def _save(figure, path: Union[str, Path]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=200)
    _pyplot().close(figure)
    return target


def stard_flow(
    stages: Sequence[Tuple[str, int]],
    exclusions: Sequence[Tuple[str, str, int]],
    path: Union[str, Path],
    title: str = "STARD participant flow",
) -> Path:
    """Participant flow: ``stages`` are ``(label, n_remaining)`` boxes and
    ``exclusions`` are ``(stage, reason_code, n)`` annotations beside them.

    Both come from the ledger, so the diagram and the counts cannot disagree.
    """
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(figsize=(7.5, 1.6 * max(len(stages), 1) + 1.2))
    axes.set_axis_off()
    axes.set_xlim(0, 10)
    axes.set_ylim(0, max(len(stages), 1))

    by_stage: dict = {}
    for stage, reason, count in exclusions:
        by_stage.setdefault(stage, []).append((reason, count))

    for position, (label, remaining) in enumerate(stages):
        y = len(stages) - position - 0.5
        axes.text(
            2.4,
            y,
            f"{label}\nn = {remaining:,}",
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "black"},
        )
        if position + 1 < len(stages):
            axes.annotate(
                "",
                xy=(2.4, y - 0.55),
                xytext=(2.4, y - 0.28),
                arrowprops={"arrowstyle": "->", "color": "black"},
            )
        dropped = by_stage.get(label, [])
        if dropped:
            body = "\n".join(f"{reason}: {count:,}" for reason, count in sorted(dropped))
            axes.text(
                6.2,
                y - 0.42,
                f"excluded\n{body}",
                ha="left",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "0.95", "edgecolor": "0.6"},
            )
    axes.set_title(title)
    return _save(figure, path)


def regression_scatter(series: Sequence[Tuple[str, Any, Any]], path: Union[str, Path]) -> Path:
    """Predicted against observed, one panel per arm, with the identity line."""
    plt = _pyplot()
    figure, axes = plt.subplots(1, len(series), figsize=(4 * len(series), 4), squeeze=False)
    for axis, (label, y, p) in zip(axes[0], series):
        axis.scatter(y, p, s=4, alpha=0.3)
        lo, hi = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
        axis.plot([lo, hi], [lo, hi], color="gray", linewidth=1)
        axis.set_title(label, fontsize=8)
        axis.set_xlabel("observed")
        axis.set_ylabel("predicted")
    return _save(figure, path)


def roc_pr_curves(curves: Sequence[Curve], path: Union[str, Path]) -> Path:
    """ROC and precision-recall panels for a set of labelled predictions."""
    from mival.metrics import auprc, auroc, pr_points, roc_points

    pyplot = _pyplot()
    figure, (roc_axes, pr_axes) = pyplot.subplots(1, 2, figsize=(10, 4.5))
    for label, y_true, y_prob in curves:
        fpr, tpr, _ = roc_points(y_true, y_prob)
        if fpr.size:
            roc_axes.plot(fpr, tpr, label=f"{label} (AUROC {auroc(y_true, y_prob):.3f})")
        recall, precision = pr_points(y_true, y_prob)
        if recall.size:
            pr_axes.plot(recall, precision, label=f"{label} (AP {auprc(y_true, y_prob):.3f})")
    roc_axes.plot([0, 1], [0, 1], linestyle=":", color="0.5", label="chance")
    roc_axes.set_xlabel("1 - specificity")
    roc_axes.set_ylabel("sensitivity")
    roc_axes.set_title("ROC")
    pr_axes.set_xlabel("recall")
    pr_axes.set_ylabel("precision")
    pr_axes.set_title("Precision-recall")
    for axes in (roc_axes, pr_axes):
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1.02)
        axes.legend(fontsize=8, loc="lower right" if axes is roc_axes else "upper right")
    return _save(figure, path)


def calibration_curves(
    curves: Sequence[Curve], path: Union[str, Path], n_knots: int = 4
) -> Path:
    """Flexible (spline) calibration curves — moderate calibration, Van Calster 2019.

    The diagonal is perfect calibration. A histogram of predicted risk sits
    underneath because a calibration curve without the score distribution
    invites reading the sparse right-hand tail as if it were populated.
    """
    from mival.metrics import flexible_calibration_curve

    pyplot = _pyplot()
    figure, (curve_axes, hist_axes) = pyplot.subplots(
        2, 1, figsize=(5.5, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    for label, y_true, y_prob in curves:
        predicted, observed = flexible_calibration_curve(y_true, y_prob, n_knots=n_knots)
        if predicted.size:
            curve_axes.plot(predicted, observed, label=label)
        hist_axes.hist(y_prob, bins=30, histtype="step", label=label)
    curve_axes.plot([0, 1], [0, 1], linestyle=":", color="0.5", label="perfect")
    curve_axes.set_ylabel("observed risk")
    curve_axes.set_title("Flexible calibration")
    curve_axes.legend(fontsize=8, loc="upper left")
    hist_axes.set_xlabel("predicted risk")
    hist_axes.set_ylabel("count")
    hist_axes.set_yscale("log")
    return _save(figure, path)


def decision_curves(
    curves: Sequence[Tuple[str, Sequence[Mapping[str, float]]]],
    path: Union[str, Path],
    band: Optional[Tuple[float, float]] = None,
) -> Path:
    """Decision curves against treat-all and treat-none.

    ``band`` shades the clinically plausible threshold range; for STEMI the
    spec puts it at roughly 1-10%, and performance outside it is not evidence
    of clinical usefulness.
    """
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(figsize=(6, 4.5))
    treat_all_drawn = False
    lowest = 0.0
    for label, rows in curves:
        thresholds = [row["threshold"] for row in rows]
        axes.plot(thresholds, [row["net_benefit"] for row in rows], label=label)
        if not treat_all_drawn:
            axes.plot(
                thresholds,
                [row["treat_all"] for row in rows],
                linestyle="--",
                color="0.4",
                label="treat all",
            )
            axes.axhline(0.0, linestyle=":", color="0.6", label="treat none")
            treat_all_drawn = True
        lowest = min(lowest, min([row["net_benefit"] for row in rows] or [0.0]))
    if band is not None:
        axes.axvspan(band[0], band[1], color="0.9", zorder=0, label="plausible band")
    axes.set_xlabel("threshold probability")
    axes.set_ylabel("net benefit")
    axes.set_title("Decision curve")
    axes.set_ylim(max(lowest, -0.05), None)
    axes.legend(fontsize=8)
    return _save(figure, path)


def degradation_curve(
    series: Mapping[str, Sequence[Estimate]],
    path: Union[str, Path],
    metric_label: str = "metric",
) -> Path:
    """One line per model over an ordered set of conditions, with intervals.

    Used for the perturbation degradation figure, but it knows nothing about
    perturbations: it plots labelled estimates in the order given.
    """
    pyplot = _pyplot()
    figure, axes = pyplot.subplots(figsize=(7, 4.5))
    for name, estimates in series.items():
        labels = [entry[0] for entry in estimates]
        values = [entry[1] for entry in estimates]
        positions = list(range(len(values)))
        axes.plot(positions, values, marker="o", label=name)
        for position, (_, value, lo, hi) in zip(positions, estimates):
            if lo == lo and hi == hi:  # not NaN
                axes.vlines(position, lo, hi, color="0.6", linewidth=1)
        axes.set_xticks(positions)
        axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes.set_ylabel(metric_label)
    axes.set_title("Degradation")
    axes.legend(fontsize=8)
    return _save(figure, path)


def forest_plot(
    estimates: Sequence[Estimate],
    path: Union[str, Path],
    reference: Optional[float] = None,
    metric_label: str = "metric",
    suppressed: Optional[Iterable[bool]] = None,
) -> Path:
    """Estimates with intervals, one row each; suppressed rows drawn hollow.

    Suppressed rows are drawn rather than dropped (spec §4.5): a subgroup too
    small to conclude from is still evidence about how much of the cohort is
    too small to conclude from.
    """
    pyplot = _pyplot()
    flags = list(suppressed) if suppressed is not None else [False] * len(estimates)
    figure, axes = pyplot.subplots(figsize=(6.5, 0.4 * max(len(estimates), 1) + 1.5))
    for row, ((label, value, lo, hi), is_suppressed) in enumerate(zip(estimates, flags)):
        y = len(estimates) - row - 1
        axes.plot(
            [value],
            [y],
            marker="o",
            color="black",
            markerfacecolor="white" if is_suppressed else "black",
        )
        if lo == lo and hi == hi:
            axes.hlines(y, lo, hi, color="0.4", linewidth=1.2)
        axes.text(
            0.0,
            y,
            label + (" *" if is_suppressed else ""),
            transform=axes.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
        )
    if reference is not None:
        axes.axvline(reference, linestyle=":", color="0.5")
    axes.set_yticks([])
    axes.set_ylim(-0.8, len(estimates) - 0.2)
    axes.set_xlabel(metric_label)
    axes.set_title("Subgroup estimates (* suppressed: fewer than the minimum events)")
    return _save(figure, path)


def rank_reversal(
    values: Mapping[str, Sequence[Tuple[str, float]]],
    path: Union[str, Path],
    metric_label: str = "metric",
) -> Path:
    """Model ranking under two or more conditions, drawn as connected lines.

    Lines that cross are the point: a ranking that survives only one operating
    condition is not a ranking. F1 is the usual subject because it moves with
    the threshold and with prevalence.
    """
    pyplot = _pyplot()
    conditions = list(values)
    figure, axes = pyplot.subplots(figsize=(6, 4.5))
    models: dict = {}
    for index, condition in enumerate(conditions):
        for model, value in values[condition]:
            models.setdefault(model, []).append((index, value))
    for model, points in models.items():
        axes.plot([x for x, _ in points], [v for _, v in points], marker="o", label=model)
        axes.annotate(
            model,
            xy=points[-1],
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            va="center",
        )
    axes.set_xticks(range(len(conditions)))
    axes.set_xticklabels(conditions, rotation=20, ha="right", fontsize=8)
    axes.set_ylabel(metric_label)
    axes.set_title("Rank stability")
    return _save(figure, path)


def case_waveform(
    samples,
    path: Union[str, Path],
    leads: Optional[Sequence[str]] = None,
    title: str = "",
    annotations: Sequence[str] = (),
    attribution=None,
    sampling_rate_hz: Optional[float] = None,
) -> Path:
    """One case: the model's input signal, lead by lead, with its context.

    ``samples`` is ``(n_leads, n_samples)`` — the tensor the model was actually
    given, not the source recording. A reviewer judging why a model called a
    trace abnormal has to see the trace the model saw; resampling, rescaling and
    lead selection all happened before it.

    ``attribution`` is optional and has the same shape as ``samples``. It is
    drawn as a shaded band behind each trace rather than as a recoloured line,
    so that the ECG morphology stays readable — the morphology is what the
    reviewer is judging, and a saliency-coloured trace hides it.
    """
    pyplot = _pyplot()
    rows = list(samples)
    n_leads = len(rows)
    if n_leads == 0:
        raise ValueError("case_waveform needs at least one lead")
    names = (
        [str(name) for name in leads]
        if leads and len(leads) == n_leads
        else [f"lead {index}" for index in range(n_leads)]
    )
    n_samples = len(rows[0])
    duration = n_samples / float(sampling_rate_hz) if sampling_rate_hz else float(n_samples)
    x_label = "time (s)" if sampling_rate_hz else "sample"

    height = 0.85 * n_leads + 1.6 + 0.22 * len(annotations)
    figure, axes = pyplot.subplots(
        n_leads, 1, sharex=True, figsize=(9, height), squeeze=False
    )
    column = [pair[0] for pair in axes]
    scale = None
    if attribution is not None:
        magnitudes = [abs(float(value)) for row in attribution for value in row]
        peak = max(magnitudes) if magnitudes else 0.0
        scale = peak if peak > 0 else None
    for index, axis in enumerate(column):
        values = [float(value) for value in rows[index]]
        times = [position * duration / max(n_samples - 1, 1) for position in range(len(values))]
        if scale is not None:
            weights = [abs(float(value)) / scale for value in list(attribution)[index]]
            axis.imshow(
                [weights],
                aspect="auto",
                cmap="Reds",
                vmin=0.0,
                vmax=1.0,
                alpha=0.45,
                extent=(0.0, duration, min(values), max(values) if max(values) > min(values) else min(values) + 1e-9),
            )
        axis.plot(times, values, linewidth=0.7, color="black")
        axis.set_ylabel(names[index], rotation=0, ha="right", va="center", fontsize=8)
        axis.tick_params(labelsize=7)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    column[-1].set_xlabel(x_label, fontsize=8)
    if title:
        figure.suptitle(title, fontsize=10)
    if annotations:
        figure.text(
            0.01,
            0.005,
            "\n".join(str(line) for line in annotations),
            fontsize=7,
            va="bottom",
            family="monospace",
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0.02 + 0.018 * len(annotations), 1, 0.97))
    figure.savefig(target, dpi=150)
    pyplot.close(figure)
    return target
