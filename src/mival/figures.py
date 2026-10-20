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


#: The figure most recently written by :func:`_save`, kept so a test can inspect
#: what was drawn (the file alone cannot tell an empty figure from a full one).
_last_figure: list = [None]


def _save(figure, path: Union[str, Path]) -> Path:
    _last_figure[0] = figure
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=200)
    _pyplot().close(figure)
    return target


def _count(value: Any) -> str:
    """A count as text. A masked count ("<10") is shown as given."""
    if isinstance(value, str):
        return value
    return f"{value:,}"


def stard_flow(
    stages: Sequence[Tuple[str, Any]],
    exclusions: Sequence[Tuple[str, str, Any]],
    path: Union[str, Path],
    title: str = "STARD participant flow",
    side: Sequence[Tuple[str, Any]] = (),
) -> Path:
    """Participant flow: ``stages`` are ``(label, n_remaining)`` boxes and
    ``exclusions`` are ``(stage, reason_code, n)`` annotations beside them.

    Both come from the ledger, so the diagram and the counts cannot disagree.

    ``side`` boxes branch off the box before the last one and sit beside the
    last box, with no arrow into it: a set that was set aside (for example the
    development split) rather than filtered down into the analysed set.
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
            f"{label}\nn = {_count(remaining)}",
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
            body = "\n".join(
                f"{reason}: {_count(count)}" for reason, count in sorted(dropped, key=lambda r: r[0])
            )
            axes.text(
                6.2,
                y - 0.42,
                f"excluded\n{body}",
                ha="left",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.4", "facecolor": "0.95", "edgecolor": "0.6"},
            )
    if side and stages:
        last = 0.5
        source = last + 1.0 if len(stages) > 1 else last
        for index, (label, count) in enumerate(side):
            y = last - 0.45 * index
            axes.text(
                7.8,
                y,
                f"{label}\nn = {_count(count)}",
                ha="center",
                va="center",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "black",
                      "linestyle": "--"},
            )
            axes.annotate(
                "",
                xy=(6.9, y),
                xytext=(3.4, source - 0.3),
                arrowprops={"arrowstyle": "->", "color": "0.4", "linestyle": "--"},
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

    ``band`` shades the clinically plausible threshold range when the study
    states one (``evaluate.decision_band``); nothing is shaded otherwise.
    Performance outside that range is not evidence of clinical usefulness.
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
    axes.set_title("Subgroup estimates (* suppressed: too few events or non-events)")
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


# ---------------------------------------------------------------------------
# Figures drawn from the exported tables (curves_long / metrics_long).
#
# These read the tables, not predictions, so a cell masked on export (NaN, or
# a "<10" count) stays masked in the figure. NaN points are skipped, and a
# count column is never assumed numeric.
# ---------------------------------------------------------------------------

_ARM_AXES = ("site", "model_id", "training_mode", "recipe_id", "perturbation_id", "label_def")


def _numeric(frame, columns):
    import pandas as pd

    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _arm_labels(frame) -> dict:
    """Arm key -> label built from the axes that differ between arms."""
    axes = [a for a in _ARM_AXES if a in frame.columns]
    varying = [a for a in axes if frame[a].astype(str).nunique() > 1]
    if not varying:
        varying = [a for a in ("model_id",) if a in frame.columns]
    labels: dict = {}
    for _, row in frame[axes].drop_duplicates().iterrows():
        key = tuple(str(row[a]) for a in axes)
        labels[key] = " / ".join(str(row[a]) for a in varying) or "all"
    return labels


class _Ctx:
    """The filter a table figure was drawn under, for its title and "no data" note."""

    def __init__(self, outcome, subgroup, update):
        from mival.stages.evaluate import SUBGROUP_ALL, UPDATE_NONE

        self.outcome, self.subgroup, self.update = outcome, subgroup, update
        parts = []
        if outcome is not None:
            parts.append(f"outcome={outcome}")
        if subgroup not in (None, SUBGROUP_ALL):
            parts.append(f"subgroup={subgroup}")
        if update not in (None, UPDATE_NONE):
            parts.append(f"update={update}")
        self.tag = ", ".join(parts)

    def title(self, base):
        return f"{base} ({self.tag})" if self.tag else base


def _slice(frame, curve, subgroup=None, update=None, outcome=None):
    """Rows of one curve, one update, one subgroup and one outcome, plus the filter used.

    ``subgroup``/``update`` default to the engine's whole-cohort and
    not-recalibrated values. With several outcomes present, one must be chosen.
    """
    from mival.stages.evaluate import SUBGROUP_ALL, UPDATE_NONE

    subgroup = SUBGROUP_ALL if subgroup is None else subgroup
    update = UPDATE_NONE if update is None else update
    rows = frame[frame["curve"] == curve]
    if "subgroup" in rows.columns:
        rows = rows[rows["subgroup"] == subgroup]
    if "update" in rows.columns:
        rows = rows[rows["update"] == update]
    if "outcome" in rows.columns and len(rows):
        present = sorted(set(rows["outcome"].astype(str)))
        if outcome is None:
            if len(present) > 1:
                raise ValueError(
                    f"{curve}: several outcomes present ({', '.join(present)}); pass outcome="
                )
            outcome = present[0]
        rows = rows[rows["outcome"].astype(str) == str(outcome)]
    # With no rows the outcome was never resolved: say so rather than omit it.
    outcome = "any" if outcome is None else outcome
    ctx = _Ctx(outcome, subgroup, update)
    ctx.curve = curve
    ctx.filters = f"subgroup={subgroup}, update={update}, outcome={outcome}"
    return _numeric(rows, ("x", "value", "ci_lo", "ci_hi", "n", "n_events")), ctx


def _warn_empty(rows, ctx, drawn, *axes):
    """Say on the figure and to the caller that nothing could be drawn."""
    import warnings

    if drawn:
        return
    if len(rows):
        text = f"no plottable points for {ctx.filters} ({ctx.curve}): all masked"
    else:
        text = f"no rows for {ctx.filters} ({ctx.curve})"
    for axis in axes:
        _empty_note(axis, text)
    warnings.warn(text, stacklevel=3)


def _arm_series(rows, metric):
    """Yield (label, points) per arm for one metric, NaN points dropped."""
    labels = _arm_labels(rows)
    axes = [a for a in _ARM_AXES if a in rows.columns]
    sub = rows[rows["metric"] == metric]
    for key, part in sub.groupby([sub[a].astype(str) for a in axes], sort=True):
        key = key if isinstance(key, tuple) else (key,)
        part = part.dropna(subset=["x", "value"]).sort_values("x")
        yield labels.get(tuple(key), "/".join(key)), part


def _wide(rows, metrics):
    """Per arm: x-aligned frame with one column per metric (NaN kept)."""
    out = {}
    labels = _arm_labels(rows)
    axes = [a for a in _ARM_AXES if a in rows.columns]
    for key, part in rows[rows["metric"].isin(metrics)].groupby(
        [rows[a].astype(str) for a in axes], sort=True
    ):
        key = key if isinstance(key, tuple) else (key,)
        wide = part.pivot_table(index="x", columns="metric", values="value", aggfunc="first")
        out[labels.get(tuple(key), "/".join(key))] = wide.sort_index()
    return out


def _empty_note(axes, text="no data"):
    axes.text(0.5, 0.5, text, ha="center", va="center", transform=axes.transAxes, color="0.5")


def grid_roc_pr(frame, path, update=None, outcome=None, subgroup=None) -> Path:
    """ROC (1 - specificity, sensitivity) and PR (sensitivity, PPV) from the threshold grid."""
    pyplot = _pyplot()
    rows, ctx = _slice(frame, "threshold_grid", subgroup, update, outcome)
    figure, (roc, pr) = pyplot.subplots(1, 2, figsize=(9.5, 4.2))
    drawn = False
    for label, wide in _wide(rows, ("sensitivity", "specificity", "ppv")).items():
        if {"sensitivity", "specificity"} <= set(wide.columns):
            pts = wide[["sensitivity", "specificity"]].dropna()
            if len(pts):
                roc.plot(1.0 - pts["specificity"].to_numpy(), pts["sensitivity"].to_numpy(),
                         label=label)
                drawn = True
        if {"sensitivity", "ppv"} <= set(wide.columns):
            pts = wide[["sensitivity", "ppv"]].dropna()
            if len(pts):
                pr.plot(pts["sensitivity"].to_numpy(), pts["ppv"].to_numpy(), label=label)
    if drawn:
        roc.plot([0, 1], [0, 1], linestyle=":", color="0.6")
    roc.set_xlabel("1 - specificity")
    roc.set_ylabel("sensitivity")
    roc.set_title(ctx.title("ROC (threshold grid)"), fontsize=9)
    pr.set_xlabel("sensitivity (recall)")
    pr.set_ylabel("PPV (precision)")
    pr.set_title(ctx.title("Precision-recall (threshold grid)"), fontsize=9)
    for axis in (roc, pr):
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
    if drawn:
        roc.legend(fontsize=7)
    _warn_empty(rows, ctx, drawn, roc, pr)
    return _save(figure, path)


def _band_line(axes, part, label, draw_band=True, **style):
    """One arm's line with its interval band where the interval is present."""
    (line,) = axes.plot(part["x"], part["value"], label=label, **style)
    if draw_band:
        ci = part.dropna(subset=["ci_lo", "ci_hi"])
        if len(ci):
            axes.fill_between(ci["x"], ci["ci_lo"], ci["ci_hi"], alpha=0.2,
                              color=line.get_color(), linewidth=0)
    return line


def _net_benefit_axes(axes, rows, draw_band):
    """Net benefit per arm, treat-all and treat-none; True if an arm was drawn."""
    drawn = False
    for label, part in _arm_series(rows, "net_benefit"):
        if len(part):
            _band_line(axes, part, label, draw_band)
            drawn = True
    if not drawn:
        return False
    for _, part in list(_arm_series(rows, "treat_all"))[:1]:
        if len(part):
            axes.plot(part["x"], part["value"], linestyle="--", color="0.4", label="treat all")
    axes.axhline(0.0, color="black", linewidth=0.8, label="treat none")
    axes.legend(fontsize=7)
    return True


def grid_decision_curve(frame, path, band=None, update=None, outcome=None,
                        subgroup=None) -> Path:
    """Net benefit with its interval, treat-all and treat-none.

    ``band`` None draws the interval wherever one is present; False omits it.
    """
    pyplot = _pyplot()
    rows, ctx = _slice(frame, "threshold_grid", subgroup, update, outcome)
    figure, axes = pyplot.subplots(figsize=(6.0, 4.4))
    drawn = _net_benefit_axes(axes, rows, True if band is None else bool(band))
    axes.set_xlabel("threshold probability")
    axes.set_ylabel("net benefit")
    axes.set_title(ctx.title("Decision curve"), fontsize=9)
    _warn_empty(rows, ctx, drawn, axes)
    return _save(figure, path)


def calibration_band(frame, path, update=None, outcome=None, subgroup=None) -> Path:
    """Observed risk against predicted risk with the interval band, over the risk histogram."""
    pyplot = _pyplot()
    rows, ctx = _slice(frame, "calibration", subgroup, update, outcome)
    dist, _ = _slice(frame, "risk_distribution", subgroup, update, outcome)
    figure, (top, bottom) = pyplot.subplots(
        2, 1, figsize=(5.6, 6.2), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    drawn = False
    for label, part in _arm_series(rows, "observed"):
        if len(part):
            _band_line(top, part, label)
            drawn = True
    if drawn:
        top.plot([0, 1], [0, 1], linestyle=":", color="0.5")
        top.legend(fontsize=7)
    top.set_ylabel("observed risk")
    top.set_xlim(0, 1)
    top.set_ylim(0, 1)
    top.set_title(ctx.title("Calibration"), fontsize=9)
    wide = _wide(dist, ("events", "nonevents"))
    width = 0.05 / max(len(wide), 1)
    for k, (label, table) in enumerate(wide.items()):
        total = table.sum(axis=1, min_count=1).dropna()
        if len(total):
            bottom.bar(total.index.to_numpy() + width * (k + 0.5), total.to_numpy(),
                       width=width, align="center", label=label)
    bottom.set_xlabel("predicted risk")
    bottom.set_ylabel("records")
    _warn_empty(rows, ctx, drawn, top)
    return _save(figure, path)


def regression_calibration_plot(frame, path, update=None, outcome=None,
                                subgroup=None) -> Path:
    """Observed mean truth against mean prediction per equal-count bin, with the identity line."""
    pyplot = _pyplot()
    rows, ctx = _slice(frame, "regression_calibration", subgroup, update, outcome)
    figure, axes = pyplot.subplots(figsize=(5.2, 5.0))
    lows, highs = [], []
    for label, part in _arm_series(rows, "observed"):
        if not len(part):
            continue
        _band_line(axes, part, label, marker="o", markersize=3)
        lows += [part["x"].min(), part["value"].min()]
        highs += [part["x"].max(), part["value"].max()]
    if lows:
        lo, hi = min(lows), max(highs)
        axes.plot([lo, hi], [lo, hi], linestyle=":", color="0.5")
        axes.legend(fontsize=7)
    axes.set_xlabel("mean predicted")
    axes.set_ylabel("mean observed")
    axes.set_title(ctx.title("Regression calibration"), fontsize=9)
    _warn_empty(rows, ctx, bool(lows), axes)
    return _save(figure, path)


def km_plot(frame, path, update=None, outcome=None, subgroup=None) -> Path:
    """Kaplan-Meier curves per risk quarter (colour), one line style per arm."""
    pyplot = _pyplot()
    rows, ctx = _slice(frame, "km_by_risk_group", subgroup, update, outcome)
    figure, axes = pyplot.subplots(figsize=(6.2, 4.4))
    groups = sorted({m for m in rows["metric"].astype(str) if m.startswith("survival@")})
    styles = ["-", "--", "-.", ":"]
    cmap = pyplot.get_cmap("viridis")
    drawn = False
    for g, metric in enumerate(groups):
        color = cmap(g / max(len(groups) - 1, 1) * 0.9)
        for a, (label, part) in enumerate(_arm_series(rows, metric)):
            if not len(part):
                continue
            axes.step(part["x"], part["value"], where="post", color=color,
                      linestyle=styles[a % len(styles)],
                      label=f"{metric.split('@')[1]} {label}")
            drawn = True
    if drawn:
        axes.legend(fontsize=6, ncol=2)
    axes.set_xlabel("days since ECG")
    axes.set_ylabel("event-free survival")
    axes.set_ylim(0, 1.02)
    axes.set_title(ctx.title("Kaplan-Meier by predicted-risk quarter"), fontsize=9)
    _warn_empty(rows, ctx, drawn, axes)
    return _save(figure, path)


def survival_decision_curve(frame, path, update=None, outcome=None, subgroup=None) -> Path:
    """Censoring-weighted net benefit at the horizon against treat-all and treat-none."""
    pyplot = _pyplot()
    rows, ctx = _slice(frame, "survival_threshold_grid", subgroup, update, outcome)
    figure, axes = pyplot.subplots(figsize=(6.0, 4.4))
    drawn = _net_benefit_axes(axes, rows, True)
    axes.set_xlabel("threshold probability")
    axes.set_ylabel("net benefit")
    axes.set_title(ctx.title("Survival decision curve"), fontsize=9)
    _warn_empty(rows, ctx, drawn, axes)
    return _save(figure, path)


def forest_from_metrics(metrics, metric, path, update=None, outcome=None) -> Path:
    """One metric across subgroup rows; a row with a ``reason`` is grey.

    ``update`` and ``outcome`` select among several when the table has them;
    with several outcomes present one must be chosen.
    """
    pyplot = _pyplot()
    from mival.stages.evaluate import UPDATE_NONE

    update = UPDATE_NONE if update is None else update
    rows = metrics[metrics["metric"] == metric]
    if "update" in rows.columns:
        rows = rows[rows["update"].astype(str) == str(update)]
    if "outcome" in rows.columns and len(rows):
        present = sorted(set(rows["outcome"].astype(str)))
        if outcome is None:
            if len(present) > 1:
                raise ValueError(
                    f"{metric}: several outcomes present ({', '.join(present)}); pass outcome="
                )
            outcome = present[0]
        rows = rows[rows["outcome"].astype(str) == str(outcome)]
    rows = _numeric(rows, ("value", "ci_lo", "ci_hi"))
    labels = _arm_labels(rows) if len(rows) else {}
    axes_cols = [a for a in _ARM_AXES if a in rows.columns]
    many = len(labels) > 1
    figure, axes = pyplot.subplots(figsize=(6.5, 0.4 * max(len(rows), 1) + 1.5))
    records = list(rows.to_dict("records"))
    plotted = 0
    for index, row in enumerate(records):
        y = len(records) - index - 1
        name = str(row.get("subgroup", ""))
        if many:
            name += " [" + labels.get(tuple(str(row[a]) for a in axes_cols), "") + "]"
        reason = row.get("reason")
        masked = reason is not None and not (isinstance(reason, float) and reason != reason) \
            and str(reason) != ""
        color = "0.65" if masked else "black"
        value, lo, hi = row.get("value"), row.get("ci_lo"), row.get("ci_hi")
        if value is not None and value == value:
            axes.plot([value], [y], marker="o", color=color)
            plotted += 1
            if lo == lo and hi == hi and lo is not None and hi is not None:
                axes.hlines(y, lo, hi, color=color if masked else "0.4", linewidth=1.2)
        axes.text(0.0, y, name + (" *" if masked else ""), transform=axes.get_yaxis_transform(),
                  ha="right", va="center", fontsize=8, color="0.45" if masked else "black")
    axes.set_yticks([])
    axes.set_ylim(-0.8, len(records) - 0.2)
    axes.set_xlabel(metric)
    outcome = "any" if outcome is None else outcome
    tag = f", outcome={outcome}" + ("" if update == UPDATE_NONE else f", update={update}")
    axes.set_title(f"{metric}: subgroup estimates{tag} (grey *: withheld, see reason)", fontsize=9)
    if not plotted:
        import warnings

        text = f"no rows for metric={metric}, update={update}, outcome={outcome}"
        if records:
            text = f"no plottable points for metric={metric}: all masked"
        _empty_note(axes, text)
        warnings.warn(text, stacklevel=2)
    return _save(figure, path)


def stard_from_counts(stages, exclusions, path, title="STARD participant flow", side=()) -> Path:
    """STARD flow from a counts table; a masked count ("<10") is printed as text."""
    return stard_flow(stages, exclusions, path, title=title, side=side)
