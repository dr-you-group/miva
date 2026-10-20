"""Reporting tier of each metric (Van Calster/STRATOS core set, Metrics Reloaded).

``core`` is what a validation report leads with; ``descriptive`` is the
threshold-dependent picture at one operating point; ``supplementary`` is every
other measure Metrics Reloaded allows, reported in full but not headlined.

Like :func:`mival.metrics.category_of`, a parameterized name such as
``net_benefit@0.05`` is looked up by its stem, so a new parameter value never
needs registering. A stem missing from the table is an error rather than a
silent default: a metric nobody placed in a tier is a metric nobody decided how
to report.
"""

from __future__ import annotations

from typing import Dict, Tuple

from . import PARAMETER_SEP

CORE = "core"
DESCRIPTIVE = "descriptive"
SUPPLEMENTARY = "supplementary"

TIER_STEMS: Dict[str, Tuple[str, ...]] = {
    CORE: (
        "auroc",
        "oe_ratio",
        "calibration_intercept_offset",
        "calibration_slope",
        "mae",
        "rmse",
        "r2",
        "reg_citl",
        "reg_calibration_slope",
        "reg_calibration_intercept",
        "bias",
        "loa_lo",
        "loa_hi",
        "uno_c",
        "auroc_ipcw",
        "oe_ratio_horizon",
        "brier_ipcw",
        "ipa",
    ),
    DESCRIPTIVE: (
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "tp",
        "fp",
        "tn",
        "fn",
        "auroc_cut",
    ),
    SUPPLEMENTARY: (
        "auprc",
        "f1",
        "mcc",
        "balanced_accuracy",
        "brier",
        "brier_scaled",
        "eci",
        "calibration_intercept",
        "citl",
        "brier_binned",
        "brier_reliability",
        "brier_resolution",
        "brier_uncertainty",
        "net_benefit",
        "net_benefit_treat_all",
        "delta_net_benefit_vs_treat_all",
        "auroc_below",
        "c_index",
        "auroc_horizon_cc",
    ),
}

_TIER_OF_STEM: Dict[str, str] = {
    stem: tier for tier, stems in TIER_STEMS.items() for stem in stems
}


def tier_of(metric: str) -> str:
    """``core``, ``descriptive`` or ``supplementary`` for a metric name."""
    stem = metric.split(PARAMETER_SEP, 1)[0]
    try:
        return _TIER_OF_STEM[stem]
    except KeyError as exc:
        raise KeyError(
            f"metric {metric!r} has no reporting tier; known stems are {sorted(_TIER_OF_STEM)}"
        ) from exc
