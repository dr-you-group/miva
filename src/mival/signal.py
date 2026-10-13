"""Canonical in-memory ECG representation.

Internal layout is always (n_leads, n_samples), float32, millivolts.
Model-specific layouts are produced by adapters, never here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

LEADS_12: Tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

# Einthoven and Goldberger relations. Each entry maps a derivable lead to the
# coefficients applied to leads I and II.
DERIVABLE_LEADS: Dict[str, Dict[str, float]] = {
    "III": {"I": -1.0, "II": 1.0},
    "aVR": {"I": -0.5, "II": -0.5},
    "aVL": {"I": 1.0, "II": -0.5},
    "aVF": {"I": -0.5, "II": 1.0},
}


@dataclass(frozen=True)
class Signal:
    data: np.ndarray
    leads: Tuple[str, ...]
    sampling_rate_hz: float
    unit: str

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"signal data must be 2-D, got {self.data.ndim}-D")
        if self.data.dtype != np.float32:
            raise ValueError(f"signal data must be float32, got {self.data.dtype}")
        if self.data.shape[0] != len(self.leads):
            raise ValueError(
                f"lead count mismatch: data has {self.data.shape[0]} rows "
                f"but {len(self.leads)} lead names were given"
            )

    @property
    def n_leads(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sampling_rate_hz


@dataclass(frozen=True)
class SourceMetadata:
    """What the Profile stage observed about a record, before preprocessing."""

    leads: Tuple[str, ...]
    sampling_rate_hz: float
    n_samples: int
    unit: Optional[str]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sampling_rate_hz
