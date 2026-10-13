"""Deterministic synthetic ECG.

No RNG: the waveform is a closed-form function of record_index and lead index,
so golden hashes stay stable across numpy versions and machines.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from mival.signal import LEADS_12, Signal, SourceMetadata

GOLDEN_SOURCE = SourceMetadata(
    leads=LEADS_12, sampling_rate_hz=500.0, n_samples=5000, unit="mV"
)


def synthetic_ecg(
    record_index: int,
    leads: Tuple[str, ...] = LEADS_12,
    sampling_rate_hz: float = 500.0,
    n_samples: int = 5000,
) -> Signal:
    t = np.arange(n_samples, dtype=np.float64) / sampling_rate_hz
    heart_rate_hz = 1.0 + 0.05 * (record_index % 10)
    rows = []
    for lead_index in range(len(leads)):
        phase = 0.13 * lead_index + 0.07 * record_index
        amplitude = 0.5 + 0.1 * ((lead_index + record_index) % 5)
        wave = (
            amplitude * np.sin(2 * np.pi * heart_rate_hz * t + phase)
            + 0.30 * amplitude * np.sin(2 * np.pi * 3 * heart_rate_hz * t + 2 * phase)
            + 0.10 * amplitude * np.sin(2 * np.pi * 7 * heart_rate_hz * t + 3 * phase)
        )
        rows.append(wave)
    return Signal(
        data=np.ascontiguousarray(np.stack(rows), dtype=np.float32),
        leads=tuple(leads),
        sampling_rate_hz=sampling_rate_hz,
        unit="mV",
    )
