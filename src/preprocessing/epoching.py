"""Fixed-length epoching with per-epoch z-score and amplitude rejection (README §7.5)."""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np


@dataclass
class EpochingResult:
    data: np.ndarray  # (n_kept_epochs, n_channels, n_samples), z-scored
    n_total: int
    n_rejected: int
    rejected_indices: list[int]


def epoch_continuous(
    raw: mne.io.BaseRaw,
    epoch_seconds: float = 2.0,
    overlap: float = 0.5,
    reject_ptp_uv: float = 200.0,
) -> EpochingResult:
    """Slice a continuous Raw into fixed windows, reject epochs whose
    peak-to-peak amplitude exceeds `reject_ptp_uv` on any channel, then
    per-channel z-score the survivors.
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data()  # volts, (n_channels, n_samples)
    epoch_len = int(epoch_seconds * sfreq)
    hop = max(int(epoch_len * (1 - overlap)), 1)

    reject_v = reject_ptp_uv * 1e-6  # MNE data is in volts

    kept, rejected_indices = [], []
    n_total = 0
    start = 0
    while start + epoch_len <= data.shape[1]:
        seg = data[:, start : start + epoch_len]
        ptp = seg.max(axis=1) - seg.min(axis=1)
        if np.any(ptp > reject_v):
            rejected_indices.append(n_total)
        else:
            kept.append(_zscore(seg))
        n_total += 1
        start += hop

    stacked = np.stack(kept, axis=0) if kept else np.empty((0, data.shape[0], epoch_len))
    return EpochingResult(
        data=stacked,
        n_total=n_total,
        n_rejected=len(rejected_indices),
        rejected_indices=rejected_indices,
    )


def _zscore(epoch: np.ndarray) -> np.ndarray:
    mean = epoch.mean(axis=1, keepdims=True)
    std = epoch.std(axis=1, keepdims=True) + 1e-12
    return (epoch - mean) / std
