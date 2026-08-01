"""Phase Locking Value connectivity features (README §8.3)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from src.data.synthetic import BANDS

CONNECTIVITY_BANDS = {"theta": BANDS["theta"], "alpha": BANDS["alpha"]}


def _instantaneous_phase(epoch: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    sos = butter(4, band, btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, epoch, axis=-1)
    analytic = hilbert(filtered, axis=-1)
    return np.angle(analytic)


def plv_matrix(epoch: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Pairwise Phase Locking Value for one band. Returns `(n_channels, n_channels)`,
    symmetric, diagonal == 1.
    """
    phase = _instantaneous_phase(epoch, sfreq, band)  # (n_channels, n_samples)
    n_channels = phase.shape[0]
    plv = np.ones((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            diff = phase[i] - phase[j]
            value = np.abs(np.mean(np.exp(1j * diff)))
            plv[i, j] = plv[j, i] = value
    return plv


def connectivity_features(
    epoch: np.ndarray, sfreq: float, bands: dict[str, tuple[float, float]] = CONNECTIVITY_BANDS
) -> dict[str, np.ndarray]:
    """PLV matrices for each requested band, keyed `plv_{band}`."""
    return {
        f"plv_{band}": plv_matrix(epoch, sfreq, freq_range) for band, freq_range in bands.items()
    }


def frontal_parietal_connectivity(
    epoch: np.ndarray,
    sfreq: float,
    frontal_idx: list[int],
    parietal_idx: list[int],
    band: tuple[float, float] = BANDS["theta"],
) -> float:
    """Mean PLV between frontal and parietal/posterior channel groups — the
    single scalar summary README §8.3 highlights as tracking workload."""
    plv = plv_matrix(epoch, sfreq, band)
    values = [plv[i, j] for i in frontal_idx for j in parietal_idx]
    return float(np.mean(values)) if values else 0.0
