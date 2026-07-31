"""Spectral (band-power) features via Welch's method (README §8.1)."""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from src.data.synthetic import BANDS


def band_powers(
    epoch: np.ndarray,
    sfreq: float,
    bands: dict[str, tuple[float, float]] = BANDS,
) -> dict[str, np.ndarray]:
    """Per-channel absolute, relative, and log power for each canonical band.

    `epoch` is `(n_channels, n_samples)`. Returns a dict with keys
    `{band}_abs`, `{band}_rel`, `{band}_log`, each an `(n_channels,)` array.
    """
    nperseg = min(256, epoch.shape[-1])
    freqs, psd = welch(epoch, fs=sfreq, window="hamming", nperseg=nperseg, noverlap=nperseg // 2, axis=-1)

    total_power = psd.sum(axis=-1) + 1e-12
    out: dict[str, np.ndarray] = {}
    for band, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        abs_power = psd[..., mask].sum(axis=-1)
        out[f"{band}_abs"] = abs_power
        out[f"{band}_rel"] = abs_power / total_power
        out[f"{band}_log"] = np.log10(abs_power + 1e-12)
    return out


def band_powers_matrix(epoch: np.ndarray, sfreq: float, bands: dict[str, tuple[float, float]] = BANDS) -> np.ndarray:
    """Same as `band_powers` but stacked as `(n_channels, n_bands)` absolute
    power — the layout `src/xai/spectral_attribution.py` explains SHAP over.
    """
    powers = band_powers(epoch, sfreq, bands)
    return np.stack([powers[f"{band}_abs"] for band in bands], axis=-1)
