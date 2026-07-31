"""Concatenates spectral + temporal + connectivity views into one flat
feature vector per epoch, for the classical (non-deep) baseline (README §8)."""

from __future__ import annotations

import numpy as np

from src.data.synthetic import CHANNEL_NAMES
from src.features.connectivity import CONNECTIVITY_BANDS, plv_matrix
from src.features.spectral import band_powers
from src.features.temporal import temporal_features


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def extract_features(
    epoch: np.ndarray,
    sfreq: float,
    channel_names: tuple[str, ...] = CHANNEL_NAMES,
) -> np.ndarray:
    """Flatten spectral (per-channel, per-band) + temporal (per-channel) +
    connectivity (upper-triangle PLV per band) features into one vector.
    Deterministic ordering, safe to call once per epoch for a full dataset.
    """
    parts: list[np.ndarray] = []

    spectral = band_powers(epoch, sfreq)
    for key in sorted(spectral):  # sorted -> stable column order
        parts.append(spectral[key])

    temporal = temporal_features(epoch)
    for key in sorted(temporal):
        parts.append(temporal[key])

    for band_name, band_range in sorted(CONNECTIVITY_BANDS.items()):
        plv = plv_matrix(epoch, sfreq, band_range)
        parts.append(_upper_triangle(plv))

    return np.concatenate(parts).astype(np.float64)


def feature_names(n_channels: int, channel_names: tuple[str, ...] = CHANNEL_NAMES) -> list[str]:
    """Column names matching `extract_features`'s deterministic ordering —
    used to label SHAP/feature-importance plots for the classical baseline.
    """
    from src.data.synthetic import BANDS

    names: list[str] = []
    spectral_keys = sorted(f"{band}_{suffix}" for band in BANDS for suffix in ("abs", "log", "rel"))
    for key in spectral_keys:
        for ch in channel_names:
            names.append(f"{key}_{ch}")

    temporal_keys = sorted(
        [
            "mean",
            "std",
            "kurtosis",
            "skewness",
            "hjorth_activity",
            "hjorth_mobility",
            "hjorth_complexity",
            "line_length",
            "zero_crossing_rate",
        ]
    )
    for key in temporal_keys:
        for ch in channel_names:
            names.append(f"{key}_{ch}")

    for band_name in sorted(("theta", "alpha")):
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                names.append(f"plv_{band_name}_{channel_names[i]}-{channel_names[j]}")

    return names
