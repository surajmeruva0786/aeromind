"""Per-channel temporal-domain features (README §8.2)."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, skew


def hjorth_parameters(epoch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hjorth activity, mobility, complexity — each `(n_channels,)`."""
    d1 = np.diff(epoch, axis=-1)
    d2 = np.diff(d1, axis=-1)

    activity = np.var(epoch, axis=-1)
    var_d1 = np.var(d1, axis=-1)
    var_d2 = np.var(d2, axis=-1)

    mobility = np.sqrt(var_d1 / (activity + 1e-12))
    mobility_d1 = np.sqrt(var_d2 / (var_d1 + 1e-12))
    complexity = mobility_d1 / (mobility + 1e-12)

    return activity, mobility, complexity


def line_length(epoch: np.ndarray) -> np.ndarray:
    """Sum of absolute first differences — a simple measure of signal
    complexity/energy, cheap to compute and robust to DC offset."""
    return np.sum(np.abs(np.diff(epoch, axis=-1)), axis=-1)


def zero_crossing_rate(epoch: np.ndarray) -> np.ndarray:
    signs = np.sign(epoch)
    signs[signs == 0] = 1
    crossings = np.diff(signs, axis=-1) != 0
    return crossings.sum(axis=-1) / epoch.shape[-1]


def temporal_features(epoch: np.ndarray) -> dict[str, np.ndarray]:
    """All temporal features for one epoch `(n_channels, n_samples)`, each
    value an `(n_channels,)` array."""
    activity, mobility, complexity = hjorth_parameters(epoch)
    return {
        "mean": epoch.mean(axis=-1),
        "std": epoch.std(axis=-1),
        "kurtosis": kurtosis(epoch, axis=-1),
        "skewness": skew(epoch, axis=-1),
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "line_length": line_length(epoch),
        "zero_crossing_rate": zero_crossing_rate(epoch),
    }
