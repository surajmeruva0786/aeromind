"""SHAP over spectral (band-power) features (README §12.3): a class x
(channel, band) attribution matrix, using `shap.TreeExplainer` on a
RandomForest trained on flattened per-channel band-power features
(`src.features.spectral.band_powers_matrix`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

from src.data.synthetic import BANDS, CHANNEL_NAMES, SyntheticEpoch
from src.features.spectral import band_powers_matrix


@dataclass
class SpectralAttributionResult:
    shap_values: np.ndarray  # (n_samples, n_channels, n_bands, n_classes)
    channel_names: tuple[str, ...]
    band_names: tuple[str, ...]

    def mean_abs_matrix(self, class_index: int) -> np.ndarray:
        """(n_channels, n_bands) mean |SHAP| for one class — the standard
        "which (channel, band) pairs matter for this class" summary."""
        return np.abs(self.shap_values[..., class_index]).mean(axis=0)


def build_band_power_dataset(
    epochs: list[SyntheticEpoch], sfreq: float = 256.0
) -> tuple[np.ndarray, np.ndarray]:
    """Flattens each epoch's `(n_channels, n_bands)` absolute band-power
    matrix into a feature row. Returns `(X, y)` with `y` the workload label."""
    matrices = np.stack([band_powers_matrix(ep.data, sfreq) for ep in epochs], axis=0)  # (N, C, B)
    n, c, b = matrices.shape
    return matrices.reshape(n, c * b), np.array([ep.workload for ep in epochs])


def fit_spectral_classifier(X: np.ndarray, y: np.ndarray, seed: int = 42) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1)
    clf.fit(X, y)
    return clf


def compute_spectral_attribution(
    clf: RandomForestClassifier,
    X: np.ndarray,
    n_channels: int = len(CHANNEL_NAMES),
    band_names: tuple[str, ...] = tuple(BANDS.keys()),
    channel_names: tuple[str, ...] = CHANNEL_NAMES,
) -> SpectralAttributionResult:
    """`X`: `(n_samples, n_channels * n_bands)`, in the same row-major
    (channel-major, then band) flatten order as `build_band_power_dataset`.
    """
    explainer = shap.TreeExplainer(clf)
    raw = explainer.shap_values(X)  # (n_samples, n_features, n_classes)
    n_samples, n_features, n_classes = raw.shape
    n_bands = n_features // n_channels
    reshaped = raw.reshape(n_samples, n_channels, n_bands, n_classes)
    return SpectralAttributionResult(
        shap_values=reshaped, channel_names=channel_names, band_names=band_names
    )
