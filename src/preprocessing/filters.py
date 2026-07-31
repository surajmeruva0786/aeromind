"""Bandpass/notch filtering (README §7.1)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos


def bandpass_filter(
    data: np.ndarray, sfreq: float, low: float = 0.5, high: float = 45.0, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass, applied along the last axis.

    `data` may be (n_channels, n_samples) or (n_samples,).
    """
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch_filter(data: np.ndarray, sfreq: float, freq: float = 50.0, quality: float = 30.0) -> np.ndarray:
    """Zero-phase notch filter to remove mains interference (50 Hz in India)."""
    b, a = iirnotch(freq, quality, sfreq)
    sos = tf2sos(b, a)
    return sosfiltfilt(sos, data, axis=-1)


def apply_standard_filters(
    data: np.ndarray,
    sfreq: float,
    bandpass: tuple[float, float] = (0.5, 45.0),
    notch_freq: float = 50.0,
) -> np.ndarray:
    """Bandpass then notch, matching the README §7 preprocessing spec."""
    out = bandpass_filter(data, sfreq, low=bandpass[0], high=bandpass[1])
    out = notch_filter(out, sfreq, freq=notch_freq)
    return out
