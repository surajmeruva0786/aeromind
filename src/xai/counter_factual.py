"""Counter-factual probe (README §12.4): attenuate frontal-midline theta
power on a high-workload epoch and check whether the model's predicted
workload class shifts toward lower workload. A model that fails this probe
is flagged as potentially overfitting to spurious, non-theta features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt

from src.data.synthetic import BANDS, CHANNEL_NAMES, FRONTAL_CHANNELS


@dataclass
class CounterFactualResult:
    original_class: int
    attenuated_class: int
    original_probs: np.ndarray
    attenuated_probs: np.ndarray
    passed: bool  # True if the probe behaved as theoretically expected


def attenuate_band(
    epoch: np.ndarray,
    sfreq: float,
    channels: tuple[str, ...],
    channel_names: tuple[str, ...] = CHANNEL_NAMES,
    band: tuple[float, float] = BANDS["theta"],
    factor: float = 0.5,
) -> np.ndarray:
    """Returns a copy of `epoch` (n_channels, n_samples) with the given
    frequency band's contribution scaled by `factor` (0.5 = attenuated by
    half) on the specified channels only; other channels/bands untouched.
    """
    lo, hi = band
    sos = butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    band_component = sosfiltfilt(sos, epoch, axis=-1)

    attenuated = epoch.copy()
    idx = [channel_names.index(ch) for ch in channels]
    attenuated[idx] = epoch[idx] - (1.0 - factor) * band_component[idx]
    return attenuated


def run_counter_factual_probe(
    model: nn.Module,
    sequence: torch.Tensor,
    sfreq: float,
    channel_names: tuple[str, ...] = CHANNEL_NAMES,
    band: tuple[float, float] = BANDS["theta"],
    channels: tuple[str, ...] = FRONTAL_CHANNELS,
    factor: float = 0.5,
) -> CounterFactualResult:
    """`sequence`: `(1, L, C, T)` — a single sequence window, ideally one
    the model currently predicts as high workload. Only the *last* epoch in
    the window (the one the heads predict from) is attenuated.

    "Passed" means either the predicted class label moved down (toward
    lower workload) or, if the label didn't flip, the model's confidence in
    its original class dropped — either is consistent with the model
    actually using frontal theta as workload evidence.
    """
    model.eval()
    with torch.no_grad():
        original_logits = model(sequence)["workload_logits"]
        original_probs = torch.softmax(original_logits, dim=-1)[0].cpu().numpy()
        original_class = int(original_probs.argmax())

        last_epoch = sequence[0, -1].cpu().numpy()
        attenuated_last = attenuate_band(last_epoch, sfreq, channels, channel_names, band, factor)

        attenuated_seq = sequence.clone()
        attenuated_seq[0, -1] = torch.from_numpy(attenuated_last.astype(np.float32))

        attenuated_logits = model(attenuated_seq)["workload_logits"]
        attenuated_probs = torch.softmax(attenuated_logits, dim=-1)[0].cpu().numpy()
        attenuated_class = int(attenuated_probs.argmax())

    passed = bool(
        (attenuated_class < original_class)
        or (attenuated_probs[original_class] < original_probs[original_class])
    )

    return CounterFactualResult(
        original_class=original_class,
        attenuated_class=attenuated_class,
        original_probs=original_probs,
        attenuated_probs=attenuated_probs,
        passed=passed,
    )
