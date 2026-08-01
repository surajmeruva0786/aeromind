"""Streaming inference engine (README §13): sliding-window buffering of raw
per-sample readings into epochs, epoch buffering into the Bi-LSTM's
sequence context window, model inference, and EWMA-smoothed predictions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class SlidingWindowBuffer:
    """Accumulates `(n_channels,)` samples one at a time; yields a complete
    `(n_channels, window_samples)` window every `hop_samples` new samples
    once full — the "2s window / 0.5s hop" streaming spec from README §13.
    """

    def __init__(self, n_channels: int, window_samples: int, hop_samples: int):
        if hop_samples <= 0 or window_samples <= 0:
            raise ValueError("window_samples and hop_samples must be positive")
        self.n_channels = n_channels
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        self._buffer: deque[np.ndarray] = deque(maxlen=window_samples)
        self._since_last_window = hop_samples  # allow emitting as soon as the first window fills

    def push(self, sample: np.ndarray) -> np.ndarray | None:
        """`sample`: `(n_channels,)`. Returns a full window when one is
        ready (buffer full AND `hop_samples` have elapsed since the last
        emission), else `None`."""
        if sample.shape != (self.n_channels,):
            raise ValueError(f"expected sample shape ({self.n_channels},), got {sample.shape}")
        self._buffer.append(sample)
        self._since_last_window += 1
        if len(self._buffer) == self.window_samples and self._since_last_window >= self.hop_samples:
            self._since_last_window = 0
            return np.stack(self._buffer, axis=1)  # (n_channels, window_samples)
        return None


class EWMASmoother:
    """Exponentially-weighted moving average over probability vectors
    (README §13's "exponentially-weighted prediction smoothing")."""

    def __init__(self, alpha: float = 0.3):
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def update(self, probs: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = probs.astype(np.float64).copy()
        else:
            self._state = self.alpha * probs + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self) -> None:
        self._state = None


def zscore_epoch(epoch: np.ndarray) -> np.ndarray:
    """Per-channel z-score, matching the preprocessing/training spec."""
    mean = epoch.mean(axis=1, keepdims=True)
    std = epoch.std(axis=1, keepdims=True) + 1e-8
    return (epoch - mean) / std


@dataclass
class PredictionEvent:
    timestamp: float
    workload_probs: np.ndarray
    fatigue_probs: np.ndarray
    smoothed_workload_probs: np.ndarray
    smoothed_fatigue_probs: np.ndarray


class StreamingEngine:
    """Ties a `SlidingWindowBuffer` (raw samples -> epochs), a rolling
    sequence of the last `sequence_length` epochs (the 30s Bi-LSTM
    context), a model, and two `EWMASmoother`s together. Call `push_sample`
    once per incoming raw sample; get back a `PredictionEvent` whenever a
    new epoch completes *and* enough context has accumulated, else `None`.
    """

    def __init__(
        self,
        model: nn.Module,
        sequence_length: int,
        window_samples: int,
        hop_samples: int,
        n_channels: int = 7,
        device: str = "cpu",
        ewma_alpha: float = 0.3,
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        self.window_buffer = SlidingWindowBuffer(n_channels, window_samples, hop_samples)
        self.epoch_sequence: deque[np.ndarray] = deque(maxlen=sequence_length)
        self.workload_smoother = EWMASmoother(ewma_alpha)
        self.fatigue_smoother = EWMASmoother(ewma_alpha)

    def reset(self) -> None:
        self.epoch_sequence.clear()
        self.workload_smoother.reset()
        self.fatigue_smoother.reset()

    @torch.no_grad()
    def push_sample(self, sample: np.ndarray, timestamp: float) -> PredictionEvent | None:
        window = self.window_buffer.push(sample)
        if window is None:
            return None

        self.epoch_sequence.append(zscore_epoch(window).astype(np.float32))
        if len(self.epoch_sequence) < self.sequence_length:
            return None  # not enough context for the Bi-LSTM window yet

        x = torch.from_numpy(np.stack(self.epoch_sequence, axis=0)).unsqueeze(0).to(self.device)  # (1, L, C, T)
        out = self.model(x)
        workload_probs = torch.softmax(out["workload_logits"], dim=-1)[0].cpu().numpy()
        fatigue_probs = torch.softmax(out["fatigue_logits"], dim=-1)[0].cpu().numpy()

        return PredictionEvent(
            timestamp=timestamp,
            workload_probs=workload_probs,
            fatigue_probs=fatigue_probs,
            smoothed_workload_probs=self.workload_smoother.update(workload_probs),
            smoothed_fatigue_probs=self.fatigue_smoother.update(fatigue_probs),
        )
