"""PyTorch Dataset wrapping epoched EEG + workload/fatigue labels.

Two entry points are supported:
  - `EpochWindowDataset(epochs)`      — flat epoch classification (single epoch -> label)
  - `SequenceEpochDataset(epochs, L)` — L consecutive epochs per subject -> one label,
    matching the 30 s Bi-LSTM context window used by AeroMind-CapsNet.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.synthetic import SyntheticEpoch


def _zscore(epoch: np.ndarray) -> np.ndarray:
    """Per-channel z-score, matching the preprocessing spec (README §7.5)."""
    mean = epoch.mean(axis=1, keepdims=True)
    std = epoch.std(axis=1, keepdims=True) + 1e-8
    return (epoch - mean) / std


class EpochWindowDataset(Dataset):
    """One epoch -> (workload_label, fatigue_label). Used by CNN-LSTM/EEGNet
    baselines and for unit-testing the capsule layers in isolation."""

    def __init__(self, epochs: list[SyntheticEpoch], normalize: bool = True):
        self.epochs = epochs
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.epochs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ep = self.epochs[idx]
        data = _zscore(ep.data) if self.normalize else ep.data
        x = torch.from_numpy(data.astype(np.float32))
        return (
            x,
            torch.tensor(ep.workload, dtype=torch.long),
            torch.tensor(ep.fatigue, dtype=torch.long),
        )


class SequenceEpochDataset(Dataset):
    """`sequence_length` consecutive same-subject epochs -> one label pair,
    matching the 30 s Bi-LSTM context window (README §9.1). Labels are taken
    from the last epoch in the window (the "current" state being predicted)."""

    def __init__(
        self,
        epochs: list[SyntheticEpoch],
        sequence_length: int = 15,
        normalize: bool = True,
    ):
        self.sequence_length = sequence_length
        self.normalize = normalize

        by_subject: dict[int, list[SyntheticEpoch]] = defaultdict(list)
        for ep in epochs:
            by_subject[ep.subject_id].append(ep)

        self.sequences: list[list[SyntheticEpoch]] = []
        for subj_epochs in by_subject.values():
            for start in range(0, len(subj_epochs) - sequence_length + 1):
                self.sequences.append(subj_epochs[start : start + sequence_length])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        frames = []
        for ep in seq:
            data = _zscore(ep.data) if self.normalize else ep.data
            frames.append(data.astype(np.float32))
        x = torch.from_numpy(np.stack(frames, axis=0))  # (L, C, T)
        last = seq[-1]
        return (
            x,
            torch.tensor(last.workload, dtype=torch.long),
            torch.tensor(last.fatigue, dtype=torch.long),
        )
