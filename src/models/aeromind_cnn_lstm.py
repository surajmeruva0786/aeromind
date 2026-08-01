"""AeroMind-CNN-LSTM (README §9.2): baseline 1 — stacked 1D convolutions per
epoch feeding the same Bi-LSTM + dual-head structure as AeroMind-CapsNet,
but with a plain conv/pool encoder instead of capsules. Used to measure how
much the capsule routing structure actually buys over a conventional CNN
front-end on the same downstream architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEpochEncoder(nn.Module):
    """Per-epoch encoder: raw (C, T) epoch -> fixed-size feature vector."""

    def __init__(self, n_channels: int = 7, conv_channels: int = 64, feature_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, conv_channels, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.pool1 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(conv_channels, feature_dim, kernel_size=9, padding=4)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: (B, C, T) -> (B, feature_dim)."""
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool1(h)
        h = F.relu(self.bn3(self.conv3(h)))
        h = self.pool2(h).squeeze(-1)
        return h


class AeroMindCNNLSTM(nn.Module):
    """Multi-task baseline: shared per-epoch CNN encoder, Bi-LSTM over a
    15-epoch (30s) window, two classification heads. No reconstruction head
    (no capsule presence-vector to decode from) — `losses.multi_task_loss`
    handles models that omit the `reconstruction` output key.
    """

    def __init__(
        self,
        n_channels: int = 7,
        n_samples: int = 512,
        conv_channels: int = 64,
        feature_dim: int = 128,
        lstm_hidden: int = 64,
        n_workload_classes: int = 3,
        n_fatigue_classes: int = 2,
    ):
        super().__init__()
        del n_samples  # unused: conv encoder is adaptive-pooled, doesn't need fixed length
        self.encoder = CNNEpochEncoder(n_channels, conv_channels, feature_dim)
        self.lstm = nn.LSTM(feature_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.workload_head = nn.Linear(lstm_hidden * 2, n_workload_classes)
        self.fatigue_head = nn.Linear(lstm_hidden * 2, n_fatigue_classes)

    def forward(
        self, x: torch.Tensor, workload_target: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """`x`: (B, L, C, T). `workload_target` is accepted for interface
        parity with AeroMindCapsNet but unused (no reconstruction head)."""
        del workload_target
        batch, seq_len, n_channels, n_samples = x.shape
        flat = x.reshape(batch * seq_len, n_channels, n_samples)

        features = self.encoder(flat)  # (B*L, feature_dim)
        seq_features = features.reshape(batch, seq_len, -1)
        lstm_out, _ = self.lstm(seq_features)
        last = lstm_out[:, -1, :]

        return {
            "workload_logits": self.workload_head(last),
            "fatigue_logits": self.fatigue_head(last),
        }
