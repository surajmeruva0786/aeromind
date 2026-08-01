"""AeroMind-EEGNet (README §9.3): baseline 2 — a compact EEGNet-style
front-end (Lawhern et al., 2018) per epoch, feeding the same Bi-LSTM +
dual-head structure used by the other two models.

This is a from-scratch EEGNet-architecture implementation (temporal conv ->
depthwise spatial conv -> separable conv), not a pretrained-weights transfer
load — no public EEGNet checkpoint is trained on a compatible 7-channel,
256 Hz montage, so "transfer learning" here means reusing the published
architecture, matching how EEGNet is normally applied to a new dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNetEpochEncoder(nn.Module):
    """Per-epoch encoder: raw (C, T) epoch -> fixed-size feature vector,
    using the standard EEGNet block structure operating on a (1, C, T)
    pseudo-image (channels-as-height, time-as-width)."""

    def __init__(
        self,
        n_channels: int = 7,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ):
        super().__init__()
        f_depth = f1 * depth_multiplier

        # Block 1: temporal conv, then depthwise spatial conv across channels.
        self.temporal_conv = nn.Conv2d(
            1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise_conv = nn.Conv2d(
            f1, f_depth, kernel_size=(n_channels, 1), groups=f1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(f_depth)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: separable conv (depthwise temporal + pointwise).
        self.separable_depthwise = nn.Conv2d(
            f_depth, f_depth, kernel_size=(1, 16), padding=(0, 8), groups=f_depth, bias=False
        )
        self.separable_pointwise = nn.Conv2d(f_depth, f2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.drop2 = nn.Dropout(dropout)

        self.feature_dim = f2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: (B, C, T) -> (B, f2)."""
        h = x.unsqueeze(1)  # (B, 1, C, T)
        h = self.bn1(self.temporal_conv(h))
        h = self.act1(self.bn2(self.depthwise_conv(h)))
        h = self.drop1(self.pool1(h))
        h = self.separable_pointwise(self.separable_depthwise(h))
        h = self.act2(self.bn3(h))
        h = self.drop2(self.pool2(h))
        return h.flatten(1)


class AeroMindEEGNet(nn.Module):
    """Multi-task baseline: shared per-epoch EEGNet encoder, Bi-LSTM over a
    15-epoch (30s) window, two classification heads. No reconstruction head."""

    def __init__(
        self,
        n_channels: int = 7,
        n_samples: int = 512,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        lstm_hidden: int = 64,
        n_workload_classes: int = 3,
        n_fatigue_classes: int = 2,
    ):
        super().__init__()
        del n_samples  # unused: encoder is adaptive-pooled, doesn't need fixed length
        self.encoder = EEGNetEpochEncoder(n_channels, f1, depth_multiplier, f2)
        self.lstm = nn.LSTM(
            self.encoder.feature_dim, lstm_hidden, batch_first=True, bidirectional=True
        )
        self.workload_head = nn.Linear(lstm_hidden * 2, n_workload_classes)
        self.fatigue_head = nn.Linear(lstm_hidden * 2, n_fatigue_classes)

    def forward(
        self, x: torch.Tensor, workload_target: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """`x`: (B, L, C, T). `workload_target` accepted for interface parity, unused."""
        del workload_target
        batch, seq_len, n_channels, n_samples = x.shape
        flat = x.reshape(batch * seq_len, n_channels, n_samples)

        features = self.encoder(flat)
        seq_features = features.reshape(batch, seq_len, -1)
        lstm_out, _ = self.lstm(seq_features)
        last = lstm_out[:, -1, :]

        return {
            "workload_logits": self.workload_head(last),
            "fatigue_logits": self.fatigue_head(last),
        }
