"""AeroMind-CapsNet (README §9.1): Conv1D -> Primary/Digit Capsules -> Bi-LSTM
over a 15-epoch (30s) context window -> workload + fatigue heads, with a
lightweight reconstruction decoder used as a regularizer.

Design note vs. README: the reconstruction head decodes a downsampled
(avg-pooled) version of the input epoch rather than the full raw waveform.
Reconstructing the full `(7, 512)` waveform from a 16-d capsule would need a
~4M-parameter decoder alone, dwarfing the rest of the ~720k-parameter model
and making the reconstruction loss numerically dominate the multi-task
objective for no real benefit — the coarse target still regularizes the
digit capsule to retain class-relevant signal shape while keeping the
decoder (and the model's total parameter count) proportionate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import DigitCapsuleRouting, PrimaryCapsule1D


def _conv1d_output_length(length: int, kernel_size: int, stride: int, padding: int) -> int:
    return (length + 2 * padding - kernel_size) // stride + 1


class CapsNetEncoder(nn.Module):
    """Per-epoch encoder: raw (C, T) epoch -> digit capsules (n_workload_classes, digit_dim).

    `n_samples` (the epoch length T) must be known up front: it determines
    how many primary-capsule instances the strided conv produces, which
    fixes the dynamic-routing weight matrix's shape. Routing is therefore
    built eagerly in `__init__` rather than lazily on first forward — lazy
    construction would leave the routing weights (a large chunk of the
    model's parameters) missing from the optimizer if `model.parameters()`
    is called before the first forward pass, a classic PyTorch footgun.
    """

    PRIMARY_KERNEL = 9
    PRIMARY_STRIDE = 16
    PRIMARY_PADDING = 4

    def __init__(
        self,
        n_channels: int = 7,
        n_samples: int = 512,
        conv_channels: int = 64,
        primary_types: int = 32,
        primary_dim: int = 8,
        digit_capsules: int = 3,
        digit_dim: int = 16,
        routing_iters: int = 3,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, conv_channels, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.pool = nn.MaxPool1d(2)

        self.primary_caps = PrimaryCapsule1D(
            conv_channels,
            num_types=primary_types,
            capsule_dim=primary_dim,
            kernel_size=self.PRIMARY_KERNEL,
            stride=self.PRIMARY_STRIDE,
            padding=self.PRIMARY_PADDING,
        )

        pooled_length = n_samples // 2
        primary_t_out = _conv1d_output_length(
            pooled_length, self.PRIMARY_KERNEL, self.PRIMARY_STRIDE, self.PRIMARY_PADDING
        )
        n_in = primary_types * primary_t_out
        self.routing = DigitCapsuleRouting(
            n_in=n_in,
            dim_in=primary_dim,
            n_out=digit_capsules,
            dim_out=digit_dim,
            n_iterations=routing_iters,
        )
        self.digit_capsules = digit_capsules
        self.digit_dim = digit_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`x`: (B, C, T) -> digit capsules (B, n_digit, digit_dim), coupling (B, n_in, n_digit)."""
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.pool(h)  # (B, conv_channels, T/2)

        primary = self.primary_caps(h)  # (B, n_in, primary_dim)
        digit, coupling = self.routing(primary)
        return digit, coupling


class ReconstructionDecoder(nn.Module):
    """Reconstructs a coarse (avg-pooled) view of the input epoch from the
    predicted-class digit capsule, masking out the other classes (standard
    CapsNet reconstruction regularization)."""

    def __init__(self, digit_dim: int, n_channels: int, pooled_length: int = 32, hidden: int = 64):
        super().__init__()
        self.pooled_length = pooled_length
        self.n_channels = n_channels
        out_dim = n_channels * pooled_length
        self.net = nn.Sequential(
            nn.Linear(digit_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, digit_capsules: torch.Tensor, class_indices: torch.Tensor) -> torch.Tensor:
        """`digit_capsules`: (B, n_classes, digit_dim); `class_indices`: (B,) the
        class to reconstruct from (target during training, argmax at inference)."""
        batch = digit_capsules.size(0)
        selected = digit_capsules[torch.arange(batch, device=digit_capsules.device), class_indices]
        out = self.net(selected)
        return out.view(batch, self.n_channels, self.pooled_length)

    def target(self, epoch: torch.Tensor) -> torch.Tensor:
        """`epoch`: (B, C, T) -> avg-pooled (B, C, pooled_length) reconstruction target."""
        return F.adaptive_avg_pool1d(epoch, self.pooled_length)


class AeroMindCapsNet(nn.Module):
    """Full multi-task model: shared per-epoch CapsNet encoder, Bi-LSTM over
    a sequence of epochs, two classification heads, one reconstruction head.
    """

    def __init__(
        self,
        n_channels: int = 7,
        n_samples: int = 512,
        primary_types: int = 32,
        primary_dim: int = 8,
        digit_dim: int = 16,
        routing_iters: int = 3,
        lstm_hidden: int = 64,
        n_workload_classes: int = 3,
        n_fatigue_classes: int = 2,
    ):
        super().__init__()
        self.n_workload_classes = n_workload_classes
        self.encoder = CapsNetEncoder(
            n_channels=n_channels,
            n_samples=n_samples,
            primary_types=primary_types,
            primary_dim=primary_dim,
            digit_capsules=n_workload_classes,
            digit_dim=digit_dim,
            routing_iters=routing_iters,
        )
        self.decoder = ReconstructionDecoder(digit_dim=digit_dim, n_channels=n_channels)

        lstm_input_dim = n_workload_classes * digit_dim
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.workload_head = nn.Linear(lstm_hidden * 2, n_workload_classes)
        self.fatigue_head = nn.Linear(lstm_hidden * 2, n_fatigue_classes)

    def forward(
        self, x: torch.Tensor, workload_target: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """`x`: (B, L, C, T) — L consecutive epochs (30s context window).
        `workload_target`: (B,) ground-truth class to drive reconstruction
        during training; if None, uses the model's own prediction (inference).
        """
        batch, seq_len, n_channels, n_samples = x.shape
        flat = x.reshape(batch * seq_len, n_channels, n_samples)

        digit_capsules, coupling = self.encoder(flat)  # (B*L, n_workload, digit_dim)
        capsule_lengths = digit_capsules.norm(dim=-1)  # (B*L, n_workload) — class "presence" score

        seq_features = digit_capsules.reshape(batch, seq_len, -1)
        lstm_out, _ = self.lstm(seq_features)
        last = lstm_out[:, -1, :]  # final timestep -> "current" state prediction

        workload_logits = self.workload_head(last)
        fatigue_logits = self.fatigue_head(last)

        # Reconstruction uses the *last* epoch in the window (the one the
        # heads are predicting for) and its digit capsule.
        last_epoch = x[:, -1]  # (B, C, T)
        last_digit_capsules = digit_capsules.reshape(batch, seq_len, self.n_workload_classes, -1)[
            :, -1
        ]
        recon_class = (
            workload_target if workload_target is not None else workload_logits.argmax(dim=-1)
        )
        reconstruction = self.decoder(last_digit_capsules, recon_class)
        reconstruction_target = self.decoder.target(last_epoch)

        return {
            "workload_logits": workload_logits,
            "fatigue_logits": fatigue_logits,
            "capsule_lengths": capsule_lengths.reshape(batch, seq_len, -1)[:, -1],
            "reconstruction": reconstruction,
            "reconstruction_target": reconstruction_target,
            "coupling": coupling.reshape(batch, seq_len, *coupling.shape[1:])[:, -1],
        }
