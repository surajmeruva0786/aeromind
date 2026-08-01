"""Capsule Network building blocks (README §9.1): squash activation, 1D
primary capsules, and dynamic routing-by-agreement to digit capsules.

Reference: Sabour, Frosst & Hinton (2017), "Dynamic Routing Between Capsules".
Adapted from 2D conv capsules to 1D (time) for EEG epochs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Non-linearity that shrinks short vectors to ~0 and long vectors to
    ~unit length, so a capsule's output *length* can be read as a presence
    probability while its *orientation* encodes instantiation parameters.
    """
    squared_norm = (tensor**2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * tensor / torch.sqrt(squared_norm + eps)


class PrimaryCapsule1D(nn.Module):
    """Conv1D projection reshaped into `num_types` capsule types, each with
    `capsule_dim` channels, at every surviving temporal position — mirroring
    how the original 2D CapsNet treats (channel-group, spatial-position) as
    the primary capsule index.
    """

    def __init__(
        self,
        in_channels: int,
        num_types: int = 32,
        capsule_dim: int = 8,
        kernel_size: int = 9,
        stride: int = 16,
        padding: int = 4,
    ):
        super().__init__()
        self.num_types = num_types
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv1d(
            in_channels,
            num_types * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: (B, C_in, T) -> (B, num_types * T', capsule_dim), squashed."""
        out = self.conv(x)  # (B, num_types*capsule_dim, T')
        batch, _, t_out = out.shape
        out = out.view(batch, self.num_types, self.capsule_dim, t_out)
        out = out.permute(0, 1, 3, 2).reshape(batch, self.num_types * t_out, self.capsule_dim)
        return squash(out, dim=-1)


class DigitCapsuleRouting(nn.Module):
    """Dynamic routing-by-agreement from `n_in` primary capsules to `n_out`
    digit capsules over `n_iterations` rounds.
    """

    def __init__(self, n_in: int, dim_in: int, n_out: int, dim_out: int, n_iterations: int = 3):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_iterations = n_iterations
        # W[i, j] projects primary capsule i's dim_in vector into digit
        # capsule j's dim_out space.
        self.W = nn.Parameter(0.01 * torch.randn(1, n_in, n_out, dim_out, dim_in))

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`u`: (B, n_in, dim_in) -> digit capsules (B, n_out, dim_out) plus
        the final coupling coefficients (B, n_in, n_out) for interpretability."""
        batch = u.size(0)
        u = u.unsqueeze(2).unsqueeze(-1)  # (B, n_in, 1, dim_in, 1)
        W = self.W.expand(batch, -1, -1, -1, -1)  # (B, n_in, n_out, dim_out, dim_in)
        u_hat = torch.matmul(W, u).squeeze(-1)  # (B, n_in, n_out, dim_out)
        u_hat_detached = u_hat.detach()

        b = torch.zeros(batch, self.n_in, self.n_out, device=u.device, dtype=u.dtype)
        v = None
        c = None
        for it in range(self.n_iterations):
            c = F.softmax(b, dim=2)  # (B, n_in, n_out)
            s = (c.unsqueeze(-1) * (u_hat_detached if it < self.n_iterations - 1 else u_hat)).sum(
                dim=1
            )
            v = squash(s, dim=-1)  # (B, n_out, dim_out)
            if it < self.n_iterations - 1:
                agreement = torch.einsum("bijd,bjd->bij", u_hat_detached, v)
                b = b + agreement

        return v, c
