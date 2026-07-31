"""Epoch-level augmentations (README §10 — Augmentation).

All transforms operate on a single `(C, T)` numpy array / torch tensor and
are composable. Mixup operates on a batch and is applied in the training
loop rather than per-sample.
"""

from __future__ import annotations

import numpy as np
import torch


class ChannelDropout:
    """Randomly zero one channel per epoch with probability `p`."""

    def __init__(self, p: float = 0.1, seed: int | None = None):
        self.p = p
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.p:
            ch = self.rng.integers(0, x.shape[0])
            x = x.copy()
            x[ch, :] = 0.0
        return x


class TimeShift:
    """Circular shift by up to `max_shift_ms` milliseconds."""

    def __init__(self, max_shift_ms: float = 50.0, sfreq: float = 256.0, seed: int | None = None):
        self.max_shift_samples = max(1, int(max_shift_ms / 1000.0 * sfreq))
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        shift = int(self.rng.integers(-self.max_shift_samples, self.max_shift_samples + 1))
        if shift == 0:
            return x
        return np.roll(x, shift, axis=-1)


class GaussianNoise:
    """Additive Gaussian noise, sigma expressed relative to z-scored signal."""

    def __init__(self, sigma: float = 0.05, seed: int | None = None):
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x + self.rng.normal(0.0, self.sigma, size=x.shape)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


def default_train_transforms(sfreq: float = 256.0, seed: int | None = None) -> Compose:
    return Compose(
        [
            ChannelDropout(p=0.1, seed=seed),
            TimeShift(max_shift_ms=50.0, sfreq=sfreq, seed=seed),
            GaussianNoise(sigma=0.05, seed=seed),
        ]
    )


def mixup_batch(
    x: torch.Tensor,
    workload: torch.Tensor,
    fatigue: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup restricted to same-class pairs (README §10): shuffles within each
    workload class and blends. Returns (x_mixed, workload, fatigue,
    workload_perm, fatigue_perm, lam) — lam == 1.0 for unmatched samples that
    could not be paired (kept unmixed rather than blended across classes).
    """
    batch_size = x.size(0)
    device = x.device
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0

    perm = torch.arange(batch_size, device=device)
    for cls in torch.unique(workload):
        idx = (workload == cls).nonzero(as_tuple=True)[0]
        if len(idx) > 1:
            shuffled = idx[torch.randperm(len(idx), device=device)]
            perm[idx] = shuffled

    x_mixed = lam * x + (1 - lam) * x[perm]
    return x_mixed, workload, fatigue, workload[perm], fatigue[perm], lam
