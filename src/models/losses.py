"""Loss functions for AeroMind's multi-task training (README §9.1, §10):
CapsNet margin loss for the workload head when capsule lengths are
available, cross-entropy otherwise, plus a weighted fatigue CE term and an
optional reconstruction MSE term.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.utils.config import TrainConfig


def margin_loss(
    capsule_lengths: torch.Tensor,
    targets: torch.Tensor,
    m_plus: float = 0.9,
    m_minus: float = 0.1,
    lambda_: float = 0.5,
) -> torch.Tensor:
    """Standard CapsNet margin loss (Sabour et al. 2017, eq. 4).

    `capsule_lengths`: (B, n_classes) digit-capsule norms (presence scores).
    `targets`: (B,) integer class labels.
    """
    n_classes = capsule_lengths.size(1)
    one_hot = F.one_hot(targets, num_classes=n_classes).to(capsule_lengths.dtype)

    present = one_hot * torch.clamp(m_plus - capsule_lengths, min=0.0) ** 2
    absent = (1.0 - one_hot) * lambda_ * torch.clamp(capsule_lengths - m_minus, min=0.0) ** 2
    return (present + absent).sum(dim=1).mean()


def multi_task_loss(
    outputs: dict[str, torch.Tensor],
    workload_target: torch.Tensor,
    fatigue_target: torch.Tensor,
    config: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine workload + fatigue (+ reconstruction, if present) losses per
    the weights in `config`. Returns the total loss and a dict of the
    individual (unweighted) component values for logging.

    Uses `margin_loss` on `capsule_lengths` when the model provides them
    (AeroMind-CapsNet); falls back to cross-entropy on `workload_logits`
    for baselines that don't produce capsule presence vectors.
    """
    components: dict[str, float] = {}

    if "capsule_lengths" in outputs:
        workload_loss = margin_loss(outputs["capsule_lengths"], workload_target)
    else:
        workload_loss = F.cross_entropy(outputs["workload_logits"], workload_target)
    components["workload_loss"] = workload_loss.item()

    fatigue_loss = F.cross_entropy(outputs["fatigue_logits"], fatigue_target)
    components["fatigue_loss"] = fatigue_loss.item()

    total = config.workload_weight * workload_loss + config.fatigue_weight * fatigue_loss

    if "reconstruction" in outputs and config.reconstruction_weight > 0:
        recon_loss = F.mse_loss(outputs["reconstruction"], outputs["reconstruction_target"])
        components["reconstruction_loss"] = recon_loss.item()
        total = total + config.reconstruction_weight * recon_loss

    components["total_loss"] = total.item()
    return total, components
