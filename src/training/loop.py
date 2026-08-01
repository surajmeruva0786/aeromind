"""Train/validation epoch loops (README §10). Works for all three model
variants via the shared `forward(x, workload_target=None) -> dict` interface
and `src.models.losses.multi_task_loss`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.losses import multi_task_loss
from src.utils.config import TrainConfig
from src.utils.metrics import ClassificationReport, compute_classification_report


@dataclass
class EpochResult:
    loss: float
    components: dict[str, float]
    workload_report: ClassificationReport
    fatigue_report: ClassificationReport


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    config: TrainConfig,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Runs one training epoch, returns the mean of each loss component."""
    model.train()
    totals: dict[str, float] = {}
    n_batches = 0

    for x, workload, fatigue in loader:
        x, workload, fatigue = x.to(device), workload.to(device), fatigue.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and scaler.is_enabled():
            with torch.autocast(device_type=device.type):
                out = model(x, workload_target=workload)
                loss, components = multi_task_loss(out, workload, fatigue, config)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x, workload_target=workload)
            loss, components = multi_task_loss(out, workload, fatigue, config)
            loss.backward()
            optimizer.step()

        for k, v in components.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate_epoch(
    model,
    loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    n_workload_classes: int = 3,
    n_fatigue_classes: int = 2,
) -> EpochResult:
    """Runs one evaluation pass (no gradient), returns mean loss plus full
    classification reports for both heads. Uses the model's own predicted
    class (`workload_target=None`) for the CapsNet reconstruction path,
    matching how the model behaves at real inference time."""
    model.eval()
    totals: dict[str, float] = {}
    n_batches = 0

    workload_labels, workload_preds, workload_probs = [], [], []
    fatigue_labels, fatigue_preds, fatigue_probs = [], [], []

    for x, workload, fatigue in loader:
        x, workload, fatigue = x.to(device), workload.to(device), fatigue.to(device)
        out = model(x)
        loss, components = multi_task_loss(out, workload, fatigue, config)
        for k, v in components.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

        w_probs = torch.softmax(out["workload_logits"], dim=-1).cpu().numpy()
        f_probs = torch.softmax(out["fatigue_logits"], dim=-1).cpu().numpy()
        workload_labels.append(workload.cpu().numpy())
        workload_preds.append(w_probs.argmax(axis=1))
        workload_probs.append(w_probs)
        fatigue_labels.append(fatigue.cpu().numpy())
        fatigue_preds.append(f_probs.argmax(axis=1))
        fatigue_probs.append(f_probs)

    workload_report = compute_classification_report(
        np.concatenate(workload_labels),
        np.concatenate(workload_preds),
        np.concatenate(workload_probs),
        n_classes=n_workload_classes,
    )
    fatigue_report = compute_classification_report(
        np.concatenate(fatigue_labels),
        np.concatenate(fatigue_preds),
        np.concatenate(fatigue_probs),
        n_classes=n_fatigue_classes,
    )
    mean_components = {k: v / max(n_batches, 1) for k, v in totals.items()}

    return EpochResult(
        loss=mean_components["total_loss"],
        components=mean_components,
        workload_report=workload_report,
        fatigue_report=fatigue_report,
    )
