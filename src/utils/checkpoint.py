"""Model/optimizer checkpoint save & load, used by the training and
evaluation CLIs (README §10, §11)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    epoch: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model_state": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return {"epoch": payload.get("epoch"), "extra": payload.get("extra", {})}
