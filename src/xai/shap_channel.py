"""SHAP channel attribution (README §12.1): wraps `shap.GradientExplainer`
around a trained AeroMind model to produce a per-channel attribution score
for the workload prediction, averaged over the time dimension.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shap
import torch
import torch.nn as nn


class WorkloadLogitWrapper(nn.Module):
    """Adapts a full AeroMind model's dict-output forward to a plain-tensor
    forward over workload logits only — the interface SHAP explainers
    expect (a single input tensor -> a single output tensor)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["workload_logits"]


@dataclass
class ChannelAttribution:
    channel_values: np.ndarray  # (n_inputs, n_channels)
    predicted_classes: np.ndarray  # (n_inputs,)
    channel_names: tuple[str, ...]


def compute_channel_attributions(
    model: nn.Module,
    background: torch.Tensor,
    inputs: torch.Tensor,
    channel_names: tuple[str, ...],
) -> ChannelAttribution:
    """`background`/`inputs`: (N, L, C, T) sequence windows.

    Runs `shap.GradientExplainer` over the wrapped model, then for each
    input takes the SHAP values for *its own predicted class* (not a fixed
    class), and averages over the sequence (L) and time (T) axes to yield
    one signed attribution score per channel — matching README §12.1's
    "averaged over the time dimension" definition.
    """
    model.eval()
    wrapped = WorkloadLogitWrapper(model)
    wrapped.eval()

    with torch.no_grad():
        predicted_classes = wrapped(inputs).argmax(dim=-1).cpu().numpy()

    explainer = shap.GradientExplainer(wrapped, background)
    shap_values = explainer.shap_values(inputs)  # (N, L, C, T, n_classes)

    n_inputs, _, n_channels, _, _ = shap_values.shape
    channel_values = np.zeros((n_inputs, n_channels), dtype=np.float64)
    for i in range(n_inputs):
        cls = predicted_classes[i]
        # mean over sequence (axis 0) and time (axis -1 after selecting class)
        channel_values[i] = shap_values[i, :, :, :, cls].mean(axis=(0, 2))

    return ChannelAttribution(
        channel_values=channel_values, predicted_classes=predicted_classes, channel_names=channel_names
    )
