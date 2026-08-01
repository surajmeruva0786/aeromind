"""Model factory: `ModelConfig.name` -> instantiated `nn.Module`.

Keeps `train.py`/`evaluate.py` decoupled from the concrete model classes so
adding a new architecture only requires registering it here.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.aeromind_capsnet import AeroMindCapsNet
from src.models.aeromind_cnn_lstm import AeroMindCNNLSTM
from src.models.aeromind_eegnet import AeroMindEEGNet
from src.utils.config import DataConfig, ModelConfig

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "aeromind_capsnet": AeroMindCapsNet,
    "aeromind_cnn_lstm": AeroMindCNNLSTM,
    "aeromind_eegnet": AeroMindEEGNet,
}


def build_model(model_config: ModelConfig, data_config: DataConfig) -> nn.Module:
    """Instantiate the model named in `model_config.name` with hyperparameters
    drawn from both configs (data config supplies input shape)."""
    if model_config.name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_config.name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    cls = MODEL_REGISTRY[model_config.name]
    n_samples = int(data_config.epoch_seconds * data_config.sfreq)

    common = dict(
        n_channels=data_config.n_channels,
        n_samples=n_samples,
        lstm_hidden=model_config.lstm_hidden,
        n_workload_classes=model_config.n_workload_classes,
        n_fatigue_classes=model_config.n_fatigue_classes,
    )

    if model_config.name == "aeromind_capsnet":
        return cls(
            **common,
            primary_types=model_config.primary_capsules,
            primary_dim=model_config.primary_capsule_dim,
            digit_dim=model_config.digit_capsule_dim,
            routing_iters=model_config.routing_iters,
        )
    return cls(**common)
