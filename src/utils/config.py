"""YAML config loading into typed dataclasses used by training/eval CLIs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    dataset: str = "synthetic"
    processed_dir: str = "data/processed/synthetic"
    n_channels: int = 7
    sfreq: float = 256.0
    epoch_seconds: float = 2.0
    sequence_length: int = 15  # epochs per Bi-LSTM context window


@dataclass
class ModelConfig:
    name: str = "aeromind_capsnet"
    primary_capsules: int = 32
    primary_capsule_dim: int = 8
    digit_capsule_dim: int = 16
    routing_iters: int = 3
    lstm_hidden: int = 64
    n_workload_classes: int = 3
    n_fatigue_classes: int = 2


@dataclass
class TrainConfig:
    optimizer: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 5e-4
    batch_size: int = 64
    epochs: int = 150
    early_stop_patience: int = 20
    lr_patience: int = 8
    lr_factor: float = 0.5
    mixed_precision: bool = True
    seed: int = 42
    protocol: str = "loso"  # loso | subject_dependent | cross_dataset
    workload_weight: float = 0.6
    fatigue_weight: float = 0.3
    reconstruction_weight: float = 0.1


@dataclass
class AeroMindConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "runs/default"

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)


def load_config(path: str | Path) -> AeroMindConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return AeroMindConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
        output_dir=raw.get("output_dir", "runs/default"),
    )
