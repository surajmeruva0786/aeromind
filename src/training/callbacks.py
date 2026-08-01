"""Training callbacks: early stopping and run/metrics logging (README §10)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EarlyStopping:
    """Stops training once `patience` consecutive epochs fail to improve.

    `mode="min"` for loss-like scores (lower is better), `mode="max"` for
    metrics like macro-F1 (higher is better).
    """

    patience: int = 20
    mode: str = "min"
    best_score: float | None = field(default=None, init=False)
    counter: int = field(default=0, init=False)
    should_stop: bool = field(default=False, init=False)

    def step(self, score: float) -> bool:
        """Record `score` for the current epoch. Returns True if it's the
        new best (i.e. the caller should checkpoint)."""
        is_better = (
            self.best_score is None
            or (self.mode == "min" and score < self.best_score)
            or (self.mode == "max" and score > self.best_score)
        )
        if is_better:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return is_better


class RunLogger:
    """Appends per-epoch train/val metrics to `<output_dir>/metrics.json`,
    flushing after every epoch so a killed/crashed run still leaves a usable
    partial log (README roadmap step 74)."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []

    def log_epoch(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
        self.history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        self._flush()

    def log_summary(self, summary: dict) -> None:
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def _flush(self) -> None:
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.history, f, indent=2)
