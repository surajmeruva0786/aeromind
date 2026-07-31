"""Classification metrics used across training, evaluation, and reports."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass
class ClassificationReport:
    accuracy: float
    macro_f1: float
    kappa: float
    confusion: np.ndarray
    roc_auc: float | None = None
    ece: float | None = None
    per_class_f1: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict:
        return {
            "accuracy": float(self.accuracy),
            "macro_f1": float(self.macro_f1),
            "kappa": float(self.kappa),
            "confusion_matrix": self.confusion.tolist(),
            "roc_auc": None if self.roc_auc is None else float(self.roc_auc),
            "ece": None if self.ece is None else float(self.ece),
            "per_class_f1": self.per_class_f1.tolist(),
        }


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error over the top-predicted-class confidence."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if not np.any(in_bin):
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    probs: np.ndarray | None = None,
    n_classes: int | None = None,
) -> ClassificationReport:
    """Compute the standard AeroMind evaluation bundle for one task head."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    n_classes = n_classes or int(max(labels.max(), predictions.max()) + 1)

    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    per_class_f1 = f1_score(
        labels, predictions, average=None, labels=list(range(n_classes)), zero_division=0
    )
    kappa = cohen_kappa_score(labels, predictions)
    cm = confusion_matrix(labels, predictions, labels=list(range(n_classes)))

    roc_auc = None
    ece = None
    if probs is not None:
        probs = np.asarray(probs)
        try:
            if n_classes == 2:
                roc_auc = roc_auc_score(labels, probs[:, 1])
            else:
                roc_auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except ValueError:
            roc_auc = None
        ece = expected_calibration_error(probs, labels)

    return ClassificationReport(
        accuracy=acc,
        macro_f1=macro_f1,
        kappa=kappa,
        confusion=cm,
        roc_auc=roc_auc,
        ece=ece,
        per_class_f1=per_class_f1,
    )
