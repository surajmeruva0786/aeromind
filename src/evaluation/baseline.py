"""Classical (non-deep) baseline: RandomForest/SVM on engineered features.

A sanity-check comparison point for the deep models — if AeroMind-CapsNet
can't beat a RandomForest on hand-crafted spectral/temporal/connectivity
features, something upstream is wrong (README §8, "classical baseline").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.data.synthetic import SyntheticEpoch
from src.features.pipeline import extract_features
from src.utils.metrics import ClassificationReport, compute_classification_report


@dataclass
class BaselineDataset:
    X: np.ndarray
    workload: np.ndarray
    fatigue: np.ndarray


def build_feature_dataset(epochs: list[SyntheticEpoch], sfreq: float = 256.0) -> BaselineDataset:
    X = np.stack([extract_features(ep.data, sfreq) for ep in epochs], axis=0)
    workload = np.array([ep.workload for ep in epochs])
    fatigue = np.array([ep.fatigue for ep in epochs])
    return BaselineDataset(X=X, workload=workload, fatigue=fatigue)


def make_classifier(kind: str = "random_forest", seed: int = 42):
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=12, random_state=seed, n_jobs=-1)
    if kind == "svm":
        return SVC(kernel="rbf", probability=True, random_state=seed)
    raise ValueError(f"Unknown classifier kind: {kind}")


def train_and_evaluate_baseline(
    train: BaselineDataset,
    test: BaselineDataset,
    target: str = "workload",
    kind: str = "random_forest",
    seed: int = 42,
) -> ClassificationReport:
    y_train = getattr(train, target)
    y_test = getattr(test, target)

    clf = make_classifier(kind, seed)
    clf.fit(train.X, y_train)

    preds = clf.predict(test.X)
    probs = clf.predict_proba(test.X) if hasattr(clf, "predict_proba") else None
    n_classes = 3 if target == "workload" else 2
    return compute_classification_report(y_test, preds, probs, n_classes=n_classes)
