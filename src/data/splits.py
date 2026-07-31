"""Evaluation protocol splitters (README §11.1).

- subject_dependent: random 80/20 split *within* each subject
- loso: leave-one-subject-out, one subject held out as test at a time
- cross_dataset: train split is one dataset's epochs, test split another's
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.data.synthetic import SyntheticEpoch


@dataclass
class Split:
    train: list[SyntheticEpoch]
    test: list[SyntheticEpoch]
    name: str


def subject_dependent_split(
    epochs: list[SyntheticEpoch], test_fraction: float = 0.2, seed: int = 42
) -> Split:
    rng = np.random.default_rng(seed)
    by_subject: dict[int, list[SyntheticEpoch]] = defaultdict(list)
    for ep in epochs:
        by_subject[ep.subject_id].append(ep)

    train, test = [], []
    for subj_epochs in by_subject.values():
        idx = rng.permutation(len(subj_epochs))
        n_test = max(1, int(len(subj_epochs) * test_fraction))
        test_idx = set(idx[:n_test].tolist())
        for i, ep in enumerate(subj_epochs):
            (test if i in test_idx else train).append(ep)

    return Split(train=train, test=test, name="subject_dependent")


def loso_splits(epochs: list[SyntheticEpoch]) -> list[Split]:
    """One split per subject, with that subject entirely held out."""
    subjects = sorted({ep.subject_id for ep in epochs})
    splits = []
    for held_out in subjects:
        train = [ep for ep in epochs if ep.subject_id != held_out]
        test = [ep for ep in epochs if ep.subject_id == held_out]
        splits.append(Split(train=train, test=test, name=f"loso_subject_{held_out}"))
    return splits


def cross_dataset_split(
    train_epochs: list[SyntheticEpoch], test_epochs: list[SyntheticEpoch], name: str = "cross_dataset"
) -> Split:
    return Split(train=list(train_epochs), test=list(test_epochs), name=name)


def class_balance(epochs: list[SyntheticEpoch]) -> dict[str, dict[int, int]]:
    """Diagnostic: label distribution, used to sanity-check a split isn't
    pathologically imbalanced before training."""
    workload_counts: dict[int, int] = defaultdict(int)
    fatigue_counts: dict[int, int] = defaultdict(int)
    for ep in epochs:
        workload_counts[ep.workload] += 1
        fatigue_counts[ep.fatigue] += 1
    return {"workload": dict(workload_counts), "fatigue": dict(fatigue_counts)}
