"""Shared pytest fixtures — small synthetic datasets for fast tests."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for CI and any XAI plotting tests

import pytest

from src.data.synthetic import generate_dataset


@pytest.fixture(scope="session")
def small_epoch_dataset():
    """~2 subjects x short duration -> fast to generate, used across many tests."""
    return generate_dataset(n_subjects=2, duration_s=20.0, sfreq=256.0, seed=7)


@pytest.fixture(scope="session")
def multi_subject_epoch_dataset():
    """More subjects, still short, for split/LOSO tests that need >1 subject."""
    return generate_dataset(n_subjects=4, duration_s=16.0, sfreq=256.0, seed=11)
