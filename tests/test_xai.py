"""Phase 8 tests: SHAP channel/spectral attribution shapes, topomap
rendering, and counter-factual probe logic (README roadmap steps 92-93)."""

from __future__ import annotations

import numpy as np
import torch

from src.data.synthetic import BANDS, CHANNEL_NAMES, FRONTAL_CHANNELS
from src.models import AeroMindEEGNet
from src.training.train import run as run_training
from src.utils.config import AeroMindConfig, DataConfig, ModelConfig, TrainConfig
from src.xai.counter_factual import attenuate_band, run_counter_factual_probe
from src.xai.explain import run_explain
from src.xai.shap_channel import compute_channel_attributions
from src.xai.spectral_attribution import (
    build_band_power_dataset,
    compute_spectral_attribution,
    fit_spectral_classifier,
)
from src.xai.topomap import build_montage_info, plot_channel_topomap


def test_compute_channel_attributions_shape():
    torch.manual_seed(0)
    model = AeroMindEEGNet()
    background = torch.randn(6, 4, 7, 512)
    inputs = torch.randn(3, 4, 7, 512)

    result = compute_channel_attributions(model, background, inputs, CHANNEL_NAMES)
    assert result.channel_values.shape == (3, len(CHANNEL_NAMES))
    assert result.predicted_classes.shape == (3,)
    assert np.isfinite(result.channel_values).all()


def test_build_montage_info_has_all_channels():
    info = build_montage_info(CHANNEL_NAMES)
    assert info["ch_names"] == list(CHANNEL_NAMES)
    assert info.get_montage() is not None


def test_plot_channel_topomap_returns_figure():
    values = np.random.default_rng(0).normal(size=len(CHANNEL_NAMES))
    fig = plot_channel_topomap(values, CHANNEL_NAMES, title="test")
    assert fig is not None
    assert len(fig.axes) >= 1


def test_spectral_attribution_pipeline_shapes(small_epoch_dataset):
    X, y = build_band_power_dataset(small_epoch_dataset)
    n_channels = len(CHANNEL_NAMES)
    n_bands = len(BANDS)
    assert X.shape == (len(small_epoch_dataset), n_channels * n_bands)

    clf = fit_spectral_classifier(X, y)
    result = compute_spectral_attribution(clf, X[:5], n_channels=n_channels)
    assert result.shap_values.shape[0] == 5
    assert result.shap_values.shape[1] == n_channels
    assert result.shap_values.shape[2] == n_bands

    mean_abs = result.mean_abs_matrix(class_index=0)
    assert mean_abs.shape == (n_channels, n_bands)
    assert np.all(mean_abs >= 0)


def test_attenuate_band_changes_signal_only_on_target_channels():
    rng = np.random.default_rng(1)
    epoch = rng.normal(size=(len(CHANNEL_NAMES), 512)).astype(np.float64)
    attenuated = attenuate_band(
        epoch, sfreq=256.0, channels=FRONTAL_CHANNELS, band=BANDS["theta"], factor=0.5
    )

    assert attenuated.shape == epoch.shape
    frontal_idx = [CHANNEL_NAMES.index(ch) for ch in FRONTAL_CHANNELS]
    other_idx = [i for i in range(len(CHANNEL_NAMES)) if i not in frontal_idx]

    assert not np.allclose(attenuated[frontal_idx], epoch[frontal_idx])
    assert np.allclose(attenuated[other_idx], epoch[other_idx])


def test_run_counter_factual_probe_returns_valid_probabilities():
    torch.manual_seed(0)
    model = AeroMindEEGNet()
    sequence = torch.randn(1, 4, 7, 512)

    result = run_counter_factual_probe(model, sequence, sfreq=256.0)
    assert 0 <= result.original_class < 3
    assert 0 <= result.attenuated_class < 3
    assert np.isclose(result.original_probs.sum(), 1.0, atol=1e-4)
    assert np.isclose(result.attenuated_probs.sum(), 1.0, atol=1e-4)
    assert isinstance(result.passed, bool)


def test_run_explain_end_to_end(tmp_path):
    """Integration test for the explain.py CLI's `run_explain`: trains a
    tiny checkpoint, then runs the full XAI pipeline (channel attribution +
    topomap, spectral SHAP, counter-factual probe) against it."""
    config = AeroMindConfig(
        data=DataConfig(processed_dir=str(tmp_path / "no_such_dir"), sequence_length=3),
        model=ModelConfig(name="aeromind_eegnet"),
        train=TrainConfig(
            protocol="subject_dependent",
            epochs=1,
            batch_size=4,
            early_stop_patience=1,
            mixed_precision=False,
        ),
        output_dir=str(tmp_path / "run"),
    )
    run_training(config, n_subjects=3, duration_s=90.0)
    checkpoint_path = tmp_path / "run" / "subject_dependent" / "best.ckpt"
    assert checkpoint_path.exists()

    output_dir = tmp_path / "xai_out"
    summary = run_explain(
        checkpoint_path,
        None,
        subject_id=0,
        n_subjects=3,
        duration_s=90.0,
        output_dir=output_dir,
        n_background=3,
        n_explain=2,
    )

    assert (output_dir / "xai_summary.json").exists()
    assert (output_dir / "channel_attribution_topomap.png").exists()
    assert len(summary["channel_attribution"]["values"]) == 2
    assert "counter_factual_probe" in summary
