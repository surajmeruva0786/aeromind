"""Phase 6 tests: training loop runs, loss decreases, checkpointing, and
protocol orchestration produce sane outputs (README roadmap step 75)."""

from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.dataset import SequenceEpochDataset
from src.models.registry import build_model
from src.training.callbacks import EarlyStopping, RunLogger
from src.training.loop import evaluate_epoch, train_one_epoch
from src.training.train import get_protocol_splits, run, run_fold
from src.utils.config import AeroMindConfig, DataConfig, ModelConfig, TrainConfig
from src.utils.seed import set_seed


def _make_loader(dataset, sequence_length=4, batch_size=4):
    seq_ds = SequenceEpochDataset(dataset, sequence_length=sequence_length)
    return DataLoader(seq_ds, batch_size=batch_size, shuffle=True)


def test_train_one_epoch_reduces_loss_over_several_epochs(small_epoch_dataset):
    set_seed(0)
    loader = _make_loader(small_epoch_dataset)
    data_cfg = DataConfig(sequence_length=4)
    model = build_model(ModelConfig(name="aeromind_cnn_lstm"), data_cfg)
    optimizer = AdamW(model.parameters(), lr=1e-2)
    train_cfg = TrainConfig(epochs=5)
    device = torch.device("cpu")

    losses = []
    for _ in range(5):
        metrics = train_one_epoch(model, loader, optimizer, train_cfg, device)
        losses.append(metrics["total_loss"])

    assert losses[-1] < losses[0]


def test_evaluate_epoch_returns_reports(small_epoch_dataset):
    loader = _make_loader(small_epoch_dataset)
    data_cfg = DataConfig(sequence_length=4)
    model = build_model(ModelConfig(name="aeromind_eegnet"), data_cfg)
    train_cfg = TrainConfig()
    result = evaluate_epoch(model, loader, train_cfg, torch.device("cpu"))
    assert 0.0 <= result.workload_report.accuracy <= 1.0
    assert 0.0 <= result.fatigue_report.accuracy <= 1.0
    assert "total_loss" in result.components


def test_early_stopping_triggers_after_patience_exceeded():
    stopper = EarlyStopping(patience=2, mode="min")
    assert stopper.step(1.0) is True
    assert stopper.step(1.1) is False
    assert stopper.step(1.2) is False
    assert stopper.should_stop is True


def test_run_logger_writes_metrics_json(tmp_path):
    run_logger = RunLogger(tmp_path)
    run_logger.log_epoch(0, {"total_loss": 1.0}, {"loss": 0.9})
    run_logger.log_epoch(1, {"total_loss": 0.8}, {"loss": 0.7})
    assert (tmp_path / "metrics.json").exists()
    assert len(run_logger.history) == 2


def test_get_protocol_splits_subject_dependent_and_loso(multi_subject_epoch_dataset):
    data_cfg = DataConfig()
    sd_splits = get_protocol_splits("subject_dependent", multi_subject_epoch_dataset, data_cfg, seed=1, n_subjects=4, duration_s=16.0)
    assert len(sd_splits) == 1

    loso = get_protocol_splits("loso", multi_subject_epoch_dataset, data_cfg, seed=1, n_subjects=4, duration_s=16.0)
    n_subjects = len({ep.subject_id for ep in multi_subject_epoch_dataset})
    assert len(loso) == n_subjects


def test_run_fold_end_to_end_subject_dependent(small_epoch_dataset, tmp_path):
    # LOSO-style split by subject_id (rather than a raw index slice) so each
    # side has a full, contiguous per-subject epoch run for sequence building.
    data_cfg = DataConfig(sequence_length=2)
    train_cfg = TrainConfig(epochs=2, batch_size=4, early_stop_patience=1, mixed_precision=False)
    model_cfg = ModelConfig(name="aeromind_cnn_lstm")

    subject_ids = sorted({ep.subject_id for ep in small_epoch_dataset})
    held_out = subject_ids[0]
    train_epochs = [ep for ep in small_epoch_dataset if ep.subject_id != held_out]
    test_epochs = [ep for ep in small_epoch_dataset if ep.subject_id == held_out]

    result = run_fold(
        model_cfg, train_cfg, data_cfg, train_epochs, test_epochs, "fold0", tmp_path, torch.device("cpu")
    )
    assert result["skipped"] is False
    assert "workload" in result["test"]
    assert (tmp_path / "fold0" / "best.ckpt").exists()
    assert (tmp_path / "fold0" / "metrics.json").exists()


def test_run_end_to_end_subject_dependent_smoke(tmp_path):
    config = AeroMindConfig(
        data=DataConfig(processed_dir=str(tmp_path / "no_such_dir"), sequence_length=2),
        model=ModelConfig(name="aeromind_eegnet"),
        train=TrainConfig(protocol="subject_dependent", epochs=2, batch_size=4, early_stop_patience=1, mixed_precision=False),
        output_dir=str(tmp_path / "run"),
    )
    summary = run(config, n_subjects=3, duration_s=60.0)
    assert summary["n_folds_completed"] == 1
    assert "workload_accuracy_mean" in summary
    assert (tmp_path / "run" / "config.yaml").exists()
    assert (tmp_path / "run" / "summary.json").exists()
