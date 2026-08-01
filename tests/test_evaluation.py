"""Phase 7 tests: evaluate.py reconstructs a deterministic fold split and
reproduces the training run's own held-out test metrics exactly, plus a
markdown report generator sanity check (README roadmap step 82)."""

from __future__ import annotations

from src.evaluation.evaluate import evaluate_checkpoint, find_run_config
from src.evaluation.metrics_report import generate_report
from src.training.train import run
from src.utils.config import AeroMindConfig, DataConfig, ModelConfig, TrainConfig, load_config


def _train_tiny_run(tmp_path, protocol="subject_dependent"):
    config = AeroMindConfig(
        data=DataConfig(processed_dir=str(tmp_path / "no_such_dir"), sequence_length=2),
        model=ModelConfig(name="aeromind_eegnet"),
        train=TrainConfig(
            protocol=protocol, epochs=2, batch_size=4, early_stop_patience=1, mixed_precision=False
        ),
        output_dir=str(tmp_path / "run"),
    )
    summary = run(config, n_subjects=3, duration_s=60.0)
    return config, summary


def test_find_run_config_locates_sibling_config_yaml(tmp_path):
    _train_tiny_run(tmp_path)
    checkpoint_path = tmp_path / "run" / "subject_dependent" / "best.ckpt"
    assert checkpoint_path.exists()
    found = find_run_config(checkpoint_path)
    assert found == tmp_path / "run" / "config.yaml"
    assert found.exists()


def test_evaluate_checkpoint_reproduces_training_run_test_metrics(tmp_path):
    _, summary = _train_tiny_run(tmp_path)
    checkpoint_path = tmp_path / "run" / "subject_dependent" / "best.ckpt"
    config = load_config(tmp_path / "run" / "config.yaml")

    result, meta = evaluate_checkpoint(
        config, checkpoint_path, fold_name="subject_dependent", n_subjects=3, duration_s=60.0
    )

    fold_result = summary["folds"][0]
    assert result.workload_report.accuracy == fold_result["test"]["workload"]["accuracy"]
    assert result.fatigue_report.accuracy == fold_result["test"]["fatigue"]["accuracy"]
    assert meta["fold"] == "subject_dependent"
    assert meta["model"] == "aeromind_eegnet"


def test_evaluate_checkpoint_loso_fold(tmp_path):
    _, summary = _train_tiny_run(tmp_path, protocol="loso")
    config = load_config(tmp_path / "run" / "config.yaml")
    fold_names = [f["fold"] for f in summary["folds"] if not f["skipped"]]
    assert fold_names, "expected at least one completed LOSO fold"
    fold_name = fold_names[0]

    checkpoint_path = tmp_path / "run" / fold_name / "best.ckpt"
    result, meta = evaluate_checkpoint(
        config, checkpoint_path, fold_name=fold_name, n_subjects=3, duration_s=60.0
    )
    assert meta["fold"] == fold_name
    assert 0.0 <= result.workload_report.accuracy <= 1.0


def test_generate_report_contains_key_sections(tmp_path):
    _, summary = _train_tiny_run(tmp_path)
    checkpoint_path = tmp_path / "run" / "subject_dependent" / "best.ckpt"
    config = load_config(tmp_path / "run" / "config.yaml")
    result, meta = evaluate_checkpoint(config, checkpoint_path, "subject_dependent", 3, 60.0)

    report = generate_report(result.workload_report, result.fatigue_report, meta)
    assert "Workload classification" in report
    assert "Fatigue classification" in report
    assert "Confusion matrix" in report
    assert meta["config_hash"] in report
