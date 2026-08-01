"""Training CLI (README §10, §16). Orchestrates subject-dependent, LOSO, and
cross-dataset protocols over any registered model, on the synthetic dataset
(in-memory or materialized via `scripts/make_synthetic_dataset.py`).

Example:
    python -m src.training.train --model aeromind_capsnet --protocol subject_dependent \\
        --epochs 5 --n_subjects 4 --duration_s 60 --output_dir runs/smoke_test

Honesty note: real STEW/MAUS ingestion into this training loop's on-disk
epoch format is not yet wired (see `data/README.md` for their current
download-only status) — `--dataset synthetic` is the only fully
end-to-end-validated path today, per the project honesty note at the top of
ROADMAP.md. `--protocol cross_dataset` uses two synthetic cohorts generated
with different seeds as a structural stand-in (same generator, different
"population"), clearly logged as such in the run summary.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.dataset import SequenceEpochDataset
from src.data.splits import Split, class_balance, loso_splits, subject_dependent_split
from src.data.synthetic import SyntheticEpoch, generate_dataset
from src.models.registry import build_model
from src.training.callbacks import EarlyStopping, RunLogger
from src.training.loop import evaluate_epoch, train_one_epoch
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import AeroMindConfig, DataConfig, ModelConfig, TrainConfig
from src.utils.logging_utils import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _load_npz_epochs(processed_dir: Path) -> list[SyntheticEpoch]:
    epochs: list[SyntheticEpoch] = []
    for f in sorted(processed_dir.glob("sub-*.npz")):
        subject_id = int(f.stem.split("-")[1])
        npz = np.load(f)
        data, workload, fatigue = npz["data"], npz["workload"], npz["fatigue"]
        for i in range(len(workload)):
            epochs.append(
                SyntheticEpoch(
                    data=data[i],
                    workload=int(workload[i]),
                    fatigue=int(fatigue[i]),
                    subject_id=subject_id,
                )
            )
    return epochs


def load_epochs(
    data_config: DataConfig, n_subjects: int, duration_s: float, seed: int
) -> list[SyntheticEpoch]:
    """Loads epochs from `data_config.processed_dir` if it holds a
    materialized synthetic dataset (`scripts/make_synthetic_dataset.py`
    output), otherwise generates one in-memory (fast path for smoke tests)."""
    processed = Path(data_config.processed_dir)
    if processed.exists() and any(processed.glob("sub-*.npz")):
        logger.info("Loading materialized synthetic dataset from %s", processed)
        return _load_npz_epochs(processed)
    logger.info(
        "No materialized dataset at %s — generating %d synthetic subjects in-memory",
        processed,
        n_subjects,
    )
    return generate_dataset(
        n_subjects=n_subjects,
        duration_s=duration_s,
        sfreq=data_config.sfreq,
        epoch_seconds=data_config.epoch_seconds,
        seed=seed,
    )


def get_protocol_splits(
    protocol: str,
    epochs: list[SyntheticEpoch],
    data_config: DataConfig,
    seed: int,
    n_subjects: int,
    duration_s: float,
) -> list[Split]:
    if protocol == "subject_dependent":
        return [subject_dependent_split(epochs, test_fraction=0.2, seed=seed)]
    if protocol == "loso":
        return loso_splits(epochs)
    if protocol == "cross_dataset":
        # Structural stand-in: a second synthetic cohort from a different
        # seed offset, standing in for a genuinely different dataset/device
        # until real cross-dataset (MAUS/STEW) ingestion is wired up.
        cross_epochs = generate_dataset(
            n_subjects=max(2, n_subjects // 2),
            duration_s=duration_s,
            sfreq=data_config.sfreq,
            epoch_seconds=data_config.epoch_seconds,
            seed=seed + 10_000,
        )
        return [Split(train=epochs, test=cross_epochs, name="cross_dataset_synthetic_proxy")]
    raise ValueError(f"Unknown protocol: {protocol}")


def run_fold(
    model_config: ModelConfig,
    train_config: TrainConfig,
    data_config: DataConfig,
    train_epochs: list[SyntheticEpoch],
    test_epochs: list[SyntheticEpoch],
    fold_name: str,
    output_dir: Path,
    device: torch.device,
) -> dict:
    """Trains one fold: carves a validation set out of `train_epochs`
    (never touching `test_epochs`), trains with early stopping on val loss,
    reloads the best checkpoint, and evaluates once on the held-out test
    epochs. Returns a JSON-serializable result dict."""
    val_split = subject_dependent_split(train_epochs, test_fraction=0.15, seed=train_config.seed)
    fold_train, fold_val = val_split.train, val_split.test

    seq_len = data_config.sequence_length
    train_ds = SequenceEpochDataset(fold_train, sequence_length=seq_len)
    val_ds = SequenceEpochDataset(fold_val, sequence_length=seq_len)
    test_ds = SequenceEpochDataset(test_epochs, sequence_length=seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        logger.warning(
            "Fold '%s' skipped: insufficient epochs for sequence_length=%d "
            "(train=%d, val=%d, test=%d sequences)",
            fold_name,
            seq_len,
            len(train_ds),
            len(val_ds),
            len(test_ds),
        )
        return {"fold": fold_name, "skipped": True}

    train_loader = DataLoader(train_ds, batch_size=train_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=train_config.batch_size)

    model = build_model(model_config, data_config).to(device)
    optimizer = AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=train_config.lr_factor, patience=train_config.lr_patience
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=train_config.mixed_precision and device.type == "cuda"
    )
    early_stopping = EarlyStopping(patience=train_config.early_stop_patience, mode="min")

    fold_dir = output_dir / fold_name
    run_logger = RunLogger(fold_dir)
    ckpt_path = fold_dir / "best.ckpt"

    start = time.time()
    epochs_run = 0
    for epoch in range(train_config.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            train_config,
            device,
            scaler if scaler.is_enabled() else None,
        )
        val_result = evaluate_epoch(
            model,
            val_loader,
            train_config,
            device,
            model_config.n_workload_classes,
            model_config.n_fatigue_classes,
        )
        scheduler.step(val_result.loss)
        run_logger.log_epoch(
            epoch,
            train_metrics,
            {
                "loss": val_result.loss,
                "workload_accuracy": val_result.workload_report.accuracy,
                "workload_macro_f1": val_result.workload_report.macro_f1,
            },
        )
        epochs_run = epoch + 1

        is_best = early_stopping.step(val_result.loss)
        if is_best:
            save_checkpoint(ckpt_path, model, optimizer, epoch, extra={"val_loss": val_result.loss})
        if early_stopping.should_stop:
            logger.info("Fold '%s': early stopping at epoch %d", fold_name, epoch)
            break

    if ckpt_path.exists():
        load_checkpoint(ckpt_path, model)
    test_result = evaluate_epoch(
        model,
        test_loader,
        train_config,
        device,
        model_config.n_workload_classes,
        model_config.n_fatigue_classes,
    )

    result = {
        "fold": fold_name,
        "skipped": False,
        "epochs_run": epochs_run,
        "wall_seconds": round(time.time() - start, 2),
        "test": {
            "loss": test_result.loss,
            "workload": test_result.workload_report.to_dict(),
            "fatigue": test_result.fatigue_report.to_dict(),
        },
    }
    run_logger.log_summary(result)
    return result


def run(config: AeroMindConfig, n_subjects: int, duration_s: float) -> dict:
    set_seed(config.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    epochs = load_epochs(
        config.data, n_subjects=n_subjects, duration_s=duration_s, seed=config.train.seed
    )
    logger.info("Loaded %d epochs, class balance: %s", len(epochs), class_balance(epochs))

    splits = get_protocol_splits(
        config.train.protocol, epochs, config.data, config.train.seed, n_subjects, duration_s
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.yaml")

    fold_results = []
    for split in splits:
        result = run_fold(
            config.model,
            config.train,
            config.data,
            split.train,
            split.test,
            split.name,
            output_dir,
            device,
        )
        fold_results.append(result)

    valid = [r for r in fold_results if not r["skipped"]]
    summary: dict = {
        "protocol": config.train.protocol,
        "model": config.model.name,
        "config_hash": config.config_hash(),
        "n_folds": len(splits),
        "n_folds_completed": len(valid),
        "folds": fold_results,
    }
    if valid:
        for key in ("accuracy", "macro_f1", "kappa"):
            values = [r["test"]["workload"][key] for r in valid]
            summary[f"workload_{key}_mean"] = float(np.mean(values))
            summary[f"workload_{key}_std"] = float(np.std(values))
            values = [r["test"]["fatigue"][key] for r in valid]
            summary[f"fatigue_{key}_mean"] = float(np.mean(values))
            summary[f"fatigue_{key}_std"] = float(np.std(values))

    RunLogger(output_dir).log_summary(summary)
    logger.info("Run complete: %s", {k: v for k, v in summary.items() if not k == "folds"})
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=None, help="Optional YAML config path (configs/*.yaml)")
    parser.add_argument(
        "--model",
        default="aeromind_capsnet",
        choices=["aeromind_capsnet", "aeromind_cnn_lstm", "aeromind_eegnet"],
    )
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic"])
    parser.add_argument(
        "--protocol",
        default="subject_dependent",
        choices=["subject_dependent", "loso", "cross_dataset"],
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence_length", type=int, default=15)
    parser.add_argument(
        "--n_subjects",
        type=int,
        default=8,
        help="Used only when generating synthetic data in-memory",
    )
    parser.add_argument(
        "--duration_s", type=float, default=300.0, help="Per-subject synthetic recording length"
    )
    parser.add_argument("--processed_dir", default="data/processed/synthetic")
    parser.add_argument("--output_dir", default="runs/default")
    return parser


def main() -> int:
    args = build_argparser().parse_args()

    if args.config:
        config = AeroMindConfig(
            data=DataConfig(
                dataset=args.dataset,
                processed_dir=args.processed_dir,
                sequence_length=args.sequence_length,
            ),
            model=ModelConfig(name=args.model),
            train=TrainConfig(
                protocol=args.protocol,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
            ),
            output_dir=args.output_dir,
        )
        from src.utils.config import load_config

        loaded = load_config(args.config)
        config.data, config.model, config.train = loaded.data, loaded.model, loaded.train
    else:
        config = AeroMindConfig(
            data=DataConfig(
                dataset=args.dataset,
                processed_dir=args.processed_dir,
                sequence_length=args.sequence_length,
            ),
            model=ModelConfig(name=args.model),
            train=TrainConfig(
                protocol=args.protocol,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
            ),
            output_dir=args.output_dir,
        )

    run(config, n_subjects=args.n_subjects, duration_s=args.duration_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
