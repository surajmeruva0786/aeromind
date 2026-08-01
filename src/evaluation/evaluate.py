"""Evaluation CLI (README §11, roadmap step 79): loads a trained checkpoint
and its run config, reconstructs the exact fold split used at training time
(same seed/protocol/data params -> deterministic), and reports held-out
test metrics via `src.evaluation.metrics_report`.

Example:
    python -m src.evaluation.evaluate \\
        --checkpoint runs/synthetic_smoke_test_capsnet/subject_dependent/best.ckpt \\
        --fold subject_dependent \\
        --n_subjects 8 --duration_s 180 \\
        --output_dir results/eval_capsnet_smoke_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SequenceEpochDataset
from src.evaluation.metrics_report import generate_report
from src.models.registry import build_model
from src.training.loop import EpochResult, evaluate_epoch
from src.training.train import get_protocol_splits, load_epochs
from src.utils.checkpoint import load_checkpoint
from src.utils.config import AeroMindConfig, load_config
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def find_run_config(checkpoint_path: Path) -> Path:
    """A checkpoint lives at `<output_dir>/<fold_name>/best.ckpt`; the run's
    `config.yaml` is one level up."""
    candidate = checkpoint_path.parent.parent / "config.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find config.yaml next to checkpoint at {candidate}. "
            "Pass --config explicitly if the run directory layout is non-standard."
        )
    return candidate


def evaluate_checkpoint(
    config: AeroMindConfig,
    checkpoint_path: Path,
    fold_name: str,
    n_subjects: int,
    duration_s: float,
    batch_size: int | None = None,
) -> tuple[EpochResult, EpochResult, dict]:
    """Rebuilds the deterministic fold split, loads the checkpoint, and
    returns `(test_result, train_side_result_placeholder, meta)`.

    `train_side_result_placeholder` is unused (kept for API symmetry) —
    only the held-out test split is evaluated, matching the training run's
    own final test-set evaluation exactly (same seed -> identical split).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = load_epochs(
        config.data, n_subjects=n_subjects, duration_s=duration_s, seed=config.train.seed
    )
    splits = get_protocol_splits(
        config.train.protocol, epochs, config.data, config.train.seed, n_subjects, duration_s
    )
    matching = [s for s in splits if s.name == fold_name]
    if not matching:
        available = [s.name for s in splits]
        raise ValueError(f"Fold '{fold_name}' not found among reconstructed splits: {available}")
    split = matching[0]

    test_ds = SequenceEpochDataset(split.test, sequence_length=config.data.sequence_length)
    if len(test_ds) == 0:
        raise ValueError(
            f"Fold '{fold_name}' has zero test sequences for sequence_length={config.data.sequence_length}"
        )

    loader = DataLoader(test_ds, batch_size=batch_size or config.train.batch_size)

    model = build_model(config.model, config.data).to(device)
    load_checkpoint(checkpoint_path, model, map_location=str(device))

    result = evaluate_epoch(
        model,
        loader,
        config.train,
        device,
        config.model.n_workload_classes,
        config.model.n_fatigue_classes,
    )

    meta = {
        "model": config.model.name,
        "protocol": config.train.protocol,
        "fold": fold_name,
        "checkpoint": str(checkpoint_path),
        "n_test_sequences": len(test_ds),
        "config_hash": config.config_hash(),
    }
    return result, meta


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a best.ckpt saved by src.training.train"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: inferred from checkpoint path)",
    )
    parser.add_argument(
        "--fold",
        default="subject_dependent",
        help="Fold/split name matching the checkpoint (e.g. loso_subject_3)",
    )
    parser.add_argument(
        "--n_subjects",
        type=int,
        default=8,
        help="Must match the training run's --n_subjects for a deterministic split rebuild",
    )
    parser.add_argument(
        "--duration_s", type=float, default=180.0, help="Must match the training run's --duration_s"
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output_dir", default="results/eval_run")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else find_run_config(checkpoint_path)
    config = load_config(config_path)

    result, meta = evaluate_checkpoint(
        config, checkpoint_path, args.fold, args.n_subjects, args.duration_s, args.batch_size
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_md = generate_report(result.workload_report, result.fatigue_report, meta)
    (output_dir / "evaluation_report.md").write_text(report_md, encoding="utf-8")

    metrics = {
        "meta": meta,
        "loss": result.loss,
        "workload": result.workload_report.to_dict(),
        "fatigue": result.fatigue_report.to_dict(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Evaluation complete: %s", {k: v for k, v in meta.items()})
    logger.info(
        "Workload accuracy=%.3f macro-F1=%.3f | Fatigue accuracy=%.3f macro-F1=%.3f",
        result.workload_report.accuracy,
        result.workload_report.macro_f1,
        result.fatigue_report.accuracy,
        result.fatigue_report.macro_f1,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
