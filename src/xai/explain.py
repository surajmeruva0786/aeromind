"""XAI CLI (README §12, roadmap step 91): ties together SHAP channel
attribution, topographic rendering, spectral SHAP, and the counter-factual
probe for one trained checkpoint + subject.

Example:
    python -m src.xai.explain \\
        --checkpoint runs/synthetic_smoke_test_capsnet/subject_dependent/best.ckpt \\
        --n_subjects 8 --duration_s 180 --subject_id 0 \\
        --output_dir results/xai/sub-00
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SequenceEpochDataset
from src.data.synthetic import BANDS, CHANNEL_NAMES
from src.evaluation.evaluate import find_run_config
from src.models.registry import build_model
from src.training.train import load_epochs
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.xai.counter_factual import run_counter_factual_probe
from src.xai.shap_channel import compute_channel_attributions
from src.xai.spectral_attribution import (
    build_band_power_dataset,
    compute_spectral_attribution,
    fit_spectral_classifier,
)
from src.xai.topomap import plot_channel_topomap

logger = get_logger(__name__)


def run_explain(
    checkpoint_path: Path,
    config_path: Path | None,
    subject_id: int,
    n_subjects: int,
    duration_s: float,
    output_dir: Path,
    n_background: int = 8,
    n_explain: int = 4,
) -> dict:
    config = load_config(config_path or find_run_config(checkpoint_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = load_epochs(config.data, n_subjects=n_subjects, duration_s=duration_s, seed=config.train.seed)
    subject_epochs = [ep for ep in epochs if ep.subject_id == subject_id]
    if not subject_epochs:
        raise ValueError(f"No epochs found for subject_id={subject_id}")

    seq_ds = SequenceEpochDataset(subject_epochs, sequence_length=config.data.sequence_length)
    if len(seq_ds) < n_background + 1:
        raise ValueError(
            f"Subject {subject_id} has only {len(seq_ds)} sequences; need at least {n_background + 1}"
        )

    model = build_model(config.model, config.data).to(device)
    load_checkpoint(checkpoint_path, model, map_location=str(device))
    model.eval()

    loader = DataLoader(seq_ds, batch_size=len(seq_ds), shuffle=False)
    all_x, all_workload, _ = next(iter(loader))
    all_x = all_x.to(device)

    background = all_x[:n_background]
    explain_set = all_x[n_background : n_background + n_explain]

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. SHAP channel attribution + topomap ---------------------------
    attribution = compute_channel_attributions(model, background, explain_set, CHANNEL_NAMES)
    fig = plot_channel_topomap(
        attribution.channel_values[0],
        CHANNEL_NAMES,
        title=f"SHAP channel attribution (subj {subject_id}, predicted class {attribution.predicted_classes[0]})",
    )
    topomap_path = output_dir / "channel_attribution_topomap.png"
    fig.savefig(topomap_path, dpi=120)
    plt.close(fig)

    # --- 2. Spectral (band-power) SHAP -----------------------------------
    X, y = build_band_power_dataset(subject_epochs, sfreq=config.data.sfreq)
    spectral_result = None
    if len(set(y.tolist())) >= 2:
        clf = fit_spectral_classifier(X, y, seed=config.train.seed)
        spectral_result = compute_spectral_attribution(clf, X, n_channels=len(CHANNEL_NAMES))
        mean_abs = spectral_result.mean_abs_matrix(class_index=int(y[-1]))
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(mean_abs, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(spectral_result.band_names)))
        ax.set_xticklabels(spectral_result.band_names, rotation=45)
        ax.set_yticks(range(len(CHANNEL_NAMES)))
        ax.set_yticklabels(CHANNEL_NAMES)
        ax.set_title("Mean |SHAP| — spectral features")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        spectral_path = output_dir / "spectral_attribution.png"
        fig.savefig(spectral_path, dpi=120)
        plt.close(fig)
    else:
        logger.warning("Subject %d has only one workload class present — skipping spectral SHAP", subject_id)

    # --- 3. Counter-factual probe on the highest-workload sequence -------
    high_idx = int(all_workload.argmax())
    probe_sequence = all_x[high_idx : high_idx + 1]
    probe_result = run_counter_factual_probe(model, probe_sequence, sfreq=config.data.sfreq)

    summary = {
        "subject_id": subject_id,
        "checkpoint": str(checkpoint_path),
        "model": config.model.name,
        "n_sequences": len(seq_ds),
        "channel_attribution": {
            "channel_names": list(CHANNEL_NAMES),
            "values": attribution.channel_values.tolist(),
            "predicted_classes": attribution.predicted_classes.tolist(),
        },
        "counter_factual_probe": {
            "original_class": probe_result.original_class,
            "attenuated_class": probe_result.attenuated_class,
            "original_probs": probe_result.original_probs.tolist(),
            "attenuated_probs": probe_result.attenuated_probs.tolist(),
            "passed": probe_result.passed,
        },
    }
    (output_dir / "xai_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "XAI run complete for subject %d: counter-factual probe passed=%s (class %d -> %d)",
        subject_id, probe_result.passed, probe_result.original_class, probe_result.attenuated_class,
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--subject_id", type=int, default=0)
    parser.add_argument("--n_subjects", type=int, default=8)
    parser.add_argument("--duration_s", type=float, default=180.0)
    parser.add_argument("--output_dir", default="results/xai/sub-00")
    parser.add_argument("--n_background", type=int, default=8)
    parser.add_argument("--n_explain", type=int, default=4)
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    run_explain(
        Path(args.checkpoint),
        Path(args.config) if args.config else None,
        args.subject_id,
        args.n_subjects,
        args.duration_s,
        Path(args.output_dir),
        args.n_background,
        args.n_explain,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
