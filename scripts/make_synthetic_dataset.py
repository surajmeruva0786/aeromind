#!/usr/bin/env python
"""Materialize a full synthetic EEG dataset on disk.

Writes one `.npz` per subject under `<output_dir>/sub-XX.npz` containing
epoched data + labels, plus an `index.csv` summary. This is the fixture
used by default across preprocessing demos, training smoke tests, and the
Streamlit replay mode — no external download required.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.data.synthetic import CHANNEL_NAMES, generate_subject_record, record_to_epochs
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/processed/synthetic")
    parser.add_argument("--n_subjects", type=int, default=8)
    parser.add_argument("--duration_s", type=float, default=300.0)
    parser.add_argument("--sfreq", type=float, default=256.0)
    parser.add_argument("--epoch_seconds", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for subj in range(args.n_subjects):
        record = generate_subject_record(
            subject_id=subj,
            duration_s=args.duration_s,
            sfreq=args.sfreq,
            seed=args.seed + subj,
        )
        epochs = record_to_epochs(record, epoch_seconds=args.epoch_seconds, overlap=args.overlap)

        data = np.stack([ep.data for ep in epochs], axis=0)  # (n_epochs, C, T)
        workload = np.array([ep.workload for ep in epochs], dtype=np.int64)
        fatigue = np.array([ep.fatigue for ep in epochs], dtype=np.int64)

        subj_path = out / f"sub-{subj:02d}.npz"
        np.savez_compressed(
            subj_path,
            data=data,
            workload=workload,
            fatigue=fatigue,
            channel_names=np.array(CHANNEL_NAMES),
            sfreq=args.sfreq,
        )
        index_rows.append(
            {
                "subject": f"sub-{subj:02d}",
                "n_epochs": len(epochs),
                "workload_counts": np.bincount(workload, minlength=3).tolist(),
                "fatigue_counts": np.bincount(fatigue, minlength=2).tolist(),
            }
        )
        logger.info("Wrote %s (%d epochs)", subj_path, len(epochs))

    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["subject", "n_epochs", "workload_counts", "fatigue_counts"]
        )
        writer.writeheader()
        writer.writerows(index_rows)

    logger.info("Synthetic dataset complete: %d subjects -> %s", args.n_subjects, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
