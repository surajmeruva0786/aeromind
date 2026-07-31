"""CLI entrypoint: raw EEG -> cleaned, epoched, z-scored `.npz` files.

    python -m src.preprocessing.run --dataset synthetic --output_dir data/processed/synthetic_preprocessed
    python -m src.preprocessing.run --dataset edf --input_dir data/raw/stew --output_dir data/processed/stew

`--dataset synthetic` generates data in-process via `src/data/synthetic.py`
(wrapped as an MNE RawArray so it exercises the exact same filter / PREP /
ICA / epoching code path real recordings go through). `--dataset edf` reads
every `.edf`/`.bdf`/`.fif` file under `--input_dir` (one file per subject).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np

from src.data.synthetic import CHANNEL_NAMES, generate_subject_record
from src.preprocessing.epoching import epoch_continuous
from src.preprocessing.filters import apply_standard_filters
from src.preprocessing.ica_artefact import remove_artefacts
from src.preprocessing.prep_pipeline import detect_bad_channels, interpolate_bad_channels, robust_average_reference
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

mne.set_log_level("ERROR")


def _make_montage_info(channel_names: list[str], sfreq: float) -> mne.Info:
    info = mne.create_info(ch_names=list(channel_names), sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="warn")
    return info


def synthetic_raw(subject_id: int, duration_s: float, sfreq: float, seed: int) -> mne.io.RawArray:
    record = generate_subject_record(subject_id=subject_id, duration_s=duration_s, sfreq=sfreq, seed=seed)
    info = _make_montage_info(list(record.channel_names), sfreq)
    # synthetic amplitudes are microvolt-scale; MNE expects volts.
    raw = mne.io.RawArray(record.data * 1e-6, info, verbose=False)
    return raw


def load_edf_raw(path: Path) -> mne.io.BaseRaw:
    if path.suffix.lower() in (".edf",):
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    elif path.suffix.lower() in (".bdf",):
        raw = mne.io.read_raw_bdf(path, preload=True, verbose=False)
    elif path.suffix.lower() in (".fif",):
        raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported raw format: {path.suffix}")
    return raw


def preprocess_raw(
    raw: mne.io.BaseRaw,
    sfreq_target: float,
    window: float,
    overlap: float,
) -> np.ndarray:
    if abs(raw.info["sfreq"] - sfreq_target) > 1e-6:
        raw = raw.copy().resample(sfreq_target, verbose=False)

    filtered_data = apply_standard_filters(raw.get_data(), raw.info["sfreq"])
    raw = mne.io.RawArray(filtered_data, raw.info, verbose=False)

    bad_report = detect_bad_channels(raw)
    if bad_report.all_bad:
        logger.info("Interpolating bad channels: %s", bad_report.all_bad)
        raw = interpolate_bad_channels(raw, bad_report)
    raw = robust_average_reference(raw)

    cleaned, ica_report = remove_artefacts(raw)
    logger.info(
        "ICA (%s) excluded %d/%d components", ica_report.method, len(ica_report.excluded), ica_report.n_components
    )

    result = epoch_continuous(cleaned, epoch_seconds=window, overlap=overlap)
    logger.info("Kept %d/%d epochs (%d rejected)", result.n_total - result.n_rejected, result.n_total, result.n_rejected)
    return result.data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synthetic", "edf"], default="synthetic")
    parser.add_argument("--input_dir", default=None, help="Required for --dataset edf")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sfreq", type=float, default=256.0)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--n_subjects", type=int, default=8, help="Only used for --dataset synthetic")
    parser.add_argument("--duration_s", type=float, default=300.0, help="Only used for --dataset synthetic")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset == "synthetic":
        for subj in range(args.n_subjects):
            raw = synthetic_raw(subj, args.duration_s, args.sfreq, seed=args.seed + subj)
            data = preprocess_raw(raw, args.sfreq, args.window, args.overlap)
            np.savez_compressed(
                out / f"sub-{subj:02d}.npz",
                data=data,
                channel_names=np.array(list(CHANNEL_NAMES)),
                sfreq=args.sfreq,
            )
            logger.info("Wrote %s", out / f"sub-{subj:02d}.npz")
    else:
        if not args.input_dir:
            parser.error("--input_dir is required for --dataset edf")
        input_dir = Path(args.input_dir)
        raw_files = sorted(
            [p for p in input_dir.rglob("*") if p.suffix.lower() in (".edf", ".bdf", ".fif")]
        )
        if not raw_files:
            logger.error("No .edf/.bdf/.fif files found under %s", input_dir)
            return 1
        for i, path in enumerate(raw_files):
            raw = load_edf_raw(path)
            data = preprocess_raw(raw, args.sfreq, args.window, args.overlap)
            np.savez_compressed(
                out / f"sub-{i:02d}.npz",
                data=data,
                channel_names=np.array(raw.ch_names),
                sfreq=args.sfreq,
                source_file=str(path),
            )
            logger.info("Wrote %s <- %s", out / f"sub-{i:02d}.npz", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
