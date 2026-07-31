#!/usr/bin/env python
"""Fetch the STEW EEG workload dataset (README §6, "Secondary").

STEW is a genuine 14-channel Emotiv-EPOC EEG dataset. Two acquisition paths
are supported:

1. **Automated** (default): downloads the MONSTER-repackaged, pre-epoched
   copy from the public Hugging Face dataset repo `monster-monash/STEW`
   (CC BY 4.0, no authentication required). This version is 128 Hz, 14
   channels, 2 s epochs, with binary high/low workload labels already
   attached — convenient but already-epoched, so it skips this repo's own
   preprocessing pipeline.
2. **Manual/raw**: the original raw STEW `.txt` per-subject recordings are
   distributed via IEEE DataPort and require a free account
   (https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset).
   Download the archive yourself and pass --archive_path to organize it.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

HF_REPO_ID = "monster-monash/STEW"


def download_from_huggingface(output_dir: str) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. Run `pip install huggingface_hub` "
            "or use --archive_path with a manually downloaded IEEE DataPort archive."
        )
        return 1

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from Hugging Face Hub (public, CC BY 4.0)...", HF_REPO_ID)
    try:
        local_path = snapshot_download(repo_id=HF_REPO_ID, repo_type="dataset", local_dir=str(out))
    except Exception as exc:  # network/HF errors — surface clearly, don't fabricate success
        logger.error("Download failed: %s", exc)
        logger.error(
            "Fall back to a manual raw download from IEEE DataPort and re-run with "
            "--archive_path."
        )
        return 1

    logger.info("STEW (MONSTER repack) downloaded to %s", local_path)
    return 0


def organize_manual_archive(archive_path: str, output_dir: str) -> int:
    archive = Path(archive_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        logger.error("Archive not found: %s", archive)
        return 1
    logger.info("Extracting %s -> %s", archive, out)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    else:
        shutil.unpack_archive(str(archive), str(out))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive_path",
        default=None,
        help="Path to a manually downloaded raw STEW archive from IEEE DataPort. "
        "If omitted, downloads the public Hugging Face repack instead.",
    )
    parser.add_argument("--output_dir", default="data/raw/stew")
    args = parser.parse_args()

    if args.archive_path:
        return organize_manual_archive(args.archive_path, args.output_dir)
    return download_from_huggingface(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
