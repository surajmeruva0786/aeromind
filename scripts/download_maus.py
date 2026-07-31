#!/usr/bin/env python
"""Organize the MAUS wearable physiological dataset.

CORRECTION vs. earlier drafts of this project's docs: MAUS ("A Dataset for
Mental Workload Assessment on N-back Task Using Wearable Sensor", Beh et al.
2021, https://arxiv.org/abs/2111.02561) provides **ECG, fingertip-PPG,
wrist-PPG, and GSR** signals from 22 subjects — it does *not* contain EEG.
It is hosted on IEEE DataPort (free account required to obtain a download
link), not on Zenodo, and is not anonymously downloadable.

Because AeroMind's preprocessing/model pipeline consumes EEG, MAUS is not
wired into that pipeline. It is kept in this repo only as a candidate
dataset for the HRV/PPG-fusion future work described in README §22. This
script validates a manually downloaded archive and reports what it found;
it does not (and cannot) download the data itself.

Real EEG data for this project comes from STEW (see `download_stew.py`) or
the built-in synthetic generator (`src/data/synthetic.py`, no download).
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

EXPECTED_MODALITIES = ("ecg", "ppg", "gsr")


def organize(archive_path: str | None, output_dir: str) -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if archive_path:
        archive = Path(archive_path)
        if not archive.exists():
            logger.error("Archive not found: %s", archive)
            return 1
        logger.info("Extracting %s -> %s", archive, out)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(out)
        else:
            shutil.unpack_archive(str(archive), str(out))

    found = [p for p in out.rglob("*") if p.is_file()]
    logger.info("Found %d files under %s", len(found), out)

    if not found:
        logger.warning(
            "No files present. MAUS requires a free IEEE DataPort account:\n"
            "  https://ieee-dataport.org/open-access/"
            "maus-dataset-mental-workload-assessment-n-back-task-using-wearable-sensor\n"
            "Download the archive manually, then re-run with --archive_path."
        )
        return 1

    logger.info(
        "MAUS is ECG/PPG/GSR, not EEG — not consumed by src/preprocessing. "
        "Retained for future HRV-fusion work (README §22)."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive_path", default=None, help="Manually downloaded MAUS archive")
    parser.add_argument("--output_dir", default="data/raw/maus")
    args = parser.parse_args()
    return organize(args.archive_path, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
