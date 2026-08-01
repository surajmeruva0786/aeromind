#!/usr/bin/env python
"""Validate a manually obtained DEAP dataset layout (README §6, "Tertiary").

DEAP requires a signed EULA emailed to the dataset owners
(https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) — there is no anonymous or
programmatic download, so this script never fetches anything. It only
checks that the expected per-subject `.dat` (Python pickle, `data_preprocessed_python/`
variant) files are present under `data/raw/deap/` and reports what's
missing, so the rest of the pipeline can fail fast with a clear message.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

EXPECTED_SUBJECTS = [f"s{i:02d}.dat" for i in range(1, 33)]  # 32 subjects


def validate(input_dir: str) -> int:
    root = Path(input_dir)
    if not root.exists():
        logger.error(
            "%s does not exist. Register and download DEAP manually from "
            "https://www.eecs.qmul.ac.uk/mmv/datasets/deap/ then place the "
            "`data_preprocessed_python` .dat files here.",
            root,
        )
        return 1

    present = {p.name for p in root.glob("*.dat")}
    missing = [s for s in EXPECTED_SUBJECTS if s not in present]

    logger.info(
        "Found %d/%d expected subject files in %s", len(present), len(EXPECTED_SUBJECTS), root
    )
    if missing:
        logger.warning("Missing: %s", ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
        return 1

    logger.info("All expected DEAP subject files present.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="data/raw/deap")
    args = parser.parse_args()
    return validate(args.input_dir)


if __name__ == "__main__":
    raise SystemExit(main())
