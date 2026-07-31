"""Lightweight structured logging shared by CLIs and the Streamlit app."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Configuration only happens once per process so repeated calls (e.g. one
    per module import) don't attach duplicate handlers.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
