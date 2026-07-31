#!/usr/bin/env python
"""Check that the runtime environment has what AeroMind needs.

Run this first after `pip install -r requirements.txt` to get a clear
pass/fail report instead of a mid-pipeline ImportError.
"""

from __future__ import annotations

import importlib
import sys

REQUIRED = [
    "numpy",
    "scipy",
    "sklearn",
    "pandas",
    "torch",
    "mne",
    "shap",
    "matplotlib",
    "yaml",
]

OPTIONAL = [
    "mne_icalabel",
    "streamlit",
    "pyedflib",
    "pylsl",
]


def _check(modules: list[str]) -> dict[str, str | None]:
    status: dict[str, str | None] = {}
    for mod in modules:
        try:
            m = importlib.import_module(mod)
            status[mod] = getattr(m, "__version__", "unknown version")
        except ImportError as exc:
            status[mod] = f"MISSING ({exc})"
    return status


def main() -> int:
    print("AeroMind environment check")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")

    ok = True
    print("\nRequired packages:")
    for mod, ver in _check(REQUIRED).items():
        marker = "OK " if not str(ver).startswith("MISSING") else "FAIL"
        if marker == "FAIL":
            ok = False
        print(f"  [{marker}] {mod:<15} {ver}")

    print("\nOptional packages (real-time / EDF / ICLabel):")
    for mod, ver in _check(OPTIONAL).items():
        marker = "OK  " if not str(ver).startswith("MISSING") else "skip"
        print(f"  [{marker}] {mod:<15} {ver}")

    try:
        import torch

        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("No CUDA device detected — training will run on CPU (slower).")
    except ImportError:
        pass

    print("\n" + ("PASS: all required packages present." if ok else "FAIL: missing required packages."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
