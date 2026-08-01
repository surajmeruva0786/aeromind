"""Optional LSL (Lab Streaming Layer) live source (README §13). The
`pylsl` import is guarded: it requires both the Python package *and* the
native `liblsl` shared library, neither of which is available on most
dev/CI machines without EEG hardware attached — importing this module must
never fail the rest of the package.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

try:
    import pylsl

    LSL_AVAILABLE = True
except Exception:  # pragma: no cover - hardware/environment dependent
    pylsl = None
    LSL_AVAILABLE = False


def stream_lsl(
    stream_type: str = "EEG", timeout: float = 5.0
) -> Iterator[tuple[np.ndarray, float]]:
    """Yields `(sample, lsl_timestamp)` pairs from the first resolved LSL
    stream of the given type. Raises `RuntimeError` if `pylsl`/`liblsl`
    isn't usable in this environment, or if no matching stream is found
    within `timeout` seconds — callers should catch this and fall back to
    `src.inference.replay` (as the Streamlit app's `--source lsl` does).
    """
    if not LSL_AVAILABLE:
        raise RuntimeError(
            "pylsl is not usable in this environment (missing package or "
            "native liblsl library) — live LSL streaming is unavailable. "
            "Use --source replay instead."
        )
    streams = pylsl.resolve_byprop("type", stream_type, timeout=timeout)
    if not streams:
        raise RuntimeError(f"No LSL stream of type '{stream_type}' found within {timeout}s")

    inlet = pylsl.StreamInlet(streams[0])
    while True:
        sample, timestamp = inlet.pull_sample()
        yield np.asarray(sample, dtype=np.float64), float(timestamp)
