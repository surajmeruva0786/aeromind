"""Offline replay sources (README §13): replay a continuous recording
sample-by-sample, optionally paced at real-time rate, feeding
`src.inference.stream.StreamingEngine`.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np

from src.data.synthetic import SyntheticSubjectRecord, generate_subject_record


def replay_array(
    data: np.ndarray, sfreq: float, realtime: bool = True, speed: float = 1.0
) -> Iterator[tuple[np.ndarray, float]]:
    """`data`: `(n_channels, n_samples)`. Yields `(sample, timestamp_seconds)`
    pairs one sample at a time. With `realtime=True`, sleeps between
    samples to pace playback at `sfreq / speed`; `realtime=False` (used for
    latency benchmarking and tests) yields as fast as the consumer allows.
    """
    n_samples = data.shape[1]
    dt = 1.0 / sfreq
    start = time.monotonic()
    for i in range(n_samples):
        t = i * dt
        if realtime:
            target = start + t / speed
            now = time.monotonic()
            if target > now:
                time.sleep(target - now)
        yield data[:, i].copy(), t


def replay_record(
    record: SyntheticSubjectRecord, realtime: bool = True, speed: float = 1.0
) -> Iterator[tuple[np.ndarray, float]]:
    yield from replay_array(record.data, record.sfreq, realtime=realtime, speed=speed)


def replay_synthetic(
    subject_id: int = 0,
    duration_s: float = 60.0,
    sfreq: float = 256.0,
    seed: int = 0,
    realtime: bool = True,
    speed: float = 1.0,
) -> Iterator[tuple[np.ndarray, float]]:
    """Generates a fresh synthetic subject recording and replays it — the
    zero-setup default replay source (works with no external data)."""
    record = generate_subject_record(
        subject_id=subject_id, duration_s=duration_s, sfreq=sfreq, seed=seed
    )
    yield from replay_record(record, realtime=realtime, speed=speed)


def replay_file(
    path: str, realtime: bool = True, speed: float = 1.0
) -> Iterator[tuple[np.ndarray, float]]:
    """Replays a real `.edf`/`.bdf`/`.fif` recording via MNE. Only
    exercised when a real file is supplied (README §13's offline replay
    mode) — not covered by unit tests, which use `replay_synthetic`."""
    import mne

    raw = mne.io.read_raw(path, preload=True, verbose=False)
    yield from replay_array(raw.get_data(), raw.info["sfreq"], realtime=realtime, speed=speed)
