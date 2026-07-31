"""Synthetic multi-channel EEG generator.

Produces signals with a *known* ground-truth relationship between spectral
band power and cognitive workload/fatigue, mirroring the well-established
findings the rest of this repository is built around: frontal-midline theta
power rises with workload, posterior alpha power falls with attentional
demand, and fatigue is associated with a broadband slowing (delta/theta up,
beta down). This gives every downstream module (preprocessing, features,
models, XAI counter-factual probes) something real to work against without
requiring any external dataset or GPU.

Not intended to be a physiologically faithful EEG simulator — it is a
verifiable stand-in that makes the whole pipeline runnable and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt

# 7-channel frontal-centric montage, matching the MAUS-style layout described
# in the README (suitable for headband-style deployment).
CHANNEL_NAMES: tuple[str, ...] = ("Fp1", "Fp2", "Fz", "C3", "C4", "Pz", "Oz")

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

FRONTAL_CHANNELS = ("Fp1", "Fp2", "Fz")
POSTERIOR_CHANNELS = ("Pz", "Oz")

WORKLOAD_LABELS = ("low", "medium", "high")
FATIGUE_LABELS = ("alert", "fatigued")


@dataclass
class SyntheticEpoch:
    """One 2-second (by default) epoch of synthetic EEG plus its labels."""

    data: np.ndarray  # (n_channels, n_samples)
    workload: int  # 0=low, 1=medium, 2=high
    fatigue: int  # 0=alert, 1=fatigued
    subject_id: int


@dataclass
class SyntheticSubjectRecord:
    """A full continuous synthetic recording for one subject."""

    subject_id: int
    data: np.ndarray  # (n_channels, n_samples)
    sfreq: float
    channel_names: tuple[str, ...]
    workload_per_sample: np.ndarray  # int labels, len == n_samples
    fatigue_per_sample: np.ndarray
    blink_mask: np.ndarray = field(repr=False)  # bool, len == n_samples


def _band_limited_noise(
    n_samples: int, sfreq: float, low: float, high: float, rng: np.random.Generator
) -> np.ndarray:
    white = rng.standard_normal(n_samples)
    sos = butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, white)


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Approximate 1/f background via frequency-domain shaping of white noise."""
    white = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]  # avoid divide-by-zero at DC
    shaped = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(shaped, n=n_samples)
    return pink / (np.std(pink) + 1e-8)


def _band_power_multipliers(workload: int, fatigue: int) -> dict[str, float]:
    """Ground-truth band power scaling used to encode workload/fatigue.

    workload: 0 (low) .. 2 (high) increases theta and beta, decreases alpha.
    fatigue: 0 (alert) / 1 (fatigued) increases delta/theta, decreases beta.
    """
    mult = {band: 1.0 for band in BANDS}
    mult["theta"] *= 1.0 + 0.6 * workload
    mult["alpha"] *= 1.0 - 0.25 * workload
    mult["beta"] *= 1.0 + 0.2 * workload

    if fatigue == 1:
        mult["delta"] *= 1.4
        mult["theta"] *= 1.3
        mult["beta"] *= 0.7
    return mult


def _channel_gain(channel: str, band: str) -> float:
    """Spatial weighting so the workload/fatigue signature is topographically
    plausible (frontal theta, posterior alpha) rather than uniform across
    the scalp — this is what makes SHAP channel attribution meaningful."""
    if band == "theta" and channel in FRONTAL_CHANNELS:
        return 1.6
    if band == "alpha" and channel in POSTERIOR_CHANNELS:
        return 1.6
    return 1.0


def generate_subject_record(
    subject_id: int,
    duration_s: float = 300.0,
    sfreq: float = 256.0,
    channel_names: tuple[str, ...] = CHANNEL_NAMES,
    segment_s: float = 10.0,
    seed: int | None = None,
) -> SyntheticSubjectRecord:
    """Generate one continuous synthetic recording with a randomised, blocky
    workload/fatigue schedule (segments of `segment_s` seconds each), plus
    injected eye-blink artefacts on frontal channels.
    """
    rng = np.random.default_rng(seed if seed is not None else subject_id)

    n_samples = int(duration_s * sfreq)
    n_channels = len(channel_names)
    n_segments = int(np.ceil(duration_s / segment_s))
    seg_samples = int(segment_s * sfreq)

    data = np.zeros((n_channels, n_samples))
    workload_per_sample = np.zeros(n_samples, dtype=int)
    fatigue_per_sample = np.zeros(n_samples, dtype=int)
    blink_mask = np.zeros(n_samples, dtype=bool)

    # Fatigue drifts on: mostly alert early, increasing fatigue probability
    # later in the recording (mirrors a real time-on-task effect).
    for seg in range(n_segments):
        start = seg * seg_samples
        end = min(start + seg_samples, n_samples)
        length = end - start
        if length <= 0:
            continue

        workload = int(rng.integers(0, 3))
        fatigue_prob = 0.15 + 0.55 * (seg / max(n_segments - 1, 1))
        fatigue = int(rng.random() < fatigue_prob)

        workload_per_sample[start:end] = workload
        fatigue_per_sample[start:end] = fatigue
        mult = _band_power_multipliers(workload, fatigue)

        # Shared common-mode component across all channels, approximating the
        # volume-conduction correlation real EEG channels exhibit (without
        # this, independently-generated per-channel noise is implausibly
        # uncorrelated and trips up correlation-based bad-channel detection).
        common_mode = 0.5 * _pink_noise(length, rng)
        for band, (lo, hi) in BANDS.items():
            common_mode += 0.4 * mult[band] * _band_limited_noise(length, sfreq, lo, hi, rng)

        for ch_idx, ch_name in enumerate(channel_names):
            signal = common_mode + 0.3 * _pink_noise(length, rng)
            for band, (lo, hi) in BANDS.items():
                amp = mult[band] * _channel_gain(ch_name, band)
                signal += amp * _band_limited_noise(length, sfreq, lo, hi, rng)
            data[ch_idx, start:end] += signal

        # Occasional eye blinks on frontal channels: slow ~0.3s deflections.
        n_blinks = rng.poisson(segment_s / 8.0)
        for _ in range(n_blinks):
            blink_start = start + int(rng.uniform(0, max(length - int(0.3 * sfreq), 1)))
            blink_len = int(0.3 * sfreq)
            blink_end = min(blink_start + blink_len, n_samples)
            t = np.linspace(-1, 1, blink_end - blink_start)
            blink_wave = 8.0 * np.exp(-4 * t**2)
            for fch in FRONTAL_CHANNELS:
                if fch in channel_names:
                    idx = channel_names.index(fch)
                    data[idx, blink_start:blink_end] += blink_wave
            blink_mask[blink_start:blink_end] = True

    # Per-channel unit scaling to microvolt-like amplitudes.
    data *= 15.0

    return SyntheticSubjectRecord(
        subject_id=subject_id,
        data=data,
        sfreq=sfreq,
        channel_names=channel_names,
        workload_per_sample=workload_per_sample,
        fatigue_per_sample=fatigue_per_sample,
        blink_mask=blink_mask,
    )


def record_to_epochs(
    record: SyntheticSubjectRecord,
    epoch_seconds: float = 2.0,
    overlap: float = 0.5,
) -> list[SyntheticEpoch]:
    """Slice a continuous record into fixed-length epochs with a single
    workload/fatigue label per epoch (majority vote over samples)."""
    sfreq = record.sfreq
    epoch_len = int(epoch_seconds * sfreq)
    hop = int(epoch_len * (1 - overlap))
    hop = max(hop, 1)

    epochs: list[SyntheticEpoch] = []
    n_samples = record.data.shape[1]
    start = 0
    while start + epoch_len <= n_samples:
        end = start + epoch_len
        seg = record.data[:, start:end]
        workload = int(np.round(np.mean(record.workload_per_sample[start:end])))
        fatigue = int(np.round(np.mean(record.fatigue_per_sample[start:end])))
        epochs.append(
            SyntheticEpoch(
                data=seg.copy(),
                workload=min(workload, 2),
                fatigue=min(fatigue, 1),
                subject_id=record.subject_id,
            )
        )
        start += hop
    return epochs


def generate_dataset(
    n_subjects: int = 8,
    duration_s: float = 300.0,
    sfreq: float = 256.0,
    epoch_seconds: float = 2.0,
    overlap: float = 0.5,
    seed: int = 42,
) -> list[SyntheticEpoch]:
    """Generate a full multi-subject synthetic dataset already sliced into
    epochs — the fastest path for tests and CI."""
    all_epochs: list[SyntheticEpoch] = []
    for subj in range(n_subjects):
        record = generate_subject_record(
            subject_id=subj,
            duration_s=duration_s,
            sfreq=sfreq,
            seed=seed + subj,
        )
        all_epochs.extend(record_to_epochs(record, epoch_seconds=epoch_seconds, overlap=overlap))
    return all_epochs
