"""Simplified PREP-style bad-channel detection + interpolation (README §7.2, §7.4).

A full PREP implementation (Bigdely-Shamlo et al., 2015) includes robust
referencing via a RANSAC-based bad channel search. This module implements
the two criteria that matter most for short, sparse-montage recordings like
the ones used here — deviation and correlation — then hands off to MNE's
spherical-spline interpolation, which is the same mechanism PREP itself
uses to fix flagged channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np


@dataclass
class BadChannelReport:
    deviation_bad: list[str]
    correlation_bad: list[str]

    @property
    def all_bad(self) -> list[str]:
        return sorted(set(self.deviation_bad) | set(self.correlation_bad))


def detect_bad_channels(
    raw: mne.io.BaseRaw,
    deviation_z_thresh: float = 5.0,
    correlation_thresh: float = 0.15,
    max_bad_fraction: float = 0.4,
) -> BadChannelReport:
    """Flag channels that are outliers in overall amplitude (deviation) or
    that fail to correlate with their neighbours (correlation) — the two
    core PREP bad-channel criteria.

    `max_bad_fraction` caps how much of the montage can be flagged: with a
    sparse 7-channel headband montage, naive per-channel statistics are
    fragile, and interpolating "bad" channels when there aren't enough good
    ones left to interpolate *from* is worse than doing nothing. If the
    naive criteria would flag more than this fraction, only the single
    worst offender per criterion is kept.
    """
    data = raw.get_data()  # (n_channels, n_samples)
    channel_names = raw.ch_names
    n_channels = len(channel_names)

    # Deviation criterion: robust z-score of per-channel standard deviation.
    stds = data.std(axis=1)
    median = np.median(stds)
    mad = np.median(np.abs(stds - median)) + 1e-12
    robust_z = 0.6745 * (stds - median) / mad
    deviation_bad = [
        channel_names[i] for i in range(n_channels) if abs(robust_z[i]) > deviation_z_thresh
    ]

    # Correlation criterion: a channel that is weakly correlated with the
    # average of all other channels is likely disconnected/noisy.
    correlation_bad = []
    mean_abs_corr = np.ones(n_channels)
    if n_channels > 2:
        corr_matrix = np.corrcoef(data)
        for i in range(n_channels):
            others = np.delete(corr_matrix[i], i)
            mean_abs_corr[i] = np.nanmean(np.abs(others))
            if mean_abs_corr[i] < correlation_thresh:
                correlation_bad.append(channel_names[i])

    max_bad = max(1, int(n_channels * max_bad_fraction))
    if len(deviation_bad) > max_bad:
        worst = np.argsort(-np.abs(robust_z))[:max_bad]
        deviation_bad = [channel_names[i] for i in worst]
    if len(correlation_bad) > max_bad:
        worst = np.argsort(mean_abs_corr)[:max_bad]
        correlation_bad = [channel_names[i] for i in worst]

    return BadChannelReport(deviation_bad=deviation_bad, correlation_bad=correlation_bad)


def interpolate_bad_channels(raw: mne.io.BaseRaw, report: BadChannelReport) -> mne.io.BaseRaw:
    """Mark flagged channels bad and spherical-spline interpolate them from
    the rest of the montage. No-op if nothing was flagged, or if every
    channel was flagged (nothing good left to interpolate from — in that
    case the recording should be rejected outright, not silently patched).
    """
    if not report.all_bad:
        return raw
    if len(report.all_bad) >= len(raw.ch_names):
        return raw
    raw = raw.copy()
    raw.info["bads"] = list(set(raw.info["bads"]) | set(report.all_bad))
    raw.interpolate_bads(reset_bads=True, verbose=False)
    return raw


def robust_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Compute average reference over "good" (non-bad) channels, matching
    the PREP philosophy of referencing before final bad-channel handling
    isn't applied to already-flagged noise."""
    raw = raw.copy()
    raw.set_eeg_reference(ref_channels="average", verbose=False)
    return raw
