"""Topographic scalp map rendering for channel attributions (README §12.2),
via `mne.viz.plot_topomap` on the project's 7-channel frontal-centric
montage (a subset of the standard 10-20 system)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def build_montage_info(channel_names: tuple[str, ...], sfreq: float = 256.0) -> mne.Info:
    """Builds an `mne.Info` with 10-20 standard positions for the given
    channel names, required by `plot_topomap` to interpolate a scalp map."""
    info = mne.create_info(list(channel_names), sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)
    return info


def plot_channel_topomap(
    values: np.ndarray,
    channel_names: tuple[str, ...],
    title: str = "",
    ax: Axes | None = None,
    cmap: str = "RdBu_r",
) -> Figure:
    """`values`: (n_channels,) scalar per channel (e.g. a SHAP attribution
    score). Returns the containing `Figure` with a colorbar."""
    info = build_montage_info(channel_names)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure

    vlim = max(abs(values.min()), abs(values.max()), 1e-8)
    im, _ = mne.viz.plot_topomap(
        values, info, axes=ax, show=False, cmap=cmap, vlim=(-vlim, vlim)
    )
    ax.set_title(title, fontsize=9, wrap=True)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    return fig
