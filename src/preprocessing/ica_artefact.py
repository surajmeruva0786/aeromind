"""ICA-based artefact removal (README §7.3).

Uses extended Infomax ICA via MNE. Component classification prefers
`mne-icalabel`'s ICLabel model when it's installed; otherwise falls back to
simple heuristics (kurtosis for eye-blink-like transients, high-frequency
power ratio for muscle-like components) so the pipeline still runs without
the optional dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mne
import numpy as np
from scipy.stats import kurtosis

ICLABEL_REJECT_CATEGORIES = {
    "eye blink",
    "muscle artifact",
    "line noise",
    "heart beat",
    "channel noise",
}
ICLABEL_REJECT_PROB = 0.8


@dataclass
class ICAArtefactReport:
    n_components: int
    excluded: list[int]
    method: str
    labels: dict[int, str] = field(default_factory=dict)


def fit_ica(
    raw: mne.io.BaseRaw, n_components: int | None = None, random_state: int = 42
) -> mne.preprocessing.ICA:
    # Average referencing and/or bad-channel interpolation upstream (see
    # prep_pipeline.py) both reduce the true rank of the data below
    # n_channels; requesting more ICA components than that rank produces a
    # numerically unstable, near-singular mixing matrix. Use MNE's own rank
    # estimate as the cap.
    if n_components is None:
        try:
            rank = mne.compute_rank(raw, tol="auto", verbose=False)["eeg"]
        except Exception:
            rank = len(raw.ch_names) - 1
        n_components = max(1, min(rank, len(raw.ch_names) - 1))
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=random_state,
        max_iter="auto",
    )
    ica.fit(raw, verbose=False)
    return ica


def _classify_with_iclabel(
    raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA
) -> ICAArtefactReport | None:
    try:
        from mne_icalabel import label_components
    except ImportError:
        return None

    result = label_components(raw, ica, method="iclabel")
    labels = result["labels"]
    probs = result["y_pred_proba"]

    excluded = [
        i
        for i, (label, prob) in enumerate(zip(labels, probs))
        if label in ICLABEL_REJECT_CATEGORIES and prob >= ICLABEL_REJECT_PROB
    ]
    return ICAArtefactReport(
        n_components=ica.n_components_,
        excluded=excluded,
        method="iclabel",
        labels=dict(enumerate(labels)),
    )


def _classify_with_heuristics(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    kurtosis_thresh: float = 5.0,
    hf_power_ratio_thresh: float = 0.5,
) -> ICAArtefactReport:
    """Fallback when mne-icalabel is unavailable: flag components with
    heavy-tailed (blink-like) distributions or a high-frequency-dominated
    spectrum (muscle-like)."""
    sources = ica.get_sources(raw).get_data()  # (n_components, n_samples)
    sfreq = raw.info["sfreq"]

    excluded = []
    labels = {}
    for i, source in enumerate(sources):
        kurt = kurtosis(source)
        freqs = np.fft.rfftfreq(len(source), d=1.0 / sfreq)
        power = np.abs(np.fft.rfft(source)) ** 2
        hf_ratio = power[freqs > 30].sum() / (power.sum() + 1e-12)

        if kurt > kurtosis_thresh:
            excluded.append(i)
            labels[i] = "eye blink (heuristic)"
        elif hf_ratio > hf_power_ratio_thresh:
            excluded.append(i)
            labels[i] = "muscle (heuristic)"
        else:
            labels[i] = "brain (heuristic)"

    return ICAArtefactReport(
        n_components=ica.n_components_, excluded=excluded, method="heuristic", labels=labels
    )


def remove_artefacts(
    raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA | None = None
) -> tuple[mne.io.BaseRaw, ICAArtefactReport]:
    """Fit (if needed), classify, exclude flagged components, and return the
    cleaned Raw plus a report of what was removed and why."""
    ica = ica or fit_ica(raw)

    report = _classify_with_iclabel(raw, ica)
    if report is None:
        report = _classify_with_heuristics(raw, ica)

    ica.exclude = report.excluded
    cleaned = ica.apply(raw.copy(), verbose=False)
    return cleaned, report
