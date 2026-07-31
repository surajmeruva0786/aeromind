import mne
import numpy as np
import pytest

from src.preprocessing.epoching import epoch_continuous
from src.preprocessing.filters import apply_standard_filters, bandpass_filter, notch_filter
from src.preprocessing.ica_artefact import remove_artefacts
from src.preprocessing.prep_pipeline import (
    BadChannelReport,
    detect_bad_channels,
    interpolate_bad_channels,
    robust_average_reference,
)
from src.preprocessing.run import preprocess_raw, synthetic_raw

mne.set_log_level("ERROR")


@pytest.fixture
def raw_60s():
    return synthetic_raw(subject_id=0, duration_s=60.0, sfreq=256.0, seed=1)


def test_bandpass_filter_removes_out_of_band_energy():
    sfreq = 256.0
    t = np.arange(0, 5, 1 / sfreq)
    low_freq = np.sin(2 * np.pi * 0.1 * t)  # below 0.5 Hz cutoff
    in_band = np.sin(2 * np.pi * 10 * t)  # inside 0.5-45 Hz
    signal = (low_freq + in_band)[None, :]

    filtered = bandpass_filter(signal, sfreq, low=0.5, high=45.0)
    assert filtered.std() < signal.std()


def test_notch_filter_reduces_mains_line():
    sfreq = 256.0
    t = np.arange(0, 5, 1 / sfreq)
    line_noise = np.sin(2 * np.pi * 50 * t)[None, :]
    filtered = notch_filter(line_noise, sfreq, freq=50.0)
    assert filtered.std() < 0.5 * line_noise.std()


def test_apply_standard_filters_shape_preserved():
    data = np.random.randn(7, 256 * 5)
    out = apply_standard_filters(data, sfreq=256.0)
    assert out.shape == data.shape
    assert not np.isnan(out).any()


def test_detect_bad_channels_returns_report(raw_60s):
    report = detect_bad_channels(raw_60s)
    assert isinstance(report, BadChannelReport)
    assert isinstance(report.all_bad, list)


def test_detect_bad_channels_never_flags_all_channels(raw_60s):
    report = detect_bad_channels(raw_60s)
    assert len(report.all_bad) < len(raw_60s.ch_names)


def test_interpolate_bad_channels_noop_when_clean(raw_60s):
    empty_report = BadChannelReport(deviation_bad=[], correlation_bad=[])
    out = interpolate_bad_channels(raw_60s, empty_report)
    assert out is raw_60s


def test_robust_average_reference_zero_mean_across_channels(raw_60s):
    referenced = robust_average_reference(raw_60s)
    data = referenced.get_data()
    channel_mean_at_each_sample = data.mean(axis=0)
    assert np.allclose(channel_mean_at_each_sample, 0.0, atol=1e-10)


def test_remove_artefacts_returns_cleaned_raw_and_report(raw_60s):
    filtered_data = apply_standard_filters(raw_60s.get_data(), raw_60s.info["sfreq"])
    raw = mne.io.RawArray(filtered_data, raw_60s.info, verbose=False)
    raw = robust_average_reference(raw)
    cleaned, report = remove_artefacts(raw)
    assert cleaned.get_data().shape == raw.get_data().shape
    assert report.n_components > 0
    assert report.method in ("iclabel", "heuristic")


def test_epoch_continuous_rejects_high_amplitude_epochs():
    sfreq = 256.0
    info = mne.create_info(["Fz"], sfreq, "eeg")
    data = np.zeros((1, int(sfreq * 10)))
    # inject one epoch-worth of huge amplitude that should be rejected
    data[0, 256:512] = 1.0  # 1V = 1e6 uV, way above 200uV threshold
    raw = mne.io.RawArray(data, info, verbose=False)
    result = epoch_continuous(raw, epoch_seconds=2.0, overlap=0.0, reject_ptp_uv=200.0)
    assert result.n_rejected >= 1
    assert result.data.shape[0] == result.n_total - result.n_rejected


def test_preprocess_raw_end_to_end_on_synthetic(raw_60s):
    data = preprocess_raw(raw_60s, sfreq_target=256.0, window=2.0, overlap=0.5)
    assert data.ndim == 3  # (n_epochs, n_channels, n_samples)
    assert data.shape[1] == 7
    assert not np.isnan(data).any()
    assert not np.isinf(data).any()
