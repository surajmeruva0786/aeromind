import numpy as np

from src.data.synthetic import (
    CHANNEL_NAMES,
    generate_dataset,
    generate_subject_record,
    record_to_epochs,
)


def test_generate_subject_record_shape():
    rec = generate_subject_record(subject_id=0, duration_s=10.0, sfreq=256.0, seed=1)
    assert rec.data.shape == (len(CHANNEL_NAMES), int(10.0 * 256.0))
    assert not np.isnan(rec.data).any()
    assert not np.isinf(rec.data).any()


def test_record_to_epochs_shapes_and_overlap():
    rec = generate_subject_record(subject_id=0, duration_s=10.0, sfreq=256.0, seed=1)
    epochs = record_to_epochs(rec, epoch_seconds=2.0, overlap=0.5)
    expected_len = int(2.0 * 256.0)
    assert all(ep.data.shape == (len(CHANNEL_NAMES), expected_len) for ep in epochs)
    # 50% overlap over 10s with 2s windows -> hop 1s -> ~9 epochs
    assert 7 <= len(epochs) <= 10


def test_epoch_labels_within_range():
    epochs = generate_dataset(n_subjects=2, duration_s=30.0, seed=3)
    assert all(ep.workload in (0, 1, 2) for ep in epochs)
    assert all(ep.fatigue in (0, 1) for ep in epochs)


def test_deterministic_with_seed():
    a = generate_subject_record(subject_id=0, duration_s=5.0, seed=99)
    b = generate_subject_record(subject_id=0, duration_s=5.0, seed=99)
    np.testing.assert_array_equal(a.data, b.data)


def test_workload_increases_frontal_theta_power():
    """Ground-truth sanity check: high-workload epochs should show more
    frontal theta power than low-workload epochs, on average — this is the
    signature the whole XAI counter-factual probe (README §12.4) depends on.
    """
    from scipy.signal import welch

    epochs = generate_dataset(n_subjects=3, duration_s=120.0, seed=5)
    fz_idx = CHANNEL_NAMES.index("Fz")

    def theta_power(sig, sfreq=256.0):
        freqs, psd = welch(sig, fs=sfreq, nperseg=min(256, len(sig)))
        band = (freqs >= 4) & (freqs <= 8)
        return psd[band].mean()

    low = [theta_power(ep.data[fz_idx]) for ep in epochs if ep.workload == 0]
    high = [theta_power(ep.data[fz_idx]) for ep in epochs if ep.workload == 2]
    assert len(low) > 5 and len(high) > 5
    assert np.mean(high) > np.mean(low)
