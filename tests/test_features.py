import numpy as np

from src.data.synthetic import BANDS, CHANNEL_NAMES, generate_dataset
from src.features.connectivity import connectivity_features, frontal_parietal_connectivity, plv_matrix
from src.features.pipeline import extract_features, feature_names
from src.features.spectral import band_powers, band_powers_matrix
from src.features.temporal import hjorth_parameters, temporal_features, zero_crossing_rate


def test_band_powers_shapes():
    epoch = np.random.randn(7, 512)
    powers = band_powers(epoch, sfreq=256.0)
    for band in BANDS:
        assert powers[f"{band}_abs"].shape == (7,)
        assert powers[f"{band}_rel"].shape == (7,)
        assert (powers[f"{band}_rel"] >= 0).all()
        assert (powers[f"{band}_rel"] <= 1.0 + 1e-6).all()


def test_band_powers_matrix_shape():
    epoch = np.random.randn(7, 512)
    matrix = band_powers_matrix(epoch, sfreq=256.0)
    assert matrix.shape == (7, len(BANDS))


def test_workload_increases_theta_power_via_feature_extractor():
    epochs = generate_dataset(n_subjects=3, duration_s=120.0, seed=5)
    fz_idx = CHANNEL_NAMES.index("Fz")

    low_theta = [band_powers(ep.data, 256.0)["theta_abs"][fz_idx] for ep in epochs if ep.workload == 0]
    high_theta = [band_powers(ep.data, 256.0)["theta_abs"][fz_idx] for ep in epochs if ep.workload == 2]
    assert np.mean(high_theta) > np.mean(low_theta)


def test_hjorth_parameters_shapes_and_sanity():
    epoch = np.random.randn(7, 512)
    activity, mobility, complexity = hjorth_parameters(epoch)
    assert activity.shape == mobility.shape == complexity.shape == (7,)
    assert (activity >= 0).all()


def test_zero_crossing_rate_bounds():
    epoch = np.sin(np.linspace(0, 20 * np.pi, 512))[None, :]
    zcr = zero_crossing_rate(epoch)
    assert 0.0 <= zcr[0] <= 1.0


def test_temporal_features_keys():
    epoch = np.random.randn(7, 512)
    feats = temporal_features(epoch)
    expected_keys = {
        "mean", "std", "kurtosis", "skewness", "hjorth_activity",
        "hjorth_mobility", "hjorth_complexity", "line_length", "zero_crossing_rate",
    }
    assert set(feats.keys()) == expected_keys
    assert all(v.shape == (7,) for v in feats.values())


def test_plv_matrix_symmetric_and_bounded():
    epoch = np.random.randn(7, 512)
    plv = plv_matrix(epoch, sfreq=256.0, band=(4.0, 8.0))
    assert plv.shape == (7, 7)
    np.testing.assert_allclose(plv, plv.T)
    assert np.allclose(np.diag(plv), 1.0)
    assert (plv >= 0).all() and (plv <= 1.0 + 1e-9).all()


def test_plv_identical_signal_is_perfectly_locked():
    t = np.linspace(0, 2, 512)
    signal = np.sin(2 * np.pi * 6 * t)
    epoch = np.stack([signal, signal], axis=0)
    plv = plv_matrix(epoch, sfreq=256.0, band=(4.0, 8.0))
    assert plv[0, 1] > 0.99


def test_connectivity_features_keys():
    epoch = np.random.randn(7, 512)
    feats = connectivity_features(epoch, sfreq=256.0)
    assert set(feats.keys()) == {"plv_theta", "plv_alpha"}


def test_frontal_parietal_connectivity_scalar():
    epoch = np.random.randn(7, 512)
    value = frontal_parietal_connectivity(epoch, sfreq=256.0, frontal_idx=[0, 1], parietal_idx=[5, 6])
    assert 0.0 <= value <= 1.0


def test_extract_features_deterministic_and_matches_names():
    epoch = np.random.randn(7, 512)
    x1 = extract_features(epoch, sfreq=256.0)
    x2 = extract_features(epoch, sfreq=256.0)
    np.testing.assert_array_equal(x1, x2)
    assert not np.isnan(x1).any()

    names = feature_names(n_channels=7)
    assert len(names) == len(x1)
