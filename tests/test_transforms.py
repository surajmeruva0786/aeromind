import numpy as np
import torch

from src.data.transforms import ChannelDropout, Compose, GaussianNoise, TimeShift, mixup_batch


def test_channel_dropout_zeros_a_channel():
    x = np.ones((7, 100))
    aug = ChannelDropout(p=1.0, seed=0)  # always trigger
    out = aug(x)
    zeroed_channels = np.where(~out.any(axis=1))[0]
    assert len(zeroed_channels) == 1


def test_channel_dropout_noop_when_p_zero():
    x = np.ones((7, 100))
    aug = ChannelDropout(p=0.0, seed=0)
    out = aug(x)
    np.testing.assert_array_equal(out, x)


def test_time_shift_preserves_shape_and_values():
    x = np.arange(7 * 100, dtype=float).reshape(7, 100)
    aug = TimeShift(max_shift_ms=50.0, sfreq=256.0, seed=0)
    out = aug(x)
    assert out.shape == x.shape
    # circular shift: same multiset of values per row
    np.testing.assert_array_equal(np.sort(out[0]), np.sort(x[0]))


def test_gaussian_noise_changes_signal():
    x = np.zeros((7, 100))
    aug = GaussianNoise(sigma=0.1, seed=0)
    out = aug(x)
    assert not np.allclose(out, x)
    assert out.std() < 1.0  # sigma=0.1 shouldn't blow up the scale


def test_compose_chains_transforms():
    x = np.ones((7, 100))
    aug = Compose([ChannelDropout(p=1.0, seed=0), GaussianNoise(sigma=0.01, seed=0)])
    out = aug(x)
    assert out.shape == x.shape


def test_mixup_batch_same_class_only():
    torch.manual_seed(0)
    x = torch.randn(8, 7, 100)
    workload = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    fatigue = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    x_mixed, w, f, w_perm, f_perm, lam = mixup_batch(x, workload, fatigue, alpha=0.2)
    assert x_mixed.shape == x.shape
    assert torch.equal(w, workload)
    assert torch.equal(w_perm, workload)  # permutation stays within same-class group
    assert 0.0 <= lam <= 1.0
