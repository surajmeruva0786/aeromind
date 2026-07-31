from src.data.dataset import EpochWindowDataset, SequenceEpochDataset
from src.data.splits import class_balance, cross_dataset_split, loso_splits, subject_dependent_split


def test_epoch_window_dataset(small_epoch_dataset):
    ds = EpochWindowDataset(small_epoch_dataset)
    assert len(ds) == len(small_epoch_dataset)
    x, workload, fatigue = ds[0]
    assert x.shape == small_epoch_dataset[0].data.shape
    assert workload.item() in (0, 1, 2)
    assert fatigue.item() in (0, 1)


def test_epoch_window_dataset_zscored(small_epoch_dataset):
    ds = EpochWindowDataset(small_epoch_dataset, normalize=True)
    x, _, _ = ds[0]
    # per-channel z-score -> ~zero mean, ~unit std per channel
    means = x.mean(dim=1)
    stds = x.std(dim=1)
    assert (means.abs() < 1e-3).all()
    assert ((stds - 1.0).abs() < 1e-2).all()


def test_sequence_epoch_dataset(multi_subject_epoch_dataset):
    seq_len = 3
    ds = SequenceEpochDataset(multi_subject_epoch_dataset, sequence_length=seq_len)
    assert len(ds) > 0
    x, workload, fatigue = ds[0]
    assert x.shape[0] == seq_len
    assert x.ndim == 3  # (L, C, T)


def test_subject_dependent_split(multi_subject_epoch_dataset):
    split = subject_dependent_split(multi_subject_epoch_dataset, test_fraction=0.2, seed=0)
    assert len(split.train) + len(split.test) == len(multi_subject_epoch_dataset)
    train_subjects = {ep.subject_id for ep in split.train}
    test_subjects = {ep.subject_id for ep in split.test}
    # subject-dependent: same subjects appear in both train and test
    assert train_subjects == test_subjects


def test_loso_splits_hold_out_disjoint_subjects(multi_subject_epoch_dataset):
    splits = loso_splits(multi_subject_epoch_dataset)
    subjects = {ep.subject_id for ep in multi_subject_epoch_dataset}
    assert len(splits) == len(subjects)
    for split in splits:
        train_subjects = {ep.subject_id for ep in split.train}
        test_subjects = {ep.subject_id for ep in split.test}
        assert train_subjects.isdisjoint(test_subjects)
        assert len(test_subjects) == 1


def test_cross_dataset_split(small_epoch_dataset, multi_subject_epoch_dataset):
    split = cross_dataset_split(small_epoch_dataset, multi_subject_epoch_dataset)
    assert split.train == small_epoch_dataset
    assert split.test == multi_subject_epoch_dataset


def test_class_balance(small_epoch_dataset):
    balance = class_balance(small_epoch_dataset)
    assert "workload" in balance and "fatigue" in balance
    assert sum(balance["workload"].values()) == len(small_epoch_dataset)
