from src.data.splits import subject_dependent_split
from src.data.synthetic import generate_dataset
from src.evaluation.baseline import build_feature_dataset, train_and_evaluate_baseline


def test_baseline_trains_and_evaluates_above_chance():
    epochs = generate_dataset(n_subjects=4, duration_s=180.0, seed=21)
    split = subject_dependent_split(epochs, test_fraction=0.2, seed=0)

    train_ds = build_feature_dataset(split.train)
    test_ds = build_feature_dataset(split.test)

    report = train_and_evaluate_baseline(train_ds, test_ds, target="workload", kind="random_forest")
    # 3-class chance level is ~0.33; the synthetic generator's ground-truth
    # theta/alpha signature should let a RandomForest clear that easily.
    assert report.accuracy > 0.4
    assert report.confusion.shape == (3, 3)
