# Feature engineering

Three views, matching README §8, each operating on one epoch `(n_channels, n_samples)`:

- `spectral.py` — Welch PSD band powers (delta/theta/alpha/beta/gamma), absolute/relative/log, per channel.
- `temporal.py` — mean/std/kurtosis/skew, Hjorth activity/mobility/complexity, line length, zero-crossing rate, per channel.
- `connectivity.py` — Phase Locking Value matrices in theta/alpha bands.

`pipeline.py` concatenates all three into one flat vector per epoch
(`extract_features`) with a deterministic, named column order
(`feature_names`) — this is what `src/evaluation/baseline.py`'s
RandomForest/SVM baseline consumes. The deep models (`src/models`) consume
raw z-scored epochs directly and do not use this flat vector.

See `notebooks/01_dataset_eda.ipynb` and `notebooks/03_feature_visualisation.ipynb`
for executed examples confirming the ground-truth workload signature
(frontal theta up, posterior alpha down) is recoverable from these features.
