# Preprocessing pipeline

`python -m src.preprocessing.run` turns raw or synthetic continuous EEG into
cleaned, epoched, per-channel z-scored `.npz` files under `--output_dir`
(one file per subject: `data` of shape `(n_epochs, n_channels, n_samples)`).

```bash
# No download needed — generates synthetic subjects in-process
python -m src.preprocessing.run --dataset synthetic --output_dir data/processed/synthetic_preprocessed --n_subjects 8

# Real recordings: any .edf/.bdf/.fif files under --input_dir, one per subject
python -m src.preprocessing.run --dataset edf --input_dir data/raw/stew --output_dir data/processed/stew --sfreq 128 --window 2.0 --overlap 0.5
```

Pipeline stages (`preprocess_raw` in `run.py`), matching README §7:

1. Resample to `--sfreq` if needed.
2. Bandpass 0.5-45 Hz + 50 Hz notch (`filters.py`).
3. Bad-channel detection (deviation + correlation criteria) and spherical-
   spline interpolation, then robust average reference (`prep_pipeline.py`).
4. Extended Infomax ICA with ICLabel component classification — falls back
   to kurtosis/high-frequency heuristics if `mne-icalabel` isn't installed
   (`ica_artefact.py`). ICA component count is capped by the data's
   estimated rank (average referencing and interpolation both reduce it
   below `n_channels`) to avoid an unstable mixing matrix.
5. Fixed-length epoching with peak-to-peak amplitude rejection (default
   200 µV) and per-channel z-score (`epoching.py`).

See `tests/test_preprocessing.py` for a fully executed example, and
`notebooks/02_preprocessing_demo.ipynb` for a walkthrough with plots.
