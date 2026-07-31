# AeroMind Datasets

This project ships with tooling for three public EEG datasets, plus a
synthetic generator that requires no download at all.

## 0. Synthetic (no download, used by default in CI/tests/demo)

`src/data/synthetic.py` generates multi-channel EEG-like signals with a
graded workload signature (frontal-theta power increases, posterior-alpha
power decreases with workload level) and injectable eye-blink/muscle
artefacts. Every module in this repository (preprocessing, features,
models, training, XAI, the Streamlit app) is validated end-to-end against
this generator, so the full pipeline is runnable with zero external data
access. Generate a full on-disk synthetic dataset with:

```bash
python scripts/make_synthetic_dataset.py --output_dir data/processed/synthetic --n_subjects 8
```

## 1. MAUS (primary, open access)

- Source: Zenodo — https://zenodo.org/record/5085848 (Mental Arithmetic
  Tasks for Assessing Cognitive Workload)
- License: CC BY 4.0 — no registration required.
- Download: `python scripts/download_maus.py --output_dir data/raw/maus`

## 2. STEW (secondary)

- Source: IEEE DataPort — Simultaneous Task EEG Workload dataset.
- Access: free but requires an IEEE DataPort account to obtain a direct
  download URL. `scripts/download_stew.py` automates the unpack/organize
  step once you have downloaded the archive manually and provide its path
  with `--archive_path`.

## 3. DEAP (tertiary, fatigue subset)

- Source: https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
- Access: **requires manual registration** with the dataset owners (a
  signed EULA is emailed to you). This cannot be automated.
- `scripts/download_deap.py` does not download anything; it validates that
  you have placed the manually obtained files under `data/raw/deap/` in the
  expected layout and reports what is missing.

## Directory layout

```
data/
├── raw/<dataset>/       # untouched, as downloaded (gitignored)
├── processed/<dataset>/ # epoched .fif files produced by src/preprocessing (gitignored)
└── README.md
```

Raw and processed data are never committed to git (see `.gitignore`) —
they are large binary files and, for STEW/DEAP, redistribution is not
permitted by the original license terms.
