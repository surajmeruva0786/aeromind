# AeroMind Datasets

This project ships with tooling for one real EEG dataset (STEW), one
non-EEG wearable dataset kept for future multimodal work (MAUS), one
gated EEG dataset (DEAP), and a synthetic generator that requires no
download at all and is what the test suite, CI, and demo app use by
default.

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

> **Correction (2026-08-01)**: earlier drafts of this project's docs
> described MAUS as an open-access, 7-channel EEG dataset on Zenodo. That
> was wrong. The real MAUS dataset provides **ECG/PPG/GSR, not EEG**, and is
> hosted on IEEE DataPort behind a free account, not Zenodo. It has been
> repositioned below accordingly — the real EEG dataset used by this
> repo's pipeline is **STEW**.

## 1. STEW (primary real-EEG dataset)

- Source: IEEE DataPort — "STEW: Simultaneous Task EEG Workload Dataset"
  (Lim, Sourina & Wang, 2018), 48 subjects, 14-channel Emotiv EPOC, 128 Hz,
  SIMKAP multitasking paradigm.
- Automated path (no account needed): `python scripts/download_stew.py`
  pulls the pre-epoched MONSTER repack from the public Hugging Face dataset
  `monster-monash/STEW` (CC BY 4.0). This version is already 128 Hz /
  14-channel / 2 s epochs with binary high-low labels, so it bypasses this
  repo's own preprocessing pipeline.
- Raw path (for the full pipeline, with continuous recordings): register at
  IEEE DataPort, download the archive manually, then run
  `python scripts/download_stew.py --archive_path <path/to/archive>`.

## 2. MAUS (reference only — not EEG, not wired into the pipeline)

- Source: IEEE DataPort — "MAUS: A Dataset for Mental Workload Assessment
  on N-back Task Using Wearable Sensor" (Beh et al., 2021). Provides
  ECG, fingertip-PPG, wrist-PPG, and GSR from 22 subjects — **no EEG
  channels**.
- Kept only as a candidate dataset for the HRV/PPG-fusion future work in
  README §22; `scripts/download_maus.py` validates a manually downloaded
  archive but does not feed `src/preprocessing`.

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
