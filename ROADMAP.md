# AeroMind Build Roadmap

This file tracks the step-by-step build-out of AeroMind from the architecture
described in `README.md` to a working, deployable system. Each step is
implemented, committed, and pushed individually. Checked items are done.

> **Honesty note**: This project builds a fully functional pipeline validated
> end-to-end with a synthetic EEG generator (`src/data/synthetic.py`) so every
> module is runnable without GPU hardware or gated dataset access. Real
> dataset download scripts are wired up where the data is genuinely open
> (MAUS). Any accuracy numbers reported in this repo from an actual executed
> run are labeled "measured (synthetic smoke test)" or "measured (MAUS)" —
> the aspirational numbers in the README §18 are targets from literature, not
> claims about a specific checkpoint in this repo, and are labeled as such.

## Session handoff (2026-08-01, resumed and continuing through Phase 12)

**Done and pushed (commits `c51fc85`..`0b5bcea`, Phases 0-4, steps 1-52, plus
Phase 5 layers/capsnet):** project scaffolding, config system, the synthetic
EEG generator + dataset/transforms/splits, real dataset download scripts
(with a factual correction: MAUS is IEEE DataPort ECG/PPG/GSR, not open EEG
— STEW is now the primary real-EEG dataset, with an automated Hugging Face
download path), the full MNE-based preprocessing pipeline (filters,
PREP-style bad-channel handling, real ICLabel-backed ICA, epoching), and
spectral/temporal/connectivity feature extraction plus a classical
RandomForest baseline, and the CapsNet layers + AeroMind-CapsNet model. All
of this has passing tests and executed notebooks with real output. An
isolated `.venv` (see below) has all deps installed.

**Phase 5 completed this session:** `aeromind_cnn_lstm.py` (baseline 1),
`aeromind_eegnet.py` (baseline 2, from-scratch EEGNet-style encoder),
`losses.py` (margin loss + multi-task weighted sum, branches on whether a
model provides `capsule_lengths`/`reconstruction`), `registry.py`
(`build_model(model_config, data_config)`), `src/utils/checkpoint.py`,
`src/models/__init__.py` exports, and `tests/test_models.py` (14 tests:
capsule layer shapes, all-3-model forward passes, registry, parameter
counts, loss values, gradient-flow/no-NaN backward pass). See
`src/models/README.md` for measured parameter counts (capsnet ~460k,
cnn_lstm ~215k, eegnet ~44k).

**Phase 6 completed this session:** `src/training/loop.py`
(`train_one_epoch`/`evaluate_epoch`, works across all 3 models via the
shared forward interface), `callbacks.py` (`EarlyStopping`, `RunLogger`),
`train.py` (CLI: `--protocol subject_dependent|loso|cross_dataset`,
subject-aware validation carving so early stopping never sees test data,
AdamW + `ReduceLROnPlateau`, CUDA-conditional AMP), `tests/test_training.py`
(7 tests). Full suite: 61 tests passing. Executed a real 25-epoch smoke
test (`AeroMind-CapsNet`, subject-dependent protocol, 8 synthetic subjects,
556s on CPU) — results, including an honestly-reported workload-head
overfitting/majority-class-collapse finding, are in
`results/synthetic_smoke_test.md`.

**Environment note:** always use `Z:\aeromind\.venv\Scripts\python.exe`
(or activate `.venv`) — do NOT `pip install` into the global Python
interpreter. That was tried once this session, upgraded the global torch,
and broke version pins for unrelated packages (torchvision/torchaudio/
facenet-pytorch) already on this machine; it was reverted. `.venv` has
everything needed (torch, mne, mne-icalabel, shap, streamlit, nbformat,
nbconvert, ipykernel — kernel registered as `aeromind-venv`) already
installed via `requirements-dev.txt`.

## Phase 0 — Project scaffolding (1-8)
- [x] 1. Directory skeleton (`app/, configs/, data/, notebooks/, results/, scripts/, src/, tests/, .github/`)
- [x] 2. `.gitignore`
- [x] 3. `LICENSE` (MIT)
- [x] 4. `requirements.txt` + `requirements-dev.txt`
- [x] 5. `pyproject.toml` (packaging + tool config: black/ruff/pytest)
- [x] 6. `src/__init__.py` package skeleton for all subpackages
- [x] 7. `.editorconfig`, `CONTRIBUTING.md`
- [x] 8. `data/README.md` — dataset access instructions

## Phase 1 — Config & utils (9-16)
- [x] 9. `src/utils/seed.py` — reproducibility
- [x] 10. `src/utils/logging_utils.py`
- [x] 11. `src/utils/metrics.py` — accuracy/F1/kappa/ECE
- [x] 12. `src/utils/config.py` — YAML config loader + dataclasses
- [x] 13. `configs/aeromind_capsnet.yaml`
- [x] 14. `configs/aeromind_cnn_lstm.yaml`
- [x] 15. `configs/aeromind_eegnet.yaml`
- [x] 16. `scripts/verify_environment.py`

## Phase 2 — Data layer (17-28)
- [x] 17. `src/data/synthetic.py` — synthetic multi-channel EEG generator with graded workload signal
- [x] 18. `src/data/dataset.py` — PyTorch Dataset for epoched EEG + labels
- [x] 19. `src/data/transforms.py` — augmentations (channel dropout, time shift, noise, mixup)
- [x] 20. `src/data/splits.py` — subject-dependent / LOSO / cross-dataset splitters
- [x] 21. `scripts/download_maus.py` (corrected: MAUS is ECG/PPG/GSR, not EEG — reference-only)
- [x] 22. `scripts/download_stew.py` (real EEG dataset; automated HF path + manual IEEE DataPort path)
- [x] 23. `scripts/download_deap.py` (registration-gated, documents manual steps)
- [x] 24. `scripts/make_synthetic_dataset.py` — generate a full synthetic dataset on disk
- [x] 25. `data/README.md` finalize (corrected dataset facts, no checksums needed — datasets are gated/synthetic, not redistributed)
- [x] 26. Unit tests for synthetic generator
- [x] 27. Unit tests for dataset/splits
- [x] 28. Unit tests for transforms

## Phase 3 — Preprocessing pipeline (29-40)
- [x] 29. `src/preprocessing/filters.py` — bandpass/notch
- [x] 30. `src/preprocessing/prep_pipeline.py` — bad channel detection + interpolation
- [x] 31. `src/preprocessing/ica_artefact.py` — ICA artefact rejection (real ICLabel via mne-icalabel, heuristic fallback)
- [x] 32. `src/preprocessing/epoching.py` — windowing + z-score + rejection
- [x] 33. `src/preprocessing/run.py` — CLI entrypoint `python -m src.preprocessing.run`
- [x] 34. Preprocessing config validation (argparse choices + fail-fast checks in run.py)
- [x] 35. Unit tests: filters
- [x] 36. Unit tests: epoching
- [x] 37. Unit tests: end-to-end preprocessing on synthetic raw
- [x] 38. `notebooks/02_preprocessing_demo.ipynb` (executed, real output committed)
- [x] 39. Preprocessing CLI docs (`src/preprocessing/README.md`)
- [x] 40. Preprocessing performance pass (vectorized scipy sosfiltfilt; ICA rank capped to avoid unstable/slow fits)

## Phase 4 — Feature engineering (41-52)
- [x] 41. `src/features/spectral.py` — Welch PSD, band powers
- [x] 42. `src/features/temporal.py` — Hjorth, kurtosis, skew, line length, ZCR
- [x] 43. `src/features/connectivity.py` — PLV theta/alpha
- [x] 44. `src/features/pipeline.py` — feature concatenation for classical baseline
- [x] 45. Unit tests: spectral
- [x] 46. Unit tests: temporal
- [x] 47. Unit tests: connectivity
- [x] 48. `notebooks/01_dataset_eda.ipynb` (executed, real output committed)
- [x] 49. `notebooks/03_feature_visualisation.ipynb` (executed, real output committed)
- [x] 50. Classical baseline model (sklearn RandomForest/SVM on features) — measured >40% on 3-class synthetic (chance ~33%), see tests/test_baseline.py
- [x] 51. `src/evaluation/baseline.py`
- [x] 52. Feature engineering docs (`src/features/README.md`)

## Phase 5 — Model architectures (53-66)
- [x] 53. `src/models/layers.py` — squash, PrimaryCapsule, DigitCapsule, dynamic routing
- [x] 54. `src/models/aeromind_capsnet.py` (smoke-tested: forward pass runs, ~460k params — see session handoff note at top of this file for why that differs from README's ~720k figure)
- [x] 55. `src/models/aeromind_cnn_lstm.py`
- [x] 56. `src/models/aeromind_eegnet.py`
- [x] 57. `src/models/losses.py` — margin loss + multi-task weighted loss
- [x] 58. `src/models/registry.py` — model factory from config name
- [x] 59. Unit tests: capsule layers shapes
- [x] 60. Unit tests: forward pass all 3 models on synthetic batch
- [x] 61. Unit tests: loss functions
- [x] 62. Parameter count assertions (measured values, tolerance-based — see `src/models/README.md` for the ~720k README-target vs ~460k measured discrepancy explanation)
- [x] 63. Gradient flow smoke test (backward pass, no NaNs)
- [x] 64. Model architecture docs in `src/models/README.md`
- [x] 65. `src/utils/checkpoint.py` — save/load checkpoints
- [x] 66. `src/models/__init__.py` exports

## Phase 6 — Training pipeline (67-78)
- [x] 67. `src/training/train.py` — CLI, AdamW, ReduceLROnPlateau
- [x] 68. `src/training/loop.py` — train/val epoch loops, early stopping
- [x] 69. `src/training/callbacks.py` — checkpointing, logging, early stop
- [x] 70. Mixed precision support (fp16, CUDA-conditional — inert on this CPU-only dev machine, gated on `device.type == "cuda"`)
- [x] 71. LOSO training orchestration (`--protocol loso`)
- [x] 72. Subject-dependent training orchestration
- [x] 73. Cross-dataset training orchestration (synthetic-cohort proxy — see honesty note in `src/training/README.md`; real MAUS/STEW cross-dataset ingestion is not yet wired)
- [x] 74. Run-config hashing + logging (`runs/<name>/config.yaml`, `metrics.json`)
- [x] 75. Unit tests: training loop runs on synthetic data, loss decreases sanity check (`tests/test_training.py`, 7 tests)
- [x] 76. **Execute real smoke-test training run** on synthetic dataset (25 epochs, AeroMind-CapsNet, 556s wall-clock on CPU)
- [x] 77. Record measured smoke-test results in `results/synthetic_smoke_test.md`
- [x] 78. Training docs + troubleshooting section (`src/training/README.md`)

## Phase 7 — Evaluation (79-86)
- [ ] 79. `src/evaluation/evaluate.py` — CLI, loads checkpoint + dataset
- [ ] 80. `src/evaluation/metrics_report.py` — confusion matrix, ROC-AUC, calibration/ECE
- [ ] 81. Cross-dataset evaluation path (MAUS→STEW)
- [ ] 82. Unit tests: evaluate.py on synthetic checkpoint
- [ ] 83. `notebooks/04_xai_topographic_analysis.ipynb` stub wired to evaluate output
- [ ] 84. Evaluation report generator (markdown/HTML summary)
- [ ] 85. **Execute real evaluation** on smoke-test checkpoint, save `results/`
- [ ] 86. Evaluation docs

## Phase 8 — Explainability / XAI (87-95)
- [ ] 87. `src/xai/shap_channel.py` — GradientExplainer wrapper for multi-channel epochs
- [ ] 88. `src/xai/topomap.py` — MNE topographic scalp map rendering
- [ ] 89. `src/xai/spectral_attribution.py` — SHAP over band-power features
- [ ] 90. `src/xai/counter_factual.py` — frontal-theta attenuation probe
- [ ] 91. `src/xai/explain.py` — CLI entrypoint
- [ ] 92. Unit tests: SHAP wrapper runs on synthetic model + batch
- [ ] 93. Unit tests: counter-factual probe logic
- [ ] 94. **Execute real XAI run** on smoke-test checkpoint, save sample topomap PNG to `results/xai/`
- [ ] 95. XAI docs

## Phase 9 — Real-time inference (96-102)
- [ ] 96. `src/inference/stream.py` — sliding window engine, EWMA smoothing
- [ ] 97. `src/inference/replay.py` — offline `.edf`/synthetic replay source at real-time rate
- [ ] 98. `src/inference/lsl_source.py` — optional LSL live source (guarded import)
- [ ] 99. `src/inference/websocket_server.py` — downstream dashboard feed
- [ ] 100. Unit tests: streaming window buffer logic
- [ ] 101. Latency benchmark script + measured result on this machine (CPU)
- [ ] 102. Inference docs

## Phase 10 — Streamlit demo app (103-108)
- [ ] 103. `app/streamlit_app.py` — live EEG plot, probability bars, fatigue indicator
- [ ] 104. Scalp topomap panel wired to `src/xai`
- [ ] 105. Session CSV export
- [ ] 106. Replay-mode wiring to synthetic/sample data (works with zero setup)
- [ ] 107. App smoke test (headless import / syntax check)
- [ ] 108. App usage docs

## Phase 11 — Tests, CI, quality (109-114)
- [ ] 109. `pytest.ini` / test config, `tests/conftest.py` fixtures
- [ ] 110. Full test suite run — fix failures
- [ ] 111. `.github/workflows/ci.yml` — lint + test on push
- [ ] 112. `.github/workflows/docker.yml` — build check
- [ ] 113. Pre-commit config (ruff/black)
- [ ] 114. Coverage report + badge

## Phase 12 — Packaging & deployment (115-120)
- [ ] 115. `Dockerfile` (CPU) + `.dockerignore`
- [ ] 116. `docker-compose.yml` for local stack
- [ ] 117. Streamlit Community Cloud deployment config (`app` entrypoint, `runtime.txt`, secrets template)
- [ ] 118. `DEPLOYMENT.md` — deploy instructions (Docker, Streamlit Cloud, HF Spaces)
- [ ] 119. Final README sync (fill placeholders, link measured results, badges)
- [ ] 120. Tag release `v0.1.0`, final CHANGELOG entry

---
**Status legend**: unchecked = pending, this file is updated and committed as each step completes.
