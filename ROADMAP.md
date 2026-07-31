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
- [ ] 29. `src/preprocessing/filters.py` — bandpass/notch
- [ ] 30. `src/preprocessing/prep_pipeline.py` — bad channel detection + interpolation
- [ ] 31. `src/preprocessing/ica_artefact.py` — ICA artefact rejection (ICLabel optional)
- [ ] 32. `src/preprocessing/epoching.py` — windowing + z-score + rejection
- [ ] 33. `src/preprocessing/run.py` — CLI entrypoint `python -m src.preprocessing.run`
- [ ] 34. Preprocessing config validation
- [ ] 35. Unit tests: filters
- [ ] 36. Unit tests: epoching
- [ ] 37. Unit tests: end-to-end preprocessing on synthetic raw
- [ ] 38. `notebooks/02_preprocessing_demo.ipynb`
- [ ] 39. Preprocessing CLI docs
- [ ] 40. Preprocessing performance pass (vectorization check)

## Phase 4 — Feature engineering (41-52)
- [ ] 41. `src/features/spectral.py` — Welch PSD, band powers
- [ ] 42. `src/features/temporal.py` — Hjorth, kurtosis, skew, line length, ZCR
- [ ] 43. `src/features/connectivity.py` — PLV theta/alpha
- [ ] 44. `src/features/pipeline.py` — feature concatenation for classical baseline
- [ ] 45. Unit tests: spectral
- [ ] 46. Unit tests: temporal
- [ ] 47. Unit tests: connectivity
- [ ] 48. `notebooks/01_dataset_eda.ipynb`
- [ ] 49. `notebooks/03_feature_visualisation.ipynb`
- [ ] 50. Classical baseline model (sklearn RandomForest/SVM on features) for sanity-check comparison
- [ ] 51. `src/evaluation/baseline.py`
- [ ] 52. Feature engineering docs

## Phase 5 — Model architectures (53-66)
- [ ] 53. `src/models/layers.py` — squash, PrimaryCapsule, DigitCapsule, dynamic routing
- [ ] 54. `src/models/aeromind_capsnet.py`
- [ ] 55. `src/models/aeromind_cnn_lstm.py`
- [ ] 56. `src/models/aeromind_eegnet.py`
- [ ] 57. `src/models/losses.py` — margin loss + multi-task weighted loss
- [ ] 58. `src/models/registry.py` — model factory from config name
- [ ] 59. Unit tests: capsule layers shapes
- [ ] 60. Unit tests: forward pass all 3 models on synthetic batch
- [ ] 61. Unit tests: loss functions
- [ ] 62. Parameter count assertions (~720k for CapsNet)
- [ ] 63. Gradient flow smoke test (backward pass, no NaNs)
- [ ] 64. Model architecture docs in `src/models/README.md`
- [ ] 65. `src/utils/checkpoint.py` — save/load checkpoints
- [ ] 66. `src/models/__init__.py` exports

## Phase 6 — Training pipeline (67-78)
- [ ] 67. `src/training/train.py` — CLI, AdamW, ReduceLROnPlateau
- [ ] 68. `src/training/loop.py` — train/val epoch loops, early stopping
- [ ] 69. `src/training/callbacks.py` — checkpointing, logging, early stop
- [ ] 70. Mixed precision support (fp16, CUDA-conditional)
- [ ] 71. LOSO training orchestration (`--protocol loso`)
- [ ] 72. Subject-dependent training orchestration
- [ ] 73. Cross-dataset training orchestration
- [ ] 74. Run-config hashing + logging (`runs/<name>/config.yaml`, `metrics.json`)
- [ ] 75. Unit tests: training loop runs 1 epoch on synthetic data, loss decreases sanity check
- [ ] 76. **Execute real smoke-test training run** on synthetic dataset (few epochs, log actual metrics)
- [ ] 77. Record measured smoke-test results in `results/synthetic_smoke_test.md`
- [ ] 78. Training docs + troubleshooting section

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
