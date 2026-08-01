# AeroMind Build Roadmap

**Status: COMPLETE — all 120 steps done, tagged `v0.1.0`.**

This file tracks the step-by-step build-out of AeroMind from the architecture
described in `README.md` to a working, deployable system. Each step is
implemented, committed, and pushed individually. Checked items are done.

> **Honesty note**: This project builds a fully functional pipeline validated
> end-to-end with a synthetic EEG generator (`src/data/synthetic.py`) so every
> module is runnable without GPU hardware or gated dataset access. Real
> dataset download scripts are wired up where the data is genuinely open
> (STEW, via an automated Hugging Face path — MAUS turned out to be IEEE
> DataPort ECG/PPG/GSR, not open EEG, and is reference-only; see the Phase
> 0-4 session handoff below for that correction). Any accuracy numbers
> reported in this repo from an actual executed run are labeled "measured
> (synthetic smoke test)" — the aspirational numbers in README §18 are
> targets from literature, not claims about a specific checkpoint in this
> repo, and are labeled as such. Docker/deployment artifacts carry their
> own honesty notes where a step couldn't be fully verified in this
> session's environment — see Phase 12 below.

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
(7 tests). Executed a real 25-epoch smoke test (`AeroMind-CapsNet`,
subject-dependent protocol, 8 synthetic subjects, 556s on CPU) — results,
including an honestly-reported workload-head overfitting/
majority-class-collapse finding, are in `results/synthetic_smoke_test.md`.

**Phase 7 completed this session:** `src/evaluation/evaluate.py` (CLI:
deterministically reconstructs a training run's fold split from its saved
`config.yaml` + seed, loads a checkpoint, evaluates the held-out test
split), `metrics_report.py` (markdown report: confusion matrix, per-class
F1, ROC-AUC, ECE), `tests/test_evaluation.py` (4 tests, including an exact
reproduction check against `src.training.train`'s own self-reported test
metrics), `notebooks/04_xai_topographic_analysis.ipynb` (executed stub
that trains+evaluates a checkpoint and plots its confusion matrix — full
SHAP/topomap rendering lands in Phase 8), and `src/evaluation/README.md`.
Ran the CLI for real against the Phase 6 smoke-test checkpoint —
`results/eval_capsnet_smoke_test/` — and confirmed its numbers exactly
match the training run's own reported test metrics.

**Phase 8 completed this session:** `src/xai/shap_channel.py`
(`shap.GradientExplainer` wrapper, per-channel attribution for each
input's own predicted class), `topomap.py` (`mne.viz.plot_topomap` on the
7-channel `standard_1020` montage), `spectral_attribution.py`
(`shap.TreeExplainer` over a RandomForest trained on flattened band-power
features — a class x (channel, band) matrix), `counter_factual.py`
(frontal-theta attenuation probe), `explain.py` (CLI tying all of the
above together), `tests/test_xai.py` (6 tests), and `src/xai/README.md`.
Ran the CLI for real against the Phase 6 smoke-test checkpoint
(`results/xai/sub-00/`, `results/xai/README.md`) — the counter-factual
probe reported `passed: false`, which is expected and honestly documented
given that checkpoint's known workload-head majority-class collapse
(`results/synthetic_smoke_test.md`); the classical spectral-SHAP path
(a separate model, unaffected by the deep checkpoint's training quality)
did recover the expected theta-dominant signature.

**Phase 9 completed this session:** `src/inference/stream.py`
(`SlidingWindowBuffer`, `EWMASmoother`, `StreamingEngine` tying both to a
loaded model), `replay.py` (synthetic + real-file offline replay, paced or
fast), `lsl_source.py` (guarded `pylsl` import, clear `RuntimeError` when
unavailable), `websocket_server.py` (`PredictionBroadcastServer` on the
`websockets` library), `tests/test_inference.py` (8 tests, including a
real WebSocket round-trip), and `src/inference/README.md`. Ran
`scripts/benchmark_latency.py` for real on this CPU-only machine — all
three models comfortably clear the 500ms (0.5s hop) real-time budget:
CapsNet p95 16ms, CNN-LSTM p95 11ms, EEGNet p95 7ms
(`results/latency_benchmark.json`).

**Phase 10 completed this session:** `app/streamlit_app.py` — a live EEG
plot, EWMA-smoothed workload/fatigue panels, an on-demand SHAP topomap
button, and CSV session export, driven by `StreamingEngine` +
`src.inference.replay`. Zero-setup by default (synthetic source); with no
checkpoint given it runs on a freshly initialized model and shows a
persistent "untrained model" warning rather than silently presenting
demo output as real. All `st.*` calls are inside `main()` so `import
app.streamlit_app` never touches the Streamlit runtime.
`tests/test_app.py` (5 tests) uses `streamlit.testing.v1.AppTest` to
actually render the app and click controls headlessly; also manually
verified with a real `streamlit run` server. `app/README.md` documents
usage.

**Phase 11 completed this session:** normalized formatting across the
whole codebase (`ruff --fix` + `black`, both were already configured in
`pyproject.toml` since Phase 0 but hadn't been enforced), added
`tests/test_lsl_source.py` (guarded-import failure path) and
`tests/test_xai.py::test_run_explain_end_to_end` (full `explain.py`
CLI integration test) to close coverage gaps, `.github/workflows/ci.yml`
(ruff + black + pytest-cov on every push/PR), `.github/workflows/docker.yml`
(build-health check, no-ops until Phase 12 adds a Dockerfile),
`.pre-commit-config.yaml` (ruff/black + standard hygiene hooks, verified
via `pre-commit run --all-files`), and a measured 86% coverage badge in
`README.md`. `CONTRIBUTING.md` documents all of this. Full suite: **86
tests passing**.

*Note on test flakiness*: one intermittent failure in
`tests/test_training.py` was observed during this phase's full-suite run
(pre-formatting). Both affected tests use fully seeded, independent RNGs
(`np.random.default_rng`, not global state) for every random decision in
their code path, so the failure was not reproducible despite ~10 repeated
full-suite and targeted re-runs afterward. Logged here rather than
silently ignored — if it recurs, the fully-seeded/deterministic design
argument above should be re-examined, not assumed to still hold.

**Phase 12 completed this session — project complete (all 120 steps):**
`Dockerfile` (CPU-only, explicit `--index-url .../whl/cpu` torch install)
+ `.dockerignore`, `docker-compose.yml`, Streamlit Community Cloud config
(`runtime.txt`, `.streamlit/config.toml`, `.streamlit/secrets.toml.example`
— the app needs no secrets in its default mode), `DEPLOYMENT.md` (Docker /
Streamlit Community Cloud / Hugging Face Spaces, each with an honest
resource/verification note), a full README sync (real repo URL and
contact info in place of placeholders, §16 usage examples corrected to
match actual implemented CLI flags rather than aspirational ones, §18
Results links to every `results/` measured-output doc, Docker install
path added, Project Structure section brought up to date), and
`CHANGELOG.md`. Tagged `v0.1.0`.

**Honesty note carried into the release**: the Dockerfile was not
build-verified locally (no reachable Docker daemon in this session's
environment) — `.github/workflows/docker.yml`'s first run against it on
GitHub Actions is its real first build check; verify that run before
depending on the image for anything beyond casual local use. Everything
else in this repo (86 tests, every CLI, the Streamlit app, the training/
evaluation/XAI/inference pipelines) was executed for real at least once
during this build, with measured results recorded under `results/` and
distinguished throughout from the literature-target numbers in
`README.md` §18.

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
- [x] 79. `src/evaluation/evaluate.py` — CLI, loads checkpoint + dataset
- [x] 80. `src/evaluation/metrics_report.py` — confusion matrix, ROC-AUC, calibration/ECE
- [x] 81. Cross-dataset evaluation path (synthetic-cohort proxy — see honesty note in `src/evaluation/README.md`; real MAUS→STEW not yet wired)
- [x] 82. Unit tests: evaluate.py on synthetic checkpoint (`tests/test_evaluation.py`, 4 tests — verifies bit-for-bit reproduction of the training run's own test metrics)
- [x] 83. `notebooks/04_xai_topographic_analysis.ipynb` stub wired to evaluate output (executed, real output committed)
- [x] 84. Evaluation report generator (markdown summary) — `generate_report()` in `metrics_report.py`
- [x] 85. **Execute real evaluation** on smoke-test checkpoint, save `results/eval_capsnet_smoke_test/`
- [x] 86. Evaluation docs (`src/evaluation/README.md`)

## Phase 8 — Explainability / XAI (87-95)
- [x] 87. `src/xai/shap_channel.py` — GradientExplainer wrapper for multi-channel epochs
- [x] 88. `src/xai/topomap.py` — MNE topographic scalp map rendering
- [x] 89. `src/xai/spectral_attribution.py` — SHAP over band-power features
- [x] 90. `src/xai/counter_factual.py` — frontal-theta attenuation probe
- [x] 91. `src/xai/explain.py` — CLI entrypoint
- [x] 92. Unit tests: SHAP wrapper runs on synthetic model + batch
- [x] 93. Unit tests: counter-factual probe logic
- [x] 94. **Execute real XAI run** on smoke-test checkpoint, save sample topomap PNG to `results/xai/` (PNGs gitignored per `results/**/*.png`, regenerate via the command in `results/xai/README.md`; `xai_summary.json` and `README.md` are tracked)
- [x] 95. XAI docs (`src/xai/README.md`)

## Phase 9 — Real-time inference (96-102)
- [x] 96. `src/inference/stream.py` — sliding window engine, EWMA smoothing
- [x] 97. `src/inference/replay.py` — offline `.edf`/synthetic replay source at real-time rate
- [x] 98. `src/inference/lsl_source.py` — optional LSL live source (guarded import)
- [x] 99. `src/inference/websocket_server.py` — downstream dashboard feed
- [x] 100. Unit tests: streaming window buffer logic (`tests/test_inference.py`, 8 tests)
- [x] 101. Latency benchmark script + measured result on this machine (CPU) — `scripts/benchmark_latency.py`, `results/latency_benchmark.json` (all 3 models well under the 500ms hop budget)
- [x] 102. Inference docs (`src/inference/README.md`)

## Phase 10 — Streamlit demo app (103-108)
- [x] 103. `app/streamlit_app.py` — live EEG plot, probability bars, fatigue indicator
- [x] 104. Scalp topomap panel wired to `src/xai` (on-demand button, not per-frame — `shap.GradientExplainer` is too expensive to run every refresh tick)
- [x] 105. Session CSV export
- [x] 106. Replay-mode wiring to synthetic/sample data (works with zero setup)
- [x] 107. App smoke test (headless import / syntax check) — `tests/test_app.py` (5 tests, using `streamlit.testing.v1.AppTest`); also manually verified with a real `streamlit run` server (HTTP 200)
- [x] 108. App usage docs (`app/README.md`)

## Phase 11 — Tests, CI, quality (109-114)
- [x] 109. `pytest.ini` / test config, `tests/conftest.py` fixtures (test config lives in `pyproject.toml`'s `[tool.pytest.ini_options]`, already in place since Phase 0; fixtures already in `tests/conftest.py` since Phase 2 — this step formalizes/documents both, see `CONTRIBUTING.md`)
- [x] 110. Full test suite run — fix failures (ran `ruff --fix` + `black` across `src/app/tests/scripts`, normalizing formatting that had drifted since earlier phases; one intermittent, non-reproducible failure was investigated across ~10 repeat runs and could not be reproduced with a root cause — see note below)
- [x] 111. `.github/workflows/ci.yml` — lint + test on push
- [x] 112. `.github/workflows/docker.yml` — build check (gracefully no-ops if `Dockerfile` isn't present yet — see Phase 12)
- [x] 113. Pre-commit config (ruff/black) — `.pre-commit-config.yaml`, verified clean via `pre-commit run --all-files`
- [x] 114. Coverage report + badge — measured 86% (`pytest --cov=src --cov=app`), static badge in `README.md`

## Phase 12 — Packaging & deployment (115-120)
- [x] 115. `Dockerfile` (CPU) + `.dockerignore` — **not build-verified in this session** (no reachable Docker daemon in this sandboxed environment; see honesty note in `DEPLOYMENT.md` — `.github/workflows/docker.yml`'s first real run on GitHub Actions is this Dockerfile's actual first build verification)
- [x] 116. `docker-compose.yml` for local stack
- [x] 117. Streamlit Community Cloud deployment config (`runtime.txt`, `.streamlit/config.toml`, `.streamlit/secrets.toml.example`)
- [x] 118. `DEPLOYMENT.md` — deploy instructions (Docker, Streamlit Cloud, HF Spaces)
- [x] 119. Final README sync (fixed placeholder repo URL/author/contact, corrected §16 usage examples to match actual CLI flags, added Docker install path, linked measured results in §18, CI+coverage badges from Phase 11)
- [x] 120. Tag release `v0.1.0`, final `CHANGELOG.md` entry

---
**Status legend**: unchecked = pending, this file is updated and committed as each step completes.
