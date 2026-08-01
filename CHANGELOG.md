# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/); this project
doesn't yet follow strict SemVer beyond the `v0.1.0` tag below, since
`0.x` is still pre-stable API territory.

## [v0.1.0] — 2026-08-01

Initial complete build: the full pipeline described in `README.md`, from
raw EEG ingestion through preprocessing, feature engineering, three model
architectures, training, evaluation, explainability, real-time inference,
a Streamlit demo, and deployment packaging — implemented, tested, and
documented across 12 phases (120 tracked steps, see `ROADMAP.md` for the
complete phase-by-phase log with commit references and measured-result
links).

### Added

- **Phase 0-1** — project scaffolding, config system (`src/utils/config.py`),
  reproducibility/logging/metrics utilities, environment verification script.
- **Phase 2** — synthetic multi-channel EEG generator with a designed
  ground-truth workload/fatigue signature (`src/data/synthetic.py`), the
  PyTorch dataset/transform/split layer, and dataset download scripts
  (STEW automated, MAUS/DEAP reference-only per their actual access terms).
- **Phase 3** — MNE-based preprocessing pipeline: filtering, PREP-style bad
  channel handling, real ICLabel-backed ICA artefact rejection, epoching,
  a CLI entrypoint, and an executed demo notebook.
- **Phase 4** — spectral/temporal/connectivity feature extraction and a
  classical RandomForest/SVM baseline, with executed EDA and feature
  visualization notebooks.
- **Phase 5** — three model architectures sharing a common interface:
  `AeroMind-CapsNet` (dynamic-routing capsule network + Bi-LSTM, ~460k
  params), `AeroMind-CNN-LSTM` (~215k params), `AeroMind-EEGNet` (~44k
  params) — plus the multi-task margin/CE/reconstruction loss, a model
  registry, and checkpoint utilities.
- **Phase 6** — the training pipeline: AdamW + `ReduceLROnPlateau`, early
  stopping, subject-dependent/LOSO/cross-dataset protocol orchestration
  with leak-free validation carving, CUDA-conditional mixed precision, and
  run-config hashing/logging.
- **Phase 7** — the evaluation CLI, which deterministically reconstructs a
  training run's exact fold split from its saved config and reports
  held-out test metrics via a markdown report generator.
- **Phase 8** — explainability: SHAP channel attribution
  (`shap.GradientExplainer`), `mne`-based topographic scalp-map rendering,
  SHAP over spectral (band-power) features, and a frontal-theta
  counter-factual probe, tied together by an `explain.py` CLI.
- **Phase 9** — real-time inference: a sliding-window streaming engine
  with EWMA-smoothed predictions, offline replay sources (synthetic and
  real `.edf`/`.bdf`/`.fif`), a guarded optional LSL live-hardware source,
  and a WebSocket broadcast server for downstream dashboards.
- **Phase 10** — the Streamlit demo app: live EEG plot, workload/fatigue
  panels, on-demand SHAP topomap explanation, CSV session export, and a
  zero-setup synthetic replay default with an explicit warning when no
  trained checkpoint is loaded.
- **Phase 11** — CI (GitHub Actions: lint + test + coverage, Docker build
  check), pre-commit hooks (ruff/black + hygiene checks), and a measured
  ~86% test-coverage badge.
- **Phase 12** — packaging and deployment: a CPU-only `Dockerfile` +
  `docker-compose.yml`, Streamlit Community Cloud config
  (`runtime.txt`, `.streamlit/config.toml`, secrets template), a
  `DEPLOYMENT.md` covering Docker/Streamlit Cloud/Hugging Face Spaces, and
  this final README/changelog sync.

### Testing

86 tests across the full pipeline (unit + integration), ~86% coverage of
`src/` and `app/`. Every phase that produces a runnable artifact (training
run, evaluation report, XAI output, latency benchmark) was executed for
real at least once, with measured — not just aspirational — results
recorded under `results/`.

### Known limitations (see README §21 for the full list)

- All measured numeric results in this repo come from the synthetic
  generator, not real EEG data — real-dataset (STEW) ingestion into the
  training/evaluation pipeline is implemented at the download-script level
  but not yet wired into a common on-disk epoch format for training.
- The `cross_dataset` training/evaluation protocol uses a second synthetic
  cohort as a structural stand-in for genuine cross-dataset (MAUS→STEW)
  transfer.
- Docker image build was not verified in this session's execution
  environment (no reachable Docker daemon); the Dockerfile follows
  standard, previously-tested patterns but should be build-verified before
  a production deployment.
- Live LSL hardware streaming and real `.edf`/`.bdf`/`.fif` replay are
  implemented but untested against real hardware/files, for lack of
  access to either in this environment.
