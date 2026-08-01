# Model architecture notes

**Status: Phase 5 complete.** All three model variants, the multi-task loss,
the model registry, and checkpoint utilities are implemented and tested
(`tests/test_models.py`, 14 tests).

All three models share the same interface: `forward(x, workload_target=None)`
where `x` is `(B, L, C, T)` — `L` consecutive epochs (the 30 s / 15-epoch
Bi-LSTM context window from README §9.1) — and the return value is a dict
with at least `workload_logits` `(B, n_workload_classes)` and
`fatigue_logits` `(B, n_fatigue_classes)`. `AeroMindCapsNet` additionally
returns `capsule_lengths`, `reconstruction`, `reconstruction_target`, and
`coupling`; `src/models/losses.multi_task_loss` branches on the presence of
these keys so the same training loop works for all three models unmodified.

## AeroMind-CapsNet (`aeromind_capsnet.py`)

Per-epoch encoder (`CapsNetEncoder`): two Conv1D+BN+ReLU blocks, MaxPool,
then `PrimaryCapsule1D` (a strided Conv1D reshaped into 32 capsule "types"
across the surviving temporal positions) feeding `DigitCapsuleRouting`
(3-iteration dynamic routing to `n_workload_classes` digit capsules).

The full model runs this encoder over each epoch in a 15-epoch (30s)
window, feeds the flattened digit capsules through a Bi-LSTM, and reads
workload/fatigue predictions off the last timestep. A small
`ReconstructionDecoder` reconstructs an avg-pooled view of the last epoch
from its predicted-class digit capsule, for use as a regularizer — see the
module docstring for why it targets a downsampled view rather than the
raw waveform (parameter-count proportionality).

Measured (not README-target) parameter count with default hyperparameters
(`n_channels=7, n_samples=512`): **~460k**, verified via
`sum(p.numel() for p in model.parameters())` on an actual instantiated
model and asserted (with tolerance) in `tests/test_models.py`. This differs
from the README's ~720k figure because of the reconstruction decoder design
choice above, not a training result.

## AeroMind-CNN-LSTM (`aeromind_cnn_lstm.py`)

Baseline 1 (README §9.2): the same Bi-LSTM + dual-head structure as
AeroMind-CapsNet, but with a plain 3-layer Conv1D + BatchNorm + ReLU +
pooling encoder (`CNNEpochEncoder`) in place of the capsule stack. No
reconstruction head — there's no capsule presence-vector to decode from.
Measured parameter count: **~215k**.

## AeroMind-EEGNet (`aeromind_eegnet.py`)

Baseline 2 (README §9.3): a from-scratch implementation of the EEGNet
architecture (Lawhern et al. 2018) — temporal conv, depthwise spatial conv
across channels, separable conv — as the per-epoch encoder, feeding the
same Bi-LSTM + dual-head structure. This reuses the *published architecture*
rather than loading pretrained weights: no public EEGNet checkpoint exists
for a 7-channel, 256 Hz montage, so "transfer learning" here means
re-applying the architecture to this dataset, which is the standard way
EEGNet is used on a new dataset. Measured parameter count: **~44k** — the
most compact of the three models, as expected from EEGNet's design goal.

## Losses (`losses.py`)

`margin_loss` — the standard CapsNet margin loss (Sabour et al. 2017) over
digit-capsule lengths, used for the workload head when a model provides
`capsule_lengths`. `multi_task_loss` combines this (or plain cross-entropy
for baselines) with a weighted fatigue cross-entropy and, when present, a
reconstruction MSE term — weights come from `TrainConfig`
(`workload_weight`, `fatigue_weight`, `reconstruction_weight`).

## Registry (`registry.py`)

`build_model(model_config, data_config)` maps `ModelConfig.name` to one of
the three classes above via `MODEL_REGISTRY`, reading input shape from
`DataConfig`. This is what `src/training/train.py` and
`src/evaluation/evaluate.py` call — neither module imports a model class
directly.

## Checkpoints (`src/utils/checkpoint.py`)

`save_checkpoint(path, model, optimizer=None, epoch=None, extra=None)` /
`load_checkpoint(path, model, optimizer=None)` — a thin wrapper around
`torch.save`/`torch.load` storing model + optimizer state dicts plus
arbitrary run metadata (used for run-config hashes in Phase 6).
