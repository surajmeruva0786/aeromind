# Model architecture notes

**Status: Phase 5 in progress.** `layers.py` and `aeromind_capsnet.py` are
done and smoke-tested; `aeromind_cnn_lstm.py`, `aeromind_eegnet.py`,
`losses.py`, and `registry.py` are not yet written. See the "Session
handoff" note at the top of `../../ROADMAP.md` for exact next steps.

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
model. This differs from the README's ~720k figure because of the
reconstruction decoder design choice above, not a training result — see
`tests/test_models.py` (once written) for the assertion.
