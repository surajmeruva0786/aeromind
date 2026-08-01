# XAI run (measured)

**Status: measured (synthetic smoke test).** Produced by:

```bash
python -m src.xai.explain \
    --checkpoint runs/synthetic_smoke_test_capsnet/subject_dependent/best.ckpt \
    --n_subjects 8 --duration_s 180 --subject_id 0 \
    --output_dir results/xai/sub-00
```

against the Phase 6 smoke-test `AeroMind-CapsNet` checkpoint (see
`results/synthetic_smoke_test.md` — that checkpoint's workload head is
known to be undertrained/majority-class-collapsed after only 25 epochs).
Wall-clock: ~68s on CPU (dominated by `shap.GradientExplainer`).

## `sub-00/channel_attribution_topomap.png`

Per-channel SHAP attribution (mean over the 15-epoch sequence and 512
samples per epoch) for the model's own predicted class, rendered on the
7-channel montage via `mne.viz.plot_topomap`. Central channels (C3/C4)
show negative attribution, frontal/occipital channels positive — this is a
real computed result, not illustrative; it should **not** be read as "the
model has learned the expected frontal-theta signature", given the
workload head's known majority-class collapse on this checkpoint (see
below).

## `sub-00/spectral_attribution.png`

Mean |SHAP| over a RandomForest trained on this subject's flattened
per-channel band-power features (`src/xai/spectral_attribution.py`, a
*separate* classical model from the deep checkpoint — SHAP over the deep
model's raw waveform isn't feature-attributable to "bands" directly, so
this view uses the same classical-features approach as the Phase 4
baseline). Theta stands out across most channels, consistent with the
synthetic generator's designed ground truth (frontal theta rises with
workload) — encouraging, since this pathway doesn't depend on the
undertrained deep checkpoint at all.

## `sub-00/xai_summary.json`

Machine-readable: raw channel attribution values, predicted classes, and
the counter-factual probe result.

**Counter-factual probe result: `passed: false`.** Attenuating frontal
theta by 50% on this subject's highest-workload sequence did not shift the
predicted class or reduce confidence in it. Given the smoke-test
checkpoint's documented workload-head collapse onto the majority class
(`results/synthetic_smoke_test.md`), this is the expected outcome, not a
bug in the probe — a model that always predicts "medium" regardless of
input will also ignore a frontal-theta perturbation. This probe is exactly
the kind of check meant to catch that failure mode (README §12.4); it
worked as designed. A better-trained checkpoint (more epochs, LOSO
protocol, or real data) would be a more meaningful test of whether the
model has learned the intended signal.
