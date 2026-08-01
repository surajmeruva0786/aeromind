# Explainability (XAI) notes

Phase 8 implements README §12: SHAP channel attribution, topographic
scalp-map rendering, SHAP over spectral features, and a counter-factual
probe, tied together by the `explain.py` CLI.

## Modules

- **`shap_channel.py`** — `compute_channel_attributions(model, background,
  inputs, channel_names)` wraps `shap.GradientExplainer` around a
  `WorkloadLogitWrapper` (adapts a model's dict output to the single-tensor
  interface SHAP expects), then for each input takes the SHAP values for
  *that input's own predicted class* and averages over the sequence and
  time axes — one signed score per channel, matching README §12.1's
  "averaged over the time dimension" definition.
- **`topomap.py`** — `plot_channel_topomap(values, channel_names, title)`
  renders a per-channel scalar array onto the project's 7-channel montage
  via `mne.viz.plot_topomap`, using `mne.channels.make_standard_montage("standard_1020")`
  for electrode positions (all 7 channel names — Fp1/Fp2/Fz/C3/C4/Pz/Oz —
  are in the standard 10-20 set).
- **`spectral_attribution.py`** — README §12.3's class x (channel, band)
  attribution matrix. Trains a separate classical RandomForest on flattened
  per-channel band-power features (`src.features.spectral.band_powers_matrix`)
  and runs `shap.TreeExplainer` over it. This is intentionally a *different*
  model from the deep checkpoint being explained — SHAP over a deep model's
  raw waveform input isn't directly attributable to named frequency bands,
  so this view reuses the classical-features approach from the Phase 4
  baseline instead.
- **`counter_factual.py`** — `run_counter_factual_probe(model, sequence,
  sfreq)` attenuates frontal-midline theta (default: Fp1/Fp2/Fz, per
  `src.data.synthetic.FRONTAL_CHANNELS`) by 50% on the last epoch of a
  sequence window and checks whether the predicted workload class moves
  down or the model's confidence in its original prediction drops. Either
  outcome counts as "passed" — a model relying on frontal theta, as the
  synthetic generator's ground truth and the literature both suggest it
  should, is expected to react to its removal.
- **`explain.py`** — CLI entrypoint tying all of the above together for one
  checkpoint + subject.

## Quick start

```bash
python -m src.xai.explain \
    --checkpoint runs/<output_dir>/<fold_name>/best.ckpt \
    --n_subjects 8 --duration_s 180 --subject_id 0 \
    --output_dir results/xai/sub-00
```

`--n_subjects`/`--duration_s` must match the training run's values (same
determinism requirement as `src.evaluation.evaluate`, see
`src/evaluation/README.md`). Produces `channel_attribution_topomap.png`,
`spectral_attribution.png`, and `xai_summary.json` in `--output_dir`.

## Measured run

See `results/xai/README.md` — a real executed run against the Phase 6
smoke-test checkpoint, including an honest discussion of why the
counter-factual probe reported `passed: false` on that particular
(undertrained, majority-class-collapsed) checkpoint.

## Honesty note

The deep model's channel attribution (`shap_channel.py`) and the
counter-factual probe are only as informative as the checkpoint being
explained — on an undertrained model they'll faithfully report "this model
isn't using the features you'd expect," which is a correct and useful XAI
result, not a tooling failure. Don't read a `passed: false` probe result
as broken code; check `results/synthetic_smoke_test.md` for that
checkpoint's training diagnostics first.
