# Synthetic smoke-test training run (measured)

**Status: measured (synthetic smoke test)** — per the honesty note at the
top of `ROADMAP.md`, every number on this page comes from an actual
executed run in this repository, not a target/aspirational figure. This is
a pipeline-correctness smoke test (few epochs, one split, CPU), not a claim
about real-world EEG classification accuracy.

## Command

```bash
python -m src.training.train \
    --model aeromind_capsnet --protocol subject_dependent \
    --epochs 25 --n_subjects 8 --duration_s 180 --sequence_length 15 \
    --batch_size 32 --seed 42 --output_dir runs/synthetic_smoke_test_capsnet
```

- Dataset: in-memory synthetic (`src/data/synthetic.py`), 8 subjects x 180s,
  1432 total epochs (2s window, 50% overlap).
- Model: `AeroMind-CapsNet` (~460k params).
- Protocol: `subject_dependent` (single 80/20 split; 15% of the 80% train
  portion further carved out as validation for early stopping).
- Hardware: CPU only (no CUDA available on this machine) — mixed precision
  was a no-op.
- Wall-clock: **556 s (~9.3 minutes)** for 25 epochs, no early stop
  triggered (`early_stop_patience` wasn't hit within 25 epochs).

## Results (held-out test split, single fold)

| Task     | Accuracy | Macro-F1 | Kappa  | ROC-AUC | ECE    |
|----------|----------|----------|--------|---------|--------|
| Workload (3-class) | 0.345 | 0.181 | 0.002 | 0.447 | 0.001 |
| Fatigue (binary)   | 0.595 | 0.592 | 0.204 | 0.678 | 0.047 |

Chance level: workload ~0.33 (3 balanced-ish classes), fatigue ~0.50-0.61
depending on class balance (test split was 90 alert / 78 fatigued).

## What this run actually shows

- **The full pipeline runs end-to-end and is correct**: data generation,
  subject-aware splitting, sequence batching, forward/backward pass,
  multi-task loss (margin + CE + reconstruction), early-stopping/
  checkpointing, and metrics computation all execute without error and
  produce a reloadable checkpoint (`runs/synthetic_smoke_test_capsnet/subject_dependent/best.ckpt`).
- **The workload head overfit and collapsed to the majority class.**
  Training loss on the workload term dropped from 0.36 to 0.004 over 25
  epochs while validation loss *rose* from 0.47 to 0.70 — classic
  overfitting on a small subject-dependent split with no regularization
  beyond the built-in reconstruction term. The test-set confusion matrix
  shows the model predicting class 1 ("medium") for the large majority of
  epochs regardless of true label (workload per-class F1: 0.03 / 0.52 /
  0.00 for low/medium/high).
- **The fatigue head learned a real, if modest, signal**: kappa 0.20 is
  clearly above chance (0 = chance-level agreement), and per-class F1 is
  balanced (0.56 / 0.63) rather than majority-collapsed.
- This is consistent with training a capsule network from scratch on a
  fairly small amount of data for very few epochs with no hyperparameter
  tuning — it is not evidence against the architecture. Follow-up work
  (more epochs, LOSO/cross-dataset protocol runs, hyperparameter search,
  and real EEG data) is needed before drawing any conclusion about model
  quality; see README §21 Limitations.

## Reproducing

The exact `AeroMindConfig` used (including the config hash `3da50b119895`)
is saved alongside the run at
`runs/synthetic_smoke_test_capsnet/config.yaml` (gitignored — regenerate
with the command above; the synthetic generator is seeded, so results are
deterministic given the same seed/hardware).
