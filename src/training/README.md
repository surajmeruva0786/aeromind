# Training pipeline notes

Phase 6 implements the training CLI (`train.py`), the epoch loop
(`loop.py`), and callbacks (`callbacks.py`) described in README §10.

## Quick start

```bash
# Fast smoke test — in-memory synthetic data, no download needed
python -m src.training.train \
    --model aeromind_capsnet \
    --protocol subject_dependent \
    --epochs 25 \
    --n_subjects 8 \
    --duration_s 180 \
    --sequence_length 15 \
    --batch_size 32 \
    --output_dir runs/my_run
```

`--model` selects any of `aeromind_capsnet`, `aeromind_cnn_lstm`,
`aeromind_eegnet` via `src/models/registry.py`. `--protocol` selects
`subject_dependent` (single 80/20 split), `loso` (one fold per subject), or
`cross_dataset` (see honesty note below).

If `--processed_dir` (default `data/processed/synthetic`) contains a
dataset materialized by `scripts/make_synthetic_dataset.py`
(`sub-XX.npz` files), it's loaded from disk; otherwise the CLI generates an
equivalent dataset in-memory using `--n_subjects`/`--duration_s`, which is
the fast path used by tests and CI.

## What a run produces

```
runs/<output_dir>/
  config.yaml              # resolved AeroMindConfig, for reproducibility
  summary.json             # aggregate metrics across all folds
  <fold_name>/
    best.ckpt               # model+optimizer state at the best val-loss epoch
    metrics.json             # per-epoch train/val loss and accuracy history
    summary.json             # this fold's final test-set metrics
```

For `subject_dependent` there's exactly one fold (`subject_dependent/`).
For `loso` there's one fold per subject (`loso_subject_<id>/`), and
`summary.json` at the top level reports the mean ± std of accuracy/macro-F1/
kappa across all folds — the headline generalization number (README §11.1).

## How a fold is trained

`run_fold()` (`train.py`) carves a validation set out of the fold's
training epochs (`subject_dependent_split(..., test_fraction=0.15)`,
subject-aware) — the fold's held-out test epochs are never touched until
the very last step, so early stopping and learning-rate scheduling can't
leak test information. AdamW + `ReduceLROnPlateau` on validation loss,
early stopping via `EarlyStopping` (`callbacks.py`), automatic mixed
precision when running on CUDA (`--mixed_precision`, on by default,
inert on CPU). The best checkpoint (lowest val loss) is reloaded before
the final test-set evaluation.

## Honesty note: cross-dataset protocol

`--protocol cross_dataset` does not yet ingest real STEW/MAUS data into
this training loop — real-dataset preprocessing output isn't wired into a
common on-disk epoch format yet (see `data/README.md`). As a structural
stand-in, it trains on one synthetic cohort and evaluates on a second
synthetic cohort generated from a different seed offset (a different
"population" from the same generator), which is logged in `summary.json`
as `"cross_dataset_synthetic_proxy"`. Treat any numbers from this protocol
as a pipeline correctness check, not a generalization claim — wiring real
MAUS→STEW cross-dataset training is listed under README §22 Future Work.

## Measured smoke-test results

See `results/synthetic_smoke_test.md` for a real executed run (not
target/aspirational numbers) — actual metrics, wall-clock time, and the
exact command used, per the ROADMAP.md honesty note.

## Troubleshooting

- **A fold logs "insufficient epochs for sequence_length" and is
  skipped.** The per-subject epoch count after the train/val/test split is
  smaller than `--sequence_length`. Either lower `--sequence_length`,
  increase `--duration_s` (more epochs per synthetic subject), or increase
  `--n_subjects`. Rule of thumb for `subject_dependent`/`loso`: you need
  roughly `sequence_length / (0.8 * 0.85)` epochs per subject just to
  populate the validation fold; add more for training itself.
- **Training is slow on CPU.** This is expected — there's no CUDA path
  assumed. Reduce `--duration_s`/`--n_subjects`/`--epochs` for local
  iteration; `mixed_precision` is a no-op on CPU (autocast/GradScaler are
  gated on `device.type == "cuda"` in `loop.py`/`train.py`).
- **`ReduceLROnPlateau` never fires.** It's stepped on validation loss
  once per epoch inside `run_fold`; if a fold is skipped (see above) no
  scheduler step happens for it — check `summary.json`'s `"skipped"` flag.
- **Checkpoint won't load into a different model config.** `save_checkpoint`
  stores a raw `state_dict` — architecture hyperparameters (capsule dims,
  LSTM hidden size, etc.) must match between save and load. The `extra`
  dict on the checkpoint stores the val loss at save time but not the full
  config; use the sibling `config.yaml` to reconstruct the exact model.
