# Evaluation notes

Phase 7 implements the evaluation CLI (`evaluate.py`) and the markdown
report generator (`metrics_report.py`) described in README §11.
`baseline.py` (the classical RandomForest/SVM baseline) is from Phase 4.

## Quick start

```bash
python -m src.evaluation.evaluate \
    --checkpoint runs/<output_dir>/<fold_name>/best.ckpt \
    --fold <fold_name> \
    --n_subjects 8 --duration_s 180 \
    --output_dir results/my_eval
```

`--fold` must match one of the training run's fold names (`subject_dependent`
for that protocol, or `loso_subject_<id>` for a specific LOSO fold —
see `summary.json` in the training run's output directory for the exact
list). `--n_subjects`/`--duration_s` must match the values passed to
`src.training.train` for that run, so the in-memory synthetic dataset (and
therefore the train/val/test split) is reconstructed identically — the
split itself is never saved to disk, only regenerated deterministically
from the seed baked into `config.yaml`.

## How determinism is verified

`tests/test_evaluation.py::test_evaluate_checkpoint_reproduces_training_run_test_metrics`
trains a tiny model via `src.training.train.run`, then calls
`evaluate_checkpoint` with the same parameters and asserts the resulting
accuracy is bit-for-bit identical to what `run_fold` computed internally
during training. This is what makes the CLI usable as an independent
double-check of a training run's own self-reported numbers, not just a
convenience wrapper.

## What it produces

```
<output_dir>/
  evaluation_report.md   # human-readable: confusion matrix, ROC-AUC, ECE, per-class F1
  metrics.json            # same numbers, machine-readable, plus run metadata
```

## Cross-dataset evaluation path (README §11.1, roadmap step 81)

`--fold cross_dataset_synthetic_proxy` evaluates a checkpoint trained with
`--protocol cross_dataset` against its held-out synthetic "second cohort"
split — see the honesty note in `src/training/README.md`: this is a
structural stand-in for the real MAUS→STEW cross-dataset path (README
§11.1), not real cross-dataset numbers.

## Measured evaluation run

`results/eval_capsnet_smoke_test/` was produced by running the CLI above
against the Phase 6 smoke-test checkpoint
(`runs/synthetic_smoke_test_capsnet/subject_dependent/best.ckpt`, itself
gitignored — regenerate it via the command in
`results/synthetic_smoke_test.md`). Its numbers are identical to that
run's own self-reported test metrics, confirming the CLI's split
reconstruction is correct.
