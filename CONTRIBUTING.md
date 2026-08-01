# Contributing to AeroMind

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
pip install -e .
```

## Development workflow

1. Check `ROADMAP.md` for the current build phase and outstanding steps.
2. Run tests before and after changes: `pytest`.
3. Lint/format: `ruff check .` and `black .`.
4. Keep modules runnable against `src/data/synthetic.py` so CI never depends
   on gated or large external datasets.

## Tests, CI, and pre-commit

- **Test config** lives in `pyproject.toml` (`[tool.pytest.ini_options]`),
  not a separate `pytest.ini`. Shared fixtures (small synthetic datasets,
  a headless `Agg` matplotlib backend) are in `tests/conftest.py`.
- **`pytest --cov=src --cov=app --cov-report=term-missing`** runs the full
  suite with coverage; last measured total was 86% (see `ROADMAP.md` Phase
  11 for the per-module breakdown — the README coverage badge is a static
  snapshot of this number, not a live integration).
- **CI** (`.github/workflows/ci.yml`) runs on every push/PR to `main`:
  `ruff check`, `black --check`, then the full test suite with coverage,
  uploading `coverage.xml` as a build artifact.
  `.github/workflows/docker.yml` builds the Docker image (no push) as a
  build-health check; it no-ops gracefully if `Dockerfile` isn't present
  yet (only relevant during the Phase 11→12 transition in this repo's
  history).
- **Pre-commit** (`.pre-commit-config.yaml`): ruff (with `--fix`) and
  black on `src/`, `app/`, `tests/`, `scripts/`, plus trailing-whitespace/
  end-of-file/YAML/merge-conflict/large-file checks (the whitespace/EOF
  checks skip `results/`, `runs/`, `notebooks/` — generated artifacts, not
  hand-edited source). Install with `pre-commit install` after cloning to
  run these automatically on `git commit`.

## Commit style

Conventional-commit-style prefixes (`feat:`, `fix:`, `docs:`, `test:`,
`chore:`, `refactor:`) with a short imperative subject line.

## Code style

- Python 3.11, type-hinted public functions.
- No commented-out code; no TODOs without an accompanying issue reference.
- Prefer small, composable functions over deep class hierarchies, except
  where `nn.Module` structure is the natural fit (models package).
