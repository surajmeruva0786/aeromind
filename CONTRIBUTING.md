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

## Commit style

Conventional-commit-style prefixes (`feat:`, `fix:`, `docs:`, `test:`,
`chore:`, `refactor:`) with a short imperative subject line.

## Code style

- Python 3.11, type-hinted public functions.
- No commented-out code; no TODOs without an accompanying issue reference.
- Prefer small, composable functions over deep class hierarchies, except
  where `nn.Module` structure is the natural fit (models package).
