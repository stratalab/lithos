# Contributing to Lithos

Lithos is developed with [uv](https://docs.astral.sh/uv/). This is a research codebase
with production discipline: every change lands green on lint, types, and tests.

## Environment

```bash
uv sync --extra eval                  # or: make install
uv run pre-commit install             # optional: run the gates on every commit
```

Extras: `eval` (transformers/lm-eval), `cloud` (s3fs/gcsfs), `serve`, `tracking` (wandb).
`uv sync --all-extras` installs all. (Sourcing/extraction extras live in Chisel now.)

## Quality gates (must pass before you push)

```bash
make check        # ruff + mypy + pytest — mirrors CI exactly
# or individually:
make lint         # uv run ruff check .
make typecheck    # uv run mypy lithos
make test         # uv run pytest
make format       # uv run ruff format . && ruff check . --fix
```

CI runs the same three gates on every push and PR. **Do not merge red.**

## Conventions

- **Configs are typed.** Runtime configuration is validated Pydantic models loaded from
  YAML (`lithos/utils/config.py`); add a field to the model, not a loose `dict` key.
- **Reproducibility.** Runs write a `run_manifest.json`, a resolved config, and JSONL
  metrics under `runs/<ts>_<name>/`. Seed via `lithos.utils.seed`.
- **Provenance.** Every corpus record carries `metadata.source_id` resolving to a row in
  the Canon (`corpus/seed_index.csv`). A record whose `source_id` doesn't resolve is a bug.
- **Big artifacts stay out of git** — `runs/`, `artifacts/`, `data/`, `models/`, and
  binary blobs are git-ignored (see `.gitignore`). Durable storage is the object store
  (`configs/storage.yaml`), not the repo.
- **Scripts are thin.** `scripts/` holds runnable entrypoints; real logic lives in the
  `lithos` package with tests.

## Tests

New behavior needs a test. Unit tests are pure and fast; integration tests that need a
heavy dependency import it lazily or guard on availability so the core test run stays
light. Match the file's existing style, comment density, and naming.

## Commits & branches

Branch off `main`; keep commits focused; write messages that explain *why*. Open a PR and
let CI go green before merging.
