# Runs

Each training run writes a self-contained directory here:

```text
runs/
  <YYYY-MM-DD_HHMMSS>_<name>/
    resolved_config.yaml   # the exact resolved config used
    metrics.jsonl          # per-step metrics (PRD §9.7)
    run_manifest.json      # reproducibility manifest (PRD §15)
    samples/               # generated samples
    checkpoints/           # model + training-state checkpoints
    evals/                 # eval reports
```

Run outputs are **git-ignored** (PRD §20.9); only this README is tracked. Directories are
created by `lithos.train.logging.create_run_dir` and are never silently overwritten
(PRD §20.7).
