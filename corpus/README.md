# corpus/

Decontamination probes for the eval-leakage screen (`lithos/data/decontam.py`) — the one
corpus artifact the consumer/training path needs locally.

- `probes/` — per-domain n-gram probe sets the decontam gate screens the training corpus
  against (so eval-battery text can't leak into pretraining).

The **Canon** — the `seed_index.csv` "catalog of intent", acquisition specs, curation task
banks, and the sourcing doctrine — moved to **Chisel** with the producer tier (see
[`docs/chisel-producer-migration.md`](../docs/chisel-producer-migration.md)). It is the
source of truth for provenance, lives at `s3://lithos-canon/`, and Lithos records reference
it via `metadata.source_id`.
