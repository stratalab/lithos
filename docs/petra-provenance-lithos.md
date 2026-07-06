# Lithos → Petra: provenance answers (the `[→ Lithos]` items)

Response to `petra/docs/petra/chisel-provenance-questions.md` and the items
`chisel/petra-provenance-answers.md` forwarded to us. **Chisel's answers are accurate about our
seam — nothing to change there.** This fills the three items they marked `[→ Lithos]` (Q3 doc
index, Q5 mixture/data-order, Q6 dedup freeze) plus the checkpoint-cadence question Petra flagged
as ours. Grounded in the as-built pipeline (`lithos/data/*`, `lithos/train/loop.py`), not
intentions; new capabilities are **[committed]** with the build they land in.

**Scope:** all of this is for **canon-era corpora** (Chisel-produced, provenance-carrying). The
current fineweb smoke corpus predates the chain — Petra's coarse resolver
(`provenance: fineweb-coarse`) is correct for it and nothing below changes that.

## Q3 — Doc index at tokenize → **yes, committed**

The *(shard, token offset) → doc* mapping is born in our tokenize/pack step, and Chisel's records
already carry every field after the offsets (`source_id`, `record_id`, `text_sha256`). We'll
emit, per shard, alongside the `.bin`:

```
<shard>.docindex.parquet   # one row per document, in stream order:
    doc_ordinal · token_start · token_end · source_id · record_id · text_sha256
```

The pipeline processes documents in order and knows each doc's exact token span as it writes
(`ShardWriter.add` per doc), so this is **additive — no extra pass**. **[committed — lands with
the first canon-era corpus build; the smoke corpus stays coarse.]**

**One property Petra must model — packing bleed.** Packing is plain concatenation: a `seq_len`
window is chopped from the flat stream and **can span document boundaries** (cross-doc attention
bleed is the v0 default; `packing.py`). So a training example maps not to one document but to the
**set of docs whose `[token_start, token_end]` overlaps the window** — which the docindex gives
you exactly. For tier-2 influence, a window's influence is shared across the docs it spans:
attribute proportionally by overlap, or roll up to the source. (Intra-doc masking / position
reset that removes bleed is a deferred flag; if enabled, the docindex is unchanged and windows
become doc-aligned.)

## Q5 — Mixture & data-order → **split; the hard part (TracIn order) is already reconstructible**

**Mixture (per-source proportions):** `corpus_manifest.json` records `mixture` (per-source
document counts) + `num_documents` + `num_tokens` — the inventory truth today.

**Effective per-doc exposure — uniform today; do not assume up-sampling.** The corpus is a flat
stream of *unique* docs (exact-dedup drops identical text, so a source can't be up-sampled by
duplication), and the dataloader shuffles + loops over the whole stream **uniformly** — no
per-source weighting, no curriculum. A doc's effective exposure = the run's epoch count
(`steps × global_batch / total_sequences`), uniform across the corpus. **If we later add
non-uniform mixing** (weighted sampling, curriculum, per-source repeats) **[committed]** we record
the per-source effective weights in the run manifest. Build for uniform now; the field appears
when the mechanism does.

**Data-order reconstructibility (TracIn) — already fully there.** The strong yes: the dataloader
permutation is deterministic — `np.random.RandomState(seed + epoch).permutation(n)` — and **every
checkpoint already persists `dataloader_state = (epoch, position, seed)`** (`loop.py`, saved for
resume). So "the sequences seen between checkpoint *i* and *j*" is exactly replayable: rebuild the
permutation, walk positions *i → j*, map each sequence index → shard window → (via the docindex)
the docs it covers. Nothing new is needed for the capability; **[committed]** we'll additionally
surface `seed` + each checkpoint's `(epoch, position)` in the run manifest so you never have to
crack `train_state.pt`.

**Packing config (tier-2 need #1).** Shards are a **flat, `seq_len`-independent** token stream;
the windowing (`seq_len`, stride = `seq_len`) is a *train-time* parameter in the run's
`resolved_config.yaml`, and the joining policy is fixed (concat + `<bos>/<eos>`, no truncation,
cross-doc bleed — `packing.py` v0). So "what the optimizer saw" = flat shards + docindex + the
run's `seq_len` + the fixed policy — all reproducible. **[committed]** we'll stamp `seq_len` + a
`packing_policy` version into the run manifest so it's one lookup, not tribal knowledge.

## Q6 — Dedup freeze on counterfactual builds → **yes, via a keep-set replay**

Corpus-level dedup lives here (`ExactDocumentDeduper` — sha256 full-text, first-wins; MinHash
near-dedup exists but is **off by default**). Petra's concern is correct: naively re-running the
pipeline on "corpus minus source X" could let a duplicate that X had shadowed *newly surface*, so
the delta wouldn't be just the dropped sources.

**Mechanism [committed — with the counterfactual path, paired with Chisel's `chisel counterfactual`]:**
the corpus build records the **dedup keep-set** (surviving `record_id`s / `text_sha256`s) in the
manifest. A counterfactual build takes `keep-set − {dropped-source records}` and **skips the dedup
stage entirely** — a deterministic pass-through of the frozen survivors. The only delta is then
exactly the dropped sources; no near-dup newly enters, no decision re-runs. For the common case
(exact-dedup only) the delta is already just the dropped docs — the keep-set makes it provable and
covers the exact-cross-source-dup and MinHash cases too.

Division of labor: **Chisel** drops the sources from the corpus spec (deterministic, their side);
**Lithos** tokenizes the survivor set with frozen dedup (ours). Two halves of one build.

## Checkpoint cadence (Petra flagged as ours) → configurable; already TracIn-ready

Every checkpoint already carries the dataloader state, so TracIn between any two saved checkpoints
is reconstructible today. Cadence is a knob — pretrain writes every *N* steps, post-train after
each step/stage (R2 contract §6). Denser cadence = finer TracIn trajectory at more storage; for a
specific flagship attribution run Petra can **request a denser interval** — a per-run cost/benefit
call rather than a baked global default.

## Summary of Lithos commitments

| What | Status |
|---|---|
| `<shard>.docindex.parquet` (doc_ordinal, token offsets, source_id, record_id, text_sha256) | **committed — first canon-era corpus** |
| Data-order reconstructible for TracIn (deterministic perm + checkpoint-persisted position) | **already true** |
| `seed` + per-checkpoint `(epoch, position)` surfaced in the run manifest | **committed** |
| `seq_len` + `packing_policy` version in the run manifest | **committed** |
| Per-source effective weights in the manifest **if** non-uniform mixing is added | committed (conditional) |
| Dedup keep-set recorded; frozen-dedup replay on counterfactual builds | **committed — with the counterfactual path** |
| Denser checkpoint cadence on request for a specific attribution run | per-run |

The doc index, the frozen-dedup counterfactual, and the mixture/order fields are the Lithos half
of the same contract; they land alongside the canon-era corpus (the smoke corpus stays coarse, as
Petra has it). One caveat worth repeating up-front to Petra: **packing bleed** (Q3) means training
examples are multi-document — the single biggest thing to model correctly in tier-2 attribution.
