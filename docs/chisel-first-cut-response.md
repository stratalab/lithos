# Lithos → Chisel: first-cut review — the format exists; wire the second face

**From:** Lithos · **To:** Chisel · **Re:** your `docs/lithos-corpus-first-cut.md`
(OpenStax shelf → corpus projection) · **2026-07-18**

**Verdict in one line:** the instincts are right (per-book identity, subtopic
tagging, the two-faces finding) — but the first cut re-invented a target format
that already exists. The contract is `docs/chisel-lithos-r2-contract.md` §3, and
the emitter for §3.1 is the `openstax.py` adapter that migrated *to you* in the
producer-tier move (`chisel-producer-migration.md`). The "missing
`book → Lithos-corpus` stage" is not a design problem; it is wiring the migrated
adapter into your pipeline as the second face, with the enrichments below.

## 1 · The target format is already contracted — use it

Your `corpus_index.jsonl` is an *index record*, not a corpus record. The
deliverable Lithos ingests is the **§3.1 canonical document** (`.jsonl.zst`, one
doc per line):

```jsonc
{ "id", "text", "source", "subset", "language", "license", "tier", "metadata", "quality_score"? }
```

plus the **§3.2 `_catalog.json`** (the mix bill-of-materials) and the canon
`seed_index.csv` rows. Keep your index file if it's useful internally; it is not
a handoff artifact. Two notes on the contract itself:

- **`tier` was missing from the §3.1 example** (your records carry it — correct
  instinct; the executable schema `documents.py` has always required it,
  fail-closed via `unknown`). Fixed on our side in the same pass as this memo.
- Post-training input formats are §3.3–§3.6 **verbatim** — see §6 below.

## 2 · Answers to your six questions

**Q1 — Chunk unit + fields.** Module-level is right, and **do not re-cut for
uniformity**. Pretraining packs documents into fixed windows itself
(`lithos/data/packing.py`, bleed packing) — a 270-token preface and a 12k-token
section are equally fine; the 12k doc simply spans windows. Uniform chunking
would only serve retrieval, and retrieval chunking is a Lithos-side policy knob
we will sweep ourselves (rev-B R1 is in-context retrieval; chunk size is an
experiment, not an ingest property). Fields: the §3.1 set, with your enrichments
in `metadata` (see §3). Token counts: chars/4 is fine **as an estimate** — name
it `est_tokens` and stamp the method; never emit a field named `tokens`. Real
counts are tokenizer-dependent and the tokenizer is in flux (the v1 Qwen-vocab
trim); Lithos recomputes at build time.

**Q2 — Concept tagging.** Confirmed: every chunk carries the **taxonomy
subtopic**, same `_match_subtopic` as the Assay face — this is the biggest gap
and the key to the unified graph. Emit `metadata.subtopic` +
`metadata.subtopic_score` on every record and **do not hard-threshold at
emission** — a low-confidence tag with its score attached is recoverable; a
dropped tag is not. Consumers apply the bar (covers-graph edges can start at
~0.7; tunable). Keep the raw heading too, as `metadata.section_title` — it's
provenance, just not a concept key.

**Q3 — Prose + problems: two streams, one join.** Deliver **two independent
streams** (canonical docs → Lithos; fixtures → Assay) — merging them into one
artifact couples pipelines that version and ship separately. The *join* is not a
file, it's shared keys: both faces carry `source_id`, `module_id`, and
`subtopic`, so "prose + verified problems for topic X" is a query. This is also
the graph-shaped-records storage decision from Chisel's own design (stable ids +
typed reference fields → StrataDB later, loader script not migration). If you
want to materialize the covers graph, a third artifact of typed edge records is
welcome — but it's derived, not load-bearing.

**Q4 — seed_index: merge.** One `seed_index.csv` (it's the canon — the
acquisition bill-of-materials, and it's yours now). Your per-book rows **replace**
the `openstax` dump row; the 168 hand-curated restricted rows stay as they are;
add a `curation` column (`hand | generated`) so machine rows are auditable.
Granularity: **per volume** (~11 rows), not per bundle — you called it, and the
canon row id becomes the `source_id` every derived record references, so it must
name the thing a human would cite.

**Q5 — International-edition route: model it; it changes nothing for training.**
Yes, put it in the canon (`route=lawful-copy`, `license_note` naming the
edition/jurisdiction). But be precise about what it buys: **`tier` stays
`restricted`** — the taxonomy keys on whether bytes may enter the weights, and
paywalled work never does, however lawfully the copy was acquired. There is no
`epoch_cap`/`priority` implication because the gradient gate superseded
epoch-caps entirely: restricted text gets **zero** gradient, so there is nothing
to cap. What lawful purchase *does* buy is real and worth the route: (a) the
retrieval datastore's legal posture — "copies the operator lawfully holds,
consulted and cited" is literally satisfied; (b) the grounding path — a teacher
reads the lawfully-held chapter, writes its own worked problem, the sandbox
verifies, and *that* enters the weights as `synthetic-verified` +
`grounded_on`. The route strengthens both channels; it moves nothing across the
weights line.

**Q6 — What post-training ingests: §3.3–§3.6, unchanged.** SFT = messages-JSONL;
TIR-SFT = messages + `segments`, gated by `validate_tir_record` before you emit;
preferences = `{prompt, chosen, rejected}`; RL tasks = taskbank JSONL with
`year` + `family_id` stamped (the contamination split depends on them). The
"evolving post-training logic" you're guessing at is the T1+T2 adoption
(`docs/tinker-learnings.md`): Lithos post-training now runs on a canonical
per-token record — tokens + float loss-weights (+ sampler logprobs/advantages
for RL). **This changes your contract not at all, by design**: the seam is
"Chisel stops at text + structure + provenance; Lithos owns tokenization and
everything per-token." The weights vector is *derived* from what you already
ship — role structure for SFT, `tool_result` segments for TIR, tier for the
gate. That the record refactor didn't touch the handoff is evidence the seam is
in the right place. Two per-record requirements it does sharpen:
**(a) stable, deterministic `id`s** (`<source_id>/<module_id>` for corpus docs) —
`grounded_on` references and Petra attribution must survive re-extraction;
**(b) per-record `metadata.grounded_on`** on every synthetic-verified target
(the tier gate requires it — source-level lists are too coarse for attribution).
And keep **tier-homogeneous files**: the SFT build config declares tier per
source file (`SFTSourceSpec`), so never mix tiers within one JSONL.

## 3 · Concrete modifications to the first-cut record

Your example record, restated as the §3.1 canonical document:

```jsonc
{ "id": "openstax-calculus/m53472",            // deterministic; the grounding/attribution key
  "text": "…the module prose…",                // the first cut shipped no text — the index isn't the corpus
  "source": "openstax",
  "subset": "calculus",
  "language": "en",
  "license": "cc-by-4.0",
  "tier": "open",
  "quality_score": null,                        // optional; omit rather than invent
  "metadata": {
    "source_id": "openstax-calculus",          // canon row (per-volume, Q4)
    "module_id": "m53472",
    "domain": "math",
    "subtopic": "functions.review",            // taxonomy key (Q2) — the join key across faces
    "subtopic_score": 0.86,
    "section_title": "Review of Functions",    // raw heading, kept as provenance
    "est_tokens": 11881, "token_estimator": "chars/4",
    "sha256": "<full 64-hex of text bytes>",   // not truncated — dedup + attestation need the whole digest
    "provenance_url": "github.com/openstax/osbooks-calculus-bundle",
    "commit": "<pinned bundle commit>"         // the adapter already pins; carry it through
  } }
```

Deltas from what you shipped: add `text` (canonical docs, not an index); add
deterministic `id`; full-length `sha256`; `tokens` → `est_tokens` + method;
heading demoted to `section_title` with `subtopic`/`subtopic_score` added;
`book` → `source`/`subset`/`metadata.source_id`; pinned `commit` carried through.

## 4 · Architecture: the stage to build (and one thing to retire)

- **Build the corpus face as a sibling of the Assay face, from the migrated
  adapter.** `openstax.py` (CNXML → canonical records, commit-pinned) came to you
  in the producer migration — the first-cut throwaway script re-derived a worse
  version of what it already does. The real stage is: adapter → canonical docs →
  **+ subtopic tagging** (shared with the Assay face, one tagger, two consumers)
  → `_catalog.json` + canon rows. Same module list in, both faces out, keys
  shared.
- **Retire the pre-migration Lithos-side OpenStax output.** Lithos's local
  `/data/corpus-staging/openstax/` (12 books / 2,367 modules, from the
  pre-migration adapter run) becomes stale the moment your stage lands — two
  extraction paths will double-count in the mix and drift in text. When your
  canonical emission is green, that directory is deleted and your output is the
  single source of truth. Dedup in the funnel is a backstop, not a licence to
  ship the same books twice.
- **Reconcile the shelf inventory.** You report 5 bundles (~7–11 books); the
  pre-migration run had 12 books. Before the dump row is replaced, diff the two
  book lists so the canon rows cover the union and nothing silently drops.

## 5 · Fixed on our side in this pass

- `chisel-lithos-r2-contract.md` §3.1 example now includes `tier` (the executable
  schema always required it; the example was wrong, and your records were right).

## 6 · Acceptance for the real stage

Green means: (1) canonical `.jsonl.zst` per volume validating against
`documents.normalize` with zero `tier=unknown`; (2) every record carrying
`subtopic` + `subtopic_score` + stable `id`; (3) `_catalog.json` whose
`est_tokens` reconcile with the canon rows; (4) per-volume canon rows replacing
the dump row, `curation=generated`; (5) the same `subtopic` values queryable from
the Assay fixtures of the same modules — one topic query returns both faces; (6)
the Lithos-side staging copy deleted. Ship it against the calculus volume first;
we'll run it through the funnel (dedup → decontam → tier gate → tokenize) and
confirm end-to-end before you fan out to the shelf.
