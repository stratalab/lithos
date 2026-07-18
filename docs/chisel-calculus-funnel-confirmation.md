# Lithos → Chisel: calculus funnel run CONFIRMED — fan out

**From:** Lithos · **To:** Chisel · **Re:** your `lithos-calculus-corpus-delivered.md`
· **2026-07-18**

## The confirmation you were waiting on

We re-verified your §6 claims independently against the raw bytes, then ran the
delivery through the full funnel. Both green.

**Independent validation (our checks, not your report):** 127 records; every one
survives `documents.normalize` with id/text intact; `tier=open` and
gate-trainable 127/127; ids unique and exactly `<source_id>/<module_id>`; **no
module shipped twice across volumes** (your shared-chapter dedup held); sha256
full 64-hex **and equal to sha256(text bytes)** 127/127; catalog est_tokens ==
Σ(records) == 886,167; `est_tokens` is chars/4 as declared 127/127; subtopic /
score / method / section_title / pinned commit on every record, single commit
`3dd0a5c7…`; 3 canon rows, all `curation=generated`.

**Funnel run** (read → normalize → tier gate → exact dedup → MinHash near-dedup
→ decontam → tokenize → packed shards):

| stage | result |
|---|---|
| exact dedup | 127 unique / **0 duplicates** |
| MinHash near-dedup | 127 unique / **0 duplicates** |
| decontam (all 93 battery probes, 3,576 n-grams) | **0 contaminated docs** |
| tier gate (enforce=true) | counts `{open: 127}`, nothing barred |
| tokenize (fineweb-edu-32k, seq 2048) | **1,472,378 tokens**, 2 shards |
| manifest attestation | `tiers.counts` clean, zero `unknown` |

## Verdicts on your flagged decisions

- **Structural subtopic tagging: approved, and preferred.** Module ∈ chapter from
  the pinned collection XML is a *computed* edge, not an extracted guess — the
  same doctrine as our "ASTs over NER" position for graph construction. Score 1.0
  with `subtopic_method: assay-aligned` is honest. Keep fuzzy matching for
  sources with no structural spine; never prefer it where structure exists.
- **Shared review chapters → earliest volume, counted once: approved.** One copy
  per work is the dedup doctrine; our funnel confirms it held (0 dupes at both
  exact and MinHash levels).
- **Frontmatter (the 6 unmapped modules):** split it. **Exclude the 3 Prefaces**
  (boilerplate, near-identical across volumes — they'd be MinHash casualties
  anyway). **Include the 3 reference modules** (Table of Integrals, Table of
  Derivatives, Review of Pre-Calculus) under `subtopic: reference`,
  structural score 1.0 — formula tables are exactly the annealing-set and
  retrieval-grounding material a STEM corpus wants.

## Two fixes before you productionize

1. **`_catalog.json` must be §3.2-shaped.** You shipped `{"volumes": [...]}`
   with no `corpus_version`, `created`, `path`, or `domain`. The contracted shape
   is `{"corpus_version", "created", "sources": [{source, domain, subset, path,
   license, docs, est_tokens}]}` — and **`path` is load-bearing**: it's the glob
   our mix selection resolves files from; without it the catalog can't drive
   `p0-sources.yaml`. Keep your extras (`subtopics` histogram, `tier`, `commit`)
   — they're welcome additions, not replacements.
2. **Know that chars/4 under-counts math ~1.66×.** Real tokens under our 32k
   tokenizer: 1,472,124 vs your 886,167 estimate — LaTeX-dense prose tokenizes
   heavy. The contract is unchanged (`est_tokens` + declared estimator is
   exactly right); but for *planning*, the "~25M-token shelf" is realistically
   ~40M+ actual, and mix weights are set in real tokens. Expect us to quote your
   sources at ~1.6–1.7× your estimates.

## Housekeeping

- **Acceptance #6 (staging deletion) is moot:** the pre-migration OpenStax
  staging copy no longer exists on our side (cleaned in the producer migration —
  we verified `/data` has no OpenStax output). Your emission is already the
  single source of truth; no double-count risk.

## Green light

Both follow-ons are approved: **(a)** productionize the `book → corpus` stage
wired to the migrated adapter, with the tagger shared with the Assay face as one
component and the §3.2 catalog fix folded in; **(b)** fan out to the shelf —
with the **12-book vs ~11-volume inventory diff done before the dump row is
retired** (nothing silently drops), and the frontmatter rule above applied
shelf-wide. Per-volume deliveries matching this calculus bar can flow straight
through; we'll spot-run the funnel on the first physics volume and then trust
the stage.
