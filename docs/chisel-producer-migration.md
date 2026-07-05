# Lithos → Chisel: Producer-Side Migration Spec

**Audience:** the Claude instance working in the **Chisel** repo.
**Author:** the Claude instance working in the **Lithos** repo (has full Lithos context; you do not).
**Status:** proposal + executable plan. Nothing has moved yet. Read §11 (open decisions) before you start cutting.

---

## 0. What this is

Lithos (the model foundry) has accreted a lot of **data-factory** code that overlaps Chisel's mission. We're splitting the data pipeline along one seam — the **canonical document record** — and moving the *producer* half to Chisel. This doc tells you exactly which files move, what they depend on, what stays in Lithos and why, and how the two repos hand off afterward.

**The one-line principle:** anything that turns *raw sources into canonical records* or *curates/generates* corpus material is **producer** → Chisel. Anything bound to the **training contract** (tokenizer, vocab, `seq_len`, shard byte-format, the eval battery) is **consumer** → stays in Lithos. Moving the consumer half would invert the dependency and put a repo boundary through Lithos's hottest loop, so it stays put.

---

## 1. The seam: the canonical record

Both halves meet at one data structure. Everything upstream of it is producer; everything downstream is consumer.

```jsonc
// The canonical document record (currently defined in lithos/data/documents.py)
{
  "id":        "string",            // stable id
  "text":      "string",            // REQUIRED, non-empty (records without text are dropped)
  "source":    "string",            // source_name, e.g. "openstax", "the-stack-stem"
  "subset":    "string | null",     // e.g. "physics", "python/jupyter"
  "language":  "string",            // natural language; default "en"
  "license":   "string",            // e.g. "cc-by-4.0", "mit", "grey"; default "unknown"
  "metadata":  { },                 // free dict; MUST carry "source_id" (CH-12 canon anchor)
  "quality_score": 0.0              // OPTIONAL float, added by the quality classifier
}
```

Serialized as **`.jsonl.zst`** (one JSON record per line, zstandard-compressed). Every extractor in Lithos already emits exactly this — they are "born ready" to move.

**`metadata.source_id`** resolves to a row in `corpus/seed_index.csv` (the Canon). This is the CH-12 provenance guarantee you already implement in Chisel; after this migration **Chisel owns the Canon** and Lithos merely references `source_id`s.

---

## 2. What MOVES to Chisel (producer)

### 2a. Extractors — source → canonical record (`lithos/data/`)

| File | Role | Heavy deps | Notes |
|---|---|---|---|
| `markup_text.py` (EX-1) | HTML/CNXML → math-aware text | lxml | pure transform; MathML→LaTeX |
| `pdf_extract.py` (EX-2) | PDF → math-aware text (Docling) | **docling** | the biggest dep to shed from Lithos |
| `stackexchange.py` | `.7z` XML dump → Q&A records | lxml, py7zr, zstandard | streaming + SQLite join |
| `openstax.py` | CC-BY textbook CNXML → records | zstandard (imports `markup_text`) | pins to approved commits |
| `stack_python.py` (EX-6) | The-Stack Python/Jupyter → records | pyarrow, zstandard | scientific-import + license filter |

### 2b. Curation / analysis / scoring (`lithos/data/`)

| File | Role | Notes |
|---|---|---|
| `topicgraph.py` | Wikipedia topic-graph (canon mining) | numpy, networkx |
| `overlap.py` | cross-corpus overlap estimation | numpy; QA/curation |
| `quality_classifier.py` | the owned v0 quality classifier (train + score) | numpy; produces `quality_score` |
| `labeling.py` | LLM quality-labeling (prompts/parse/agreement) | curation input to the classifier |

> **Chisel scores; Lithos thresholds.** `quality_classifier.py` (produce the score) moves. `quality.py` (apply a *threshold* at mix time) **stays** — the threshold is an empirical mix decision (§3).

### 2c. Scripts (`scripts/`)

Move: `acquire/acquire.py`, `extract_stackexchange.py`, `inventory_corpus.py`, `rank_canon.py`, `run_overlap_matrix.py`, `run_topic_graph.py`, `label_quality.py`, `make_pilot_set.py`, `train_quality_classifier.py`, `validate_seed_index.py`.

### 2d. Configs / canon data

Move: `corpus/seed_index.csv` (**the Canon — Chisel owns it now**), `corpus/acquisition.yaml`, `configs/quality/rubrics.yaml`, `configs/topicgraph/seeds.yaml`, `corpus/problems/` (verifiable task banks — RL/eval curation).

### 2e. Dependencies that move out of Lithos

The `pyproject.toml` extras `data = [lxml, py7zr]` and `pdf = [docling]` move to Chisel, plus `pyarrow` (for `stack_python`) and the HF `datasets`/`hf` CLI usage (acquisition). **This is a real win:** Lithos stops needing Docling/lxml/py7zr installed to train a model.

### 2f. Tests that move (`tests/`)

`test_markup_text.py`, `test_pdf_extract.py`, `test_stackexchange.py`, `test_openstax.py`, `test_stack_python.py`, `test_overlap.py`, `test_topicgraph.py`, `test_quality_classifier.py`, `test_labeling.py`. (Bring them; they're your regression net.)

---

## 3. What STAYS in Lithos (consumer — training-coupled)

| File(s) | Why it can't move |
|---|---|
| `tokenize.py`, `packing.py`, `shard.py`, `dataloader.py` | the training byte-format: vocab, `seq_len`, dual-stream loss-mask, the memmap layout the train loop reads |
| `decontam.py` | screens against the **eval battery** — a model-safety gate that must run right before training, in the training repo |
| `manifest.py` | the *training* corpus manifest (shards + provenance) that the dataloader reads |
| `quality.py` | applies a quality **threshold** — an empirical, bpb-sweep-driven mix decision |
| `pipeline.py` | the funnel — **split, don't move** (§4) |

Also staying: all `scripts/train_*`, `run_evals.py`, `run_ablation.py`, `tokenize_corpus.py`, `build_sft_corpus.py`, `prepare_*` (post-train data prep — coupled to the model/tokenizer), the tokenizer (`lithos/tokenizer/`, `train_tokenizer.py` — it *defines* the training contract), and `configs/{train,model,eval,tokenizer}/`, `corpus/probes/` (decontam probes).

---

## 4. The one real refactor: split `pipeline.py`

Today `lithos/data/pipeline.py` runs the **whole** funnel end-to-end:

```
documents → filter → dedup → tokenize → shards
```

It imports both halves: `documents`, `filters`, `dedup`, `minhash`, `quality`, `decontam`, `manifest`, `shard`, `tokenize`. After the split it must start from **canonical records** (Chisel's output), not raw sources:

```
Lithos pipeline.py (after):   read canonical .jsonl.zst
                                → [optional] cross-source dedup / quality-threshold / decontam
                                → tokenize → pack → shard → corpus manifest
```

So `pipeline.py` **stays in Lithos** but loses its front end. The extraction/cleaning it used to trigger becomes Chisel's job; Lithos's funnel input becomes "a directory of canonical `.jsonl.zst`."

---

## 5. The shared plumbing you must bring or share

The producer modules depend on **only two** Lithos things (verified by import graph — they do *not* import any training-coupled module):

1. **`lithos/data/documents.py`** — the canonical schema (`DocumentSource`, `normalize`) and readers (`read_jsonl`, `read_parquet`, `iter_documents`). This is the **contract**. Options:
   - **(recommended)** extract the *schema + normalize* into a tiny shared spec package both repos depend on; Lithos keeps the **readers** (it consumes), Chisel gets a **writer** that conforms.
   - or: Chisel duplicates the record shape from this spec and you version it by hand (§11).
2. **`lithos/utils/io.py`** — zstandard/JSON read-write helpers. Small; copy the handful of functions the extractors use (`write_json`, the `.jsonl.zst` open/stream helpers) or fold them into the shared lib.

Nothing else. No extractor imports `tokenize`/`pack`/`shard`/`dataloader`. That's why this is a clean cut.

---

## 6. The handoff after the split

1. **Chisel produces** canonical `.jsonl.zst` per source (it already does for the extractors), tagged with `source`, `subset`, `license`, `metadata.source_id`, and optional `quality_score`. Recommendation: **normalize *every* source to canonical `.jsonl.zst`** (not raw parquet) so Lithos only needs `zstandard`, not `pyarrow`/`docling`/`lxml`. (Tradeoff in §11.)
2. **Chisel owns the Canon** (`seed_index.csv`) and the acquisition specs. It writes the raw + canonical mirrors to the shared store (R2 `s3://lithos-data-fineweb-edu`, or the local `/data2` pretrain volume).
3. **Lithos ingests** via `configs/data/p0-sources.yaml` — a catalog of `DocumentSource` entries whose `paths` point at Chisel's canonical output. **This catalog stays in Lithos** (the *mix* is Lithos's empirical decision) but references Chisel's files. Think of it as Lithos's "bill of materials," Chisel's output as the warehouse.

Net: Chisel = the factory (sources, extracts, cleans, curates, scores, generates). Lithos = the furnace (reads canonical records → dedup/decontam/mix → tokenize/pack/shard → train/eval/post-train). The canonical `.jsonl.zst` is the pallet between them.

---

## 7. Migration plan (ordered, low-risk)

1. **Stand up the shared contract** (§5.1) first — the schema + io helpers as a small package (or a versioned spec Chisel copies). Nothing else can move cleanly until this exists.
2. **Move the extractors** (§2a) — they're self-contained and already emit the contract. Bring their tests (§2f). Add the `data`/`pdf`/`pyarrow` deps to Chisel.
3. **Move curation/scoring** (§2b) + their scripts/configs (§2c/2d).
4. **Move the Canon** (`seed_index.csv` + `acquisition.yaml` + `validate_seed_index.py`) — Chisel becomes the source of truth for provenance.
5. **In Lithos:** refactor `pipeline.py` to start from canonical records (§4); drop the `data`/`pdf` extras from `pyproject.toml`; delete the moved modules; repoint `p0-sources.yaml` at Chisel's output dir.
6. **Verify both sides** (§10).

Do 1→2 first and stop; confirm Lithos still builds a corpus reading Chisel-produced `.jsonl.zst` before moving the rest. Don't big-bang it.

---

## 8. Judgment calls — decide these explicitly (don't let me pre-decide)

| Module | The call |
|---|---|
| `filters.py` | Heuristic content cleaning (length/symbol-ratio). Cleaning is producer-ish, but it's wired into Lithos's funnel. **Lean:** move the cleaning heuristics to Chisel (produce cleaner records); leave a thin mix-time filter in Lithos if needed. |
| `dedup.py` / `minhash.py` | Near-dedup is mechanical and not training-coupled, so it *could* be Chisel. But cross-source dedup naturally runs at mix-assembly, which is Lithos. **Lean:** keep in Lithos for now (mix-time), revisit if Chisel wants to ship pre-deduped sources. |
| `quality.py` | Score in Chisel, threshold in Lithos (already split above). Confirm you're happy with that seam. |
| output format | Normalize everything to `.jsonl.zst` (clean seam, Lithos sheds pyarrow) **vs** let Lithos read some raw parquet directly (fewer transforms, but Lithos keeps pyarrow + format knowledge). Recommend the former. |
| shared contract | Real shared package (import-coupled, always in sync, but two repos now share a build dep) **vs** copied-and-versioned spec (looser, but can drift). Recommend the package if your tooling makes cross-repo deps easy. |

---

## 9. Acceptance criteria

- **Lithos:** `pyproject.toml` no longer lists `docling`/`lxml`/`py7zr`; `uv run pytest` green with the producer modules + their tests removed; `pipeline.py` builds a tokenized corpus reading canonical `.jsonl.zst` produced by Chisel; a training smoke run still works.
- **Chisel:** the moved extractors + curation run and emit schema-valid canonical `.jsonl.zst`; `seed_index.csv` validates; `metadata.source_id` on every record resolves to a Canon row.
- **Interface:** a record written by Chisel round-trips through `lithos.data.documents.read_jsonl` → `normalize` without loss.

---

## 10. Non-goals / cautions

- **Don't move the tokenizer, packing, sharding, dataloader, decontam, or the mix logic.** They're the training contract. If you find yourself wanting them in Chisel, re-read §0.
- **Don't move `build_sft_corpus.py` / `prepare_*`** — post-training data prep is tokenizer/model-coupled (it packs SFT/DPO shards in the training format), even though it "feels" like data work.
- The canonical schema becomes **load-bearing across two repos.** Version it deliberately; a breaking change now costs a coordinated two-repo release.

---

*Generated from the Lithos repo at merge `d4fd97f`. The module inventory and dependency claims were verified against the tree, not recalled — but re-check before deleting anything, since Lithos keeps moving.*
