# Chisel → Lithos: R2 Output Contract & Run Flow

**Companion to** `docs/chisel-producer-migration.md` (which says *what code* moves to Chisel).
This doc says **what the finalized, training-ready artifacts are, in what format, and where they live** — a three-tier storage model (local HDD → local NVMe → R2) plus the end-to-end run flow across the rented cluster, the H100, and the local box.

**Golden rule:** raw data stays on the local HDD, work-in-progress on the local NVMe, and **only the finalized artifacts the rented GPUs need cross through R2** — each a versioned, immutable object. No file is hand-carried between machines.

---

## 1. Storage tiers — and what lives on R2

Data sits in the tier that matches its lifecycle; **only the finalized, cloud-critical tier reaches R2.** This keeps R2 small (you never pay cloud rates for bulk raw data) and matches how the pipeline runs — locally, until a rented GPU needs something.

| Tier | Medium | Holds | Why |
|---|---|---|---|
| **Raw** | 24 TB HDD (local, cold) | source mirrors (Canon-keyed) + bibliography evidence (content-addressed) | bulk, write-once, re-downloadable; no cloud box reads it |
| **WIP** | 2 TB NVMe (local, hot) | canonical records, filtered/deduped, tokenize-in-progress, working sets | fast local processing; re-derivable from raw |
| **Finalized** | R2 (cloud, durable) | tokenized shards, tokenizer, post-train JSONL, checkpoints, models | the *only* things the disposable rented GPUs read/write |

**The rule:** if a rented cloud box must read or write it, it's in R2; otherwise it stays local. Raw + WIP are re-derivable (so they don't need R2's durability); the finalized outputs are irreplaceable (so they do). On the HDD, two provenance layers coexist — `raw/` keyed by Canon `source_id` (whole sources), and `evidence/` content-addressed (the ~2,700 page-level syllabus snapshots behind adoption edges, which don't resolve to a Canon row).

### On R2 — separate buckets, IAM-enforced

Only the finalized tier lives here, in per-owner buckets; "Chisel writes / Lithos reads" is enforced by **IAM, not convention** (each side writes only its own buckets; a buggy job can't cross the line).

**Chisel-written** · *Lithos: read-only*
```
lithos-canon/       seed_index.csv, acquisition.yaml            # small provenance index (source_id resolution)
lithos-posttrain/   {sft,tir,prefs,rl-tasks}/<version>/*.jsonl  # finalized post-train inputs the H100 reads (§3.3–§3.6)
```
**Lithos-written** · *Chisel: no access*
```
lithos-tokenizers/  <name>/tokenizer.json                       # the training contract
lithos-tokenized/   <corpus_version>__<tokenizer_version>/…     # finalized pretrain shards the cluster mmaps + manifest.json
lithos-checkpoints/ <run_name>/step_NNNNNN/                     # model.safetensors + train_state.pt + meta.json
lithos-models/      <name>-<version>/  {hf_export,eval,run}/    # exported final model
```

**Not on R2 (local only):** raw mirrors + evidence → HDD; canonical pretrain corpus + all intermediate processing → NVMe. Chisel produces canonical records **locally**, Lithos tokenizes them **locally**, and only the **tokenized shards** are pushed to R2 — so the Chisel→Lithos *pretrain* handoff is local (NVMe), and R2 carries only what the cluster consumes. *(If re-extracting some source is expensive — e.g. a Docling PDF pass — mirror just that canonical output to the HDD as a cold cache; still off R2.)*

**Key boundary unchanged:** Chisel stops at **canonical records + post-training JSONL** and never tokenizes (§4) — the split just runs across local tiers now, with R2 as the local→cloud line.

---

## 2. Output contract — per pipeline stage

| Stage | Chisel produces | Format | Lands (local vs R2) | Lithos reads it with |
|---|---|---|---|---|
| **Pretrain corpus** | canonical documents | `.jsonl.zst`, schema §3.1 | **local NVMe** (WIP) | `read_jsonl` → mix → tokenize → shards to R2 |
| **Mix manifest** | catalog of the above | `_catalog.json`, §3.2 | local NVMe | drives `p0-sources.yaml` selection |
| **Canon** | provenance index | `seed_index.csv` | `lithos-canon/` (R2) | `source_id` resolution + validation |
| **SFT** | instruction conversations | messages-JSONL, §3.3 | `posttrain/sft/<v>/` | `posttrain.sft_corpus` → packed shards |
| **TIR-SFT** | tool-use reasoning traces | messages+segments, §3.4 | `posttrain/tir/<v>/` | `posttrain.sft_corpus` (TIR render) |
| **Preferences** | harvested/synthetic pairs | `{prompt,chosen,rejected}`, §3.5 | `posttrain/prefs/<v>/` | `posttrain.preference_dataset` → DPO |
| **RL tasks** | verifiable problems | taskbank JSONL, §3.6 | `posttrain/rl-tasks/<v>/` | `posttrain.taskbank.load_tasks` → GRPO |

> **Not Chisel's job:** *on-policy* verifier preferences are generated **by Lithos on the H100** (sample from the model → E1-verify). Chisel supplies the *raw ingredients* — the RL task banks and any harvested/synthetic prefs; Lithos manufactures the on-policy data during post-training.

---

## 3. Exact schemas

### 3.1 Canonical pretraining record (`.jsonl.zst`, one per line)
```jsonc
{ "id": "str", "text": "str (required, non-empty)", "source": "str", "subset": "str|null",
  "language": "en", "license": "str", "metadata": { "source_id": "<canon row>", "domain": "physics" },
  "quality_score": 0.0 }          // quality_score optional
```

### 3.2 Corpus catalog (`_catalog.json`) — the mix bill-of-materials
```jsonc
{ "corpus_version": "v0.2", "created": "<iso>",
  "sources": [ { "source": "the-stack-stem", "domain": "code", "subset": "python/jupyter",
                 "path": "corpus/v0.2/code/the-stack-stem/*.jsonl.zst",
                 "license": "permissive", "docs": 812345, "est_tokens": 4.1e9 } ] }
```
Lithos's `p0-sources.yaml` selects from this (the *mix weights* stay Lithos's empirical call; the *inventory* is Chisel's).

### 3.3 SFT (`{"messages": [...]}` per line)
```jsonc
{ "messages": [ {"role":"user","content":"..."}, {"role":"assistant","content":"..."} ],
  "source_id": "<canon row>" }        // extra keys ignored by Lithos; keep source_id for provenance
```

### 3.4 TIR-SFT traces
Same messages-JSONL, but assistant turns carry **`segments`** (`think` / `text` / `tool` / `tool_result`) using the exact string tokens `<think>`, `<|python|>`, `<|octave|>`, `<|/tool|>`, `<|tool_result|>`. **Authoritative schema: `docs/tir-format.md` §5** — write to that spec verbatim; Lithos renders + masks by token ID, so tool-result payloads must be their own segment (they're excluded from the loss).

> **Lithos-side guard (to build before the first TIR trace is ingested):** because Lithos masks by token ID, a malformed `tool_result` segment would silently poison the loss mask — the worst kind of data bug, invisible until it degrades the model. Lithos will **validate incoming TIR traces at ingestion and fail loud** on malformed/mistyped segments rather than trust the producer. The validator is a backstop, not a license to be sloppy: write to §5 exactly.

### 3.5 Preferences (`{prompt, chosen, rejected}` per line)
```jsonc
{ "prompt": [ {"role":"user","content":"..."} ], "chosen": "str", "rejected": "str" }
```

### 3.6 RL task bank (verifiable problems, per line)
```jsonc
{ "id": "opt", "prompt": "str", "kind": "numeric|symbolic|code|units",
  "answer": "str", "tests": "code-harness (kind=code)", "units": "kPa (kind=units)",
  "tol": 1e-6, "level": "opt", "year": 2024, "metadata": {} }
```
`id` is optional (Lithos derives one from the prompt); `year` drives the RLVR-pool / eval-set split, so **stamp it** if you want contamination-safe eval separation.

---

## 4. The tokenization seam (why Chisel stops at records)

Tokenizing needs the exact tokenizer + vocab = the training contract → Lithos. But you don't want the **8×H200 burning money tokenizing on boot.** Resolution: Lithos runs **tokenize/pack as a cheap prep step off-cluster** (the 4070 box or a small CPU VM), writing tokenized shards to R2 keyed by **`<corpus_version>__<tokenizer_version>`**. The cluster then just pulls those and trains.

So the derived-artifact cache is content-addressed by *both* versions — change the corpus **or** the tokenizer and you get a fresh shard set; nothing stale is ever silently reused.

---

## 5. End-to-end run flow

```
A. CHISEL (local, no GPU, whenever data changes)
   source raw → local HDD;  extract / clean / curate / generate
   → canonical records + mix catalog to local NVMe (WIP)
   → push only finalized post-train JSONL (+ canon index) to R2.

B. TOKENIZE/PACK (Lithos prep, local NVMe or a small CPU VM)
   read local canonical corpus + tokenizer  →  tokenize → pack → shard
   → push tokenized/<cv>__<tv>/ to R2.         # the finalized pretrain input; once per (mix, tokenizer)

C. PRETRAIN (rent 8×H200)
   boot → pull tokenized/<cv>__<tv>/ to local NVMe → mmap-train
   → push checkpoints/<run>/step_* to R2 every N steps
   → export models/<name>-<ver>/ to R2 → SPIN DOWN.

D. POST-TRAIN (rent 1×H100)
   boot → pull the pretrained model + posttrain/{sft,tir,prefs,rl-tasks}/
   → SFT → (generate on-policy prefs) → DPO → GRPO-TIR
   → push a checkpoint to R2 after each stage/step
   → export the final model to R2 → SPIN DOWN.

E. EVAL / INFERENCE (local 4070S)
   pull models/<name>-<ver>/ from R2 → run inference + full eval battery locally
   → scorecards land locally (optionally push to models/<ver>/eval/).
```

Everything in Lithos already speaks this via `lithos/utils/storage.py` (`Storage.get`/`.put`, fsspec over R2). GitHub carries the *code + configs*; the **local HDD + NVMe** carry *raw + WIP*; **R2** carries the *finalized data + weights*; the rented machines are stateless and disposable.

---

## 6. Versioning & immutability (the rules that keep it sane)

- `corpus_version`, `tokenizer_version`, and `model_version` are explicit and immutable. Never overwrite a version — bump it.
- Tokenized shards are keyed by `corpus × tokenizer` (§4) — the only safe cache key.
- Every record carries `metadata.source_id` → a Canon row. A record whose `source_id` doesn't resolve is a bug, not a warning (CH-12).
- Post-training checkpoints are written **after each step/stage** (the user's requirement) so a spot-terminated H100 loses at most one interval.

---

## 7. Decisions (settled 2026-07-05)

1. **Separate buckets, enforced by IAM — not convention.** Chisel creds = write on the producer buckets only; Lithos = read-only there + write on its own. The golden rule is a *permission boundary*; a buggy job can't cross it.
2. **Tokenize/pack runs on a small spot CPU VM near R2** (fast pull/push, cheap) — Lithos's operational call.
3. **Tokenized shards pull to local NVMe** before training (throughput) — Lithos's call.
4. **Scorecards mirror into `lithos-models/<ver>/eval/`** — self-describing model packages (matches `lithos-100m-v0.1`).
5. **Three-tier storage** (§1): raw → 24 TB local HDD, WIP → 2 TB local NVMe, finalized → R2. R2 is the local→cloud boundary, sized to only what the rented GPUs touch — not a mirror of everything.

Two gaps raised by Chisel and resolved into this doc: **(a)** page-level bibliography evidence had no home → its own content-addressed store on the HDD raw tier (§1); **(b)** TIR loss-mask poisoning → a loud Lithos-side validator (§3.4).

---

*Written from the Lithos repo at `d4fd97f`. Formats verified against the actual consumers (`sft_dataset`, `preference_dataset`, `taskbank`, `documents`); the TIR schema defers to `docs/tir-format.md` §5 as authoritative.*
