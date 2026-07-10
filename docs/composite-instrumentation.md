# Composite instrumentation spec

**Status: spec — not yet built.** The measurement apparatus that lets us say whether a composite
(R1 retrieval, TIR-in-server) is actually better than the naked SLM, and *why*. Implements the
experiments C0–C5 of `docs/composite-model-layer.md` §11.

Read `docs/composite-model-layer.md` first — especially **§10 (the baseline is the instrument)**.
This document is how §10 becomes code.

---

## 1. The principle

> **Instrument only where information is destroyed. Recompute everything else.**

Most per-stage logging duplicates a deterministic function of things we already keep, and then
gives us two sources of truth that eventually disagree. The question is never "should we measure at
this stage" but **"is this stage lossy?"**

By that test the pipeline has exactly **three lossy points**, and everything else is recomputable.

---

## 2. The three capture points (and the four non-captures)

### Capture

| # | What | Why lossy | Cost |
|---|---|---|---|
| **P1** | **Checkpoints on the training trajectory** at each C0 token budget | Cannot be recovered without retraining. C0 needs the *same run* at 1 / 2 / 4 epochs — separate runs confound token budget with init + data order. | ~free (we already checkpoint) |
| **P2** | **The per-token decode record** at eval time | The final string, and even the final bpb, cannot be decomposed back into per-token, per-neighbour contributions. | small — eval set only |
| **P3** | **The `runs` manifest** binding P1↔P2 | The four-tuple + `tokens_seen` is the only thing that makes two rows comparable. | trivial |

### Do **not** capture

| What | Because |
|---|---|
| Exposure (which docs seen by step *N*) | **Reconstructible**: `RandomState(seed + epoch).permutation` + the checkpoint's `dataloader_state` + `docindex.parquet`. See `docs/petra-provenance-lithos.md`. |
| bpb / benchmark aggregates | Deterministic functions of (checkpoint, dataset). Recompute; never store a second copy. |
| Per-token quantities **during training** | Terabytes, to reproduce what one forward pass regenerates on demand. |
| Neighbour lists for *every* token | See §7 — scalars for every token, neighbour lists sampled. |

**Recompute > log, whenever the function is deterministic and its inputs are retained.** This is
what the seed/`dataloader_state` discipline bought us; spend it.

---

## 3. Schemas

Three Parquet tables. `pyarrow` + `polars` are already core deps.

### 3.1 `runs` — one row per measured configuration

The `served_model_id` four-tuple (`docs/composite-model-layer.md` §7.1), made concrete.

| Column | Type | Notes |
|---|---|---|
| `run_id` | str | content hash of the rest of this row |
| `weights_sha256` | str | |
| `tokens_seen` | int64 | the C0 axis |
| `epochs_seen` | float32 | **must be a whole number for C0** (§5) |
| `n_params` | int64 | the C1 axis |
| `datastore_version` | str? | null ⟹ baseline |
| `decode_policy_version` | str | λ schedule, grammar, logit masks |
| `tool_env_sha` | str? | the pinned sandbox env (`CHECKER_IMPORT_SET`) |
| `composite` | enum | `none` / `knn_lm` / `retro` / `tir` |
| `knn_k`, `ann_params` | int, str? | |
| `eval_set_id`, `eval_set_sha256` | str | |
| `vram_bytes`, `ssd_bytes` | int64 | **C3b** — the scarce-resource comparison (§10.5) |
| `lithos_git_sha` | str | |

### 3.2 `tokens` — one row per (run_id, doc_id, pos)

Teacher-forced over the eval set. Partition by `run_id`.

| Column | Type | Notes |
|---|---|---|
| `run_id`, `doc_id`, `pos` | str, int32, int32 | |
| `episode_id` | str? | joins to `episodes`; null for plain LM eval |
| `token_id` | int32 | |
| `byte_len` | uint8 | **UTF-8 bytes of this token** — required, see below |
| `logprob_lm` | float32 | the naked LM's log p(actual token) |
| `p_knn_true` | float32? | probability the kNN distribution assigns the actual token |
| `masked` | bool | TIR `tool_result` spans — **excluded from bpb** |
| `freq_bucket` | uint8 | from a precomputed corpus unigram count (§6) |
| `is_numeric`, `is_unit` | bool | cheap tokenizer-class flags (§6) |
| `nb_dsrow` | list\<int32\>? | datastore row-ids, **sampled** (§7) |
| `nb_dist` | list\<float32\>? | |

**`byte_len` is not optional.** bpb = `Σ nll_bits / Σ byte_len`. Storing bytes-per-token is what
makes the table tokenizer-agnostic, which is the entire reason we compare in bits-per-byte rather
than perplexity — the from-scratch 500M (32k vocab) and the Qwen-lineage 4B (151k vocab) are only
comparable in bpb.

**We store `logprob_lm` and `p_knn_true` separately, never the interpolated result.** Because

```
p_composite(t) = (1-λ)·p_LM(t) + λ·p_kNN(t)
nll_bits(t)    = -log2 p_composite(t)
```

is a pure function of two stored floats and λ. **The entire λ sweep therefore becomes a query, not
a set of runs.** One forward pass over the eval set yields Δbpb(λ) for every λ.

### 3.3 `episodes` — one row per TIR / RLVR task episode

The unit of measurement for tool use is the *answer*, not the token.

| Column | Type |
|---|---|
| `run_id`, `episode_id` | str, str |
| `task_id`, `family_id`, `kind` | str, str?, enum(`numeric`/`symbolic`/`code`/`units`) |
| `verdict` | bool |
| `detail` | str? |
| `n_tool_calls`, `tool_runtimes` | int32, list\<str\> |
| `tool_wall_ms` | int32 |
| `n_tokens_emitted`, `n_tokens_masked` | int32, int32 |
| `seed` | int64 |

### 3.4 `datastore_rows` — the neighbour dictionary

`dsrow (int32) → (source_id, record_id, text_sha256)`. Keeping `nb_dsrow` as int32 rather than
64-char hex strings is a ~16× size reduction on the widest column in `tokens`.

**These three keys are the join to Petra and Chisel** — the same `source_id` / `record_id` /
`text_sha256` fixed in the reconciliation. This is why `text_sha256`-on-every-record (the open
Chisel ask) is load-bearing here and not just for Petra.

---

## 4. Every experiment is a query, not a harness

The point of one wide table: you cannot later discover you aggregated away the thing you needed.

```sql
-- bpb helper (always: WHERE NOT masked)
bpb(x) := sum(-log2(x)) / sum(byte_len)

-- C0: is the retrieval gain absorbed by more training?   [THE GATE]
SELECT epochs_seen,
       bpb(logprob_lm) - bpb(interp(logprob_lm, p_knn_true, :lambda)) AS delta_bpb
FROM tokens JOIN runs USING (run_id)
WHERE n_params = 100e6 AND NOT masked
GROUP BY epochs_seen;            -- decays -> crutch, kill R1.  plateaus -> substitution.

-- C1: does the plateau FLOOR rise as params shrink?  (read only after C0 says "plateau")
... GROUP BY n_params, epochs_seen;   -- compare floors, never peaks

-- C2: datastore size/gain knee
... GROUP BY datastore_version;       -- x-axis = |datastore|, log-spaced

-- C3b: equal-scarce-resource comparison
... GROUP BY (vram_bytes, ssd_bytes); -- 500M + N-GB store  vs  naked model N GB larger

-- The §6 discriminator: crutch is diffuse, substitution is concentrated
... GROUP BY freq_bucket;

-- C4: tool use, from `episodes`
SELECT n_params, kind, avg(verdict) FROM episodes JOIN runs USING (run_id) GROUP BY 1, 2;
```

**C3 (latency) is not in these tables.** ANN-vs-forward-pass latency is a device benchmark, not an
offline eval; it writes `p50_ms` / `p99_ms` into `runs` and nothing else. Keep it separate rather
than pretending it's the same instrument.

---

## 5. The whole-epoch constraint (a bug that would silently invalidate C0)

If the datastore is built from the **full** corpus but the 2B-token checkpoint has only *seen* a
fraction of it, then at 2B the datastore is partly **fresh facts** — we are measuring mutability —
while at 40B it is genuinely corpus-internal. That inflates the early Δ, exaggerates the apparent
decay, and makes a real substitution effect look like a crutch. **C0 would return the wrong answer
and kill R1 for the wrong reason.**

**Constraint:** every C0 checkpoint sits on a **whole-epoch boundary of a fixed corpus** — 1 / 2 /
4 epochs. Then every checkpoint has seen every document, and the only thing varying is *how well it
learned*, never *what it saw*. Our 3–4-epoch regime already has this shape.

**Assert it, don't assume it:** reconstruct the exposure set at each checkpoint (§2) and check it
covers every `text_sha256` in the datastore.

### 5.1 The other way to fool yourself: eval-in-datastore

If the eval set is inside the datastore, kNN retrieves the answer verbatim, Δbpb is enormous, and
the result is worthless. "Corpus-internal" means **the training corpus**, never the held-out set.

**Assert `datastore ∩ eval = ∅` on `text_sha256`** at run construction, and fail loudly. The
corpus-level decontam pass should already guarantee this; the assert is the re-check, and it is the
single highest-value line of code in this spec.

---

## 6. The cheap discriminator (independent of C0)

The two hypotheses of §10.3 have **different signatures in the token distribution**:

- **Crutch** — the model is undertrained, therefore bad at *everything*. Retrieval lifts it
  **diffusely, across common tokens**.
- **Substitution** — a claim about capacity for *facts*. The gain **concentrates on rare tokens**:
  numbers, entities, units, technical terms.

So stratify Δbpb by token frequency. `freq_bucket` = log-spaced decile of the token's unigram count
in the training corpus (bucket 0 = rarest). `is_numeric` / `is_unit` are cheap tokenizer-class
flags giving a second, semantic cut.

This costs **one eval pass over one checkpoint** — no extra training — and it is *independent
evidence* for the same question C0 answers with three training budgets. **Run it first.**

If the frequency-stratified picture and the token-budget sweep agree, believe the answer. **If they
disagree, the setup is broken** — and you want to know that before committing the flagship, not
after.

---

## 7. Volume, and the one place we sample

Scalars are ~40 B/row. A 5M-token eval set is ~200 MB/run — keep them for every token, always.

Neighbour lists at `k=16` are the wide column (~64 B with int32 `dsrow`, plus distances). Across
(3 token budgets × 2 sizes × several datastore sizes) they dominate.

**Rule: scalars for every token; neighbour lists for a 1% uniform sample ∪ all tokens in the two
rarest `freq_bucket`s.** The rare buckets are exactly where §6 does its work, so the sampling is
aligned with the science rather than against it. **Record the sampling rate in `runs`** — a silent
cap reads as full coverage.

---

## 8. Where it lands

| Path | What |
|---|---|
| `lithos/evals/instrument.py` | `TokenRecorder` / `EpisodeRecorder` + the Parquet schemas; `run_id` hashing |
| `lithos/evals/perplexity.py` | extend `compute_perplexity` to emit per-token rows (it already does the teacher-forced pass) |
| `lithos/evals/composite_queries.py` | C0–C5 as polars queries over the tables |
| `lithos/serve/composite.py` | the kNN-LM decode hook: emits `p_knn_true` + neighbours; **λ applied at query time, not here** |
| `configs/eval/instrument.yaml` | eval set, k, datastore version, sampling rate |
| `tests/test_instrument.py` | §9 |

Reuse `lithos/evals/ablation.py`'s scorecard pattern (intervention → proxy → eval → diff against a
named baseline) — C0 is an ablation whose intervention axis is `epochs_seen`.

---

## 9. Verifying the apparatus itself

An instrument that hasn't been calibrated measures its own bugs. Four tests, all cheap:

1. **λ=0 identity (the most valuable test).** With λ=0 the composite path must reproduce the
   baseline logprobs **bit-exactly**. Any drift means the two paths differ in context, dtype, or
   masking — and every Δbpb you ever compute is that bug.
2. **Sum check.** bpb recomputed from `tokens` equals the eval harness's reported bpb to float
   tolerance. Guards the `byte_len` / `masked` handling.
3. **Decontam assert.** `datastore ∩ eval = ∅` on `text_sha256` (§5.1). Fail the run, not the paper.
4. **Exposure assert.** Reconstructed exposure at each C0 checkpoint ⊇ datastore (§5).
5. **Determinism.** Re-running a `run_id` reproduces `nll_bits` exactly (seeded, `PYTHONHASHSEED=0`).

---

## 10. Non-goals

- **Not** a training-time profiler. No per-token logging during training.
- **Not** a serving telemetry system. This measures *architectures*, not production traffic.
- **Not** Petra. Attribution consumes these tables (join keys in §3.4) but lives in its own repo —
  see `docs/petra-composite-attribution.md` for what changes on their side.

---

## 11. Build order

1. `runs` + `episodes` + the §9 tests. **Works today** — TIR/RLVR is in the MVP, so the episode
   table has a live customer before R1 exists.
2. `tokens` + the frequency stratification (§6) against the **naked** baseline. Still no composite:
   this is pure baseline characterisation, and it is the §10 ruler.
3. `lithos/serve/composite.py` kNN-LM hook + `datastore_rows`.
4. **C0.** Gate. If Δ decays with `epochs_seen`, R1 is scaffolding — stop, and the apparatus has
   already paid for itself.
