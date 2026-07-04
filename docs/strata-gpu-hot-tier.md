# StrataDB GPU Hot Tier — Requirements (2026-07-04)

**Status: BANKED, post-MVP** (same gate as research tracks R1/R2 — nothing here
touches the P0 → mix → tokenizer → 500M critical path). This document records
*requirements*, not design: what the tier must do so that when R1/R2 kick off,
StrataDB's side of the contract is already specified. Companion documents:
`docs/moho.md` (the kernel layer that consumes this tier) and `docs/chisel.md`
(unrelated leg, recorded in the same pass).

## 1. Purpose and the one framing decision

Moho/R2's thesis — KV cache and facts live in StrataDB, not in weights or raw
VRAM — requires StrataDB to participate in the GPU's memory hierarchy. The
framing decision, made explicitly: **this is a tiered cache, not a port.** A
fully GPU-resident StrataDB would offload nothing — it would reorganize the
same scarce VRAM the KV cache already exhausts. The requirement is therefore a
**GPU-native hot tier over the existing store of record**:

| Tier | Medium | Holds | Role |
|---|---|---|---|
| **T0** | GPU device memory | hot KV pages, retrieval index (or shards), hot graph adjacency (CSR), page summaries, page table | serve the decode loop at compute speed |
| **T1** | host RAM (pinned) | warm pages, staging buffers | async promotion/demotion buffer |
| **T2** | SSD — StrataDB proper | full KV history, full graph, consolidated memories | store of record; the only durable tier |

T0/T1 are **write-back caches** — losing them loses no data, only warmth. All
durability semantics (crash consistency, WAL) remain T2-only, unchanged from
StrataDB today. This keeps the GPU work out of the correctness-critical path.

## 2. Scope

**In scope:** the three device-resident structures (page pool, index,
adjacency), the promotion/demotion machinery between tiers, and the device-side
query interface Moho's kernels call.
**Out of scope:** the kernels themselves (Moho), multi-process/multi-tenant
access (StrataDB is embedded, single-process — this simplification is
load-bearing and deliberate), network anything, and any change to T2's
on-disk format.

## 3. Functional requirements

**HT-1 · Device page pool.** Fixed-size KV pages (page size a build-time
parameter, expected 16–64 tokens, aligned to the paged-attention block size),
allocated from a pre-reserved device arena with a page table mapping logical
page id → (tier, address). Reference counting so an in-flight attention step
can never have a page evicted under it.

**HT-2 · Device-resident retrieval index.** Given query vector(s), return the
top-k candidate pages **without a host round-trip** — results land in device
memory, consumable by the next kernel in the same stream. The index form is a
design decision deferred to Moho v0 (page-summary bounds à la Quest vs. IVF vs.
graph-ANN), but the requirement stands regardless: *k*-selection latency must
fit inside a decode step (§5).

**HT-3 · Device-resident graph adjacency.** The hot neighborhood's edges in CSR
form, supporting bounded one-hop expansion of a top-k page set on device. This
is the StrataDB-specific requirement no off-the-shelf KV-cache system has —
similarity-only retrieval starves relational lookups; the edges fix that, so
the tier must serve them at the same speed as the pages.

**HT-4 · Async promotion/demotion.** T2→T1→T0 prefetch overlapping compute
(stream-ordered copies, copy-engine parallelism); score- and edge-aware
eviction (edge-driven prefetch is the latency answer banked in R2 — a page's
graph neighbors are promotion candidates the moment it's touched). No copy may
stall the decode stream; misses degrade quality (fewer retrieved pages), never
correctness.

**HT-5 · Zero-copy interop.** Pages and query results expose as CUDA tensors to
PyTorch via DLPack, stream-ordered, with no hidden synchronization. The decode
loop must contain **zero** implicit device-host syncs attributable to the tier.

**HT-6 · Write path (append).** During decode/training, newly produced KV pages
and edges append to T0 and flow down asynchronously to T2 in batches. An
explicit `flush()` defines the durability point (checkpoint integration);
between flushes, T2 lags by design.

**HT-7 · Memory budgeting and graceful degradation.** Hard per-tier byte caps
fixed at init. Under pressure the tier shrinks retrieval breadth (smaller pool,
lower k) — it must never OOM the model that hosts it. The model's own weights
and activations always take priority; the tier lives in what's left.

**HT-8 · Consumer-GPU floor.** Must run on Ampere+ consumer parts; the dev
target is the local RTX 4070 Super (12 GB, CC 8.9) — the edge thesis dies if
the tier needs datacenter hardware. GPUDirect Storage is an *optional*
acceleration where present; the required baseline path is SSD→host-pinned→device
staging. Quantized page formats (fp8/int8 KV) are a v1 requirement candidate to
stretch the 12 GB budget.

**HT-9 · Observability.** Counters sufficient for the benchmark harness: per-
tier hit rates, promotion/demotion volume, decode-stream stall time attributable
to the tier, index recall proxy. Without these the thesis is unfalsifiable.

**HT-10 · Trainable-memory mode (v2, recorded now).** For training-time offload
(R1.2+): rows of a trainable memory table live in the tier, gathered sparsely to
GPU per step, with gradient scatter and **sparse optimizer state co-located in
the same tiering** (the DLRM/ZeRO-Infinity pattern). Requirements-level note
only: the read path (HT-1..5) must not bake in read-only assumptions that
preclude this.

## 4. Interface requirement (the Moho seam)

The tier exposes primitives; Moho's kernels compose them. The contract, at
requirements level:

```
topk_pages(q, k, expand_hops=0..1, filter?) -> PageSet   # device-side result
gather(PageSet)                             -> device ptrs/tensors (DLPack)
append(kv_page, edges?, meta)               -> page_id
flush()                                     -> durability point at T2
stats()                                     -> HT-9 counters
```

Every call in the decode loop must be device-callable or stream-ordered
host-callable with no sync points. This API — not any kernel — is the actual
boundary between StrataDB and Moho; changes to it are cross-project decisions.

## 5. Non-functional targets (to validate, not promises)

- **Decode overhead budget:** retrieval + expansion + gather ≤ **20%** of a
  decode step at 500M scale on the 4070S. If the tier can't meet this, the R2
  thesis fails *on edge* and must say so plainly.
- **Effective context:** a VRAM budget that natively holds N tokens of full KV
  must serve ≥ **8×N** effective context through the tier at that overhead.
- **Quality floor:** measured against full-attention baseline on long-context
  evals; degradation budget set by the R2 experiment plan, not here.

## 6. Verification requirements

Microbenchmarks (gather bandwidth, top-k latency vs. k, expansion latency vs.
fan-out, promotion throughput) plus the end-to-end harness: tokens/sec at
context L under VRAM budget B vs. the full-KV baseline, and rare-fact bpb for
the R1 integration. **The harness is a prerequisite deliverable, not an
afterthought** — it exists before the first kernel does, so every kernel lands
against a measured baseline.

## 7. Staging and the gate

- **HT-v0** (with Moho v0): read-mostly inference cache — HT-1..5, 7..9.
- **HT-v1**: write/consolidation path — HT-6 full, eviction→consolidation hooks
  (the episodic→semantic bridge banked in R1/R2).
- **HT-v2**: trainable-memory mode — HT-10.

**Gate (restated so this doc can't be misread as a green light):** R1.1 — the
kNN-LM spike on the 100M rig — needs *none of this* (FAISS-GPU + a logit
interpolation hook suffices) and is the cheapest falsification of the whole
direction. The hot tier starts only if R1.1 survives, and after the Lithos MVP
(500M flagship) ships.

## 8. Open questions (for the design doc, later)

Index form (summaries-only vs. IVF vs. graph-ANN); page size; Rust↔CUDA story
(cudarc/cust vs. C++ FFI shim); whether HMM/unified memory on consumer parts
beats explicit staging; fp8 page format and its interaction with attention
numerics; how much of T0 the index may consume before it starves the page pool.
