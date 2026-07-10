# Moho — Requirements (2026-07-04)

> **AMENDMENT (2026-07-10) — Moho serves R2, not R1.** Rev B moved retrieval *above* the
> token stream (`docs/composite-plan.md` §1, §5.5), so R1 touches the datastore **once per
> request** and hands the model *text*. At ~10⁵ document chunks that search is <2 ms against
> a 250 ms–1 s generation: **0.2% of request latency (2% even at 10⁶ chunks), and no kernel
> makes a context token cheaper.** Putting the index in VRAM would spend the device's scarcest resource to save
> ~7 ms — Gate 5 failing on our own infrastructure. R2 is different in kind: it touches the
> store **per token, per layer, per head, inside the attention path**, and hands the model
> *tensors*. That is where §2's kernels earn their keep.
>
> **The one conditional:** if C-CTX returns `displacement`, decode-loop retrieval (kNN-LM)
> returns — and *that* is per-token — so Moho would serve both legs after all. The
> experiment picks whether Moho has one customer or two. Read §5.5 of the composite plan
> before building anything here. References to R1 below stand only in that branch.

**Moho is the boundary layer where model compute meets deep memory** — named
for the Mohorovičić discontinuity between crust and mantle. In the stack's
systems framing (Lithos = processor, StrataDB = memory hierarchy, tools =
coprocessor), **Moho is the memory controller**: the custom kernels that make
attention-over-a-database fast enough to be real. It is the enabling layer for
research tracks R1 (facts out of weights) and R2 (KV cache/attention offload
with graph expansion). **Status: BANKED, post-MVP, gated behind the R1.1
spike** (§6). Companion: `docs/strata-gpu-hot-tier.md` — the StrataDB-side
tier whose API (`topk_pages / gather / append / flush`) is the contract Moho
implements against. That API, not any kernel, is the cross-project boundary.

## 1. Thesis

Replace/extend the transformer's KV cache with StrataDB so that inference and
(later) training offload attention state, relationships, and facts to a tiny
database spanning GPU→host→SSD. The transformer keeps reasoning, language, and
orchestration in weights; working memory and facts live in the store. Two
user-stated prerequisites — a GPU-resident StrataDB mode and custom kernels —
refined as: (1) a *tiered* GPU hot tier (see companion doc: full-resident would
offload nothing), and (2) the kernel inventory below. A third prerequisite,
recorded here as first-class: **the benchmark harness and interface contract
exist before the first kernel does** (MO-6, MO-7).

## 2. Scope

**In scope:** device kernels and their orchestration — selection, gather,
expansion, prefetch overlap, and (v2) sparse trainable-memory updates.
**Out of scope:** the tier's data structures and promotion machinery (StrataDB
hot tier), the model architecture changes that consume retrieved state (R1/R2
experiment plans), and any multi-GPU/distributed concern (edge is single-GPU;
that simplification is load-bearing).

## 3. Kernel inventory (functional requirements)

**MO-1 · Sparse/paged gather-attention.** Attention over a dynamically
selected, non-contiguous set of KV pages (the `PageSet` returned by the tier).
**Build on FlashInfer** — it already handles paged/sparse layouts well; Moho
adapts and integrates rather than rewriting. Requirement: numerically
equivalent to dense attention over the same tokens (within fp tolerance), no
host sync in the decode loop.

**MO-2 · Page-summary top-k scoring.** Quest-style upper-bound scoring of the
query against per-page summaries (min/max keys), producing the candidate set
for MO-1. Fusable with the gather where profitable. Known technique; adapt
published kernels.

**MO-3 · One-hop graph expansion.** Bounded CSR traversal on device: expand the
top-k page set along StrataDB's edges before gathering. **This is the novel
kernel** — the StrataDB-specific idea no off-the-shelf system has. Similarity
retrieval starves relational lookups (multi-hop facts, code call graphs,
equation dependencies); the expansion kernel is where the graph thesis becomes
measurable. Requirement: bounded fan-out, deduplicated against the top-k set,
latency additive within the MO-8 budget.

**MO-4 · Prefetch orchestration.** Stream/copy-engine scheduling that overlaps
tier promotion (T2→T1→T0) with compute, driven by edge-based prediction (a
touched page's graph neighbors are next-step candidates). Plumbing, not
research — but it is where the "SSD-backed working memory" latency story lives
or dies.

**MO-5 · Trainable-memory kernels (v2).** For R1.2+ training-time offload:
sparse row gather to device, gradient scatter, and **fused sparse optimizer
update** (Adam/Adagrad rows co-resident with their weights in the tier). This
is the proven DLRM / ZeRO-Infinity pattern imported into the reasoning stack —
recorded now so v0/v1 interfaces don't bake in read-only assumptions.

## 4. Cross-cutting requirements

**MO-6 · The interface contract is the deliverable.** Moho consumes exactly the
hot-tier API; anything a kernel needs that the API lacks is a *cross-project
change request*, not a workaround. Kernels expose themselves to Lithos as
drop-in attention/lookup modules (same signatures as the dense paths they
replace) so R1/R2 experiments are config flips, not forks.

**MO-7 · Benchmark harness before kernels.** Microbenchmarks (gather bandwidth
vs. page scatter, top-k latency vs. k, expansion latency vs. fan-out, overlap
efficiency) plus end-to-end: tokens/sec at context L under VRAM budget B vs.
full-KV baseline; rare-fact bpb for R1 configurations. Every kernel lands
against a measured baseline with stated falsification criteria.

**MO-8 · Budgets (shared with the tier).** Selection + expansion + gather ≤
~20% of a decode step at 500M scale on the RTX 4070 Super; ≥8× effective
context at a fixed VRAM budget; quality degradation within the R2 experiment
plan's floor. If edge-class hardware can't meet these, the honest output is
"R2 is datacenter-only" — the harness must be able to say so.

**MO-9 · Triton-first.** Kernels in Triton unless profiling forces CUDA/CUTLASS
for a specific hot spot — tractability for a solo+AI team outweighs the last
20% of a kernel's ceiling until the thesis is proven. Consumer-GPU floor
(Ampere+, dev target 4070S) inherited from the tier.

## 5. Mapping to R1/R2

| Track | Needs | Moho version |
|---|---|---|
| R1.1 kNN-LM spike (100M) | **nothing** — FAISS-GPU + logit-interp hook | none |
| R1.2 RETRO-style trained-in (500M) | MO-1, MO-2 (+MO-5 if memory is trainable) | v1–v2 |
| R2 KV offload + graph expansion | MO-1..4 + hot tier | v0–v1 |
| R1↔R2 consolidation bridge | MO-3/MO-4 over consolidated stores | v1 |

## 6. Sequencing and the gate

**R1.1 first, Moho only if it survives.** The kNN-LM spike is the cheapest
falsification of the entire memory-fusion direction and needs no custom
kernels; building the memory controller before knowing the memory helps would
be backwards. Gate order: Lithos MVP (500M flagship) ships → R1.1 spike on the
100M rig → if rare-fact bpb moves, Moho v0 (MO-2, MO-1, MO-7 harness) →
hot-tier HT-v0 integration → R2 prototype → MO-3 graph expansion (the
differentiating experiment). Nothing in this document authorizes work before
that gate; it exists so the requirements are recorded while the thinking is
fresh.

## 7. Open questions (for the design doc, later)

Where selection runs per layer vs. shared across layers (Quest is per-head;
sharing saves latency at recall cost); page size coupling between MO-1 and the
tier; whether MO-3 expansion should be trained-in (learned edge weights) or
computed-only at v0; fp8 KV pages' interaction with attention numerics; how
R1's token-level retrieval and R2's page-level retrieval share (or don't share)
one index; Triton's maturity for the CSR-traversal shape of MO-3.
