# The composite: revised plan and methodology

**Status: DECIDED (2026-07-09), post-literature-sweep.** Supersedes the R1 architecture in
`docs/composite-model-layer.md` and the experiment ladder in `docs/c0-spec.md`. The doctrine is
unchanged; the *architecture* and the *sequence* both moved.

We are engineers, not researchers. We do not reproduce other people's negative results — we believe
them and spend the money on the question nobody has answered for us. This document says which
question that is.

---

## 1. The split nobody drew

Sort every result in the sweep by **which side of the token stream the mechanism lives on**:

| Mechanism | Lives | Evidence | Source |
|---|---|---|---|
| In-context RAG (prepend passages) | **Above** the stream | **Positive** | Singh et al. 2026 — *"Retrieved passages are concatenated as context"* |
| MassiveDS RIC-LM (prepend top-k) | **Above** the stream | **Positive** | Shao et al. 2024 — *"we concatenate the top k=3 documents"*, *"we prepend the retrieved documents"* |
| kNN-LM softmax interpolation | **Below** the stream | **Negative** | Xu 2023 (partly absorbed) · Nishida 2025 (no tail) · Wang 2023 (no generation) |
| RETRO chunked cross-attention | **Below** the stream | **Negative** | Norlund 2023 (gain ≈ verbatim overlap) |

**Every mechanism with positive evidence lives above the token stream. Every mechanism with negative
evidence lives below it.**

This is the same observation `composite-model-layer.md` §9 made about Claude, unprompted, before any
of this research: *"the composition you can observe sits above the token stream, not below it."* We
wrote it as a caution against guessing at other people's internals. It turns out to be the empirical
finding.

**The doctrine survives untouched.** The four impossibilities still hold — mutable, per-tenant,
exact, a guarantee. What changed is *where the mutable-facts composite belongs*. It belongs above
the line.

---

## 2. What this does to R1

R1 was: *facts out of weights, via kNN-LM → RETRO in the decode loop, over a token-level datastore,
enabled by Moho.* Every clause of that is now wrong or unnecessary.

**R1 becomes: retrieval-in-context over StrataDB, above the token stream.** One move fixes four
things:

1. **The evidence flips sign.** We stop building the mechanism with four negative results and build
   the one with two positive ones.
2. **The unresolved edge risk dissolves.** `composite-model-layer.md` §5 called the datastore size
   R1's central risk: token-level keys are `9.84e9 × 768 × 2B ≈ 15.1 TB`. Document-level chunks are
   `~10M × 768 × 2B ≈ 15 GB`, or ~1 GB quantized. **Three orders of magnitude.** Edge-feasible.
3. **Moho's R1 dependency disappears.** No per-token ANN gather in the attention path. Moho now
   serves **R2 only** — which is where it always belonged, since Moho is a systems project.
4. **Attribution gets easier, not harder.** A prepended passage is citable at the span level with no
   neighbour log, no `p_knn_true`, no per-token instrumentation. `docs/petra-composite-attribution.md`
   §4's ground-truth offer gets *cheaper* to deliver.

And it introduces exactly one new cost, which is the interesting part.

---

## 3. The question that is ours

In-context RAG spends **context**. At 500M, context is the scarcest resource the model has.

MassiveDS reports *"marginal benefits"* on reasoning-heavy MMLU and MedQA **for weaker models**, and
explains its factual wins with: *"Both TriviaQA and NQ evaluate factual recall without complex
reasoning. When the right information is provided in context, the LM only needs to extract the
answer."*

So retrieval pays for **extraction** and not for **reasoning** — and pays least for small models.
Lithos is a small STEM reasoner. That is the finding aimed directly at us.

But nobody has separated the two possible causes, because nobody cares about a 500M with a short
context window:

- **(a) Capability**: a small model simply cannot integrate a retrieved fact into a multi-step
  derivation. Retrieval is useless to a reasoner at this scale.
- **(b) Displacement**: the retrieved passages *ate the context the reasoning needed*. The model
  could have used the fact; it no longer had room to think.

These predict identical benchmark numbers and opposite architectures.

> **C-CTX — the fork.** At a 500M-scale context budget, is retrieval's benefit on STEM reasoning
> destroyed by the context it displaces? Hold the fact constant; vary how it arrives (prepended
> passage vs. oracle fact injected free vs. no fact) and vary the reasoning-token budget.

- **If (b) displacement dominates** → a *context-free* fact channel is the only way to get facts into
  a small reasoner, and kNN-LM's weak per-token gain might still win **net of displacement**. This is
  the one argument that resurrects decode-loop retrieval — and it is a **scarce-resource** argument
  (§10.5), never a capability one.
- **If (a) capability dominates** → retrieval does not serve the flagship at all. Build in-context
  RAG for the *mutability and citation* product surface, and stop there. No Moho for R1, no kNN-LM,
  no token-level datastore.

This is the experiment we run. It is cheap, it needs no new training run, and its answer picks the
architecture.

---

## 4. Methodology: the composite acceptance test

Every one of the four negative results has the same shape. A composite's gain was **credited to the
composite** when it actually came from somewhere else. These are failure modes of *attribution*, not
of retrieval, and they will recur for R2, TIR, and Verity.

**No composite's number is believed until it passes all six gates.**

| # | Gate | The failure it prevents | Learned from |
|---|---|---|---|
| **0** | **Mechanism identity.** Evidence transfers only within a mechanism class. In-context RAG results may not be cited for kNN-LM. | Citing a cousin's result as your own | The research summary made exactly this error |
| **1** | **Absorbed baseline.** Has the baseline received every trick the composite gets for free? | Crediting the datastore for a temperature schedule | Xu 2023 — the tricks fold back in |
| **2** | **Product metric.** Does the gain appear in the thing we ship (task accuracy), not only the proxy (bpb)? | Shipping a perplexity win that changes nothing | Wang 2023 — ppl improves, generation doesn't |
| **3** | **Leakage measured, not asserted.** Report the gain bucketed by overlap between the store and the eval. | Measuring copying and calling it generalization | Norlund 2023 — gain ≈ verbatim overlap |
| **4** | **Distribution, not aggregate.** What fraction of tokens are helped vs. hurt? | An aggregate carried by a minority while most regress | Wang 2023 — majority of tokens get worse |
| **5** | **Cost in the scarce resource.** Is the gain worth what it costs in the resource the target device actually lacks? | Winning on FLOPs and losing on VRAM — or on context | §10.5, and now C-CTX |

Gate 0 is the one we invented and immediately needed. Gate 5's *scarce resource* turned out not to be
VRAM or SSD, as §5 assumed, but **context**.

---

## 5. The re-sequence

Ordered by evidence strength, not by elegance.

| | Leg | Impossibility | Status | Why here |
|---|---|---|---|---|
| **1** | **TIR / sandbox** | exact | **Live (MVP)** | Untouched by any of this. A calculator beats parametric arithmetic forever. Strongest leg. |
| **2** | **R2** — KV/state offload | per-tenant | **Promote to first research track** | **Immune to the entire sweep.** All four negative results concern *fact retrieval into the output distribution*. R2 approximates *the same attention* — its success criterion is memory and latency, not bpb. It is a systems bet, and it fails only for engineering reasons. Moho serves R2 (§5.5 for the arithmetic). |
| **3** | **In-context RAG over StrataDB** | mutable | **Build; gated on C-CTX for the reasoner** | Positive evidence, cheap, no kernels, ~1 GB index. Delivers mutability + citation-by-construction. |
| **4** | **Verity** | a guarantee | Parked; seam landed | The **final authority on the support**. Applied *first*, to the raw logits — and final because every later stage (temperature/top-k/top-p) is monotone, i.e. can only *remove* mass. See §8.1. |
| **5** | **Decode-loop retrieval** (kNN-LM / RETRO) | — | **Deferred. Resurrectable only by C-CTX (b).** | Four negative results. Do not build on a scarce-resource argument we have not yet measured. |

**R2 before R1 was already banked as a possibility. The sweep makes it the decision.**

---

## 5.5 What StrataDB buys R1 — and what it doesn't

§5 says "Moho serves R2 only" without saying what StrataDB is *for* in R1. It is for four
things, and **none of them is speed.**

### The arithmetic, because the answer is counterintuitive

> **The cost of R1 is denominated in context tokens, not milliseconds. A kernel makes
> milliseconds cheaper. It cannot make a token cheaper.**

On the edge target, a 500M in fp16 is ~1 GB of weights, and decode is memory-bandwidth
bound — every generated token reads the whole model. At 100–400 GB/s that is ~3–10 ms per
token, so a 100-token answer costs **250 ms – 1 s**.

Retrieval, by contrast, happens **once per request**, over text, before prefill:

| | bytes touched | latency (100–400 GB/s) | share of a 100-token request |
|---|---|---|---|
| exact search, 10⁶ chunks × 512 d, fp32 | 2.05 GB, read once | 5.1–20.5 ms | **2.05 %** |
| exact search, 10⁵ chunks (realistic on-device) | 205 MB | 0.5–2.1 ms | **0.20 %** |
| generating 100 tokens | ~100 GB (weights × 100) | **0.25 – 1.0 s** | 100 % |

**Retrieval is a fifth of a percent of request latency at realistic on-device scale, and
about two percent even at a million chunks.** A custom kernel for it is Amdahl's law with
the numerator filled in. Worse: a GPU hot tier means putting the index in VRAM, where it
competes with the weights — spending the *scarcest resource on the device* to save ~7 ms out
of ~300. That is **Gate 5 failing on our own infrastructure**, which is a decent sign the
gates were worth writing down.

The thing worth optimising is not the search. It is the **query embedding**: a trained
encoder means shipping a *second model* to the edge, costing the same VRAM. The move worth
testing is reusing the LM itself as the encoder rather than carrying a second set of
weights. Untested; but that is the constraint that binds, not ANN throughput.

### What StrataDB does buy

| | Why numpy-and-files cannot |
|---|---|
| **Versioning / branching** | `datastore_version` is currently a content hash recomputed over every chunk. As a branch pointer, Petra's counterfactual **`scope: datastore`** (the defect caught in the handoff) becomes a cheap branch op instead of a full index rebuild. |
| **Incremental mutability** | **This is the product.** After the sweep, R1 survives on the *mutable* clause — facts that change, or facts never seen. With files, changing one fact rebuilds the whole index. That is the opposite of the feature. |
| **Graph expansion** | One-hop expansion after similarity top-k: retrieve a chunk, pull in its section siblings or cited references. Pure similarity starves relational recall. An R1 capability, above the token stream, needing **no kernel**. |
| **Persistence past RAM** | Memory-mapped, SSD-backed — which is what an edge device actually has a lot of. |

### Why Moho belongs to R2, in one sentence

**R1 touches the database once per request and hands the model *text*. R2 touches it per
token, per layer, per head, inside the attention path, and hands the model *tensors*.**
That is where a gather sits on the critical path hundreds of times per token, where memory
bandwidth is genuinely the wall, and where paged gather-attention, page-summary top-k, and
one-hop CSR expansion pay for themselves.

### The conditional, which is the interesting part

**If C-CTX returns `displacement`, the GPU-hot-tier picture becomes right after all.** In
that branch prepending is too expensive in the currency that matters, a context-free fact
channel is the only route, kNN-LM comes back — and kNN-LM *is* per-token retrieval in the
decode loop. Then the GPU-resident vectors and the custom gather are necessary, and Moho
serves both legs.

If C-CTX returns `capability`, none of that machinery is ever needed for facts.

> **The experiment does not only pick the model architecture. It picks whether Moho has one
> customer or two.**

### Today's backing store

`lithos/retrieval/index.py`: numpy + files (`vectors.npy`, `chunks.jsonl`,
`datastore_manifest.json`). No StrataDB, no FAISS, no vector DB — the same thin-seam
decision Chisel v0 made, for the same reason (StrataDB is too green to debug both sides of
the fence at once). **`VectorIndex` is the seam**: the only thing that touches vectors.
StrataDB's hot tier, or an ANN index, implements `search` and nothing above it changes.

---

## 6. The kill list

Struck, with the reason, so we do not rediscover them:

- **The 8-epoch C0-A sweep as an experiment.** Keep the *run* — relabel it the **baseline saturation
  curve**, which `composite-model-layer.md` §10.1 requires anyway. We are not paying to reproduce
  four published negative results.
- **The frequency discriminator** (`composite-instrumentation.md` §6). Refuted; already retracted.
- **The softmax-bottleneck justification for R1.** Partly absorbed by Xu 2023. Stop citing it.
- **Moho as an R1 dependency.** Moho serves R2. R1's search is 0.2–2% of request latency, and no kernel makes a *context token* cheaper (§5.5). What StrataDB buys R1 is versioning, mutability, graph expansion, and persistence — none of them speed.
- **The token-level (15 TB) datastore.** Document chunks, not token keys.
- **Per-token kNN instrumentation** — `p_knn_true`, neighbour lists, IVF-Flat, the λ-sweep-as-query.
  All of it was machinery for a mechanism we are not building. `runs` + `episodes` + a
  document-level retrieval log is the whole apparatus now. **The instrumentation spec simplifies by
  most of its length.**

---

## 7. The ladder, re-costed

| | Experiment | Needs | Cost | Decides |
|---|---|---|---|---|
| **E1** | **A0 — absorb.** Implement Xu 2023's three parametric interventions on the 100M. | Training, no datastore | ~$40–110 | Not an experiment — a **free model improvement**, kept forever, no `datastore_version`, no ANN, no attribution ambiguity. Do it regardless. |
| **E2** | **Baseline saturation curve.** One 8-epoch 110M run, checkpoints at whole epochs. | One run | ~$38 | Required by §10.1 before *any* composite. Gives the ruler. |
| **E3** | **C-CTX — the fork.** Fact held constant; delivery varied (prepend / oracle-free-injection / none) × reasoning-token budget. | Eval-only, on E2's checkpoints + a doc index | ~$0 compute | **Picks the architecture** (§3). |
| **E4** | **C0-B — usability.** Can the model use a retrieved STEM fact in a multi-step derivation? Split eval into factual-lookup vs. multi-step. | Shares E3's apparatus | ~$0 | Whether R1 serves the flagship at all. |

**Under ~$150, and E3 is the one nobody has run for us.**

Sequence: **E1 → E2 → E3 → E4.** E1 and E2 pay for themselves independent of every composite
decision, which is the mark of an experiment worth running first.

---

## 7.5 Correction found by building it (2026-07-10)

`docs/composite-model-layer.md` and Plate 01 both said **"Verity must be the last write to
the logits"**, reasoning that anything applied afterwards could reintroduce a forbidden
token. Building the walking skeleton (`lithos/serve/composite.py`) proved the *mechanism*
wrong, and a test caught it:

- **The failure.** Applied last, the policy runs after nucleus sampling. Against a
  confident model, `top_p` collapses the support to exactly one token — and if the policy
  bans that token, the constraint is unsatisfiable, even though the model had a whole
  vocabulary of allowed alternatives. (`test_banning_the_models_favourite_token_does_not_
  empty_the_nucleus`.)
- **The fix.** Apply the policy **first**, to the raw logits. It is still final, because
  every later stage — temperature, top-k, top-p — is **monotone**: each can only *remove*
  probability mass, never add it. Nothing downstream can reintroduce a banned token.
- **The invariant that makes it true** is now enforced, not assumed: a processor that
  *raises* any logit raises `ValueError` (`_apply_decode_policy`). Mass-removal is the
  property; "last" was only ever a proxy for it.

The original worry was sound and aimed at the wrong stage: the thing that could *add* mass
was **kNN-LM interpolation**, and rev B moved retrieval above the token stream, so no
mass-adding stage remains in the decode chain at all.

---

## 8. What would resurrect decode-loop retrieval

One thing, and it must be measured, not argued:

> C-CTX shows displacement (b) dominates — prepending passages costs a 500M more in displaced
> reasoning tokens than the facts are worth — **and** an oracle context-free fact channel recovers
> the loss.

Then, and only then, kNN-LM's weak, high-frequency-concentrated, non-generative per-token gain is
worth revisiting, because it is the only fact channel that costs **zero context**. It would be
justified on Gate 5 (scarce resource), never on capability. And it would still have to clear Gates
1–4 against an A0 baseline.

Anything short of that measurement, and we are building on a hope the literature already refuted.
