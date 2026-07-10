# C0 — the R1 gate experiment

**Status: spec, post-literature-sweep. The design changed materially; read §1 before §4.**

Gates R1 (`docs/composite-model-layer.md` §11). Implements against
`docs/composite-instrumentation.md`. **This document supersedes the C0 design in both** where they
conflict — the sweep invalidated three of its assumptions.

---

## 0. What the literature actually says

Sources marked ✓ were fetched and quoted directly; sources marked ○ rest on the research pass and
should be read before they are relied on.

| # | Finding | Source |
|---|---|---|
| **L1** ✓ | **Part of the kNN-LM gain has already been absorbed into parametric LMs.** The gain decomposes into (a) using a different input representation for next-token prediction, (b) approximate kNN search, (c) softmax temperature on the kNN distribution — and: *"we incorporate these insights into the model architecture or the training procedure of the standard parametric LM, improving its results **without the need for an explicit retrieval component**."* | Xu, Alon, Neubig, *Why do Nearest Neighbor Language Models Work?* (ICML 2023), [2301.02828](https://arxiv.org/abs/2301.02828) |
| **L2** ✓ | **kNN-LM does not help the tail.** *"kNN-LM does not improve prediction performance for low-frequency tokens but mainly benefits high-frequency tokens regardless of long-tail contexts in the datastore."* The paper examines product-quantization approximation error as a candidate cause. | Nishida et al., *Long-Tail Crisis in Nearest Neighbor Language Models* (NAACL 2025 Findings), [2503.22426](https://arxiv.org/abs/2503.22426) |
| **L3** ✓ | **Perplexity gains do not transfer to generation.** *"While the KNN-LM and related methods yield impressive decreases in perplexity, we discover that they do not exhibit corresponding improvements in open-ended generation quality."* Also: interpolation *increases* perplexity for the **majority** of tokens; the aggregate win comes from dramatic gains on a small subset. | Wang et al., *KNN-LM Does Not Improve Open-ended Text Generation* (2023), [2305.14625](https://arxiv.org/abs/2305.14625) |
| **L4** ✓ | **RETRO's LM gain is largely copying.** *"The performance gains from retrieval largely originate from overlapping tokens between the database and the test data, suggesting less non-trivial generalization than previously assumed... even limited token overlap may significantly decrease test-time loss."* | Norlund et al., *On the Generalization Ability of Retrieval-Enhanced Transformers* (EACL 2023 Findings), [2302.12128](https://arxiv.org/abs/2302.12128) |
| **L5** ✓ | **The closest paper to C0 is not C0.** *To Memorize or to Retrieve* sweeps pretraining tokens 1–150×/param at fixed params × retrieval-store size 1–20×, on OLMo-2 30M–3B. But: mechanism is **in-context RAG** (top-k=5 FAISS prepend), the token axis varies **unique tokens** not epochs, and the store is a **held-out DCLM slice** (corpus-*external*). Its stated direction of decrease is w.r.t. **model size** (*"as model size increases, the marginal benefit decreases"*), not tokens. Abstract headline is pro-retrieval. Contains a substitution threshold at *"∼D/N=4.14 pretraining tokens per parameter... with each retrieval token replacing multiple pretraining ones."* | Singh et al. (2026), [2604.00715](https://arxiv.org/abs/2604.00715) |
| **L6** ○ | **Retrieval wins a compute-matched tradeoff even on heavily-overtrained bases** (Pythia, OLMo-1B ~3T, Llama-3-8B ~15T). Caveats: the win concentrates on **knowledge-intensive factual QA** and is *weak/mixed on reasoning*; the accounting excludes inference/search cost and treats datastore storage as free. | Shao et al., MassiveDS (NeurIPS 2024), [2407.12854](https://arxiv.org/abs/2407.12854) |
| **L7** ○ | RETRO's own replication holds pretraining fixed at 330B tokens for every size, so its "gain shrinks with scale" result **confounds params with tokens/param** — the *most*-overtrained (smallest) model shows the *largest* gain. Cannot be read either way. | Wang et al., [2304.06762](https://arxiv.org/abs/2304.06762) |

**The token-budget axis, for kNN-LM/RETRO specifically, with a corpus-internal datastore, remains
unrun.** But that is now a much less interesting fact than it was this morning.

---

## 1. What this changes (three of our assumptions are dead)

### 1.1 The frequency discriminator is refuted. Retract it.

`composite-instrumentation.md` §6 asserted: crutch ⇒ diffuse gains on **common** tokens;
substitution ⇒ gains concentrated on **rare** tokens. **L2 measures the opposite** — kNN-LM's gains
concentrate on high-frequency tokens and it *hurts* the tail. Under our own decision rule, published
kNN-LM would be classified a crutch.

Either the rule is wrong or kNN-LM is a crutch. We do not get to assume which. **Stop using
frequency stratification as a discriminator.** Keep it as a *descriptive diagnostic*, with one
mandatory control: L2 names **product-quantization approximation error** as a candidate cause of the
tail failure. Our original spec planned PQ — which would have *manufactured the very artifact*.
Hence §5.3.

### 1.2 bpb alone cannot answer the question. L3 is fatal to a bpb-only C0.

Perplexity improves while generation does not, and the aggregate improvement is driven by a minority
of tokens while the majority get *worse*. A C0 that plateaus in aggregate Δbpb would tell us nothing
about whether R1 helps the product. **Every C0 arm reports three numbers, not one** (§5.4).

### 1.3 "Corpus-internal" is not a neutral control — it is the *copying* condition.

L4: RETRO's gain largely originates in verbatim overlap between the store and the test data. Our
§5.1 decontam assert (`datastore ∩ eval = ∅` on `text_sha256`) is therefore **not hygiene, it is
existential** — and it is not sufficient. Exact-hash disjointness does not remove *n-gram* overlap.
We must measure the gain as a function of overlap, not merely assert overlap is zero (§5.5).

---

## 2. The reframe: absorb first, then measure the residual

L1 is the doctrine's own test, run by someone else, on our exact mechanism, with a partial *yes*.
That is not a reason to quit. It is a reason to **change the baseline**.

`composite-model-layer.md` §10 already said the baseline must have had *a fair shot at the capability
the composite provides*. A naive LM has not: three of the tricks that make kNN-LM work are
architecture/training tricks, not knowledge. Comparing kNN-LM against a naive baseline therefore
measures **retrieval + free tricks**, and credits the datastore for the tricks.

> **C0's real question is not "does retrieval help?" It is: *what is left of the retrieval gain once
> everything absorbable has been absorbed?* The residual is the only quantity that can justify R1.**

This is cheap. The absorbed baseline needs **no datastore at all**.

---

## 3. Two channels, and only one of them carries the product

`composite-model-layer.md` §10.4 split these. The literature now splits them too, and they point in
**opposite directions**:

| Channel | Datastore | What it claims | Evidence |
|---|---|---|---|
| **Architecture** | ⊆ training corpus | "retrieval expresses what the softmax cannot" | **Negative** — L1 (partly absorbed), L2 (no tail), L3 (no generation), L4 (copying) |
| **Mutability** | ⊄ training corpus | "facts update without a training run" | **Positive** — L5, L6 (but factual QA, *weak on reasoning*) |

Two consequences we should say out loud.

**R1's justification moves.** It is not the softmax bottleneck. It is: *facts that change, or facts
the model never saw.* That is the mutable-substrate clause of the absorption test, and it survives —
you cannot train on a fact you do not have. But it means the architecture claim was never the load
bearer, and we should stop citing it.

**The reasoning caveat is aimed straight at us.** L6's retrieval win concentrates on factual QA and
is weak on reasoning. **Lithos is a STEM reasoner.** This is the single most threatening finding in
the sweep, and no experiment in our current plan measures it. Hence C0-B.

---

## 4. The experiment

Three arms. **A0 runs first and may make the rest unnecessary.**

### A0 — Absorb (no datastore, no retrieval)

Implement L1's three parametric interventions on the 100M and measure the gain over the naive
baseline. Cost: one training run per intervention, no retrieval infrastructure.

Whatever A0 recovers is gain we get **for free, forever, with no datastore, no ANN latency, no
`datastore_version` in the identity tuple, and no attribution ambiguity.** Every subsequent Δ is
measured against A0, not against the naive baseline.

### C0-A — The architecture channel (expected: negative)

Fixed corpus, fixed 110M params, **corpus-internal** datastore, checkpoints at **1 / 2 / 4 / 8
epochs** (whole-epoch boundaries — `composite-model-layer.md` §5).

Measures Δ(kNN-LM over **A0**) as a function of epochs. Prediction in §7.

### C0-B — The mutability channel, on reasoning (the real gate)

Datastore contains STEM content the model provably never saw (a held-out corpus slice + post-cutoff
documents, using the `family_id`-aware split from `taskbank.py`). Measures **downstream STEM task
accuracy**, not bpb, at each epoch checkpoint.

The question is *not* "does more training absorb this" — it cannot; the facts were never in the
training data. The question L6 forces on us is **"can a 110M/500M model actually *use* retrieved
STEM facts in a reasoning chain, or does retrieval only pay on factual lookup?"** Split the eval
into a factual-lookup subset and a multi-step-reasoning subset and report them separately. If
retrieval helps lookup and not reasoning, **R1 does not serve the flagship** regardless of how C0-A
lands.

---

## 5. Design

### 5.1 Corpus and checkpoints
The fixed corpus the 100M rig trains on. One training run to 8 epochs, `checkpoint_interval` set to
land exactly on epoch boundaries. **One run, not four** — separate runs confound training budget with
initialization and data order.

### 5.2 Datastore
Subsample of the **training corpus** (20M–100M token keys). A subsample still satisfies
`datastore ⊆ corpus`, so the exposure assert holds at every epoch ≥ 1. C0-A measures the *trend* of
Δ across epochs, not its absolute magnitude, so subsampling is scientifically sound. (Full-corpus
keys would be `9.84e9 × 768 × 2B ≈ 15.1 TB` — infeasible, and the reason the field uses PQ.)

### 5.3 **IVF-Flat, not IVF-PQ** ← forced by L2
L2 names PQ approximation error as a candidate cause of the tail failure. Using PQ would manufacture
the artifact we are trying to measure. Use `IndexIVFFlat` (no quantization error; only an `nprobe`
recall limit), with `nprobe` tuned until **recall@k ≥ 0.99** against brute force on a 10k-query
sample. PCA to 128 dims is permitted **only if** a full-dimension arm on a smaller subsample bounds
the PCA-induced Δ. Record `nprobe`, `recall@k`, and the PCA delta in `runs`.

### 5.4 Three metrics per arm, never one ← forced by L3
1. **Aggregate Δbpb** — the headline, and the least trustworthy.
2. **Fraction of tokens helped vs hurt** — L3 says the majority get *worse*. If our aggregate
   improves while >50% of tokens regress, we have reproduced L3 and must not report the aggregate
   alone.
3. **Downstream STEM task accuracy** (the frozen battery). The only number that speaks to the product.

Frequency stratification is reported as a **diagnostic**, explicitly not as a discriminator (§1.1).

### 5.5 Overlap, measured — not asserted ← forced by L4
- Hard assert: `datastore ∩ eval = ∅` on `text_sha256`. Necessary, **not sufficient**.
- Then *measure* the gain as a function of maximal n-gram overlap between each eval segment and its
  retrieved neighbours. Bucket eval tokens by longest-matching-n-gram (0, 1–4, 5–8, 9+) and report
  Δbpb per bucket. **If the gain lives entirely in the high-overlap buckets, we have reproduced L4
  and the gain is copying.**

### 5.6 λ is a query, not a run
Store `logprob_lm` and `p_knn_true`; sweep λ offline (`composite-instrumentation.md` §3.2). One
forward pass per checkpoint yields the whole λ curve.

### 5.7 Apparatus calibration
The §9 tests of the instrumentation spec, especially **λ=0 must reproduce the A0 baseline logprobs
bit-exactly.**

---

## 6. Cost

| Item | Estimate |
|---|---|
| 8-epoch 110M run (78.7B tokens seen) | `6ND = 6 × 1.1e8 × 7.87e10 ≈ 5.2e19` FLOPs ≈ **$38** at $733/1e21 |
| A0 interventions (≤3 extra runs, may be shorter) | ~$40–110 |
| Datastore build (one forward pass over the subsample) + IVF-Flat index | hours of CPU/GPU, negligible $ |
| Eval passes (4 checkpoints × 3 arms, teacher-forced over ~5M tokens) | negligible |

**Under ~$200 of compute.** The sanity check: the same formula puts the existing ~30B-token 100M run
at ~$14.5, matching the observed $10–15/run.

---

## 7. Pre-registered predictions

Written **before** the run, so we cannot rationalise afterwards.

| | Prediction | If wrong |
|---|---|---|
| **A0** | Recovers a material fraction of the naive-baseline kNN-LM gain (L1 says the tricks transfer) | If A0 recovers ~nothing, L1 does not replicate at 110M and the naive baseline was fair after all |
| **C0-A Δ** | Small over A0, and **decays with epochs** | A flat, material residual over A0 = the architecture claim is real and we were wrong to doubt it. **This would be the surprising, publishable result.** |
| **C0-A tokens** | Majority of tokens *hurt*; aggregate carried by a minority (L3) | If the majority are helped, L3 does not replicate here |
| **C0-A frequency** | Gains concentrate on **high**-frequency tokens (L2) | If gains concentrate in the tail with IVF-Flat, then **L2's tail failure was a PQ artifact** — a real finding, and good news for R1 |
| **C0-A overlap** | Gain concentrated in high-n-gram-overlap buckets (L4) | If the gain survives at zero overlap, RETRO's copying critique does not transfer to kNN-LM |
| **C0-B** | Retrieval helps factual lookup, **little or nothing on multi-step reasoning** (L6) | If retrieval materially helps STEM *reasoning* at 110M, that is R1's strongest possible result and the flagship case is made |

---

## 8. What each outcome kills

- **A0 large, C0-A residual ≈ 0** → the kNN-LM architecture channel is scaffolding. Absorb the
  tricks, ship no datastore. **R1's architecture justification dies; R1 survives only if C0-B does.**
- **C0-A residual decays with epochs** → confirms the absorption test on our own mechanism. Same
  conclusion, with a clean curve.
- **C0-A residual flat and material** → the architecture claim holds where the literature says it
  shouldn't. Investigate before believing; check overlap buckets first (L4).
- **C0-B flat on reasoning** → **R1 does not serve the flagship.** It may still serve a
  factual-lookup product, but that is not what Lithos is. This is the outcome the sweep says to
  expect, and the one we are least prepared for.
- **C0-B positive on reasoning** → R1 is substrate, on the mutability clause, for our actual task.
  Proceed to Moho.

---

## 9. Sequencing

1. **A0.** No datastore. Cheapest, and it redefines the baseline for everything after.
2. **C0-B.** The real gate, and the one the literature says is most at risk. Do not defer it behind
   C0-A merely because C0-A is more elegant.
3. **C0-A.** Confirmatory, cheap, and mostly a replication that calibrates our instrument against
   four published results (L1–L4). Its value is that if we *cannot* reproduce L2/L3/L4, our
   apparatus is broken — and we would rather learn that here than on the flagship.

**Note the inversion.** C0 was designed as the gate. After the sweep, **C0-B is the gate** and C0-A
is a calibration run. The doctrine survives; the experiment that tests it moved.
