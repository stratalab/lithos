# The composite model layer

**Status: doctrine — settled framing, open experiments.** Where the "model" in a Strata
deployment stops being a naked LLM and becomes a composite that still emits tokens. Written
because the framing decides which of R1/R2/TIR/Verity are moats and which are scaffolding a
future training run will delete.

Companions: `docs/tir-format.md` (the tool seam), `docs/petra-provenance-lithos.md` (the
attribution seam), `docs/chisel-f7-response.md` §3 (the sandbox determinism contract).

---

## 0. The question

> What if the model layer in a standard AI app were not a simple model that emits tokens, but a
> composite architecture that still emits tokens and is enhanced internally? The harness consumes
> the stream as if it were an LLM, but the outputs are better, faster, or cheaper.

The idea is sound and mostly already true of every production inference stack. But "composite" is
a bag that contains three very different things, and the whole strategic content of this document
is the line between them.

---

## 1. The interface insight

A token stream is an **interface**, and interfaces admit substitution. Anything that preserves

1. the streaming contract (append-only tokens, no retraction), and
2. enough of the output distribution that the caller's assumptions still hold

can hide arbitrary machinery behind it. That is a real and large design space. It is also a space
in which it is very easy to violate (2) silently, because the harness has no way to detect it.

---

## 2. The taxonomy

The dividing line is **what the composite does to the output distribution**, and it predicts
everything downstream — what leaks, what's cacheable, what's evaluable.

| Tier | Definition | Examples | Visible to harness? |
|---|---|---|---|
| **T0 — distribution-preserving** | Same output distribution, less time/money | Speculative decoding with rejection sampling; prefix/KV caching; paged attention; continuous batching | No. Free lunch. |
| **T1 — distribution-changing, trajectory-preserving** | Still one token per forward pass, but not the token the naked LM would emit | kNN-LM softmax interpolation; grammar/constrained decoding; logit-level policy masking | Only via logprobs |
| **T2 — trajectory-changing** | The stream is the *winner of a search*, not a forward pass | Best-of-n with a verifier; cascades/escalation; tool execution inside the server; draft-then-revise | Yes — in latency shape, cost, cancellation |

### T0 is a free lunch, and therefore not a moat

Speculative decoding with the standard rejection-sampling scheme provably samples from the
*target model's exact distribution* — it is strictly faster with zero semantic change. (Note that
Medusa-style *typical acceptance* relaxes this and quietly becomes T1.) Prefix caching, paged
attention, and continuous batching are likewise bit-for-bit invisible. Every serious stack does
these. They are table stakes, not differentiation.

### T1 is cheap to hide and easy to get wrong

kNN-LM interpolates the LM softmax against a datastore of nearest neighbours; constrained decoding
masks logits to a grammar. Both are a small patch on the decode step. Both mean that **the model
now lies when asked for logprobs** — and a great deal of harness machinery (classification by
logprob comparison, perplexity eval, self-consistency weighting, guided sampling) consumes
logprobs as if they were the LM's.

### T2 is where "it emits tokens like an LLM" becomes a polite fiction

Once the composite runs a search, you are streaming the winner. Consequences, all structural:

- **Time-to-first-token decouples from time-to-answer.** Best-of-n cannot stream until the winner
  is chosen. The streaming contract has no *retraction*, so a T2 composite must either buffer
  (destroying TTFT) or commit to a candidate early (destroying the benefit). Speculative decoding
  escapes this only because accepted tokens are final.
- **Cost develops a long tail**, and the caller is billed for tokens they never see. Best-of-8 is
  8× hidden generation.
- **Cancellation and budget semantics** stop being meaningful to the caller: there is a control
  loop inside the "model" that the caller cannot see, budget, or interrupt.

---

## 3. The absorption test (the doctrine)

The dangerous pattern in this whole space is that **composition-for-capability is a depreciating
asset**. The frontier's repeated move has been *scaffolding getting absorbed into weights*:

- Chain-of-thought prompting and best-of-n sampling → reasoning models trained to think, with a
  thinking-token budget as a knob.
- ReAct/Toolformer-style agent scaffolds → tool-use RL; the model just calls the tool.
- Retrieval prompting → partly eaten by long context and retrieval-aware pretraining.

Each generation, somebody's clever inference-time composite becomes the next generation's default
trained behavior. If your edge is "we wrap the model in a loop," you are renting an advantage from
the next training run.

So apply this test to any proposed composite:

> **The absorption test — could a bigger, better training run absorb this?**
> If yes, it is *scaffolding*: it will be deleted, and building on it is renting.
> If no, it is *substrate*: composition is the only way to get it, forever.

And the reason a thing survives the test is always one of four structural impossibilities: the
thing is **mutable**, **per-tenant**, **exact**, or **a guarantee**. Weights cannot be any of those.

This yields the sentence the whole doc exists to preserve:

> **In every durable composite, the weights hold the *judgment* and the composite holds the thing
> judgment operates on — a fact, a computation, a constraint. Whenever you find yourself
> compositing judgment, you are building scaffolding that a training run will delete.**

The seam is always the same:

| | Absorbed into weights (judgment) | Never absorbable (substrate) |
|---|---|---|
| Tools | *when* to call the calculator (GRPO-RLVR) | the calculator's *answer* |
| Retrieval | *how* to use a retrieved fact | the *fact* |
| Policy | the *tendency* to comply (RLHF) | the *guarantee* |
| Memory | *what* to recall | the per-tenant *state* |

---

## 4. Where composition is durable — and it is exactly the Strata legs

| Impossibility | Why weights can't | The composite | Strata leg |
|---|---|---|---|
| **Mutable facts** | You cannot retrain to fix a stale fact; facts are numerous and change | Facts live in a datastore, retrieved at decode | **R1** (kNN-LM → RETRO), StrataDB |
| **Per-tenant state** | You cannot train per-customer weights | KV/attention state offloaded, per-tenant, versioned | **R2**, StrataDB |
| **Exact computation** | A calculator beats parametric arithmetic forever; no scale closes it | The sandbox executes; the model reads the result | **TIR** + `lithos.posttrain.sandbox` |
| **Hard guarantees** | You can train a *tendency*, not a *guarantee* | Deterministic enforcement at the decode boundary | **Verity** |

This is not a coincidence discovered after the fact. It is why those four are the legs and the
rest are products. Each one is composite because the alternative is *impossible*, not because
composition is clever.

### The first shippable composite: TIR-in-the-server

Lithos today executes tool calls in the *generation loop* — which is to say, above the token
stream. Moving execution **inside the served model** turns Lithos into a T2 composite that, from
the harness's view, is simply a small model that is uncannily good at arithmetic. This is the most
concrete instance of the idea available in our own stack, and it passes the absorption test on
exactly one clause: the *judgment* to call the tool is trained in (GRPO-RLVR), the *execution*
never can be.

The precondition is determinism of the tool, and **that contract is already frozen**: Python-only,
one shared namespace, `PYTHONHASHSEED=0`, single-threaded BLAS, 5 s / 2 GB, pinned
`CHECKER_IMPORT_SET = {stdlib, numpy, scipy, sympy}` (`docs/chisel-f7-response.md` §3). The same
determinism we pinned so RLVR rewards are reproducible is what makes a composite server legal.

---

## 5. Why composition pays more at 500M — and where that argument breaks

**The case for.** Parameters are expensive (FLOPs on every token, on every device, forever) and
fixed (updating them is a training run plus a redeploy). A datastore is cheap and mutable
(updating it is a write). The trade therefore favors the model whose parametric budget is
*scarcest* — a 500M STEM reasoner with a datastore and a sandbox trades the thing it has least of
for the thing that's cheapest. The compute asymmetry runs in our favor at small scale in a way it
does not for a lab serving a frontier model.

**The case against, and it is serious.** The edge cuts both ways. We chose 500M because it fits
in device memory. kNN-LM datastores are billions of entries. A multi-GB datastore does not fit on
the device that the small model was chosen for, so R1-at-the-edge resolves to one of:

- **compress the datastore** (dimension reduction, PQ/quantized ANN, adaptive retrieval that only
  looks up "hard" tokens à la RetoMaton) — the datastore shrinks, the gain shrinks with it;
- **retrieve per-chunk, not per-token** (RETRO-style chunked cross-attention) — amortizes lookup
  cost, changes the KV, breaks caching differently;
- **put the datastore in StrataDB over the network** — which reintroduces latency and *reverses
  the edge premise entirely*.

**This tension is unresolved and is the central open question of R1.** Write it down; do not let
the elegance of the thesis paper over it.

---

## 6. The five leaks

Concrete costs, each of which has bitten a real system.

1. **Caching.** Precisely: retrieval that *enters the context* (RETRO, in-context RAG) invalidates
   the prefix cache from the insertion point onward, because what you retrieve depends on what
   you've decoded. Retrieval that only touches the *softmax* (kNN-LM) preserves the KV cache but
   adds an ANN search *per decoded token* — historically the dominant inference cost. KV reuse is
   where the serving margin lives; a composite that breaks it can be net-negative on cost even
   while being better on quality.
2. **Logprobs.** Any T1/T2 composite returns logprobs that are not the LM's. Either return them
   honestly labeled as post-composite, or refuse to return them. Silently returning them is the
   worst option and the default one.
3. **Determinism and reproducibility.** A mutable datastore means the same prompt yields a
   different answer next month. Evals acquire a shelf life. A regression can originate in the
   *corpus* rather than the weights, and without provenance you cannot bisect it.
4. **Harness collision.** If the app's harness also does retrieval and tool calls, you now have
   two memory systems and two agent loops with no arbiter, fighting over the context window.
   Double retrieval, nested loops, unattributable behavior. *Who owns the context window* becomes
   a real interface question, not a rhetorical one.
5. **Attribution collapse.** A naked LLM has one causal story: weights + prompt → tokens. A
   composite has as many stories as it has components. When the answer is wrong: the draft model,
   the neighbour, the tool result, or the reranker? (See §8 — this one cuts *for* us.)

---

## 7. What the interface must gain

If composition is real, the token-stream interface needs three additions. None are exotic; the
first is a hard rule.

**7.1 Model identity must cover the whole composite.** A served model is no longer a weights file.
Its identity is a tuple, and evals, bisects, and incident reports must record all of it:

```
served_model_id = (weights_sha256, datastore_version, decode_policy_version, tool_env_sha)
```

Pinning `datastore_version` is what makes a mutable-corpus composite evaluable at all — and it is
precisely what StrataDB's branching/time-travel exists to provide.

**7.2 Logprob honesty.** The stream must be able to declare logprobs *unavailable* or
*post-composite*, rather than emitting numbers the caller will misread as the LM's.

**7.3 An out-of-band provenance channel.** Alongside the tokens, emit which retrieved records and
which tool calls influenced which spans. This is not speculative interface design — Anthropic's
public Citations API is an existence proof that the token-stream interface has already been
widened to carry provenance for exactly this reason.

---

## 8. The precondition: attribution (and why the reconciliation work was load-bearing)

A composite model layer is only *operable* if you can bisect it. Bisecting it means:

- provenance on every retrieved datum (`source_id`, `record_id`, `text_sha256` — on **every**
  record, not just some producers);
- frozen-dedup replay, so a counterfactual rebuild is actually counterfactual;
- Petra able to say whether a behavior came from a **training source** or a **retrieved datum**.

The reconciliation work is therefore not a side quest to the composite thesis; it is the
precondition for the composite being debuggable rather than mystical.

**And the composite makes attribution strictly easier on one axis.** A retrieved fact is *citable
by construction*; a parametric fact must be excavated (Petra's three-tier evidence ladder:
exemplars → Concept-Influence-MDA → counterfactual retraining). Every fact moved out of the
weights and into the datastore converts an expensive attribution problem into a free one.

**That — not "faster or cheaper" — is R1's real pitch.**

---

## 9. On Anthropic (honest boundary)

I have no visibility into Anthropic's serving internals, and this thesis should not be built on a
guess about them. Speculative decoding is standard practice and would be invisible by design, so
its presence or absence tells you nothing.

What is worth noting is that the composition you *can* observe in Claude sits **above** the token
stream, not below it: API-layer tool orchestration, explicit user-controlled prompt caching,
citations, and the harness. And extended thinking is evidence **for** the absorption thesis rather
than for the composite one — "think longer" was scaffolding (CoT prompting, best-of-n), and it
became a *trained* model behavior with a budget knob.

---

## 10. The baseline is the measurement instrument

**Before any composite, build the best *standard* SLM we can.** Not as a hurdle for the composite
to clear — as the **ruler**. Without it, a composite's gain is uninterpretable, not merely
unimpressive.

**10.1 The done-criterion, because "best possible" is unbounded.** If R1 is gated on a perfect
baseline it never starts. The baseline is **done when it is saturated, not when it is perfect**:
it sits where predicted on the parity frontier (`docs/eval-plan.md`) and another epoch buys
nothing. That is the gate.

**10.2 The control is the same weights, composite off vs on.** Not "our 500M vs somebody's 500M."
Same checkpoint, retrieval/tool path disabled and enabled. Cheap, and the only rigorous form.

**10.3 The confound that a size sweep cannot resolve.** "Retrieval helps small models more" is
predicted equally by two hypotheses:

- **Substitution** (durable): the parameters are *physically* insufficient to hold the facts, so
  retrieval supplies what weights structurally cannot.
- **Crutch** (scaffolding — fails the absorption test): the model is undertrained or badly fed,
  and retrieval papers over a deficiency more tokens would have fixed.

Varying *model size* cannot separate them. **Varying the token budget at fixed size can.** Measure
Δbpb from retrieval on the 100M at 2B / 10B / 40B tokens seen:

- Δ **decays** with token budget → **crutch**. A bigger training run is absorbing it in front of
  you. Kill it.
- Δ **plateaus at a floor** → **substitution**. That floor is the irreducible capacity deficit,
  and it is exactly what R1 harvests.

This is **the absorption test made empirical** — not "could a bigger training run absorb this," but
*train more and watch whether it does*. It runs on the 100M mix-sweep rig with one extra axis.

**10.4 The datastore must be corpus-internal for the architecture claim.** kNN-LM's own control,
and the surprising part of that result: retrieving from *the very data the model was trained on*
still improves perplexity — clean evidence that the softmax cannot express what the representation
already encodes. Stock the datastore with facts the model never saw and you have measured **data,
not architecture** (you compared "has the facts" against "doesn't"). Both experiments are worth
running; they measure different things:

| Datastore | Measures | Legitimate claim |
|---|---|---|
| ⊆ training corpus | the **architecture** (softmax bottleneck) | "retrieval beats parameters at expressing what we learned" |
| ⊃ training corpus (fresh facts) | the **mutability** benefit | "facts update without a training run" — the product pitch |

Reporting the second as if it were the first is the easiest way to fool ourselves, and it is the
mistake much of the RAG literature makes.

**10.5 Compare at equal *scarce-resource* budget, not equal parameters.** Every kNN-LM paper
compares model+datastore against the same model; none compare against a model scaled up by the
datastore's bytes. At the edge that is the only comparison that matters — our own framing is
*capability per GB*. But bytes are not fungible: datastore bytes are cold, on SSD, touched once
per lookup; parameter bytes are hot, in VRAM, touched every token. So the currency is **GB of
whatever is scarce on the target device**. On an 8 GB-VRAM / 512 GB-SSD box, 2 GB of datastore
costs ~nothing scarce while 2 GB of parameters costs a quarter of VRAM. That asymmetry is R1's
real defense — but it must be *stated as the comparison*, not assumed.

**10.6 Falsification is cheap and early; confirmation is expensive and late.** The 100M spike can
**kill** R1 but cannot **validate** it. A null Δbpb on a *saturated* 100M with a *corpus-internal*
datastore is decisive and saves the flagship run. A large positive tells you almost nothing — a
small undertrained model benefits from retrieval trivially. Plan around the asymmetry: spike to
refute, confirm only on the saturated flagship.

---

## 11. Open questions and the experiments that settle them

Ordered by cost. C0 gates everything downstream.

| # | Question | Experiment | What a negative result kills |
|---|---|---|---|
| **C0** | Is the retrieval gain *absorbed* by more training? | Fixed 100M; Δbpb from corpus-internal kNN-LM at 2B / 10B / 40B tokens seen (§10.3) | **R1 entirely** — a decaying Δ means retrieval is scaffolding |
| **C1** | Does the *plateau floor* rise as params shrink? | Same as C0 at 100M and 500M, saturated; compare floors, not peaks | R1's small-model premise (§5) — but only readable once C0 says "plateau" |
| **C2** | Where is the datastore size/gain knee? | Δbpb vs datastore entries, log-spaced | On-device R1 viability (§5 case-against) |
| **C3** | Is per-token ANN cheaper than the 500M forward pass on the target device? | Benchmark ANN latency vs forward latency | kNN-LM as a *per-token* method — forces RETRO-style per-chunk |
| **C3b** | Does R1 beat a naked model of equal **scarce-resource** footprint? | 500M + N-GB datastore vs a naked model N GB larger, on the target device (§10.5) | The edge pitch, if VRAM is the scarce axis |
| **C4** | Does tool use close the arithmetic gap *more* for the small model? | TIR vs parametric arithmetic, 500M vs 4B | The core STEM bet's scale story |
| **C5** | Can we bisect a corpus-caused regression? | Pin `datastore_version`; inject a bad record; bisect | §7.1 model identity, and the whole eval story |

---

## 12. Status

**Decided.**
- The **absorption test** is the doctrine. Composition-for-capability is not a moat.
- The durable composites are exactly the four impossibilities (mutable / per-tenant / exact /
  guarantee), which map one-to-one onto R1, R2, TIR, Verity.
- Model identity is the four-tuple in §7.1; evals record all four.
- Weights hold judgment; the composite holds what judgment operates on.
- **No composite is built before the naked SLM is saturated (§10).** The baseline is the ruler,
  its done-criterion is saturation (not perfection), and C0 — the token-budget sweep — gates R1.

**Banked (post-MVP).**
- R1 (facts out of weights: kNN-LM → RETRO, staged), R2 (KV/state offload). R2 may go first.
- TIR-in-the-server (T2 composite) — shippable once generation-side execution moves behind the
  stream; the sandbox determinism contract is already frozen.

**Rejected.**
- Composition as a *source of capability* (best-of-n, cascades, agent scaffolds sold as "the
  model"). It fails the absorption test; the next training run deletes it.

**Unresolved.**
- §5: the edge/datastore-size tension. This is R1's central risk, not a detail.

---

## 13. Where this becomes code

- **`docs/composite-instrumentation.md`** — the measurement apparatus. Three lossy capture points,
  three Parquet tables, C0–C5 as queries. Includes the two experiment-design bugs that would
  silently invalidate C0 (whole-epoch exposure; eval-in-datastore).
- **`docs/petra-composite-attribution.md`** — what changes for Petra. One live defect (a
  counterfactual without a `scope` proves nothing once R1 exists), two schema asks, and the
  ground-truth calibration offer.
