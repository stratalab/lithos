# Lithos → Petra: attribution under a composite model

**Status: request for schema changes + a proposal.** Companion to
`docs/petra-provenance-lithos.md` (the data-pipeline contract, already agreed). This one is about
what changes in Petra's *evidence model* when the served "model" stops being a naked LLM.

Contains **one live defect**, **two schema asks**, and **one offer that we think is the most
valuable thing in this document for Petra specifically.** Read §4 even if you skip the rest.

---

## 0. Context in one paragraph

Lithos is moving toward a **composite model layer**: the thing that emits tokens is the weights
*plus* something else — a retrieval datastore (R1), offloaded KV state (R2), or a sandbox that
actually executes the model's tool calls (TIR). Doctrine and the taxonomy are in
`docs/composite-model-layer.md`. The consequence for Petra is simple to state and awkward to
handle: **a behaviour can now originate outside the weights**, and Petra's job is to say where a
behaviour came from.

**This is not hypothetical or post-MVP. TIR ships in the MVP.** The tool channel exists today.

---

## 1. The defect: a counterfactual without a scope proves nothing

**This is live and it has a deadline** — it is a defect in the counterfactual request format we
standardised with Chisel days ago.

Today the request is:

```json
{"counterfactual_of": "<name>@<version>",
 "drop": {"source_ids": [...], "record_ids": [...], "text_sha256s": [...]},
 "reason": "...", "rebuild": "identical-otherwise"}
```

Once R1 exists, **that request is ambiguous, and tier-3 evidence built on it is void.** Drop a
`source_id` from the *training corpus*, retrain, and the model still answers correctly — because
the fact is sitting in the R1 datastore, which the drop never touched. The retrain proved nothing,
and it cost a training run to prove it.

### The fix

Add a **required** `scope`, with **no default**:

```json
{"counterfactual_of": "<name>@<version>",
 "drop": {"source_ids": [...], "record_ids": [...], "text_sha256s": [...]},
 "scope": "train" | "datastore" | "both",
 "reason": "...", "rebuild": "identical-otherwise"}
```

No default, deliberately. A silent default is exactly how void evidence gets published — and the
safe-looking default (`train`) is the one that produces the silent failure above.

Consequences to encode:

- `scope` must resolve to **two** keep-sets: a training keep-set (Chisel's record cut → Lithos's
  frozen-dedup replay) and a **datastore keep-set** (the R1 index rebuild). They are different
  artifacts with different lifetimes.
- A tier-3 result **must record which components of `served_model_id` actually changed**
  (`docs/composite-model-layer.md` §7.1). If `scope` includes `datastore` but `datastore_version`
  is unchanged in the result, the counterfactual did not happen. Assert it.
- `scope: "train"` remains correct and sufficient **until R1 lands**. Adding the field now costs
  nothing; adding it after the format is frozen costs a migration and, in the meantime, a class of
  results that look valid and aren't.

---

## 2. Ask #1 — `provenance_channel` on every evidence record

```
provenance_channel: "parametric" | "retrieved" | "tool"
```

Without it, Petra will attribute a *retrieved* fact — or a **sandbox-computed number** — to the
weights, and report a confident false positive.

**The `tool` channel is live now.** If a model answers an arithmetic question correctly because
`lithos.posttrain.sandbox` executed the computation, "the model learned arithmetic from training
source X" is simply wrong: no training source produced that number, CPython did. Any tier-1
exemplar search over TIR traces hits this immediately, because the `tool_result` span is *in the
context* but was **masked from the loss by token ID** (`docs/tir-format.md` §2–§4). Tokens that
never contributed a gradient cannot have training-source attribution — and the mask is exactly the
signal that tells you so.

Suggested rule: **a span whose loss was masked is `tool` by construction.** That gives you the
channel for free on TIR data, with no inference.

---

## 3. Ask #2 — key evidence to the full `served_model_id`, not `weights_sha`

```
served_model_id = (weights_sha256, datastore_version, decode_policy_version, tool_env_sha)
```

Evidence gathered under two different `datastore_version`s, or two different sandbox environments,
is not evidence about the same system and must not be pooled. Today `weights_sha` alone is a
sufficient key; the day R1 or a sandbox bump lands, it silently stops being one, and the pooling is
invisible.

---

## 4. The offer: the composite hands Petra the ground truth it has never had

This is the part we would lead with.

Petra's cheap tiers — exemplar search (tier 1) and Concept-Influence-MDA (tier 2) — have a
structural problem that is not Petra's fault and is not solved anywhere in the literature: **there
is no ground truth for attribution.** That is precisely why tier 3 (counterfactual retraining)
exists, why it is the only tier anyone fully trusts, and why it costs a training run per question.
Tiers 1 and 2 ship today **without error bars**.

A composite fixes this, in two directions, cheaply.

**Direction A — the retrieval channel gives exact known sources.** When the model uses a retrieved
fact, we know *precisely* which record supplied it: the retrieval log records `source_id`,
`record_id`, `text_sha256`, and the distance, per token
(`docs/composite-instrumentation.md` §3.2). So: place a fact **only** in the datastore, never in
training. Have the model use it. Then ask tier 1 and tier 2 where it came from. **Every parametric
attribution they return is a measured false positive.**

**Direction B — planted canaries give parametric ground truth.** Inject synthetic facts into
exactly one training document each (known `source_id`), train normally, verify the model reproduces
them, then ask Petra's tiers to find the source. This is the standard memorisation-canary technique
and it gives **parametric** ground truth **without N retrains**. Requirements: the canaries must be
decontaminated out of every eval set, and the plant must be recorded in the manifest.

Together these give tiers 1 and 2 **measured precision and recall** — the thing they currently
lack entirely — and every point of measured precision reduces how often you must pay for tier 3.

**The honest caveat, stated up front:** calibration measured on the *retrieved* channel does not
automatically transfer to the *parametric* channel. A method could be well-behaved on retrieval
(where the causal path is short and explicit) and badly behaved on weights (where it is neither).
That assumption is why **both arms matter** — Direction B is the one that actually calibrates the
parametric channel, and Direction A is the one that catches channel confusion. Run both, report
them separately, and never quietly average them.

This also sharpens the pitch we already agreed on: **a retrieved fact is citable by construction; a
parametric fact must be excavated.** Every fact moved out of the weights converts an expensive
attribution problem into a free one. Attribution is R1's real product argument — not speed, not
cost.

---

## 5. What Lithos will hand you

From `docs/composite-instrumentation.md`:

| Artifact | Contents |
|---|---|
| `runs` | the `served_model_id` four-tuple + `tokens_seen`, `n_params`, footprint |
| `tokens` | per-token: `logprob_lm`, `p_knn_true`, `masked`, `freq_bucket`, sampled neighbour lists |
| `episodes` | per-TIR-episode: verdict, tool calls, tokens emitted vs masked |
| `datastore_rows` | `dsrow → (source_id, record_id, text_sha256)` — **the join key** |

The join keys are the same three the reconciliation fixed. **This depends on Chisel's open ask:
`text_sha256` stamped on *every* record**, not just Docling-ingested ones. Retrieval-channel
attribution is keyed on it.

---

## 6. What we need back

1. **`scope` in the counterfactual request — required, no default.** (§1. Before the format
   freezes; this is the one with a deadline.)
2. **`provenance_channel` on every evidence record.** (§2. `tool` is needed *now*, not post-MVP.)
3. **Evidence keyed to the full four-tuple.** (§3.)
4. **A calibration report** for tiers 1 and 2, both arms, precision/recall, reported separately.
   (§4.)
5. **Agreement on the canary protocol**: count, fact shape, injection point, and the decontam
   guarantee that keeps them out of every eval set.

---

## 7. Sequencing, and what survives if R1 dies

We owe you this honestly: **R1 is gated on experiment C0** (`docs/composite-model-layer.md` §11 —
does more training absorb the retrieval gain?). If C0 says the gain decays, R1 is scaffolding and
we kill it.

So, explicitly:

| Ask | Conditional on R1? |
|---|---|
| §1 `scope` | **No** — cheap now, prevents void evidence later, `scope:"train"` is correct until then |
| §2 `provenance_channel` | **No** — the `tool` channel is live in the MVP |
| §3 four-tuple key | **No** — `tool_env_sha` already varies |
| §4 Direction B (canaries) | **No** — pure parametric calibration, valuable regardless |
| §4 Direction A (retrieval ground truth) | **Yes** — needs R1 |
| §5 `tokens.p_knn_true` / neighbours | **Yes** — needs R1 |

Nothing here blocks Petra's current work. **Do not build for R1.** Build for §1–§3 and the canary
arm, all of which pay off on the naked model, and inherit the rest if C0 clears.
