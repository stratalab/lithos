# Lithos → Chisel: the acquisition-tier gate

**Status: landed on the Lithos side (green: ruff + mypy 75 files + 346 tests). Chisel work required.**

A required field plus a gate, the same shape as the F7 seams. Companions:
`docs/chisel-lithos-r2-contract.md` (record schema), `docs/tir-format.md` §2–§4 (the loss mask
this rule is derived from), `docs/composite-plan.md` (why restricted content belongs in the
datastore).

---

## 0. The change in one paragraph

We used to tier sources by **license**. We now tier them by **acquisition** — how the bytes reached
us — because that is where the exposure actually lives. And the gate applies **only to tokens that
receive a gradient**, which turns out to exempt far more than we first assumed. Restricted text may
be an SFT prompt, an RLVR problem statement, or a retrieval datastore entry. It may not be a
training *target*.

Explaining Bernoulli's theorem after reading a textbook is lawful — facts and ideas are not
copyrightable, only expression is. Downloading that textbook from a shadow library is a separate act
with separate liability, however transformatively it is later used. Dedup, epoch caps, and
regurgitation evals defend the first question. Nothing defended the second. Now something does.

**What this gate does not do: cure acquisition.** The copy was made at download time and barring the
text from the weights does not un-make it. What it buys is (a) the corpus manifest becomes an
**attestation** — provably zero `restricted` documents entered the weights — and (b) restricted
expression is confined to the retrieval channel, where every use is quotation with a citation. That
is the posture that survived Google Books, and it is what a regulated buyer asks for.

> **SCOPE AMENDMENT (2026-07-10, `docs/v1-on-qwen.md` §2).** v1 post-trains a **Qwen3 base**, so
> the attestation covers *the weights **we** trained* — the SFT / RLVR / DPO tokens — and **not**
> the base's pretraining data, which is opaque to us. The gate still governs every token we add,
> but the manifest must now carry `scope` and `base_model`: **an attestation that does not state
> what it covers is worse than none.** The unqualified claim above becomes true again in v2, when
> pretraining returns.

---

## 1. The vocabulary

| Tier | Meaning | Examples | Weights? | Datastore? |
|---|---|---|---|---|
| `open` | Explicit permissive license or public domain | USPTO, NASA/DOE/NIST, OpenStax, LibreTexts, Stack Exchange (CC-BY-SA), ODC-By datasets | ✅ | ✅ |
| `lawful` | Freely and publicly distributed **by the rightsholder**, no explicit license | arXiv, vendor datasheets, GitHub issues, NPTEL, Stanford Online | ✅ | ✅ |
| `restricted` | Paywalled or shadow-library **acquisition** | textbooks, standards bodies | ❌ | ✅ (over copies the operator lawfully holds) |
| `synthetic-verified` | Machine-generated **and** verifier-gated | teacher writes a worked problem, sandbox verifies it | ✅ *only with* `grounded_on` | ✅ |
| `unknown` | Undeclared | — | ❌ | ❌ |

**Fail-closed.** An undeclared source cannot be trained on. A policy you can state is a policy you
will eventually violate at 2am; a policy the pipeline enforces is a property of the system.

---

## 2. The rule: the gate follows the gradient, not the stage

This is the same argument as the `tool_result` loss mask (`docs/tir-format.md` §2–§4): **a span that
never contributed a gradient cannot be memorized from, and cannot carry training-source
attribution.** Applied consistently:

| Stage | Gradient-bearing tokens | Gated? |
|---|---|---|
| **pretrain** | every token | all text |
| **SFT** | the completion only (the prompt is loss-masked) | **targets only** |
| **DPO** | `chosen` and `rejected` | both |
| **RLVR / GRPO** | the policy's own rollouts | **nothing external** |
| **TIR** | all but `tool_result` (masked) | targets only |

So a `restricted` textbook problem statement is a **stimulus**, never a target:

- ✅ SFT prompt, with a `synthetic-verified` derived solution as the target.
- ✅ RLVR problem statement — GRPO's loss lands only on the policy's rollouts, and the answer key is
  a value, not expression. **RLVR needs no tier gate at all.**
- ❌ SFT assistant target. That is transcription, not teaching.

**The path from a textbook to the weights exists and is one hop:** a teacher reads the retrieved
chapter, writes a worked problem in its own words, the sandbox verifies it, and *that* is trained
on — tagged `synthetic-verified` with `grounded_on: [<the chapter's source_id>]`.

> **Counterintuitive, and worth internalising: SFT memorizes *harder* than pretraining.**
> Memorization tracks repetition, and SFT sees few tokens over several epochs with the loss
> concentrated on the target (`repeats:` upsamples deliberately). A paragraph seen once in a
> trillion-token stream is not the hazard that the same paragraph as a target, three times, is.

---

## 3. The seam: import it, don't reimplement it

```python
from lithos.data.tiers import (
    Tier, TierViolation,
    TIER_OPEN, TIER_LAWFUL, TIER_RESTRICTED, TIER_SYNTHETIC_VERIFIED, TIER_UNKNOWN,
    WEIGHTS_ALLOWED_TIERS, DATASTORE_ALLOWED_TIERS,
    tier_of, is_trainable, assert_trainable, assert_prompt_source,
)
```

`lithos/data/tiers.py` is dependency-free (stdlib + typing only), exactly like `tir_validate.py`.
Import it at the same pinned `lithos` version so "trainable" cannot drift between the repos.

- `assert_trainable(doc)` — for **gradient-bearing** text. Raises `TierViolation`.
- `assert_prompt_source(tier)` — for **loss-masked** text. Accepts `restricted`; rejects only
  `unknown`, because an undeclared provenance cannot be attested either way.

---

## 4. What Chisel must do

### 4.1 Emit `tier` as a **top-level** record field (not `metadata.tier`)

`lithos.data.documents.normalize()` now keeps **eight** top-level keys —
`{id, text, source, subset, language, license, tier, metadata}` — and reads
`record.get("tier", <source default>)`. A `metadata.tier` would survive the round-trip but be
invisible to the gate.

`license` and `tier` are independent: the first is what the rightsholder granted, the second is how
the bytes reached us. Keep both.

### 4.2 Emit `metadata.grounded_on` on every synthetic

```jsonc
{"tier": "synthetic-verified",
 "metadata": {"source_id": "...", "text_sha256": "...",
              "grounded_on": ["src:pearson-fluids-ch7"]}}
```

`grounded_on` lives in `metadata` (that is where `is_trainable` reads it). A `synthetic-verified`
record without it **fails the gate**. This is deliberate: dropping the grounding to look clean does
not sanitize anything, it makes the trace dishonest — and Petra surfaces the grounding regardless
(`docs/petra-composite-attribution.md`).

Grounding **may** point at a `restricted` source. That is the whole design. The expression never
transfers; the idea does.

### 4.3 Retier `corpus/seed_index.csv` from license to acquisition

Today the `tier` column reads `{grey: 197, green: 53, mixed: 5}`. But `grey` conflates three
different things, and the token weights make the change nearly free:

| Current | est_tokens | Rows | Becomes | Why |
|---|---:|---:|---|---|
| green | **6,572 B** (87.4%) | 53 | `open` | already explicit licenses |
| mixed | **889 B** (11.8%) | 5 | per-row | split by acquisition |
| grey — 6 bulk rows | **60.0 B** | 6 | mostly `lawful` | see below |
| grey — the textbooks | **231 M** | 191 | `restricted` | paywalled / shadow-library |

The six bulk `grey` rows are not textbooks at all:

| id | est_tokens | current note | → |
|---|---:|---|---|
| `arxiv-bulk-src` | 30 B | "per-paper author licenses" | `lawful` |
| `github-issues-prs` | 15 B | "public platform text" | `lawful` |
| `datasheets-appnotes` | 8 B | "freely published" | `lawful` |
| `nptel` | 5 B | "free-to-view; IIT copyright" | `lawful` |
| `stanford-online` | 1 B | "free-to-view" | `lawful` |
| `standards-slice` | 1 B | **"mostly paywalled-published"** | `restricted` |

**The discipline costs 0.016% of the bill of materials.** Everything actually `restricted` —
`standards-slice`, the 191 textbooks, `springer-gtm` — totals ~1.2 B of 7,522 B indexed. The 191
textbooks are 231 M tokens; a 500M model at ~100 tokens/param sees ~50 B. They were never a bulk-token
play. They were always a *pedagogy* play, and pedagogy is what retrieval and grounded synthesis
recover.

**The real hole is register, not volume.** Green physics is 6.5 B tokens; 50 B of the 56 B of "green
engineering" is *patent prose*, which is legalistic and nothing like a worked example; green
chemistry is **0**. The worked problem and the derivation that builds live almost entirely in the
restricted tier. That is exactly what `synthetic-verified` + `grounded_on` exists to recover — and it
is now the single highest-value thing Chisel produces.

### 4.4 Enforce at the publish gate

`tier` joins `source_id` / `text_sha256` as a **required** field. Reject `unknown` at publish, not at
tokenize — a bad record should never reach Lithos.

### 4.5 Post-training emission

- **SFT**: the source spec carries `tier` (of the *assistant targets*) and `prompt_tier` (of the
  prompts, which may be `restricted`). See `lithos/posttrain/sft_corpus.py::SFTSourceSpec`.
- **RLVR task banks**: no gate. Stamp `tier` for provenance so Petra can attribute, but nothing
  external receives a gradient.
- **DPO prefs**: `chosen` and `rejected` are both targets. Both gated.

---

## 5. What we need back

1. **`tier` top-level on every canonical record**, propagated from `seed_index.csv` (§4.1).
2. **`metadata.grounded_on` on every synthetic** (§4.2). Never dropped to look clean.
3. **Retier the seed index** by acquisition, using §4.3 (`grey` → `lawful` for the five bulk rows;
   `restricted` for the textbooks and `standards-slice`).
4. **Publish-gate `tier`** as required; reject `unknown` (§4.4).
5. **Import `lithos.data.tiers`** rather than reimplementing the vocabulary (§3).
6. **`prompt_tier` on SFT sources**, so a restricted problem statement can be used as the stimulus it
   is (§4.5).

---

## 6. What landed on the Lithos side

| | |
|---|---|
| `lithos/data/tiers.py` | the vocabulary + `assert_trainable` / `assert_prompt_source`; dependency-free |
| `lithos/data/documents.py` | `tier` is the 8th canonical key; `DocumentSource.tier` (pydantic `Literal` catches typos at config-load) |
| `lithos/data/pipeline.py` | `TierPolicy` (enforce=True by default); the gate runs **first** in the doc loop, before quality and dedup — a restricted record reaching the tokenizer is a config error, not a filtering decision |
| `lithos/data/manifest.py` | `tiers: {policy, counts, synthetic_grounded}` — **the attestation** |
| `lithos/posttrain/sft_corpus.py` | `SFTSourceSpec.tier` (targets) + `prompt_tier` (masked) + `grounded_on`; gate runs up front, before a single token renders |
| `tests/test_tiers.py` | 29 tests, incl. `test_restricted_prompt_with_a_verified_derived_target_is_allowed` and `test_a_restricted_prompt_cannot_rescue_a_restricted_target` |
| `configs/**` | every source now declares `tier` — fail-closed means they had to |

The manifest now carries the thing worth having:

```jsonc
"tiers": {
  "policy": {"enforce": true, "allowed": ["lawful", "open", "synthetic-verified"]},
  "counts": {"open": 12045, "lawful": 883},        // no "restricted" key. that is the point.
  "synthetic_grounded": 402                        // stated plainly, not hidden
}
```
