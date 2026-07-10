# v1 on a Qwen base — what it costs, what it buys, and how to keep v2 cheap

**Status: DECIDED (2026-07-10).** v1 drops from-scratch pretraining. The ladder becomes
Qwen3 **0.6B / 1.7B / 4B / 8B** base weights (Apache-2.0 — *verify per size*), with the full
Lithos post-training stack and composite layer on top. From-scratch pretraining returns in
**v2**, as the *attestation demonstrator*.

Supersedes the model-ladder decisions in `docs/prd.md` §6 and the "from-scratch 500M is the
first keeper" line wherever it appears.

---

## 0. Why

The product is the **tooling** to build attributable domain models cheaply. The model is the
reference implementation. **No institution will ever pretrain from scratch**, so a reference
implementation that begins with a from-scratch pretrain demonstrates a path nobody walks.

> **Improving a model you did not train is a stronger demonstration of the tooling than
> shipping one you did.** It proves the recipe, not the weights.

This was half-decided already: the 4B hero was always a continued-pretrain of Qwen3-4B. The
only question was whether the same logic reaches 0.6B. It does. The two things from-scratch
actually bought — a 32k STEM vocab and an attestable corpus — are addressed in §3 and §2.

---

## 1. What changes, and what does not

| | Status |
|---|---|
| Pretraining (data pipeline, packing, DDP, spot resume, checkpointing) | **Unused in v1.** Written, tested, proven on the 100M end-to-end run. Does not rot. |
| `lithos/serve/hf_import.py` | **Already green** — Qwen3 loads into `LithosForCausalLM` with logit parity, 8 tests. |
| Composite (`serve/composite.py`, `retrieval/`, `evals/cctx.py`) | **Untouched.** They ask the tokenizer only for `encode` / `decode` / `token_to_id`. |
| `chat_template` | **Untouched.** Resolves specials **by name**, never by ID. |
| Post-training (SFT, DPO, GRPO-RLVR, TIR) | Untouched logic; **retokenize the data**. |
| Tokenizer | Qwen's 151k, **trimmed** (§3). The 32k STEM BPE becomes a v2 asset. |
| Tier gate | **Still enforced — over the tokens we add.** Scope must be stated (§2). |
| Decontamination | **Protects a corpus we no longer train on.** See §4. |

The composite surviving a base swap untouched is the walking skeleton paying off: it was
built against protocols, not against our tokenizer.

---

## 2. Attestation is **scoped**, not off

Framing it as "off in v1, on in v2" is how the discipline lapses and the code rots. What is
actually true:

| Channel | v1 (Qwen base) | v2 (from-scratch) |
|---|---|---|
| **Retrieved facts** | cite **exactly**, by construction | same |
| **Tool results** | `tool` channel, masked from loss, attributable | same |
| **Post-training influence** (SFT/RLVR/DPO) | **counterfactually attributable** — we own the mix, the tier gate governs every token | same |
| **The base's parametric knowledge** | **opaque. We say so.** | attributable |

So attestation *runs* in v1. It simply covers the **delta**, not the base.

### The required change: an attestation must state its own scope

`manifest["tiers"]` currently reads as though it covers the weights. It must carry:

```jsonc
"tiers": {
  "scope": "post-training-only",          // or "full" once v2 pretrains from scratch
  "base_model": "Qwen/Qwen3-1.7B-Base",   // null when the weights are ours end to end
  "base_data_cutoff": "<from the model card>",
  "policy": {"enforce": true, "allowed": ["lawful", "open", "synthetic-verified"]},
  "counts": {"open": 12045, "lawful": 883},
  "synthetic_grounded": 402
}
```

**An attestation that does not state what it covers is worse than none** — it invites the
strong reading. `docs/chisel-tier-gate.md` §0 currently claims "provably zero `restricted`
documents entered the weights." That becomes "…entered the weights **we trained**."

---

## 3. The vocabulary is the one real technical cost — so pay it down

Qwen3 carries a 151,936-token vocab serving 100+ languages a STEM reasoner will never emit.
On a small model that is not a rounding error:

| model | hidden | embed params | share of total | trimmed to ~32k |
|---|---:|---:|---:|---|
| Qwen3-0.6B | 1024 | 156M | **25.9 %** | saves 123M (20.5 % of the model) |
| Qwen3-1.7B | 2048 | 311M | 18.3 % | saves 246M (14.4 %) |
| Qwen3-4B | 2560 | 389M | 9.7 % | saves 307M (7.7 %) |
| Qwen3-8B | 4096 | 1,245M | 15.6 % | saves 983M (12.3 %) |

*(dims from memory — **verify against the hub**, especially which sizes tie embeddings)*

**A quarter of Qwen3-0.6B is a lookup table for languages we do not use.** That is precisely
the capability-per-GB argument that justified a 32k vocab from scratch, and it is the only
from-scratch advantage we can keep in v1:

**Trim the vocab.** Prune to tokens the STEM+English corpus actually uses; slice the
embedding rows (and the tied `lm_head`). Qwen's BPE has byte-level fallback, so nothing
becomes unencodable — only less efficient. Days of work, not a training run.

**Consequence for the ship target:** run the *family* as an instrument, choose what ships on
capability-per-GB **after** trimming. It may well be 1.7B rather than 0.6B.

---

## 4. The new eval hazard, and the tool we already have

**`DecontaminationFilter` scans *our* corpus against the eval battery. It cannot scan Qwen's
36T tokens.** GSM8K, MATH, and MMLU are almost certainly in the base's pretraining data, and
nothing we do can remove them.

Three consequences, in order of importance:

1. **Absolute benchmark scores are no longer evidence of reasoning.** Do not quote them as
   such. This is not a new problem — it is a pre-existing problem we can now see.

2. **Δ-over-base remains perfectly valid**, because both arms inherit the same contamination.
   And this is a *better* eval story than we had: "our from-scratch 500M scores Y" always
   invited "compared to what?" Now the baseline is literally the model we started from, and
   **Gate 1 (absorbed baseline) becomes trivially satisfiable** — the baseline is the base,
   with every trick it came with.

3. **The only trustworthy absolute numbers come from tasks the base never saw.**
   `taskbank.split_by_year(tasks, cutoff_year=<the base's data cutoff>)` — the family-aware
   split we built for F7 — is now the **primary** eval instrument, not a contamination
   nicety. `base_data_cutoff` therefore becomes a first-class recorded field, and the eval
   report must separate **pre-cutoff (Δ only)** from **post-cutoff (absolute, trustworthy)**.

---

## 5. The base seam — what keeps v2 a config flip instead of a rewrite

The base must be a **recorded, swappable component**, never an assumption. Concretely, none
of the following may leak into code or data:

- **Token IDs.** Specials are resolved by name (`special_ids`, `tir_token_ids`). Any code or
  fixture that hard-codes an ID is a v2 landmine. `docs/tir-format.md` §2 documents IDs 7–15
  for *our* tokenizer; those are descriptive, not normative.
- **Vocab size.** `padded_vocab_size`, `dtype_for_vocab` already derive it.
- **`base_model` and `base_data_cutoff`** must appear in the corpus manifest, the SFT
  manifest, and every eval scorecard — otherwise the v1→v2 comparison is uncontrolled and
  the whole point of the sequencing is lost.

> The discipline in v1 buys the **option** on v2. Without it, "we'll do attestation later" is
> how the sovereignty thesis quietly dies.

---

## 6. v1 hardening — the actual work, in order

**Post-training**
1. ✅ **DONE** — `lithos/serve/tokenizer_adapt.py`. Adds the **13** required specials
   (`chat_template.REQUIRED_SPECIAL_TOKENS`: 7 core + 6 TIR) to Qwen's tokenizer, *reusing*
   any it already has (Qwen3 ships `<think>`/`</think>`, so ~11 are genuinely new). The
   embedding grows via `load_qwen3(hf, vocab_size=…)`; added rows are zero-init and trained
   during SFT. A test proves growing the vocab **preserves import parity on Qwen's original
   slice** — the property the whole decision rests on.
2. ✅ **DONE** — retokenization is a *build input*, not a data transform. The SFT/RLVR source
   data is text (messages JSONL, task banks); it tokenizes at build time. `save_augmented_
   tokenizer` / `scripts/adapt_qwen_tokenizer.py` cut an augmented `tokenizer.json` + `adapt.json`
   artifact; point each build's `tokenizer_path` at it and the existing pipeline retokenizes
   unchanged. A test drives the real SFT build under an augmented tokenizer and checks the
   shards (specials at their augmented ids, dtype widened past uint16 for Qwen-scale vocab, loss
   mask on the assistant turn). `assert_tokenizer_matches_model` guards the vocab contract.
   *Remaining:* run it against the real Qwen `tokenizer.json` (needs HF access) and repoint the
   configs — an ops step, not code. The decontam probes retokenize the same way.
3. **E2.5 — retrieval-aware SFT**: reference blocks in the mix, plus **distractor** examples
   the model must ignore. Required before any C-CTX `capability` verdict is believable
   (`docs/composite-plan.md` §3 cause (c)).
4. Vocab trim (§3).

**Composite**
5. `weights_sha256` helper; the `runs` table (episodes exist, runs does not).
6. A script to build a `Datastore` from Chisel's canonical records, with the tier gate and
   the `datastore ∩ eval = ∅` assert wired in.
7. A STEM task bank whose answers actually live in a corpus (C-CTX uses a toy fact today).

**Evals**
8. `base_model` + `base_data_cutoff` on every manifest and scorecard.
9. Eval report splits **pre-cutoff (Δ-over-base only)** from **post-cutoff (absolute)**.
10. The base is the anchor. Every headline claim is a Δ over the exact base checkpoint.

---

## 7. The v2 tripwire

From-scratch returns as the **attestation demonstrator**: the model that can prove what it
was trained on, end to end — corpus manifest, frozen-dedup replay, counterfactual rebuild,
Petra's parametric channel lit up.

It is a genuinely differentiated artifact and a *better* story as v2 than as v1. But it needs
a gate, not a hope. **Proposed gate: v2 begins when the tooling has produced a domain model
for someone who is not us.** That is the same metric that decides whether the tooling is real
(`docs/composite-plan.md`'s successor question), and it is the only condition under which
attestation has a buyer.
