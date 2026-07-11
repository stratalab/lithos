# Decode-Policy — Requirements

**Status: parked; the seam is landed.** Requirements for the composite's 4th leg — the
"hard guarantees" leg — a Lithos-internal decode-time enforcement layer. The seam exists and
is enforced (`lithos/model/generation.py::_apply_decode_policy`,
`lithos/serve/composite.py::DenyTokensPolicy`, `decode_policy_version` in the `ServedModelId`
attestation tuple); the *policy library* on top of it is parked. This doc pins what the leg
must be so it can be built without re-deriving it.

## 0. Naming — what this is, and what it is not

This leg is **not Verity.** It learns from Verity's policy thinking but carries neither its
name nor its product boundary:

- **Decode-policy (this doc)** — Lithos's own, model-*coupled* enforcement at the decode
  boundary: a forbidden **token** is unreachable, by construction. It exists only inside
  Lithos because it requires owning the decode loop, so it is a *feature of Lithos*, not a
  standalone product — a descriptive name, not a brand.
- **Verity** — a separate, standalone, model-*independent* product: an action-layer reference
  monitor that verifies each *resolved tool call* at the harness (`before_tool_call`), and so
  works on any model (Qwen / Llama / Claude / Lithos). It is **not** a composite leg; it
  integrates with Lithos *externally*, at the TIR / tool-dispatch seam.

Both enforce the same *kind* of deterministic "never do X" floor, at two different points in
the flow (**emit** vs **act**). The relationship is **align, don't couple**: the decode-policy
adopts Verity's floor taxonomy and attestation discipline for the token-expressible subset; it
shares no code and no product surface with it. See §7.

## 1. The thesis (why it's a leg)

Every learned safety mechanism trains a **tendency**: RLHF, refusal tuning, constitutional
methods shift a distribution so forbidden output becomes *less likely*. A tendency is not a
guarantee — under the right prompt, temperature, or adversary, a low-probability forbidden
emission is still reachable. For a class of requirements — never emit a secret, always stay
inside a tool-call grammar — "very unlikely" is not acceptable; the requirement is
**impossible-by-construction**.

That impossibility is the leg (`docs/composite-model-layer.md` §4): *you can train a tendency,
not a guarantee → deterministic enforcement at the decode boundary*. The decode-policy makes a
forbidden **token** unreachable, not merely improbable — mechanically, auditable, versioned,
independent of what the weights "want."

## 2. Where it sits — the final authority on the support

A **decode-boundary** mechanism. At each step it is handed the raw logits and returns the
support (the tokens that may be sampled). Its correctness rests on one structural fact
(`generation.py` module docstring; `composite-plan.md` §8.1):

> The policy is applied **first**, to the raw logits, and it is **final** because every later
> stage — temperature, top-k, top-p — is **monotone**: each can only *remove* mass, never add
> it. So no downstream stage can reintroduce a forbidden token.

Running it first (not last) lets nucleus sampling renormalize over the *allowed* set instead of
collapsing onto a token the policy would then ban. "Applied first, final by monotonicity" is
the load-bearing property; any implementation must preserve it.

## 3. The core invariants (already enforced — requirements, not aspirations)

`_apply_decode_policy` enforces two today; they are the leg's correctness contract:

- **INV-1 — may only remove mass.** A processor that *raises* any logit is rejected (it could
  reintroduce a forbidden token past a downstream stage). Enforcement: raising a logit raises.
- **INV-2 — may not mask everything.** An all-`-inf` row is an unsatisfiable constraint, not a
  sample; it fails **loudly**, never emits `NaN`.

A third is implicit and must be made explicit as the policy library grows:

- **INV-3 — deterministic.** Given (logits, generated-prefix, policy-version), the support is a
  pure function — no wall-clock, no RNG, no network. This is what makes a decision replayable
  and a violation a *bug*, not *bad luck*.

## 4. What it enforces — the token-expressible floor (and only that)

This leg owns exactly the subset of the floor that is **expressible over tokens**:

| Class | Example | Mechanism |
|---|---|---|
| **Token/span denial** | a leaked secret or canary string, a banned literal | `DenyTokensPolicy` (landed) — mask token ids |
| **Grammar / structured output** | valid JSON, a fixed schema, a tool-call envelope, a units string | grammar-constrained mask: only tokens that keep the output on a valid grammar path survive |
| **TIR structural validity** | `<|python|>…<|/tool|>` well-formed; segments in call→result order | a TIR-aware policy constraining the special-token transitions (what `tir_validate` checks *post hoc*, enforced *at decode*) |
| **Emission bans** | never *say* this API key / PII / canary | span denial at the source — the one place a secret can be stopped before it exists |

**What it does *not* own — the action-semantic floor.** `DROP TABLE prod`, `rm -rf`, financial
thresholds, tainted-path egress are **not expressible over tokens**: a structurally perfect
tool call can be catastrophic (`DROP TABLE` is valid to any grammar), and "which DB is prod" is
context the logits never see. Those rules belong to **Verity**, at the tool hook (§7). The
decode-policy guarantees well-*formed* emission; Verity guarantees safe *action*.

## 5. Functional requirements

- **FR-1 — deterministic support restriction.** Exposes a `LogitsProcessor`
  (`(logits, generated) -> logits`) that removes mass per INV-1/2/3. (Landed as the seam.)
- **FR-2 — policy as versioned data, not code.** A policy is a content-hashed artifact with a
  `version`, never hard-coded at a call site; the version is the identity by which a decision
  is attributed and replayed.
- **FR-3 — attestation.** The active policy version is part of the served-model identity
  (`ServedModelId`: `weights_sha256`, `datastore_version`, **`decode_policy_version`**,
  `tool_env_sha` — landed). Every response is attributable to the exact policy that shaped it.
- **FR-4 — grammar compilation.** A structured-output constraint compiles to a per-step token
  mask over the *live* tokenizer's vocabulary; the compiled artifact is content-hashed (FR-2).
- **FR-5 — composition.** Multiple policies (a deny set + a grammar + a TIR rule) compose into
  one support by **intersection of allowed sets**, and the composition is itself mass-removing
  (INV-1 holds under composition). Order-independent.
- **FR-6 — fail loud.** An unsatisfiable constraint (INV-2) or a malformed policy raises at the
  boundary; it never degrades to an unconstrained sample or a `NaN`. A guarantee that fails
  open is not a guarantee.
- **FR-7 — Petra as a weak input, never a dependency.** May consume a Petra signal (e.g. "a
  known-bad feature fired") as an *input* to a policy, but correctness must not depend on it —
  Petra is a weak internal signal, the decode-policy is the deterministic gate.
- **FR-8 — replayability.** Given a recorded (prompt, weights, datastore, policy-version,
  tool-env), a response — including every support restriction — replays bit-for-bit.

## 6. Non-functional requirements

- **NFR-1 — latency at the boundary.** The per-step mask runs inside the hot decode loop:
  vectorized masking, a precompiled grammar automaton (not a per-step parse). Serving it at
  speed inside the served model is a **Moho** concern.
- **NFR-2 — no false negatives.** A forbidden token must be impossible, not rare. A
  probabilistic "mostly enforced" decode-policy is a failed one.
- **NFR-3 — auditable.** Every decision is reconstructable from the attested version — a
  violation is a reproducible bug, not a one-off.
- **NFR-4 — testable by construction.** A policy ships with adversarial fixtures proving the
  forbidden set is unreachable under greedy *and* sampled decoding (the eval golden-fixture
  discipline).
- **NFR-5 — tokenizer-coupled, version-pinned.** A compiled grammar/deny-set is valid only for
  the tokenizer it was compiled against; the artifact pins the tokenizer hash.

## 7. The three fences — where the decode-policy sits among them

The Strata "never do X" floor enforces at **three distinct points**; conflating them under one
name was the drift this doc corrects (2026-07-11):

- **train-on fence → Chisel** — what the loop may train on (teacher doctrine, open-weight
  allowlist, tier gates).
- **emit fence → decode-policy (this leg)** — what a Lithos model may emit, at decode.
- **action fence → Verity** — what a deployed agent may execute (external, model-independent,
  at the harness `before_tool_call`).

The one cross-repo seam is **TIR / tool-dispatch**: when a Lithos agent is about to execute a
tool call, that resolved call is the `before_tool_call` point Verity verifies. Optionally, the
decode-policy and Verity could share a **policy-artifact format**, so a "never emit this secret"
rule authored once compiles to *both* a Verity check and a decode deny-set — an alignment
opportunity (align, don't couple), not a requirement.

## 8. Status ledger

- ✅ **Seam landed** — `LogitsProcessor` + `_apply_decode_policy` (INV-1/INV-2); threaded
  through `generate` and `tir_rollout`; `DenyTokensPolicy`; `decode_policy_version` in the
  attestation tuple (FR-1, FR-3, partial FR-6).
- ◻ **Policy-as-data + registry** (FR-2), **grammar compiler** (FR-4), **composition** (FR-5).
- ◻ **INV-3 explicit** + adversarial fixture suite (NFR-4).
- ◻ **TIR-structural decode policy** — enforce at decode what `tir_validate` checks post hoc.
- ◻ **Moho integration** (NFR-1) — serve the mask at speed inside the served model.

## 9. Open questions

1. **Grammar engine** — adopt an existing constrained-decoding library or build minimal?
   (Portability vs the no-dependency-in-the-hot-loop NFR.)
2. **Policy authoring format** — how a policy is written and content-hashed (grammar + deny
   list + tokenizer pin, compiled to one artifact).
3. **Shared artifact with Verity (§7)** — worth a common policy format across the emit and
   action fences, or keep them fully independent with only a shared attestation convention?
4. **Latency budget** — the acceptable per-token cost, which bounds Q1 and decides whether
   Moho (NFR-1) is v1 or later.
