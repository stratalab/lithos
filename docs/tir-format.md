# The Lithos TIR + Reasoning Format (D1)

The wire format for tool-integrated reasoning: how a Lithos model thinks, calls
a tool, and consumes the result — as tokens, masks, and generation rules. This
is the head of the post-training critical path: it sizes the tokenizer's
reserved special-token block, so it must freeze **before** the tokenizer v1.0
retrain (`docs/tokenizer.md` §3.3). Companion to `docs/post-training-review.md`
§2.1–2.5 (the gaps this closes) and `docs/post-training-implementation-plan.md`
(epic D1). The **defining capability** it encodes is Phase 12's whole thesis:
reason → call Python/Octave → use the verified result.

## 0. Principles

1. **Control by ID, never by string.** Every structural token (`<think>`,
   tool delimiters, result markers) is an atomic special token inserted and
   detected by ID — the exact guarantee `chat_template.py` already gives for
   chat tokens. String-parsing the rendered text for `</think>` would be
   fragile (BPE could split it, the model could drift). This makes masking
   exact and generation-time boundary detection unambiguous.
2. **Never train on the sandbox's output.** Tool-result tokens are injected by
   the harness and **masked from the SFT loss / excluded from the RL policy
   gradient + KL**. The model learns to *make* the call and to *use* the
   result; training on the result itself teaches it to hallucinate outputs
   instead of calling (the §2.1 quality-decider).
3. **One canonical format, both family members.** The from-scratch 500M (32k
   Lithos vocab) and the 4B hero (151k Qwen vocab) render from the *same*
   JSONL. String forms are chosen so each tokenizer registers them as its own
   special tokens — the data is written once (§5).
4. **Match the source data where free.** Harvested reasoning traces (R1/QwQ/
   Qwen3 distillations, OpenMathReasoning) already contain `<think>…</think>`.
   Adopting that exact string makes bulk conversion near-identity instead of a
   risky global substitution — worth one break from the `<|…|>` convention.
5. **Failures are training signal.** A tool call that errors returns its
   traceback as the result; the model learns to read the error and retry. The
   format does not distinguish success from failure — the sandbox does.

## 1. Decisions (resolved 2026-07-03)

Two choices were genuinely the user's; both are now settled. They move data prep
and (for B) pretraining — the token table and masking rules below are unchanged
either way.

**(A) No-think latency lever → DEFERRED (simpler MVP).** Reasoning is *always
generated* — it's the capability. The MVP trains **always-on thinking**; a
collapsed-trace UI is a free product-layer choice (no training cost). The format
reserves the toggle mechanism (a `no_think` system flag → empty `<think>` span)
so it can be trained in **later without a re-freeze**, if the edge-latency lever
is wanted post-MVP. No no-think SFT variants in the MVP data.

**(B) `<think>` is a loss target; length handled by a long-context pretrain
phase → NOT by capping.** Thinking is a loss target (masking it would reduce SFT
to short-answer tuning — pointless for a reasoner). The §2.3 collision
(harvested traces 4k–16k tokens vs 2048 pretraining context) is resolved by
**extending the 500M's context in pretraining** (RoPE-theta scaling + long-doc
anneal), *keeping* the full-length traces rather than dropping them. This adds a
pretraining workstream (epic **E10** in the implementation plan) and its target
context is set by measuring the trace-length distribution (feeds Part-B P1). The
format imposes no length; the rendering policy now caps episodes to the
**extended** context, not 2048.

## 2. Token assignment (the reserved-block deliverable)

Assigns concrete meaning to the reserved IDs 7–15 that `docs/tokenizer.md` §3.3
pinned. IDs 0–6 (chat + control) are unchanged.

| ID | Token | Role | SFT loss |
|---|---|---|---|
| 7 | `<think>` | open reasoning span | **learned** |
| 8 | `</think>` | close reasoning span | **learned** |
| 9 | `<\|python\|>` | open Python tool call (SymPy/NumPy/SciPy/CoolProp/python-control/pint) | **learned** |
| 10 | `<\|octave\|>` | open Octave tool call | **learned** |
| 11 | `<\|/tool\|>` | close tool call → **pause, execute** | **learned** |
| 12 | `<\|tool_result\|>` | open injected result (closed by the existing `<\|end\|>`, ID 6) | **masked** |
| 13 | `<\|assay\|>` | open **Assay** tool call — payload is the **IR as JSON** (`{"task", "inputs", "missing_inputs"}`); the template owns the method, the model routes + fills slots. *Claimed 2026-07-19.* | **learned** |
| 14–15 | reserved | FIM prefix/middle/suffix *or* future tool/control | — |

Seven new tokens carry all of TIR + thinking + Assay. Design notes:

- **Runtime identity is in the open tag** (`<|python|>` / `<|octave|>` / `<|assay|>`),
  so the harness routes by ID with no payload parsing; a **shared close** `<|/tool|>`
  keeps the tool-token count down.
- **Payload is raw source** for the code runtimes, not JSON-wrapped — the tool's
  "argument" *is* code, and JSON-escaping multiline code with quotes/backslashes
  (regex, LaTeX) is a reliability sink. R1/Qwen TIR emit raw source between
  delimiters; so do we. The **assay payload is JSON by design** — its argument is
  structured data (an IR), not code, so the code-escaping rationale doesn't apply.
- **The result reuses `<|end|>` as its closer**, matching how every turn closes;
  no separate result-close token needed.
- **Recommendation to feed back into `docs/tokenizer.md`:** either commit
  FIM to 13–15 now (code-infill is plausibly wanted, and adding tokens post-
  freeze is a full migration) **or widen the reserved tail to ~ID 19** for a few
  genuine spares. Vocab slots are ~free (one embedding row each at 32k); a
  re-freeze is not. I lean **widen + pre-commit FIM** — decide at freeze time.

## 3. The episode grammar

A TIR assistant turn, one or more tool calls, ending on `<|end|>`:

```
<|user|> {problem} <|end|>
<|assistant|> <think> {reasoning} </think>
{optional prose} <|python|>
{source}
<|/tool|>                          ← model STOPS; harness executes
<|tool_result|>
{captured stdout / return, or traceback on error; truncated to a cap}
<|end|>                            ← harness INJECTS (masked); model resumes
{more <think>…</think>, more tool calls, …}
{final answer} <|end|>             ← turn genuinely ends
```

Worked example (the property-lookup thesis, units-checked):

```
<|user|> Saturation pressure of water at 120 °C? <|end|>
<|assistant|> <think> Saturation pressure isn't something to recall to 5 figures —
that's a steam-table lookup. CoolProp: PropsSI('P','T',T,'Q',0,'Water'), T in
kelvin. </think> <|python|>
from CoolProp.CoolProp import PropsSI
print(PropsSI('P', 'T', 120 + 273.15, 'Q', 0, 'Water'))
<|/tool|>
<|tool_result|>
198672.6...
<|end|>
So the saturation pressure at 120 °C is about **198.7 kPa** (≈1.96 atm). <|end|>
```

Rendering rules (the harness enforces, not the model):
- **Truncate tool results** to a token cap before injection — a giant array
  print must not blow context (also a reward-hacking surface; see the E1e judge).
- **Cap tool calls per episode** (config, e.g. 4) — prevents infinite call loops.
- **Errors pass through** as the result (stderr/traceback), so retry-on-error is
  learned, not special-cased.

## 4. Loss masking & generation control

**SFT loss mask** (extends `render_conversation`'s existing rule; all by ID):

| Segment | Learned? |
|---|---|
| `<|assistant|>` header, `<|user|>`/`<|system|>` turns, BOS | masked (supplied at inference) |
| `<think>` … `</think>` + content | **learned** |
| tool-call open (`<|python|>`/`<|octave|>`) + source + `<|/tool|>` | **learned** |
| `<|tool_result|>` + result + its closing `<|end|>` | **masked** (the sandbox wrote it) |
| resumed reasoning + final answer + final `<|end|>` | **learned** |

The one new rule vs today: the `<|tool_result|>…<|end|>` span is masked. In RLVR
(E4) the identical span is excluded from the policy-gradient **and** the KL —
those tokens are the environment's move, not the policy's action.

**Generation control loop** (drives inference *and* the E4 RL rollout):

1. Generate until `<|/tool|>`, `<|end|>`, or `max_new` tokens.
2. **Stopped on `<|/tool|>`:** slice the call (from the last `<|python|>`/
   `<|octave|>` to `<|/tool|>`), route by the open-token ID, execute in the E1
   sandbox, capture output/traceback, inject `<|tool_result|> {capped} <|end|>`,
   **resume** (subject to the per-episode call cap).
3. **Stopped on `<|end|>` or `max_new`:** turn complete.

`<|/tool|>` and `<|end|>` are the two stop tokens; their distinction (execute-
and-resume vs finish) is why the tool call needs its own closer rather than
overloading `<|end|>`.

## 5. Cross-family compatibility (500M ↔ 4B hero)

One canonical JSONL feeds both, because the string forms were chosen to be
registrable as special tokens in *either* vocab:

- **`<think>`/`</think>`** are Qwen3's actual thinking strings — the hero's
  tokenizer maps them natively; the Lithos tokenizer registers them at IDs 7–8.
- **`<|python|>`/`<|octave|>`/`<|/tool|>`/`<|tool_result|>`** are Lithos-specific;
  the hero adds them as special tokens (Qwen reserves spare `<|…|>` slots for
  exactly this). **Verifying that availability is folded into the E7 Qwen-import
  spike** — if Qwen's reserved slots can't take them cleanly, the hero needs a
  render-time remap and "one canonical format" weakens to "one format + a hero
  adapter." Resolve in E7 before the claim hardens.

Net: harvested traces convert once, into this format; each family member's
tokenizer handles the ID mapping. No per-model dataset fork for the MVP.

## 6. What this unblocks (and what it deliberately doesn't decide)

**Unblocks:** the tokenizer v1.0 freeze (reserved block now has assigned
meaning, §2); E3 (template/dataset extension has an exact spec, §3–4); E1's tool
schema (`<|python|>`/`<|octave|>` + raw-source payload); bulk trace conversion
(§4 + §5 give the target format).

**Not decided here (correctly downstream):** the *target* extended context
(measure the trace-length distribution → epic E10; §2.3 resolved to
long-context extension, but the exact length is a pretraining-design call); FIM's
final inclusion (a tokenizer-freeze call, §2); the anti-gaming judge's rules
(E1e); RLVR reward shaping (E1b/E4). Those consume this format; they don't
change it.

## 7. The producer record schema (`segments` JSONL) — authoritative for Chisel

§2–§4 define the *wire* format (tokens, grammar, masking) the tokenizer sees. This section is
the **producer JSONL** Chisel writes — the structured form `chat_template.render_conversation`
turns into that wire format + loss mask. The **executable spec is
`lithos/posttrain/tir_validate.validate_tir_record`**; both repos call it as the shared
ingestion gate (R2 contract §3.4).

An SFT/TIR line is one conversation:

```
{ "messages": [ message, ... ] }          # non-empty; extra top-level keys (e.g. source_id) ignored
```

A message is either a flat turn or a segmented assistant turn (**exactly one** of
`content`/`segments`; `segments` are assistant-only):

```
{ "role": "system"|"user"|"assistant", "content": str }
{ "role": "assistant", "segments": [ segment, ... ] }
```

A segment is one of four typed shapes — every declared field a **string**:

```
{ "type": "think",       "text": str }                          # learned
{ "type": "text",        "text": str }                          # learned
{ "type": "tool",        "runtime": "python"|"octave", "code": str }   # learned
{ "type": "tool_result", "output": str }                        # MASKED (Lithos appends + masks its <|end|>)
```

The validator fails loud (never silently drops) on: both/neither of `content`/`segments`;
`segments` on a non-assistant turn; unknown segment `type` or tool `runtime`; a missing or
non-string field. **Ordering (grammar, §3):** a `tool_result` answers the immediately
preceding `tool` call — emit them in call→result order (the validator checks structure, not
this pairing, so keep it correct at the producer).

*Why a standalone validator and not just "trust the schema":* the `tool_result` span is masked
from the loss **by token ID**. A mistyped or misplaced segment would train the model on the
sandbox's own output — an invisible corruption that only surfaces as a degraded model. So the
gate runs at *both* ends against the same `validate_tir_record` (Chisel before emitting, Lithos
before rendering); golden fixtures live at `tests/fixtures/tir_golden.jsonl`, tested by both.

## Pointers

- Closes: `docs/post-training-review.md` §2.1 (tool turn), §2.2 (reasoning
  format), §2.5 (the freeze dependency).
- Sequenced as epic **D1** in `docs/post-training-implementation-plan.md`;
  downstream epics E1/E3/E4/E7 all reference this spec.
- Feeds: `docs/tokenizer.md` §3.3 (the reserved block this fills). Chat/mask
  precedent: `lithos/posttrain/chat_template.py`. Tools/runtime: Phase 12 in
  `implementation-plan.md`.
