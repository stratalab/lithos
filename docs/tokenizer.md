# The Lithos Tokenizer — Design & Retrain Plan

A tokenizer is a frozen frequency census of its training sample; its quality is
how well that census matches what the model will actually read. Ours (v0.1) was
censused on FineWeb-Edu prose, and the STEM pivot changes the population. This
doc records what stays settled, the measured case for a retrain, the design
decisions for the STEM vocab, and the gates it must pass before we freeze it.
Companion to `docs/data-construction.md` (stage 11; the §1.9 target mix defines
the training sample) and `docs/eval-plan.md` (tier-3 verdicts).

## 0. Principles

1. **Compression is capability at the edge.** Bytes/token multiplies effective
   context length and tokens-per-FLOP — for a compact on-device model, a 20%
   fertility tax on math/code is a 20% cut to how much STEM fits in the window
   and the budget. Fertility is not a vanity metric here.
2. **Freeze late, gate hard, retrain once.** A tokenizer swap invalidates every
   tokenized shard and every checkpoint. We carry the FineWeb-Edu v0.1 tokenizer
   until the STEM corpus mix (§1.9 of data-construction) is decided, retrain
   once against that mix, gate it (§4), and freeze as v1.0 for the whole ladder.
3. **The sample is the spec.** BPE training is unsupervised frequency counting;
   there is no other knob that matters as much as what text goes into the
   200k-doc training sample. Curating that sample *is* designing the tokenizer.
4. **Downstream decides.** Tier 1–2 metrics (`scripts/eval_tokenizer.py`) are
   cheap gates and regression tests; the verdict is per-domain **bits-per-byte
   on the 100M rig** — the same doctrine as the quality classifiers, and bpb is
   used precisely because perplexity is not comparable across tokenizers.
5. **Special-token IDs are forever.** Pinned low IDs (0–6) survive retrains by
   construction (PRD §7.1); chat templates and eval harnesses never move.

## 1. Settled (carried from v0.1 unchanged)

| Decision | Value | Why |
|---|---|---|
| Algorithm | byte-level BPE, no `<unk>` | 256-byte base alphabet ⇒ lossless on any input (68/68 adversarial roundtrip, incl. bidi/ZWJ/CRLF); deterministic; ecosystem-standard |
| Digits | `individual_digits: true` | consistent number segmentation helps arithmetic — a first-class concern for a STEM model; verified in segmentation probes (`12345` → 5 tokens) |
| Special tokens | 7 pinned at IDs 0–6 | stability across retrains (PRD §7.1) |
| `add_prefix_space` | false | lossless roundtrip |
| `min_frequency` | 2 | standard; no evidence for change |
| Trainer | `scripts/train_tokenizer.py`, ~200k-doc sample | pair frequencies converge fast; sample *distribution* is the lever, not size |

**Vocab size stays 32,000 for the 100M and 500M rungs.** The embedding matrix
is 32k × 768 = 24.6M of the 100M model's ~110M params (tied) — already ~22% of
the model. Doubling to 64k adds another 24.6M params of lookup table at the
expense of nothing else; at these scales, merges are cheaper than rows. Revisit
at the 1B/3B rungs where the embedding share falls to a few percent (§6). Hard
ceiling regardless: **65,536**, the uint16 shard boundary
(`lithos/data/shard.py::dtype_for_vocab`) — crossing it doubles corpus bytes on
disk and in R2.

## 2. The measured case for a retrain (2026-07, `corpus/probes/`)

`fineweb-edu-32k` vs references on identical bytes (ratio = our tokens / theirs;
probe-sized samples, directional):

| domain | bytes/tok | vs gpt2 | vs Qwen2.5 (151k) |
|---|---|---|---|
| general | 4.92 | 1.04x | 1.02x |
| physics | 2.71 | 1.14x | 1.14x |
| engineering | 2.90 | 1.13x | 1.11x |
| math | 2.11 | 1.15x | 1.20x |
| code | 1.98 | 1.14x | **1.75x** |

Diagnosis (segmentation probes): `\alpha` → `\|al|pha`, `\frac{` → 4 tokens,
and an 8-space indent costs **8 tokens** — the vocab contains no multi-space
runs, no LaTeX commands, no code idioms, because zero of the ~31k merges saw
them. On prose the 32k vocab is within 2–4% of vocabularies 4–5× its size, so
the format is fine; the census population was wrong. Note the GPT-2 pre-token
regex *does* permit whitespace-run tokens (`\s+(?!\S)`); their absence is purely
a training-data fact, so the fix is the sample, not the pre-tokenizer.

## 3. Design: `lithos-stem-32k` v1.0

**3.1 Training sample = the §1.9 target mix, post-filter.** Draw the ~200k–500k
doc sample from the *cleaned* corpus (post stage-3 heuristics, post stage-4
thresholds where available) in the decided mix proportions. Rationale: the
model reads the filtered mix, so the census should count it; sampling raw would
spend merges on boilerplate we later delete. This sequences the retrain *after*
the mix decision and at least a first-pass filtered corpus — it is not blocked
on the full 10-stage pipeline.

**3.2 Consider a code/LaTeX floor, decide empirically.** If the final mix is
prose-heavy, mix-proportional sampling may still under-allocate merges to
indentation and LaTeX. Candidate B = same sample with code+math floored at
~2× their mix share. Train both (minutes of CPU), run tier 1–2 on both, send
the winner — or both, if they split the gates — to the 100M ablation. Merges
are zero-sum, so this is exactly the trade the gates exist to referee.

**3.3 Reserve a special-token block now — now with assigned meaning (`docs/tir-format.md` D1).**
Pin IDs 7–15 before training v1.0. The TIR format decision (which had to precede
this freeze — that's the §2.5-analog sequencing constraint) assigns the first
six: `<think>`(7) `</think>`(8) `<|python|>`(9) `<|octave|>`(10) `<|/tool|>`(11)
`<|tool_result|>`(12), leaving 13–15 for FIM or spares. **Recommendation carried
from D1:** either pre-commit FIM (prefix/middle/suffix) to 13–15 now, or **widen
the tail to ~ID 19** for genuine spares — vocab slots are ~free (one embedding
row each), a post-freeze re-migration is not. This is the exact failure the
pinned-ID rule exists to prevent, extended one step: tool/think/FIM markers get
stable IDs *without* a later vocab migration.

**3.4 Everything else per §1.** Same algorithm, digits, prefix-space,
min-frequency, 32k size. One new config: `configs/tokenizer/stem-32k.yaml`
(the `data:` block becomes a jsonl sample exported from the corpus pipeline
rather than a raw HF stream).

## 4. Gates (in order; each gate cheap enough to iterate)

**Tier 1–2** — `scripts/eval_tokenizer.py`, probe sets + large held-out
`--sample` slices per domain:

- Roundtrip: 68/68 lossless, non-negotiable.
- Specials: `stable_low_ids` OK with the extended 16-token list.
- Compression: math and code ratios vs Qwen2.5 improve to ≤ 1.10x and ≤ 1.30x
  respectively; **general regresses ≤ 5%** vs fineweb-edu-32k on the same
  probes (the prose we're willing to pay).
- Segmentation checklist (diff vs the v0.1 report): `\frac{`, `\alpha`,
  `\begin{` ≤ 2 tokens each; 8-space indent ≤ 2 tokens; digits still
  individual; no multi-digit merges leaked.
- Vocab health on the large samples: dead-token rate < ~10%; undertrained
  candidates reviewed by eye (they predict glitch tokens).

**Tier 3** — the verdict: two 100M runs, identical data and byte content,
identical *byte* budget (token budgets differ by construction), compare
per-domain bpb via the existing ablation harness. Adopt iff STEM-domain bpb
improves and general bpb holds within noise. Then freeze v1.0.

## 5. Migration checklist (when v1.0 freezes)

1. Push tokenizer to R2; record eval_report.json numbers into §2's table here
   (artifacts/ is gitignored — the doc is the durable record).
2. Retokenize all corpora — shards are tokenizer-specific; manifests carry
   `tokenizer_name`, so stale mixes fail loudly rather than silently.
3. Old checkpoints are incompatible (embedding rows reindexed); the ladder
   restarts from the v1.0 tokenizer, which is why this precedes the 500M run.
4. Chat template, eval harness, serving: no changes — pinned IDs.
5. `corpus/probes/` segmentation diff checked in with the retrain PR.

## 6. Open questions (banked, navigate empirically)

- **Digit chunking.** Individual digits vs right-to-left 3-digit groups
  (Llama-3-style): revisit when arithmetic evals exist at 500M; chunking
  triples number compression but risks inconsistent splits.
- **48–64k vocab at 1B/3B.** Embedding share falls with scale; a larger vocab
  mostly buys prose/multilingual headroom we may not want. Re-run this doc's
  §3–4 process if revisited; uint16 ceiling stands.
- **Math-symbol coverage.** Unicode math (∂, ℝ, ±, °, Ω) currently costs 2–3
  tokens each via byte fallback; whether arXiv-heavy sampling fixes this
  organically or the segmentation gate needs explicit lines — check on v1.0.

## Pointers

- Code: `lithos/tokenizer/{tokenizer_config,train_tokenizer,evaluate,inspect_tokenizer}.py`,
  `scripts/{train_tokenizer,eval_tokenizer}.py`, `corpus/probes/` (+ README),
  `lithos/data/shard.py` (uint16 boundary).
- Configs: `configs/tokenizer/fineweb-edu-32k.yaml` (current), `stem-32k.yaml` (to write).
- Docs: `data-construction.md` §1.2 stage 11 + §1.9 (mix), `eval-plan.md` (ablation harness).
- Recipes: GPT-2/tiktoken (byte-level BPE), StarCoder2 (code tokenization),
  Llama-3 §3.2 (digit handling), Qwen2.5 (the compression bar we measure against).
