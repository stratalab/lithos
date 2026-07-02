# Lithos Eval Plan — the measuring stick

How we test every model in the family, at every lifecycle stage. Companion to
`lithos-implementation-plan.md` (Phase 9 built the harness skeleton; Phase 12
adds the TIR battery) and `docs/data-construction.md` (Part 3 cross-refs here;
the corpus build produces the bpb sets and problem banks this plan consumes).

## 0. Principles (non-negotiable)

1. **Frozen + versioned.** A battery never changes mid-comparison; additions create v2. Scores are only comparable within a battery version.
2. **Anchors make the harness trustworthy.** Known open models (Qwen2.5-0.5B/1.5B, later Qwen3-0.6B/1.7B and instruct variants) run on *our* harness must reproduce published numbers within ~1–2%. When our model's score moves, it's the model, not the harness. (Validated on the 100M shakedown: clean monotonic ordering, anchors on-spec.)
3. **Decontaminate training data against every eval** — 13-gram matching *plus* paraphrase-aware checks for exam content (study sites leak reworded versions past n-grams).
4. **Executable scoring wherever possible.** Code runs, values match, units balance (`pint`). No judge ambiguity; the sandbox is the grader.
5. **One verifier, three consumers, disjoint pools.** The same sandbox scores evals, provides RLVR reward, and filters synthetic data — but eval problems are never in the RLVR training pool. Problem banks are **split by year**: train on pre-2024, eval on 2024+ (renewable annually, contamination-resistant by construction).
6. **Regression gates.** Every post-training stage must not collapse the previous stage's scores (base battery + bpb after SFT; SFT battery after RL). Catastrophic forgetting is a test failure, not a surprise.
7. **Evaluate the artifact that ships.** Keeper evals re-run on the quantized GGUF on target hardware — quantization can silently break tool-call formatting.
8. **Watch real rollouts, not the reward curve** (banked Phase-11 lesson). Score deltas get spot-checked against actual transcripts before being believed.

## 1. The instruments

| Instrument | What it measures | Scoring | Status |
|---|---|---|---|
| **Per-domain bpb** | pretrain quality per slice (code/math/physics/eng/general) | bits-per-byte on held-out sets diverted at corpus build | sets = corpus-build output (Phase 10) |
| **Frozen battery** (v1: sciq, arc, piqa, hellaswag, winogrande, lambada, obqa; v2 adds MMLU-STEM, GSM8K) | general + STEM knowledge vs anchors | lm-eval 0-shot (few-shot at ≥500M) | v1 ✅ + anchors ✅ |
| **Executable STEM battery** | code + math problem-solving | HumanEval/MBPP+ executed in sandbox (pass@1); GSM8K/MATH answer-checked | verifier = Phase 12 sandbox |
| **TIR battery** | the defining capability: reason → call Python/Octave → use result | sandbox verifies value **and** units; disjoint from RLVR pool | Phase 12 |
| **Tool-uplift metric** | the product thesis, quantified | solve rate *with* sandbox − *without*, same problems; goes on the model card | Phase 12 |
| **Transfer probes** | cross-domain reasoning (the pivot's actual target) | derive-then-implement (physics → code), verified end-to-end | Phase 12 |
| **Eng/physics exam sets** | the gap no public benchmark covers | held-out `kind=problems` acquisitions (FE-style, quals, olympiad), value+units checked | from problem-bank acquisition |
| **IFEval + format checks** | instruction following, chat template sanity | rule-based | post-training stage |
| **Judged comparisons** | helpfulness/explanation quality where execution can't reach | LLM-judge vs same-size instruct anchors (closed judge = build tool, doctrine-clean) | post-training stage |
| **Regurgitation eval** | doctrine caveat on grey-tier works | verbatim-continuation probe over indexed grey works | to build (§1.5 promise) |
| **Edge suite** | deployment reality | tokens/s, memory ceiling, TIR battery re-run on 4-bit GGUF on target hardware | Phase 12 acceptance |

## 2. Lifecycle: which instruments, which gates

| Stage | Question | Instruments | Gate |
|---|---|---|---|
| **Pretrain (continuous)** | is training healthy? | per-domain bpb on checkpoints; loss curves | bpb monotone-ish; no slice regressing |
| **Base keeper** | did pretraining work? | full frozen battery vs anchors; bpb; regurgitation eval | ≥ anchor-relative expectation for size; regurgitation clean |
| **Mix sweep (100M)** | which data recipe? | per-domain bpb (primary — benchmarks are at chance <500M) | winning mix carried up |
| **SFT** | usable assistant? | executable STEM battery; IFEval; judged comparison; **regression gate** on base battery + bpb | no base collapse; battery jump vs base |
| **RLVR-TIR** | does the defining capability exist? | TIR battery; tool-uplift; transfer probes; exam sets; reward-hacking audit (held-out verifiers + rollout reading) | tool-uplift positive and material; audits clean |
| **Ship (GGUF/edge)** | works where it ships? | edge suite; TIR battery quantized | capability survives quantization |

Cadence/cost tiers: **continuous** (bpb per checkpoint — free), **per-experiment** (frozen battery — minutes), **keeper** (full suite + judged + regurgitation + quantized edge run).

## 3. Known gaps (honest)

- **No public benchmark measures our niche** — compact STEM + tool-integrated reasoning. GPQA floors small models, MMLU has a 25% MC guess floor, HumanEval/GSM8K are saturated/contaminated. Our exam-derived executable sets are the stopgap; the long-term answer is §4.
- Below ~500M, benchmark signal is mostly noise — bpb carries the decision weight (this is by design, not a workaround).
- Physics/eng *knowledge* (vs problem-solving) has no clean instrument anywhere; bpb on those slices is the proxy.

## 4. The public benchmark (post-MVP; design-as-if-published starts now)

Eventually we publish the battery's curated slice as a public benchmark — the niche has no scoreboard, and whoever defines the benchmark defines the category. Self-built benchmarks earn their grain of salt from four specific suspicions; each gets a structural answer:

1. **"You trained on it"** → rolling year-partitioned sections (each year's new exams are post-cutoff for *every* evaluated model — contamination impossible by construction, LiveCodeBench-style).
2. **"You graded it favorably"** → executable verification only; no judge anywhere in scoring.
3. **"You designed it around your model"** → coverage derived from the curriculum taxonomy (topic-graph families), not model behavior; benchmark version locked and published **before** our flagship results, including categories we lose.
4. **"Sandbagged baselines"** → one-command reproducible harness, full logs, best-settings baselines — and a competitor topping the leaderboard stays up. That's what makes it a benchmark, not an ad.

Economics: the benchmark is ~free — it's the internal battery with a README, versioning, and a leaderboard. It also compounds: the harness is a Lithos+sandbox demo, green-tier problems overlap the StrataDB sample-data exports, and the same acquisition feeds RLVR/internal/public uses through disjoint pools. **Standing discipline from today:** build the battery as if it will be published — clean task specs, versioned, no internal shortcuts.

## 5. Status ledger

- ✅ Harness + frozen battery v1 + Qwen anchors + decontam wiring + ablation loop (Phase 9, validated on the 100M).
- ⏳ From corpus build: per-domain bpb sets, problem-bank year splits.
- ◻ To build: executable battery wiring (sandbox = Phase 12), TIR battery + tool-uplift protocol, transfer probes, judged-comparison harness, regurgitation eval, edge suite.
