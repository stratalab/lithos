# Lithos Eval Plan — the measuring stick

How we test every model in the family, at every lifecycle stage. Companion to
`implementation-plan.md` (Phase 9 built the harness skeleton; Phase 12
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
| **Frozen battery** (v1 ✅; v2/v3 composition per domain: §2) | general + STEM knowledge vs anchors | lm-eval 0-shot (few-shot at ≥500M) | v1 ✅ + anchors ✅ |
| **Executable STEM battery** | code + math problem-solving | HumanEval/MBPP+ executed in sandbox (pass@1); GSM8K/MATH answer-checked | verifier = Phase 12 sandbox |
| **TIR battery** | the defining capability: reason → call Python/Octave → use result | sandbox verifies value **and** units; disjoint from RLVR pool | Phase 12 |
| **Tool-uplift metric** | the product thesis, quantified | solve rate *with* sandbox − *without*, same problems; goes on the model card | Phase 12 |
| **Transfer probes** | cross-domain reasoning (the pivot's actual target) | derive-then-implement (physics → code), verified end-to-end | Phase 12 |
| **Eng/physics exam sets** | the gap no public benchmark covers | held-out `kind=problems` acquisitions (FE-style, quals, olympiad), value+units checked | from problem-bank acquisition |
| **IFEval + format checks** | instruction following, chat template sanity | rule-based | post-training stage |
| **Judged comparisons** | helpfulness/explanation quality where execution can't reach | LLM-judge vs same-size instruct anchors (closed judge = build tool, doctrine-clean) | post-training stage |
| **Regurgitation eval** | doctrine caveat on grey-tier works | verbatim-continuation probe over indexed grey works | to build (§1.5 promise) |
| **Edge suite** | deployment reality | tokens/s, memory ceiling, TIR battery re-run on 4-bit GGUF on target hardware | Phase 12 acceptance |
| **Parity matrix** | the win condition: which weight class each task plays in | full battery over the §3 anchor tiers → task × weight-class table, losses included; + capability-per-GB | flagship keeper |

## 2. Battery composition ladder (v1 → v3)

Versions freeze at keeper gates (principle 1). A benchmark enters the battery only at the rung where it produces signal for our sizes — below its floor it can't move a decision, so it stays out (that's most famous benchmarks; bpb decides, §5).

- **v1 (frozen ✅, signal at 100M):** sciq, arc, piqa, hellaswag, winogrande, lambada, obqa.
- **v2 (freezes at the 500M keeper):** v1 + **GSM8K paired with GSM-Symbolic** (perturbed templates; the score gap between the pair *measures memorization* — run both, always) + SVAMP/ASDiv (easier word problems; earliest math signal) + MMLU-STEM (25% MC guess floor: small deltas = noise, secondary) + **HumanEval+/MBPP+** (EvalPlus variants — the originals' weak tests pass wrong code) + **CRUXEval** (predict function output/input: code *reasoning* without long generation, signal earlier than generation benchmarks) + IFEval (post-SFT stages only).
- **v3 (freezes before the 1B/3B + hero keepers):** v2 + MATH-500 + **LiveCodeBench** (rolling window — contamination-resistant, and the design model for §6) + SciBench + TheoremQA + JEEBench + the first frozen slice of our exam-derived physics/eng sets + the TIR battery.

Per-domain inventory, with honest signal floors:

| Domain | Public instruments | Signal floor | The real instrument |
|---|---|---|---|
| Math | GSM8K+GSM-Symbolic, SVAMP/ASDiv (v2); MATH-500 (v3) | ~300M easy sets; ~500M GSM8K (SFT'd); ~1B MATH | year-split exam bank, difficulty-laddered via the `level` column |
| Code | HumanEval+/MBPP+, CRUXEval (v2); LiveCodeBench (v3); BigCodeBench (hero only) | ~500M generation; CRUXEval earlier | rolling own sets (LiveCodeBench pattern) + TIR battery |
| Physics | SciQ/ARC (v1); MMLU-physics (v2, MC-floor caveat); SciBench/TheoremQA/JEEBench (v3 — free-response numeric, fits the executable doctrine); GPQA model-card-only (floors everything <~30B; run once at hero, expect ~chance, don't steer by it) | knowledge ~100M; problem-solving ~1B | FE-style / quals / olympiad sets, value+units checked |
| Engineering | MMLU-EE (tiny, MC) — otherwise the public landscape is **empty** | — | **ours is the instrument**: FE-style value+units sets + TIR battery + tool-uplift; the vacuum is exactly the §6 opportunity |

Pending a license-and-quality pass before v3 entry: OlympiadBench (text-only slice), UGPhysics, PHYBench. Standing rules: (1) every benchmark named here goes into the decontamination probe list **now**, before the big corpus builds — not as a pre-flagship scramble; (2) benchmarks get the same license check as training data before battery entry; (3) below ~500M, per-domain bpb remains the decision metric — battery scores are confirmation.

## 3. The parity frontier — anchor tiers and the headline artifact

**Win condition (settled 2026-07-03):** Lithos does not chase frontier comparability. The win is **weight-class jumping** — the 3B delivering 8–13B-class performance on specific tasks. The battery's job is therefore to *map the parity frontier*: for each task, which weight class does Lithos actually play in? That requires three anchor tiers, not one:

1. **Size-matched** (Qwen 0.5B/1.5B/4B, later Qwen3 equivalents) — validates the harness and sets the size-expected baseline. Existing role (principle 2), unchanged.
2. **Weight-class-above** (Qwen3-8B, Llama-3.1-8B, gemma-2-9B-class) — **the parity claim is measured here.** Instruct-vs-instruct, best settings, on our harness: the §6 anti-sandbagging discipline applies to our own comparisons first.
3. **Same-size specialists** (Qwen2.5-Math-1.5B, Qwen2.5-Coder-3B, Phi-4-mini-class) — the honesty tier. A math-tuned model beating 8B *generalists* on math is table stakes (Qwen2.5-Math-1.5B already does it); the novel claim is **one compact model holding specialist parity across all four domains simultaneously**, plus owning the eng/TIR/transfer surface where no competitor exists at any size.

**Headline artifact: the parity matrix** — task rows × weight-class columns, showing which column Lithos occupies per row, **including the breadth rows we lose** (MMLU-everything, open-ended chat: capacity spent on STEM instead, by design). Publishing the losses is what makes the wins credible — same discipline as §6's "categories we lose."

**The denominator is memory, not parameters.** For the edge product, comparisons also run at equal deployment footprint: capability-per-GB on target hardware, quantized artifact (principle 7). Lithos-3B-4bit (~1.7GB) matching an 8B-4bit (~4.5GB) is a ~2.6× resource claim, stated as such on the model card.

Why weight-jumping is plausible — and therefore which battlegrounds to pick: **specialization** (capacity on four domains, not breadth), **TIR** (the model+sandbox *system* competes, ToRA-style), **RLVR on verified data**. Tasks where these levers dominate are winnable; breadth-bound tasks are not. Chosen battlegrounds only.

## 4. Lifecycle: which instruments, which gates

| Stage | Question | Instruments | Gate |
|---|---|---|---|
| **Pretrain (continuous)** | is training healthy? | per-domain bpb on checkpoints; loss curves | bpb monotone-ish; no slice regressing |
| **Base keeper** | did pretraining work? | full frozen battery vs anchors; bpb; regurgitation eval | ≥ anchor-relative expectation for size; regurgitation clean |
| **Mix sweep (100M)** | which data recipe? | per-domain bpb (primary — benchmarks are at chance <500M) | winning mix carried up |
| **SFT** | usable assistant? | executable STEM battery; IFEval; judged comparison; **regression gate** on base battery + bpb | no base collapse; battery jump vs base |
| **RLVR-TIR** | does the defining capability exist? | TIR battery; tool-uplift; transfer probes; exam sets; reward-hacking audit (held-out verifiers + rollout reading) | tool-uplift positive and material; audits clean |
| **Flagship keeper (3B / 4B hero)** | does it punch above weight? | parity matrix over the §3 anchor tiers; capability-per-GB | 8–13B-column wins on chosen battlegrounds; specialist parity held; losses published |
| **Ship (GGUF/edge)** | works where it ships? | edge suite; TIR battery quantized | capability survives quantization |

Cadence/cost tiers: **continuous** (bpb per checkpoint — free), **per-experiment** (frozen battery — minutes), **keeper** (full suite + judged + regurgitation + quantized edge run).

## 5. Known gaps (honest)

- **No public benchmark measures our niche** — compact STEM + tool-integrated reasoning. GPQA floors small models, MMLU has a 25% MC guess floor, HumanEval/GSM8K are saturated/contaminated. Our exam-derived executable sets are the stopgap; the long-term answer is §6.
- Below ~500M, benchmark signal is mostly noise — bpb carries the decision weight (this is by design, not a workaround).
- Physics/eng *knowledge* (vs problem-solving) has no clean instrument anywhere; bpb on those slices is the proxy.

## 6. The public benchmark (post-MVP; design-as-if-published starts now)

Eventually we publish the battery's curated slice as a public benchmark — the niche has no scoreboard, and whoever defines the benchmark defines the category. Self-built benchmarks earn their grain of salt from four specific suspicions; each gets a structural answer:

1. **"You trained on it"** → rolling year-partitioned sections (each year's new exams are post-cutoff for *every* evaluated model — contamination impossible by construction, LiveCodeBench-style).
2. **"You graded it favorably"** → executable verification only; no judge anywhere in scoring.
3. **"You designed it around your model"** → coverage derived from the curriculum taxonomy (topic-graph families), not model behavior; benchmark version locked and published **before** our flagship results, including categories we lose.
4. **"Sandbagged baselines"** → one-command reproducible harness, full logs, best-settings baselines — and a competitor topping the leaderboard stays up. That's what makes it a benchmark, not an ad.

Economics: the benchmark is ~free — it's the internal battery with a README, versioning, and a leaderboard. It also compounds: the harness is a Lithos+sandbox demo, green-tier problems overlap the StrataDB sample-data exports, and the same acquisition feeds RLVR/internal/public uses through disjoint pools. **Standing discipline from today:** build the battery as if it will be published — clean task specs, versioned, no internal shortcuts.

## 7. Status ledger

- ✅ Harness + frozen battery v1 + Qwen anchors + decontam wiring + ablation loop (Phase 9, validated on the 100M).
- ✅ Battery v2/v3 composition + parity-frontier anchor design decided (§2–3, 2026-07-03).
- ⏳ From corpus build: per-domain bpb sets, problem-bank year splits.
- ◻ To build: executable battery wiring (sandbox = Phase 12), TIR battery + tool-uplift protocol, transfer probes, judged-comparison harness, regurgitation eval, edge suite.
- ◻ From §2–3: extend the decontam probe list with every §2-named benchmark (before the big corpus builds); license/quality pass on OlympiadBench/UGPhysics/PHYBench; tier-2/3 anchor runs (8B-class + specialists) on our harness; parity-matrix + capability-per-GB reporting in the scorecard.
