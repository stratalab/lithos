# Deep Research Prompt — How AI labs run evals, and which benchmarks are relevant to us

*Paste everything below the line into your deep-research tool. It is self-contained.*

---

## Role and objective

You are a research analyst specializing in LLM evaluation methodology. I am building a
family of **compact, from-scratch STEM reasoning models** and need to design an evaluation
program with two halves: **(a) internal evals** I steer development with, and **(b)
externally-accepted benchmarks** I report to earn credibility with a technical audience.
These answer to different masters — I optimize against the internal set, but must be careful
*not* to overfit the external one — so I need to understand both, and the line between them.

Produce a rigorous, **citation-backed** report on how AI labs actually run and report model
evaluations in 2024–2026, and a **credibility-graded map** of which benchmarks are worth
reporting for a model like mine. Prioritize primary sources: model technical reports and
model cards, benchmark papers, evaluation-harness repositories and their docs, and reputable
methodology papers. Where labs disagree or numbers fail to reproduce, say so and explain why.

## Context — the model being evaluated (so you can judge relevance)

- **What it is:** a compact cross-domain STEM reasoner — **code + math + physics + engineering**
  — targeted at *edge/on-device* deployment (quantized, small memory footprint).
- **Sizes:** a ladder of small models — ~100M (done), **500M flagship**, 1B, 3B — plus a **4B
  "hero"** that is a continued-pretrain of an open base (Qwen3-4B lineage). So relevance must be
  judged at **sub-1B through ~4B** scale, *not* frontier scale.
- **Defining capability — Tool-Integrated Reasoning (TIR):** the model reasons, then **calls a
  code sandbox (Python/Octave)** and uses the result. The core product thesis is that *STEM
  computation is done better by executing code than by parametric recall*. So how the field
  evaluates **tool use / program-aided / agentic** reasoning matters as much as static QA.
- **Win condition:** not frontier comparability. It is **weight-class jumping** — a 3B matching
  **8–13B-class** performance on specific STEM tasks, and **capability-per-GB** on edge hardware
  (quantized). We expect to *lose* on breadth (general MMLU, open-ended chat) by design.
- **Scoring doctrine:** prefer **executable/verifiable grading** (a sandbox checks the value,
  units balance) over LLM-as-judge wherever possible.
- **Contamination stance:** we decontaminate training data against every eval and split problem
  banks **by year** (train on pre-cutoff, eval on post-cutoff, renewed annually — the
  LiveCodeBench pattern), so contamination is prevented by construction.

I already have: per-domain bits-per-byte for pretraining health, a frozen general-knowledge
battery with size-matched anchors, and a decontamination pipeline. **The gaps I most need this
research to inform are: the tool-use/TIR evaluation standard, the credibility status of STEM
benchmarks at my scale, physics/engineering coverage (where public benchmarks seem thin), and
the accepted methodology + reporting norms so my numbers are trusted.**

---

## Workstreams (answer each; keep them separate in the report)

### WS1 — The lab eval playbook (operational)
How do labs that ship **small and STEM/code models** — Qwen/Alibaba, DeepSeek, Microsoft (Phi),
Google (Gemma), Meta (Llama 3.2 small), Mistral, HuggingFace (SmolLM), plus the frontier labs as
reference — actually *run and report* evals?
- **Harnesses in real use:** EleutherAI `lm-evaluation-harness`, HuggingFace `lighteval`, Stanford
  `HELM`, OpenCompass, EvalPlus, `bigcode-evaluation-harness`. Which harness is standard for
  which benchmark, and how much do results diverge between them?
- **Protocol conventions:** zero-shot vs few-shot (and how many shots is "standard" for each
  benchmark), greedy vs. sampled decoding, **pass@k**, **maj@k / self-consistency**, chat-template
  vs. raw-completion effects, and prompt-format sensitivity. Quantify how much these choices move
  scores where sources allow.
- **The reproducibility problem:** why the "same benchmark, different number" gap exists across
  reports, and what disclosure lets a reader trust a number.
- **Report vs. internal:** what actually appears on a model card / tech report vs. what labs keep
  private.

### WS2 — The externally-accepted benchmark map (compact STEM + code)
For **each** relevant benchmark below (and any I've missed), give a structured verdict:
*what it measures · current SOTA & saturation level · known contamination/leakage · **signal
floor** (does it separate sub-1B / 1–4B models, or flat-line at chance?) · license & usage
terms · is it still credible to report in 2026, or vanity/saturated?*
- **Math:** GSM8K, GSM-Symbolic, SVAMP/ASDiv, MATH / MATH-500, MinervaMath, AIME, OlympiadBench.
- **Code:** HumanEval / HumanEval+, MBPP / MBPP+, CRUXEval, LiveCodeBench, BigCodeBench,
  (SWE-bench / SWE-bench-Verified as the large-model reference point).
- **Science / physics / eng:** GPQA (+ Diamond), SciBench, TheoremQA, JEEBench, MMLU-STEM slices,
  MMLU-Pro, ARC, SciQ, and **anything covering physics *problem-solving* or engineering** (I
  believe this landscape is nearly empty — confirm and enumerate whatever exists, incl.
  UGPhysics, PHYBench, and similar).
- **General reasoning (as breadth reference):** BBH, MMLU, MMLU-Pro, MMMU.
Flag explicitly which benchmarks are **table-stakes to report** for a small STEM/code model to be
taken seriously, versus which are saturated/contaminated enough to be near-worthless.

### WS3 — Evaluating tool use / TIR / agentic reasoning (my defining capability)
This is the most important workstream. How does the field evaluate models that **use tools/code**?
- **Program-aided / TIR methods and how they report results:** PAL, PoT, ToRA, MathCoder,
  DeepSeek-Math, Qwen2.5-Math (its TIR mode) — how do they measure and present *tool-assisted* vs.
  *tool-free* performance? Is there an accepted **"tool-uplift" metric** (solve rate with tools −
  without)? If not, how is the benefit of tools quantified?
- **Tool-use / function-calling / agent benchmarks:** Berkeley Function-Calling Leaderboard (BFCL),
  ToolBench / ToolEval, API-Bank, τ-bench, AgentBench, GAIA, and any math-specific tool-use
  evals. What each measures, credibility, and whether any fit a *compact* model.
- **The standard question:** is there a *community-accepted* way to benchmark "reason-then-execute"
  STEM problem-solving? If the answer is "no accepted standard exists," say so plainly — that is a
  finding (it means we may have to define one).

### WS4 — Small-model & edge evaluation
- How are **compact/edge** models (Phi-3/4-mini, Gemma-2/3-2B, Qwen2.5-0.5–3B, SmolLM,
  Llama-3.2-1B/3B, MobileLLM) benchmarked and *marketed*? What framing do they use for
  **capability-per-parameter** or **capability-per-GB**?
- **Quantized evaluation:** does 4-bit/INT8 quantization measurably degrade benchmark scores or
  **break tool-call / structured-output formatting**? How do labs report quantized vs. full-precision?
- **On-device reporting:** tokens/s, latency, memory ceiling — what's the accepted way to report
  efficiency alongside capability?
- **The "small specialist beats big generalist" claim** (e.g. Qwen2.5-Math-1.5B vs. much larger
  generalists on math): how is it substantiated credibly, and how do skeptics attack it?

### WS5 — Eval methodology & credibility (the meta-layer)
- **Contamination detection:** n-gram/13-gram overlap, canary strings, membership-inference
  (min-K% prob, perplexity-based), and **rolling-window / time-partitioned** designs (LiveBench,
  LiveCodeBench) — how they work and how strong each is.
- **Statistical rigor:** the critique that a single benchmark number is noise — variance across
  seeds/prompts/orderings, error bars, significance. Summarize the key papers and the emerging
  norms (do labs report confidence intervals yet?).
- **LLM-as-judge:** when it's accepted, its known biases (position, verbosity, self-preference),
  and mitigations — and where the field says *not* to use it.
- **What makes a *self-published* benchmark credible** (I intend to publish one for the STEM/TIR
  niche): the design features that earn trust — executable grading, time-partitioned/rolling
  sections, pre-registration, one-command reproducible harness, published baselines that aren't
  sandbagged, and leaving competitor wins on the leaderboard. Cite examples of self-built
  benchmarks that *did* earn field acceptance and what they did right.

### WS6 — Internal-vs-external split & the reporting contract
- What do labs keep as **private internal evals** (regression suites, held-out sets, red-team)
  vs. publish, and *why*?
- **Overfitting to public benchmarks (Goodhart):** documented cases, and how careful labs guard
  against it (held-out private sets, rolling evals, refusing to train on benchmark-adjacent data).
- **Third-party leaderboards as external validation:** LMArena / Chatbot Arena, HuggingFace Open
  LLM Leaderboard, HELM, Artificial Analysis, and any STEM/code-specific boards. Which actually
  **confer credibility** in 2026, their eligibility/submission process, and whether a compact
  specialist model can meaningfully place on them.

---

## Deliverable format

1. **Executive summary** (~1 page): the 8–12 most decision-relevant findings, each one sentence.
2. **One section per workstream** (WS1–WS6), with a **table** wherever a set of benchmarks/tools
   is compared (columns: what-it-measures · SOTA/saturation · contamination · signal-floor-by-size
   · license · 2026-credibility-verdict).
3. **Synthesis — recommendations for a model like mine:**
   - The **minimal external benchmark set** to report at each rung (**500M / 1B / 3B / 4B-hero**)
     to be credible, each with a contamination + licensing verdict.
   - The **methodology to adopt** (few-shot conventions, sampling, pass@k, error bars, decontam
     protocol) so my numbers are trusted.
   - The **tool-use/TIR evaluation approach**: adopt an existing standard, or the case that we
     must define one — and what it should look like.
   - **Which third-party leaderboards** to target, and eligibility.
   - **Where the gaps are** that no external benchmark covers (I expect engineering + TIR) and
     that we'd therefore have to build and publish ourselves.
4. **Source list** — primary sources, grouped by workstream, each with a one-line note on why it's
   authoritative.

## Ground rules
- **No generic benchmark listicles.** Every benchmark named gets a *credibility verdict*:
  saturated? contaminated? does it separate models at *my* sizes? licensed for our use?
- **Judge relevance at sub-1B–4B scale**, not frontier scale. A benchmark that only separates
  30B+ models is a *negative* finding for me — say so.
- **Prefer 2024–2026 sources** — eval practice moves fast; flag anything likely stale.
- **Quantify** the effect of protocol choices (shots, sampling, harness) on scores wherever
  sources permit, rather than describing them qualitatively.
- **Cite primary sources** (tech reports, papers, harness repos, leaderboard docs). Where a claim
  is contested or a number doesn't reproduce across sources, present the disagreement, don't
  smooth it over.
- Distinguish **established practice** from **your own inference** — label the latter.
