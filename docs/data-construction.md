# Lithos Data Construction — Working Reference

Consolidates the data strategy (pre- **and** post-training) for the STEM-domain flagship. Companion to `lithos-implementation-plan.md` (Phases 9–12).

> **Scope / status.** This is a *living working reference*, grounded in the open recipes named in §1.4 and §2 — it captures the consolidated knowledge and a roadmap, not yet the deep per-report extraction (concrete thresholds, exact mixes). That deeper digest — reading each report and pulling its specific filter constants and ablation results — is the future deepening flagged at the end. **All dataset names below need a per-dataset license + provenance check before they touch a keeper run**; sizes are approximate. Sovereignty rule throughout: **open-teacher / human / self-generated only — nothing with proprietary-model provenance in the chain.**

## 0. Principles (the meta-lessons)

1. **Quality > quantity at fixed compute.** A smaller, cleaner, better-mixed corpus beats a bigger dirty one (FineWeb-Edu / DCLM / Phi). But quality must still produce *enough* volume to hit the over-train budget — hence synthetic.
2. **Everything is validated by ablation** on the 100M rig, decided on **per-domain bits-per-byte** (benchmarks flat-line below ~500M). No filter or mix ships unmeasured (the FineWeb methodology — hundreds of small-proxy ablations).
3. **Decontaminate before trusting any eval.** N-gram match training text vs the *frozen* battery; drop leaks.
4. **Provenance obsessively.** Every artifact gets a manifest: source, filters, dedup stats, mix weights, tokenizer version → reproducible + auditable, or it isn't an asset.
5. **Sovereignty.** Open-teacher / human / self-generated data only. The moat is *owning* the chain end to end (same bet as StrataDB).
6. **Verifiability is the unlock.** In code/math/physics, code runs, math checks, units balance — which makes quality filtering *executable*, synthetic *safe-and-checkable*, evals *trustworthy*, and RLVR *possible*. This single property re-pays across the whole pipeline.

---

## Part 1 — Pretraining corpus

### 1.1 The mental model: a funnel + assembly

- **Funnel:** start with a huge dirty pool → clean → dedup → decontaminate → quality-rank down to a much smaller, much better corpus.
- **Assembly:** for a *domain* corpus, gather many curated sources and blend them in deliberate proportions. Ours is mostly curated (less raw web-crawl pain, higher signal).

### 1.2 Pipeline stages

| Stage | What it does | Tools / technique | Us |
|---|---|---|---|
| 1. Acquire | source raw bytes | Common Crawl (WARC) for web; curated repos for the rest | **net-new**: multi-source ingestion |
| 2. Extract | format → clean text | HTML: `trafilatura`/`resiliparse`; **PDF/LaTeX**: arXiv source > PDF, else `Nougat`/GROBID/Marker; code: strip binaries/generated | **net-new**: LaTeX/PDF + code extraction |
| 3. Heuristic filter | drop obvious junk | C4 + Gopher rules (length, symbol/word ratio, repetition, boilerplate, stopwords) | seam built (`data/pipeline` filters) |
| 4. Model-quality | the biggest lever | LLM labels a sample → distill into a cheap classifier → score whole corpus → threshold (FineWeb-Edu / DCLM) | thresholding carried edu-score ✅; **net-new**: train our own for unscored sources |
| 5. Dedup | remove duplicates | exact (hash) + **near (MinHash/LSH)**. Scope matters: FineWeb dedups *per-snapshot*, not globally (global can hurt) | MinHash ✅ (`data/minhash.py`) |
| 6. Decontaminate | strip benchmark leaks | 13-gram match vs frozen battery | ✅ (`data/decontam.py`) |
| 7. PII / license | redact secrets, license-comply | secret-scanning (critical for **code** — API keys); permissive-only, honor opt-outs | **net-new** for code/STEM |
| 8. Domain tag + mix | control the blend | tag by domain; choose mix weights (DoReMi / data-mixing-laws / empirical sweep) | **net-new**: weighted-mix spec + tagging |
| 9. Curriculum / anneal | order matters | bulk mix → **final cooldown phase on a small, very-high-quality set** as LR→0 (Llama-3, OLMo-2, MiniCPM) — cheap, high-impact | **net-new**: annealing set |
| 10. Synthetic | multiply quality tokens | rephrase (WRAP), textbook-gen (Cosmopedia/Phi); **STEM: generate-then-verify** | **net-new**: verified synthetic (un-deferred) |
| 11. Tokenize/pack/shard | finalize | train tokenizer, pack sequences, shard + manifest | ✅ (general 32k); **net-new**: STEM tokenizer retrain |

### 1.3 STEM-specific additions

- **LaTeX/PDF extraction** for arXiv physics/math/eng (math-aware; arXiv source beats PDF OCR).
- **Code filtering**: language detection, filter generated/vendored/minified, secret-scanning, license filtering, near-dedup at file + repo level.
- **Executability/verification** as a quality signal: does the code parse/run? does the solution check out? (signal the general edu-classifier can't give).
- **Over-weight the intersections** — physics-via-code, math-as-proof-and-program (Jupyter notebooks, papers-with-code, scientific-computing repos): that's where *transfer* is taught.

### 1.4 The open recipes to read (steal from each)

- **FineWeb / FineWeb-Edu** (HuggingFace) — the gold standard; documents every ablation, the edu-classifier recipe, per-snapshot dedup.
- **DCLM** (DataComp-LM) — model-based (fastText) filtering ≫ heuristic; controlled filtering benchmark.
- **Dolma** (AI2) — open 3T corpus + toolkit; full documented pipeline.
- **RefinedWeb** (Falcon) — web-only can rival curated with aggressive filtering+dedup.
- **The Stack v2 / StarCoder2** (BigCode) — code sourcing, license filtering, secret-scanning, dedup.
- **Nemotron-CC** (NVIDIA) — classifier ensemble + synthetic rephrasing at scale.
- **STEM**: OpenWebMath, Proof-Pile-2 (Llemma), FineMath, MathPile, DeepSeekMath, Qwen2.5-Coder/Math data sections.

### 1.5 Source inventory — the five mixable slices

| Slice | Candidate sources | License posture |
|---|---|---|
| Code | The Stack v2, GitHub issues/PRs, notebooks | permissive-only, opt-outs honored, secret-scanned |
| Math | FineMath, OpenWebMath, Proof-Pile-2/AlgebraicStack, arXiv math | mostly open; check arXiv terms |
| Physics + Eng | arXiv physics/cond-mat/eng, **Stack Exchange** Q&A (CC-BY-SA), OpenStax/LibreTexts (CC), USPTO patents (public domain) | mostly CC / public domain |
| General glue (~15%) | FineWeb-Edu | ODC-By |
| Verified synthetic | generated-and-checked solutions / reasoning traces (open teacher) | self-owned, teacher disclosed |

### 1.6 What we've built vs net-new

- **Built:** heuristic-filter seam, MinHash near-dedup, 13-gram decontam, quality-score thresholding, held-out holdout, ablation harness, manifests, general 32k tokenizer, packing, sharding, R2 storage.
- **Net-new:** multi-source ingestion + LaTeX/PDF/code extraction, domain tagging, self-trained quality classifier, weighted-mix spec, per-domain bpb sets, verified synthetic, annealing set, STEM tokenizer retrain.

---

## Part 2 — Post-training datasets

Post-training is **not one dataset** — it's a different *kind* of data per stage. For a verifiable STEM model the most valuable kind isn't human-labeled; it's **verifiable problems + a checker**.

### 2.1 The stages and what each eats

1. **SFT (instruction tuning)** — base → instruction-follower. Data = (prompt → response) / multi-turn, chat-templated, loss-masked on non-assistant tokens. Biggest single base→assistant jump.
2. **RLVR (RL with verifiable rewards)** — the domain-native stage. Data = **problems with checkers** (math answers, code tests); RL generates its own rollouts (GRPO). No reward model, no human labels. *Ahead of DPO in value for us.*
3. **Preference / DPO** — (prompt, chosen, rejected) triples. DPO over PPO for a solo team. STEM pairs best generated **on-policy** (verifier: correct=chosen).
4. **Distillation** — open teacher generates SFT/reasoning data (Phase 12 Track B).

### 2.2 Dataset inventory

**General SFT backbone** (chat/format/follow):
- **Tülu 3 SFT mix** (AI2) — current open gold standard, curated + decontaminated. Best start.
- Human-written, fully clean: **FLAN**, **Dolly**, **OASST**, **Natural-Instructions**.
- The **LIMA** lesson: ~1k *excellent* examples does most of the SFT work.

**STEM SFT (the core):**
| Domain | Datasets (open-teacher / self-aligned) |
|---|---|
| Code | **StarCoder2-Instruct** (self-aligned, no proprietary teacher), **Magicoder/OSS-Instruct**, Code-Feedback |
| Math | **OpenMathInstruct-2** (Mixtral/Llama-gen), **NuminaMath** (AIMO-winning CoT), MetaMathQA, Orca-Math |
| Science/physics | **Camel-AI** (physics/chem), SciInstruct |
| Reasoning traces | **OpenR1-Math**, **OpenThoughts/2**, Bespoke-Stratos, Sky-T1 — long CoT distilled from *open* R1/QwQ |

**Preference (DPO):** Tülu 3 preference mix, **HH-RLHF** (human), SHP, PKU-SafeRLHF — but prefer **on-policy verifier-labeled** STEM pairs.

**RLVR verifiable problems (the superpower):**
- Math: **GSM8K, MATH, NuminaMath** (ground-truth answers).
- Code: **MBPP, HumanEval+, APPS, CodeContests, TACO, LiveCodeBench** (unit tests).
- "Dataset" = (problem, checker). **Overlaps almost entirely with the executable eval harness** — build the verifier once, use it for eval *and* RLVR *and* preference labeling.

### 2.3 The sovereignty filter

- **Prefer:** human-written (OASST, Dolly, HH-RLHF, FLAN), open-teacher-generated (Tülu 3, OpenMathInstruct, OpenR1), **self-generated + verifier-filtered** (the gold path).
- **Avoid:** GPT-4-generated sets — **OpenHermes, UltraFeedback, Alpaca, WizardLM, ShareGPT** — OpenAI-ToS / provenance taint vs the sovereignty thesis.
- ⚠️ "Open" varies (research-only, NC clauses, GPT taint). Per-dataset license check before any keeper run.

### 2.4 Minimal viable set + sequencing (500M flagship)

1. **Must-have — SFT:** small clean general backbone (curated Tülu 3 / LIMA-sized) **+** STEM bulk (OpenMathInstruct-2, StarCoder2-Instruct, a physics slice, reasoning traces). ~90% of the perceived quality jump.
2. **High-value — RLVR:** verifiable math+code problems + our checker. For a reasoning model, > DPO. Built on the eval verifier.
3. **Optional/second — DPO:** start from on-policy verifier-labeled pairs, not bought data.

### 2.5 The self-generation engine (most sovereign, domain-perfect)

In a verifiable domain we can **generate most of our own post-training data**: problems → solutions/traces → **keep only what the checker passes** (rejection sampling on correctness). Same generate-then-verify machinery as pretraining synthetic; fully sovereign (no proprietary teacher in provenance); reuses the eval verifier. **The open datasets are the seed and the eval; our own verified generation is the engine.**

---

## Part 3 — Eval data (cross-ref Phase 9)

- Frozen versioned battery (general) + **executable STEM battery** (HumanEval/MBPP/GSM8K/MATH) + **cross-domain transfer probe** (derive-then-implement).
- **Per-domain bpb held-out sets** (code/math/physics/eng/general) — the mix-sweep's decision metric.
- Decontamination against all of the above; Qwen-0.5B/1.5B reference anchors committed.

## Open questions we navigate empirically (not solve)

- Optimal mixing for a **transfer** objective (most mixing work optimizes loss, not transfer).
- Synthetic : real ratio; annealing-set composition for STEM.
- Whether small-scale-optimal mix holds at scale → carry **top-2** up each rung; fit a data-mixing/scaling law.

## Pointers

- Plan: `lithos-implementation-plan.md` — Phase 9 (eval), Phase 10 (corpus + mix machinery), Phase 11 (post-training), Phase 12 (mix-sweep → 500M → 1B).
- Code: `lithos/data/{pipeline,minhash,decontam,quality,documents}.py`, `lithos/evals/{benchmarks,scorecard,ablation}.py`.
- **Future deepening:** read each §1.4 report and extract its concrete filter constants, dedup scoping, and ablation deltas into this doc (tagged adopt / adapt / skip).
