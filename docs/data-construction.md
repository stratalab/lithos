# Lithos Data Construction — Working Reference

Consolidates the data strategy (pre- **and** post-training) for the STEM-domain flagship. Companion to `lithos-implementation-plan.md` (Phases 9–12).

> **Scope / status.** This is a *living working reference*, grounded in the open recipes named in §1.4 and §2 — it captures the consolidated knowledge and a roadmap, not yet the deep per-report extraction (concrete thresholds, exact mixes). That deeper digest — reading each report and pulling its specific filter constants and ablation results — is the future deepening flagged at the end. **All dataset names below need a per-dataset license + provenance check before they touch a keeper run**; sizes are approximate. Two provenance rules, at different layers: **raw pretraining text follows the §1.5 sourcing doctrine** (publicly available = ingestible; leaked/private = never); **generated data is open-teacher / human / self-generated only — nothing with proprietary-model provenance in the chain.**

## 0. Principles (the meta-lessons)

1. **Quality > quantity at fixed compute.** A smaller, cleaner, better-mixed corpus beats a bigger dirty one (FineWeb-Edu / DCLM / Phi). But quality must still produce *enough* volume to hit the over-train budget — hence synthetic.
2. **Everything is validated by ablation** on the 100M rig, decided on **per-domain bits-per-byte** (benchmarks flat-line below ~500M). No filter or mix ships unmeasured (the FineWeb methodology — hundreds of small-proxy ablations).
3. **Decontaminate before trusting any eval.** N-gram match training text vs the *frozen* battery; drop leaks.
4. **Provenance obsessively.** Every artifact gets a manifest: source, filters, dedup stats, mix weights, tokenizer version → reproducible + auditable, or it isn't an asset.
5. **Sovereignty.** The moat is *owning* the chain end to end (same bet as StrataDB). Concretely: for **generated** data, open-teacher / human / self-generated only — no proprietary-model provenance. For **raw** text, ownership means provenance manifests + the §1.5 doctrine, not a self-imposed license handicap the incumbents ignore.
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

### 1.5 Sourcing doctrine (SETTLED)

The decided policy, replacing the earlier permissive-only posture. Rationale: STEM knowledge is humanity's commons; the incumbents ingested everything and we will not run with a self-imposed handicap they ignore. The trade we accept: **enjoinment/legal risk in exchange for sovereignty** — a copyright claim against a training *use* is contested territory; a contract breach or possession of stolen material is not. Principled sourcing is enforced **per-document, not per-source** (the index, §1.7, is the enforcement surface).

**Solid lines (never cross):**
- **Leaked / proprietary / private / hacked material** — anything not made available to the public by its owner. No exceptions.
- **GPT/Claude outputs as training data** — a *provable contract breach* (ToS with a direct counterparty, logged API calls), categorically worse than copyright exposure. We also lose nothing: open reasoners are frontier-class at STEM (§2.2).

**Grey (ingest, with caveats):**
- **Copyrighted-but-published books & papers** (incl. paywalled). The idea/expression dichotomy is the legal spine: we train on the *knowledge*, we must not *reproduce the expression*. Caveats that make this real: aggressive **dedup**, **epoch caps** on any single copyrighted work, and a **regurgitation eval** (prompt with book prefixes, measure verbatim continuation) in the frozen battery.
- **Closed models as build tools** (labeling, judging, curation assistance) — fine, so long as their outputs never become training targets.

**Green (unrestricted):**
- Public domain, open-licensed (CC, ODC-By, permissive code), government/patent text.
- **Distillation of open-weights models** (Apache-2.0 Qwen, MIT DeepSeek/GLM) — the license *invites* it.

**Universal regardless of tier:** secret-scanning (an API key in a repo is private data even if the repo is public), PII redaction, provenance manifests, decontamination.

### 1.6 Supply assessment — sized for 7–13B

We size the corpus as if training a 7–13B model (a few trillion tokens-seen), so the 500M/1B/4B rungs are never data-starved and the ambition has a ceiling we've measured.

- **World supply of high-quality unique STEM text: roughly 1.5–3T tokens reachable**, of which ~1T survives serious quality filtering. By slice: **code** is the largest (The Stack v2 scale — high hundreds of B after filtering); **papers** (arXiv + the published literature) a few hundred B; the **book canon** low hundreds of B; **math is the scarce slice: ~100–200B unique** — the binding constraint.
- **Data-constrained scaling** (Muennighoff et al.): up to **~4 epochs ≈ fresh data**. So ~1T unique high-quality tokens honestly supports ~4T tokens-seen — enough for a 7–13B over-trained run.
- **Consequences:** (a) there *is* enough raw STEM in the world for the full ladder; (b) the scarce-math gap is closed by **verified synthetic** (generate-then-check multiplies the slice that verifiability makes safe to multiply); (c) epoch accounting becomes a first-class manifest column, since we will deliberately multi-epoch the best slices (and epoch-cap the grey ones, §1.5).

### 1.7 Index-first curation — the catalog of intent

**Build the index before acquiring a single byte.** The corpus starts as a *bill of materials*: a table of every work we intend to ingest, so coverage, gaps, licensing, and cost are measurable before acquisition spend — and so per-document sourcing decisions (§1.5) are auditable rather than vibes.

- **Schema (one row per work):** canonical ID (ISBN / DOI / arXiv ID) · title · domain · subfield · level (intro / UG / grad / research) · license tier (green / grey per §1.5) · est. tokens · priority · acquisition route · status.
- **Harvest existing curation instead of curating from scratch:** university syllabi, qualifying-exam reading lists, the per-field "bibles", award lists, review-article bibliographies. Humanity already ranked its STEM canon; we transcribe the ranking.
- **The index is also the enforcement surface:** license tier and epoch cap live as columns, so the §1.5 doctrine is applied mechanically at ingestion, and the regurgitation eval knows exactly which works to probe.

### 1.8 Source inventory — the five mixable slices

| Slice | Candidate sources | Doctrine tier (§1.5) |
|---|---|---|
| Code | The Stack v2, GitHub issues/PRs, notebooks | green + grey (public repos regardless of license); secret-scanned always |
| Math | FineMath, OpenWebMath, Proof-Pile-2/AlgebraicStack, arXiv math, **the math book canon** | green + grey (published books epoch-capped) |
| Physics + Eng | arXiv physics/cond-mat/eng, **Stack Exchange** Q&A (CC-BY-SA), OpenStax/LibreTexts (CC), USPTO patents (public domain), **the physics/eng book canon** | green + grey (published books epoch-capped) |
| General glue (~15%) | FineWeb-Edu | green (ODC-By) |
| Verified synthetic | generated-and-checked solutions / reasoning traces (open teacher) | green (self-owned, teacher disclosed) |

### 1.9 What we've built vs net-new

- **Built:** heuristic-filter seam, MinHash near-dedup, 13-gram decontam, quality-score thresholding, held-out holdout, ablation harness, manifests, general 32k tokenizer, packing, sharding, R2 storage.
- **Net-new:** **seed index / catalog of intent (§1.7 — the next artifact)**, multi-source ingestion + LaTeX/PDF/code extraction, domain tagging, self-trained quality classifier, weighted-mix spec, per-domain bpb sets, verified synthetic, annealing set, STEM tokenizer retrain, **regurgitation eval + epoch-cap accounting (§1.5)**.

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

**Open-reasoner distillation teachers (the verified-synthesis generator engine).** Distill *open, permissively-licensed* reasoners only — **never GPT/Claude** (their ToS forbids using outputs to train competing models: a provable contract breach, categorically worse than the books' copyright question — see §2.3). Roster by slice:
- **Code (lead): GLM-5.2** (Z.ai, open weights, MoE ~744B/40B-active) — current best *open* coding model: SWE-Bench Pro 62.1 (SoTA-open), Terminal-Bench 2.1 81.0, #2 frontend. Plus **Qwen2.5-Coder-32B**, **DeepSeek-V3**.
- **Math:** **Qwen2.5-Math-72B**. **Reasoning / TIR traces:** **DeepSeek-R1**, **QwQ-32B**, **Qwen3** (thinking).
- Always **distillation + rejection sampling**, not blind SFT-on-teacher-text: generate → **sandbox-verify** → keep only correct → tokenize with our vocab → own it (strips the teacher's hallucinated derivations; beats trusting raw traces). Capacity-gated to the flagship (4B hero / 500M) per the Phase-11 lesson that distillation transfers *style not substance* on a tiny student.

**Preference (DPO):** Tülu 3 preference mix, **HH-RLHF** (human), SHP, PKU-SafeRLHF — but prefer **on-policy verifier-labeled** STEM pairs.

**RLVR verifiable problems (the superpower):**
- Math: **GSM8K, MATH, NuminaMath** (ground-truth answers).
- Code: **MBPP, HumanEval+, APPS, CodeContests, TACO, LiveCodeBench** (unit tests).
- "Dataset" = (problem, checker). **Overlaps almost entirely with the executable eval harness** — build the verifier once, use it for eval *and* RLVR *and* preference labeling.
- **Reward-hacking defense (lifted from GLM-5.2's coding RL).** In TIR/agentic RLVR the **sandbox *is* the reward surface**, so the policy *will* try to game it (our banked Goodhart lesson, restated). Mitigation: an **LLM judge inspects each tool call for suspicious patterns** — hard-coding the expected answer, trivial/no-op calls, prompt-injecting or short-circuiting the verifier — and **returns dummy data instead of reward on suspicion**, which kills the exploit while keeping training stable. Pair with the standing rule: **watch real rollouts, not the reward curve**.

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

- Frozen versioned battery (general) + **executable STEM battery** (HumanEval/MBPP/GSM8K/MATH) + **cross-domain transfer probe** (derive-then-implement) + **regurgitation eval** (verbatim-continuation probe over indexed grey-tier works — the §1.5 caveat, made measurable).
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
